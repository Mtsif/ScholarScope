
import json
import io
import math
from PIL import Image, ImageDraw, ImageFont
import plotly.express as px
import plotly.graph_objects as go
import hashlib
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

TIMEOUT = 18
PAGE_SIZE = 100
MAX_PUBLICATION_PAGES = 50


def get_id(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError('Enter a Google Scholar profile URL or Scholar user ID.')
    if '://' in value:
        parsed = urlparse(value)
        if parsed.hostname not in ('scholar.google.com', 'scholar.google.co.uk') and not (
            parsed.hostname and parsed.hostname.startswith('scholar.google.')
        ):
            raise ValueError('Please use a Google Scholar citations profile URL.')
        value = parse_qs(parsed.query).get('user', [''])[0]
    if not re.fullmatch(r'[A-Za-z0-9_-]{5,32}', value):
        raise ValueError('Could not find a valid Google Scholar user ID.')
    return value


def read_uploaded_collection(uploaded) -> dict:
    """Validate an uploaded personal collection; never write it to server disk."""
    try:
        raw = json.loads(uploaded.getvalue().decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f'Invalid JSON file: {exc}') from exc

    # Accept the old local scholar_profiles.json format (Scholar ID -> profile).
    # Also accept an explicit wrapper {"researchers": {ID: profile}}.
    if isinstance(raw, dict) and 'researchers' in raw:
        raw = raw['researchers']
    if not isinstance(raw, dict):
        raise ValueError('Expected a JSON object mapping Google Scholar IDs to profiles.')
    cleaned = {}
    for key, profile in raw.items():
        if not isinstance(profile, dict):
            raise ValueError('Each researcher must have a JSON object of metrics.')
        try:
            sid = get_id(str(profile.get('user_id') or key))
        except ValueError as exc:
            raise ValueError(f'Invalid researcher ID {key!r}: {exc}') from exc
        cleaned[sid] = {**profile, 'user_id': sid,
                        'profile_url': f'https://scholar.google.com/citations?user={sid}&hl=en'}
    return cleaned


def export_collection(profiles: dict) -> bytes:
    return json.dumps(profiles, ensure_ascii=False, indent=2).encode('utf-8')


def number(text: str):
    match = re.search(r'\d[\d,\s.]*', text.replace('\xa0', ' '))
    if not match:
        return None
    digits = re.sub(r'\D', '', match.group())
    return int(digits) if digits else None


def scholar_page(user_id: str, *, start: int = 0, pagesize: int | None = None) -> BeautifulSoup:
    params = {'user': user_id, 'hl': 'en'}
    if pagesize is not None:
        params.update({'cstart': start, 'pagesize': pagesize})
    response = requests.get(
        'https://scholar.google.com/citations',
        params=params,
        headers={'User-Agent': 'Mozilla/5.0', 'Accept-Language': 'en-US,en;q=0.9'},
        timeout=TIMEOUT,
    )
    if response.status_code in (403, 429, 503):
        raise RuntimeError(f'Google Scholar refused the request (HTTP {response.status_code}). Try again later.')
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    if not soup.select_one('#gsc_prf_in'):
        raise RuntimeError('Google Scholar did not return a researcher profile (possibly a CAPTCHA or access restriction).')
    return soup


def count_publications(user_id: str, first_soup: BeautifulSoup) -> int:
    """Count publication rows across Scholar profile pages, not just the first 20."""
    first_rows = first_soup.select('#gsc_a_b tr.gsc_a_tr')
    # The first page typically contains 20 entries. Request pages of 100 to count all.
    if len(first_rows) < 20:
        return len(first_rows)

    total = 0
    seen_pages = set()
    for page_index in range(MAX_PUBLICATION_PAGES):
        if page_index:
            time.sleep(1.5)  # Avoid rapid-fire Scholar requests.
        soup = scholar_page(user_id, start=page_index * PAGE_SIZE, pagesize=PAGE_SIZE)
        rows = soup.select('#gsc_a_b tr.gsc_a_tr')
        if not rows:
            return total
        # Detect a silently repeated page rather than double-counting.
        page_keys = tuple((row.select_one('.gsc_a_at').get_text(' ', strip=True)
                           if row.select_one('.gsc_a_at') else row.get_text(' ', strip=True))
                          for row in rows)
        if page_keys in seen_pages:
            raise RuntimeError('Google Scholar repeated a publication page; count could not be verified.')
        seen_pages.add(page_keys)
        total += len(rows)
        if len(rows) < PAGE_SIZE:
            return total
    raise RuntimeError('Publication list exceeds the configured page limit; no partial count was saved.')


def fetch_profile(user_id: str) -> dict:
    soup = scholar_page(user_id)
    name_node = soup.select_one('#gsc_prf_in')
    rows = soup.select('#gsc_rsb_st tbody tr')
    metrics = {}
    for row in rows:
        cells = row.select('td')
        heading = row.find('th')
        if heading and len(cells) >= 2:
            label = heading.get_text(' ', strip=True).lower()
            metrics[label] = {'all': number(cells[0].get_text(' ', strip=True)),
                              'recent': number(cells[1].get_text(' ', strip=True))}
    if not metrics:
        for row in rows:
            cells = row.find_all('td', recursive=False)
            if len(cells) >= 3:
                label = cells[0].get_text(' ', strip=True).lower()
                metrics[label] = {'all': number(cells[1].get_text(' ', strip=True)),
                                  'recent': number(cells[2].get_text(' ', strip=True))}
    def pick(prefix: str):
        return next((v for k, v in metrics.items() if k.startswith(prefix)), {'all': None, 'recent': None})
    citations = pick('citations')
    h_index = pick('h-index')
    if any(x['all'] is None for x in (citations, h_index)):
        raise RuntimeError('Citation metrics could not be parsed. The Google Scholar page layout may have changed.')

    years = [number(e.get_text(' ', strip=True)) for e in soup.select('.gsc_g_t')]
    yearly_counts = [number(e.get_text(' ', strip=True)) for e in soup.select('.gsc_g_al')]
    annual = {str(y): c for y, c in zip(years, yearly_counts) if y is not None and c is not None}
    aff = soup.select_one('.gsc_prf_il')
    return {
        'user_id': user_id,
        'name': name_node.get_text(' ', strip=True),
        'affiliation': aff.get_text(' ', strip=True) if aff else '',
        'citations': citations['all'],
        'citations_recent': citations['recent'],
        'h_index': h_index['all'],
        'h_index_recent': h_index['recent'],
        'publications': count_publications(user_id, soup),
        'annual_citations': annual,
        'profile_url': f'https://scholar.google.com/citations?user={user_id}&hl=en',
        'fetched_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'source': 'Google Scholar public profile',
    }


# --- Presentation helpers --------------------------------------------------
NAVY = '#14253D'
BLUE = '#3277DB'
GOLD = '#DFAB4E'


def numeric(value):
    try:
        return int(value) if value is not None and not pd.isna(value) else None
    except (ValueError, TypeError):
        return None


def formatted(value):
    n = numeric(value)
    return f'{n:,}' if n is not None else '—'


def ui_chart(fig, height=390):
    fig.update_layout(
        height=height, template='plotly_white',
        font=dict(family='Arial, sans-serif', color=NAVY, size=13),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=15, r=20, t=30, b=30),
        hoverlabel=dict(bgcolor=NAVY, font_color='white'),
        showlegend=False,
    )
    fig.update_xaxes(showline=False, gridcolor='#E9EEF5', zeroline=False)
    fig.update_yaxes(showline=False, gridcolor='#E9EEF5', zeroline=False)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def badge_font(size, bold=False):
    candidates = (
        ['/System/Library/Fonts/Supplemental/Arial Bold.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf']
        if bold else
        ['/System/Library/Fonts/Supplemental/Arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def make_badge(profile):
    """Compact, name-free, CV-ready metrics PNG (1080 × 190)."""
    w, h = 1080, 190
    im = Image.new('RGB', (w, h), '#FFFFFF')
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((3, 3, w-4, h-4), radius=22,
                        fill='#FFFFFF', outline='#DAE4F0', width=3)
    # Original ScholarScope cap icon (not Google's official logo).
    d.rounded_rectangle((22, 27, 154, 163), radius=20, fill='#EAF2FF')
    d.polygon([(44, 82), (88, 60), (133, 82), (88, 106)], fill=BLUE)
    d.line([(63, 103), (63, 122), (88, 137), (113, 122), (113, 103)], fill=BLUE, width=7)
    d.line([(127, 87), (127, 124)], fill=BLUE, width=5)
    d.ellipse((121, 123, 133, 135), fill=GOLD)
    d.text((173, 32), 'GOOGLE', font=badge_font(22, True), fill=BLUE)
    d.text((173, 61), 'SCHOLAR', font=badge_font(22, True), fill=NAVY)
    metrics = [('CITATIONS', profile.get('citations')),
               ('H-INDEX', profile.get('h_index')),
               ('PUBLICATIONS', profile.get('publications'))]
    starts = (366, 601, 837)
    for idx, ((label, value), x) in enumerate(zip(metrics, starts)):
        if idx:
            d.line([(x-21, 36), (x-21, 154)], fill='#DFE6F0', width=2)
        d.text((x, 45), label, font=badge_font(19, True), fill='#65758B')
        d.text((x, 81), formatted(value), font=badge_font(44, True), fill=NAVY)
    buf = io.BytesIO()
    im.save(buf, format='PNG', optimize=True)
    return buf.getvalue()


st.set_page_config(page_title='ScholarScope | Researcher Analytics', page_icon='🎓', layout='wide')
st.markdown("""
<style>
.block-container {max-width: 1480px; padding-top: 2.2rem;}
h1, h2, h3 {letter-spacing: -0.035em;}
div[data-testid="stMetric"] {border:1px solid #e1e8f2; border-radius:17px; padding:18px 20px; background:#fafcff;}
div[data-testid="stMetricLabel"] {font-weight:600;}
</style>
""", unsafe_allow_html=True)
# Brand icon rendered in-memory so deployment needs only app.py.
logo = Image.new('RGBA', (112, 112), (0, 0, 0, 0))
ld = ImageDraw.Draw(logo)
ld.rounded_rectangle((2, 2, 110, 110), radius=27, fill=BLUE)
ld.polygon([(21, 47), (56, 28), (92, 47), (56, 67)], fill='white')
ld.line([(35, 60), (35, 78), (56, 90), (77, 78), (77, 60)], fill='white', width=6)
ld.line([(88, 51), (88, 82)], fill=GOLD, width=5)
ld.ellipse((83, 81, 93, 91), fill=GOLD)
brand_icon, brand_title = st.columns([0.075, 0.925], gap='small')
with brand_icon:
    st.image(logo, width=60)
with brand_title:
    st.title('ScholarScope')
st.caption('Researcher analytics · Google Scholar public-profile metrics')

if 'profiles' not in st.session_state:
    # Load the example collection once per browser session, without sharing edits.
    sample_path = Path(__file__).with_name('toy.json')
    if sample_path.exists():
        try:
            class _LocalJSON:
                def __init__(self, content):
                    self.content = content

                def getvalue(self):
                    return self.content

            st.session_state.profiles = read_uploaded_collection(
                _LocalJSON(sample_path.read_bytes())
            )
        except (OSError, ValueError) as exc:
            st.warning(f'Could not load toy.json: {exc}')
            st.session_state.profiles = {}
    else:
        st.session_state.profiles = {}
if 'loaded_upload_hash' not in st.session_state:
    st.session_state.loaded_upload_hash = None
if 'dirty' not in st.session_state:
    st.session_state.dirty = False
profiles = st.session_state.profiles

with st.sidebar:
    st.header('Your researchers')
    uploaded = st.file_uploader('Import collection (JSON)', type=['json'], key='import_json')
    if uploaded is not None:
        digest = hashlib.sha256(uploaded.getvalue()).hexdigest()
        if digest != st.session_state.loaded_upload_hash:
            try:
                imported = read_uploaded_collection(uploaded)
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.session_state.profiles = imported
                profiles = st.session_state.profiles
                st.session_state.loaded_upload_hash = digest
                st.session_state.dirty = False
                st.success(f'Imported {len(imported)} researchers.')
    st.divider()
    profile_input = st.text_input('Google Scholar profile URL or ID', placeholder='Scholar profile link or ID')
    if st.button('Add / update researcher', type='primary', use_container_width=True):
        try:
            sid = get_id(profile_input)
            with st.spinner('Retrieving Google Scholar metrics…'):
                profile = fetch_profile(sid)
            profiles[sid] = profile
            st.session_state.dirty = True
            st.success(f"Updated {profile['name']}")
        except (ValueError, RuntimeError, requests.RequestException) as exc:
            st.error(str(exc))
    if profiles and st.button('Refresh all researchers', use_container_width=True):
        updated, errors = 0, []
        progress = st.progress(0, text='Refreshing profiles…')
        for i, sid in enumerate(list(profiles)):
            if i:
                time.sleep(2)
            try:
                profiles[sid] = fetch_profile(sid)
                updated += 1
                st.session_state.dirty = True
            except (RuntimeError, requests.RequestException) as exc:
                errors.append(f"{profiles[sid].get('name', sid)}: {exc}")
            progress.progress((i + 1) / len(profiles), text=f'{i+1}/{len(profiles)} processed')
        st.success(f'Refreshed {updated}/{len(profiles)} profiles.')
        for error in errors:
            st.warning(error + ' Previous metrics retained.')
    if profiles:
        st.divider()
        selected = st.selectbox('Manage profile', list(profiles), format_func=lambda sid: profiles[sid].get('name', sid))
        st.link_button('Open Google Scholar ↗', profiles[selected]['profile_url'], use_container_width=True)
        if st.button('Remove selected researcher', use_container_width=True):
            profiles.pop(selected)
            st.session_state.dirty = True
            st.rerun()
        if st.button('Clear collection', use_container_width=True):
            st.session_state.profiles = {}
            st.session_state.dirty = True
            if uploaded is not None:
                st.session_state.loaded_upload_hash = hashlib.sha256(uploaded.getvalue()).hexdigest()
            st.rerun()
    st.divider()
    st.download_button('⬇ Download researcher collection',
                       data=export_collection(profiles), file_name='scholar_profiles.json',
                       mime='application/json', use_container_width=True,
                       on_click=lambda: st.session_state.update(dirty=False))
    if st.session_state.dirty:
        st.warning('Unsaved changes — download your updated JSON before leaving.')

main_tab, profile_tab = st.tabs(['Dashboard', 'Researcher profile'])

with main_tab:
    if not profiles:
        st.info('Add a researcher or import a saved collection to start comparing metrics.')
    else:
        data = pd.DataFrame([
            {'Researcher': p.get('name', sid), 'Affiliation': p.get('affiliation', ''),
             'Citations': numeric(p.get('citations')), 'h-index': numeric(p.get('h_index')),
             'Publications': numeric(p.get('publications')),
             'Last refreshed (UTC)': p.get('fetched_at', '')}
            for sid, p in profiles.items()
        ])
        st.subheader('Researcher overview')
        st.dataframe(data.sort_values('Citations', ascending=False, na_position='last'),
                     hide_index=True, use_container_width=True,
                     column_config={'Citations': st.column_config.NumberColumn(format='%d'),
                                    'Publications': st.column_config.NumberColumn(format='%d')})
        st.download_button('⬇ Export metrics CSV', data.to_csv(index=False).encode('utf-8'),
                           'scholar_metrics.csv', 'text/csv')

        st.subheader('Explore the metrics')
        metric = st.selectbox('Compare researchers by', ['Citations', 'h-index', 'Publications'])
        chart_data = data[['Researcher', metric, 'Affiliation', 'Citations', 'Publications', 'h-index']].copy()
        chart_data = chart_data.loc[:, ~chart_data.columns.duplicated()].dropna(subset=[metric]).sort_values(metric)
        if not chart_data.empty:
            fig = px.bar(chart_data, x=metric, y='Researcher', orientation='h',
                         custom_data=['Affiliation', 'Citations', 'Publications', 'h-index'],
                         color_discrete_sequence=[BLUE])
            fig.update_traces(marker_line_width=0, opacity=.93,
                              hovertemplate='<b>%{y}</b><br>'+metric+': %{x:,}<br>%{customdata[0]}<extra></extra>')
            fig.update_yaxes(title=None)
            fig.update_xaxes(title=None)
            ui_chart(fig, height=max(350, min(570, 95 + 49 * len(chart_data))))

        st.markdown('#### Citation impact vs. publication output')
        scatter = data.dropna(subset=['Citations', 'Publications']).copy()
        if not scatter.empty:
            scatter['h-index'] = pd.to_numeric(scatter['h-index'], errors='coerce').fillna(0)
            fig = px.scatter(scatter, x='Publications', y='Citations', text='Researcher',
                             hover_name='Researcher', hover_data=['Affiliation','h-index'],
                             size='h-index', size_max=25,
                             color_discrete_sequence=[BLUE])
            fig.update_traces(textposition='top center', marker=dict(opacity=.82, line=dict(width=1.5,color='white')))
            fig.update_xaxes(rangemode='tozero')
            fig.update_yaxes(rangemode='tozero')
            ui_chart(fig, height=465)
            st.caption('Each dot represents one researcher; dot size indicates h-index.')


with profile_tab:
    if not profiles:
        st.info('Add a researcher or import a saved collection to view their profile.')
    else:
        st.subheader('Researcher profile')
        chosen = st.selectbox('Select a researcher', list(profiles),
                              format_func=lambda sid: profiles[sid].get('name', sid), key='researcher_detail')
        profile = profiles[chosen]
        st.markdown(f"### {profile.get('name', chosen)}")
        if profile.get('affiliation'):
            st.caption(profile['affiliation'])
        c1, c2, c3 = st.columns(3)
        c1.metric('Citations', formatted(profile.get('citations')))
        c2.metric('h-index', formatted(profile.get('h_index')))
        c3.metric('Publications', formatted(profile.get('publications')))
        annual = profile.get('annual_citations') or {}
        if annual:
            annual_data = pd.DataFrame([{'Year': int(y), 'Citations': int(v)}
                                        for y, v in annual.items()]).sort_values('Year')
            fig = px.area(annual_data, x='Year', y='Citations', markers=True,
                          color_discrete_sequence=[BLUE])
            fig.update_traces(hovertemplate='%{x}: <b>%{y:,} citations</b><extra></extra>')
            fig.update_xaxes(dtick=1)
            ui_chart(fig, height=325)
        else:
            st.info('Annual citation history is not available for this profile.')


        st.markdown('#### Google Scholar metrics badge')
        badge = make_badge(profile)
        st.image(badge, width=680)
        st.download_button('Download PNG badge', data=badge,
                           file_name=f"scholar_badge_{chosen}.png", mime='image/png')
        st.caption(f"Last updated: {profile.get('fetched_at', 'Unknown')}")

with st.sidebar.expander('About ScholarScope'):
    st.write('Compare public Google Scholar researcher metrics and export a compact citation badge for an academic website or CV.')
    st.write('Import a saved JSON collection to continue your work, and download the updated collection when finished.')
