"""Interactive FocoNet focal-mechanism dashboard."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from matplotlib.patches import Circle
from obspy.imaging.beachball import beach

import config
from inference import (
    default_station_row,
    event_ids,
    load_dataset,
    load_model,
    parse_event,
    predict,
    predict_all_events,
)
from map_view import (
    load_catalog,
    load_stations,
    lookup_event_coords,
    match_predictions_to_catalog,
    plot_xq_map,
)

st.set_page_config(
    page_title='FocoNet Dashboard',
    page_icon='◉',
    layout='wide',
    initial_sidebar_state='expanded',
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
  font-family: 'IBM Plex Sans', sans-serif;
}
.stApp {
  background:
    radial-gradient(ellipse 80% 50% at 10% 0%, rgba(14, 90, 92, 0.18), transparent 55%),
    radial-gradient(ellipse 60% 40% at 90% 10%, rgba(180, 110, 40, 0.10), transparent 50%),
    linear-gradient(180deg, #0f1418 0%, #151c22 45%, #10161b 100%);
  color: #e6ecef;
}
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #121a1f 0%, #0d1317 100%);
  border-right: 1px solid rgba(140, 170, 175, 0.18);
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
  font-family: 'IBM Plex Sans', sans-serif;
  letter-spacing: 0.02em;
}
div[data-testid="stMetricValue"] {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 1.35rem;
}
.hero-title {
  font-size: 1.85rem;
  font-weight: 600;
  margin: 0;
  color: #f2f6f7;
}
.hero-sub {
  color: #9bb0b6;
  margin-top: 0.2rem;
  margin-bottom: 1.2rem;
  font-size: 0.95rem;
}
.panel {
  background: rgba(22, 32, 38, 0.78);
  border: 1px solid rgba(140, 170, 175, 0.16);
  border-radius: 10px;
  padding: 1rem 1.1rem;
}
.mono {
  font-family: 'IBM Plex Mono', monospace;
}
.stButton > button {
  border-radius: 6px;
  border: 1px solid rgba(140, 170, 175, 0.35);
  background: rgba(20, 48, 52, 0.85);
  color: #e6ecef;
}
.stButton > button:hover {
  border-color: #2ea8a4;
  color: #fff;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner='Loading FocoNet weights…')
def cached_model():
    return load_model()


@st.cache_resource(show_spinner='Loading XQ dataset…')
def cached_dataset():
    return load_dataset()


@st.cache_resource(show_spinner='Loading ISC catalog…')
def cached_catalog():
    return load_catalog()


@st.cache_resource(show_spinner='Loading XQ inventory…')
def cached_stations():
    return load_stations()


@st.cache_resource(show_spinner='Predicting all XQ events for map…')
def cached_foco_map_df(_model):
    import pandas as pd

    rows = predict_all_events(dataset=cached_dataset(), model=_model)
    pred = pd.DataFrame(rows)
    return match_predictions_to_catalog(pred, cached_catalog())


def init_event_state(eid: str, dataset: dict) -> None:
    parsed = parse_event(dataset[eid])
    st.session_state.event_id = eid
    st.session_state.station_ids = list(parsed['station_ids'])
    st.session_state.xyz = parsed['xyz'].copy()
    st.session_state.features = parsed['features'].copy()
    st.session_state.focalmech = parsed['focalmech'].copy()
    st.session_state.mag = parsed['mag']
    st.session_state.baseline = {
        'station_ids': list(parsed['station_ids']),
        'xyz': parsed['xyz'].copy(),
        'features': parsed['features'].copy(),
    }
    st.session_state.selected_station = 0
    st.session_state.bulk_xyz_offset = np.zeros(3, dtype=np.float32)
    st.session_state.bulk_feat_offset = np.zeros(11, dtype=np.float32)


def ensure_state(dataset: dict) -> None:
    ids = event_ids(dataset)
    if 'event_id' not in st.session_state or st.session_state.event_id not in dataset:
        init_event_state(ids[0], dataset)
    if 'edit_mode' not in st.session_state:
        st.session_state.edit_mode = 'Single station'
    if 'use_sliders' not in st.session_state:
        st.session_state.use_sliders = False
    if 'bulk_xyz_offset' not in st.session_state:
        st.session_state.bulk_xyz_offset = np.zeros(3, dtype=np.float32)
    if 'bulk_feat_offset' not in st.session_state:
        st.session_state.bulk_feat_offset = np.zeros(11, dtype=np.float32)


def apply_bulk_offsets() -> None:
    """Add uniform offsets to every station (relative to event baseline)."""
    base = st.session_state.baseline
    off_xyz = np.asarray(st.session_state.bulk_xyz_offset, dtype=np.float32)
    off_feat = np.asarray(st.session_state.bulk_feat_offset, dtype=np.float32)
    st.session_state.xyz = base['xyz'] + off_xyz.reshape(1, 3)
    st.session_state.features = base['features'] + off_feat.reshape(1, 11)
    # Keep polarity binary after bulk offset
    pol = np.sign(st.session_state.features[:, 0])
    pol[pol == 0] = 1.0
    st.session_state.features[:, 0] = pol


def _num_or_slider(
    label: str,
    value: float,
    lo: float,
    hi: float,
    step: float,
    *,
    help_txt: str = '',
    key: str,
    as_slider: bool,
) -> float:
    clipped = float(np.clip(value, lo, hi))
    if as_slider:
        if step > 0 and step < (hi - lo):
            n_steps = int(round((hi - lo) / step))
            if n_steps > 400:
                step = (hi - lo) / 400
        return float(
            st.slider(
                label,
                min_value=float(lo),
                max_value=float(hi),
                value=clipped,
                step=float(step),
                help=help_txt,
                key=key,
            )
        )
    return float(
        st.number_input(
            label,
            value=clipped,
            step=float(step),
            format='%.4f',
            help=help_txt,
            key=key,
        )
    )


def draw_beachball(strike: float, dip: float, rake: float, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.6, 4.6), facecolor='#121a1f')
    ax.set_facecolor('#121a1f')
    ax.set_aspect('equal')
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.axis('off')

    bb = beach(
        [strike, dip, rake],
        xy=(0, 0),
        width=2.0,
        linewidth=1.2,
        facecolor='#c47a2c',
        bgcolor='#1b2a30',
        edgecolor='#e8eef0',
        alpha=1.0,
        zorder=2,
    )
    ax.add_collection(bb)
    rim = Circle((0, 0), 1.0, fill=False, edgecolor='#9bb0b6', linewidth=1.4, zorder=3)
    ax.add_patch(rim)

    # Cardinal ticks
    for ang, lab in [(90, 'N'), (0, 'E'), (270, 'S'), (180, 'W')]:
        rad = np.deg2rad(ang)
        ax.plot(
            [1.05 * np.cos(rad), 1.15 * np.cos(rad)],
            [1.05 * np.sin(rad), 1.15 * np.sin(rad)],
            color='#9bb0b6',
            lw=1.0,
        )
        ax.text(
            1.28 * np.cos(rad),
            1.28 * np.sin(rad),
            lab,
            color='#c5d4d8',
            ha='center',
            va='center',
            fontsize=10,
            fontweight='600',
        )

    ax.set_title(title, color='#e6ecef', fontsize=12, pad=8, fontweight='600')
    fig.tight_layout()
    return fig


def station_map(
    xyz: np.ndarray,
    features: np.ndarray,
    selected: int,
    *,
    height: int = 520,
) -> go.Figure:
    pol = features[:, 0]
    colors = ['#2ea8a4' if p >= 0 else '#c45c3e' for p in pol]
    sizes = [22 if j == selected else 14 for j in range(len(pol))]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xyz[:, 0],
            y=xyz[:, 1],
            mode='markers+text',
            text=[str(j + 1) for j in range(len(xyz))],
            textposition='top center',
            textfont=dict(size=12, color='#e6ecef'),
            marker=dict(
                size=sizes,
                color=colors,
                line=dict(width=2.0, color='#e8eef0'),
                symbol='circle',
            ),
            hovertemplate=(
                'Station %{customdata[0]}<br>'
                'Δx=%{x:.2f} km<br>Δy=%{y:.2f} km<br>'
                'Δz=%{customdata[2]:.2f} km<br>'
                'Polarity=%{customdata[1]:+.0f}<extra></extra>'
            ),
            customdata=np.column_stack(
                [
                    np.arange(1, len(xyz) + 1),
                    pol,
                    xyz[:, 2],
                ]
            ),
            name='Stations',
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode='markers',
            marker=dict(size=22, color='#f0c14a', symbol='x', line=dict(width=2.5)),
            name='Hypocenter',
            hovertemplate='Hypocenter<extra></extra>',
        )
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(18,26,31,0.75)',
        margin=dict(l=40, r=30, t=50, b=40),
        height=height,
        legend=dict(font=dict(color='#c5d4d8', size=12), bgcolor='rgba(0,0,0,0)'),
        xaxis=dict(
            title=dict(text='Δx east (km)', font=dict(size=13)),
            color='#9bb0b6',
            gridcolor='rgba(155,176,182,0.18)',
            zerolinecolor='rgba(155,176,182,0.45)',
            zerolinewidth=1.5,
            scaleanchor='y',
            scaleratio=1,
        ),
        yaxis=dict(
            title=dict(text='Δy north (km)', font=dict(size=13)),
            color='#9bb0b6',
            gridcolor='rgba(155,176,182,0.18)',
            zerolinecolor='rgba(155,176,182,0.45)',
            zerolinewidth=1.5,
        ),
        font=dict(family='IBM Plex Sans', color='#c5d4d8'),
        title=dict(
            text='Station geometry (plan view, relative to hypocenter)',
            font=dict(size=15, color='#e6ecef'),
        ),
    )
    return fig


def main() -> None:
    dataset = cached_dataset()
    model = cached_model()
    ensure_state(dataset)
    ids = event_ids(dataset)

    st.markdown(
        '<p class="hero-title">FocoNet Focal Mechanism Studio</p>'
        '<p class="hero-sub">Edit station polarity, S/P ratios, SNRs and locations — '
        'watch the predicted beachball update in real time. Regional map auto-plots '
        'XQ inventory + ISC catalog + FocoNet beachballs.</p>',
        unsafe_allow_html=True,
    )

    tab_studio, tab_map = st.tabs(['🎛 Mechanism Studio', '🗺 Regional Map'])

    # ── Sidebar: event + station list ──────────────────────────────────────
    with st.sidebar:
        st.markdown('### Event')
        choice = st.selectbox(
            'Select event',
            ids,
            index=ids.index(st.session_state.event_id),
            format_func=lambda e: f'{e}  ·  M{dataset[e].get("mag", float("nan")):.1f}',
        )
        if choice != st.session_state.event_id:
            init_event_state(choice, dataset)
            st.rerun()

        mag = st.session_state.mag
        n_sta = len(st.session_state.station_ids)
        st.caption(f'{n_sta} / {config.MAX_STATIONS} stations loaded')
        if np.isfinite(mag):
            st.metric('Magnitude', f'{mag:.2f}')

        st.markdown('### Edit mode')
        edit_mode = st.radio(
            'Apply changes to',
            ['Single station', 'All stations'],
            index=0 if st.session_state.edit_mode == 'Single station' else 1,
            help='All stations: add the same offset to every station (e.g. +5 km depth).',
        )
        if edit_mode != st.session_state.edit_mode:
            st.session_state.edit_mode = edit_mode
            if edit_mode == 'All stations':
                st.session_state.bulk_xyz_offset = np.zeros(3, dtype=np.float32)
                st.session_state.bulk_feat_offset = np.zeros(11, dtype=np.float32)
                apply_bulk_offsets()
            st.rerun()
        st.session_state.edit_mode = edit_mode

        st.markdown('### Input style')
        st.session_state.use_sliders = st.toggle(
            'Use sliders',
            value=st.session_state.use_sliders,
            help='Switch number boxes to sliders for easier exploration.',
        )

        if st.button('Load demo (1 station)', use_container_width=True):
            if config.DEMO_EVENT_ID in dataset:
                init_event_state(config.DEMO_EVENT_ID, dataset)
                st.session_state.edit_mode = 'Single station'
                st.rerun()

        st.markdown('### Stations')
        if edit_mode == 'Single station':
            labels = [
                f'{j + 1}. {sid}  ({"+" if st.session_state.features[j, 0] >= 0 else "−"})'
                for j, sid in enumerate(st.session_state.station_ids)
            ]
            selected = st.radio(
                'Edit station',
                options=list(range(n_sta)),
                format_func=lambda j: labels[j],
                index=min(st.session_state.selected_station, n_sta - 1),
                label_visibility='collapsed',
            )
            st.session_state.selected_station = selected
        else:
            st.caption(f'Bulk offsets apply to all **{n_sta}** stations.')
            if st.button('Reset bulk offsets', use_container_width=True):
                st.session_state.bulk_xyz_offset = np.zeros(3, dtype=np.float32)
                st.session_state.bulk_feat_offset = np.zeros(11, dtype=np.float32)
                apply_bulk_offsets()
                st.rerun()

        if edit_mode == 'Single station':
            c1, c2 = st.columns(2)
            with c1:
                if st.button('＋ Add', use_container_width=True, disabled=n_sta >= config.MAX_STATIONS):
                    sid, xyz_row, feats_row = default_station_row(n_sta)
                    st.session_state.station_ids.append(sid)
                    st.session_state.xyz = np.vstack([st.session_state.xyz, xyz_row])
                    st.session_state.features = np.vstack([st.session_state.features, feats_row])
                    st.session_state.baseline['station_ids'].append(sid)
                    st.session_state.baseline['xyz'] = np.vstack(
                        [st.session_state.baseline['xyz'], xyz_row]
                    )
                    st.session_state.baseline['features'] = np.vstack(
                        [st.session_state.baseline['features'], feats_row]
                    )
                    st.session_state.selected_station = n_sta
                    st.rerun()
            with c2:
                if st.button('− Remove', use_container_width=True, disabled=n_sta <= 1):
                    j = st.session_state.selected_station
                    st.session_state.station_ids.pop(j)
                    st.session_state.xyz = np.delete(st.session_state.xyz, j, axis=0)
                    st.session_state.features = np.delete(st.session_state.features, j, axis=0)
                    st.session_state.baseline['station_ids'].pop(j)
                    st.session_state.baseline['xyz'] = np.delete(
                        st.session_state.baseline['xyz'], j, axis=0
                    )
                    st.session_state.baseline['features'] = np.delete(
                        st.session_state.baseline['features'], j, axis=0
                    )
                    st.session_state.selected_station = max(0, j - 1)
                    st.rerun()

        if st.button('Reset event', use_container_width=True):
            base = st.session_state.baseline
            st.session_state.station_ids = list(base['station_ids'])
            st.session_state.bulk_xyz_offset = np.zeros(3, dtype=np.float32)
            st.session_state.bulk_feat_offset = np.zeros(11, dtype=np.float32)
            apply_bulk_offsets()
            st.session_state.selected_station = 0
            st.rerun()

    # Live prediction used by both tabs
    i = st.session_state.selected_station
    feats = st.session_state.features
    xyz = st.session_state.xyz

    with tab_studio:
        result = render_studio(i, feats, xyz, model)

    with tab_map:
        render_regional_map(model, result, i)


def render_studio(i, feats, xyz, model):
    edit_mode = st.session_state.edit_mode
    use_sliders = st.session_state.use_sliders
    eid = st.session_state.event_id

    top_left, top_mid, top_right = st.columns([1.2, 1.0, 0.85], gap='medium')

    with top_left:
        if edit_mode == 'All stations':
            n_sta = len(st.session_state.station_ids)
            st.markdown('#### Bulk adjust all stations')
            st.caption(
                f'Uniform offset added to **all {n_sta} stations**. '
                f'Example: set **ΔΔz = +5** to raise every station depth by 5 km.'
            )

            st.markdown('**Location offset (km)**')
            bulk_xyz = st.session_state.bulk_xyz_offset.copy()
            loc_cols = st.columns(3)
            for j, (name, help_txt) in enumerate(zip(config.LOCATION_NAMES, config.LOCATION_HELP)):
                lo, hi, step = config.LOCATION_SLIDER_RANGES[j]
                with loc_cols[j]:
                    bulk_xyz[j] = _num_or_slider(
                        f'Δ{name}',
                        float(bulk_xyz[j]),
                        lo,
                        hi,
                        step,
                        help_txt=f'Added to every station. {help_txt}',
                        key=f'bulk_xyz_{eid}_{j}',
                        as_slider=use_sliders,
                    )
            st.session_state.bulk_xyz_offset = bulk_xyz

            with st.expander('Feature offsets (optional)', expanded=False):
                bulk_feat = st.session_state.bulk_feat_offset.copy()
                feat_cols = st.columns(2)
                for j in range(1, 11):
                    lo, hi, step = config.FEATURE_SLIDER_RANGES[j]
                    with feat_cols[(j - 1) % 2]:
                        bulk_feat[j] = _num_or_slider(
                            config.FEATURE_NAMES[j],
                            float(bulk_feat[j]),
                            lo,
                            hi,
                            step,
                            help_txt=config.FEATURE_HELP[j],
                            key=f'bulk_feat_{eid}_{j}',
                            as_slider=use_sliders,
                        )
                st.session_state.bulk_feat_offset = bulk_feat

            quick1, quick2, quick3 = st.columns(3)
            with quick1:
                if st.button('+5 Δz all', use_container_width=True, key='bulk_p5'):
                    st.session_state.bulk_xyz_offset[2] += 5.0
                    apply_bulk_offsets()
                    st.rerun()
            with quick2:
                if st.button('−5 Δz all', use_container_width=True, key='bulk_m5'):
                    st.session_state.bulk_xyz_offset[2] -= 5.0
                    apply_bulk_offsets()
                    st.rerun()
            with quick3:
                if st.button('Clear offsets', use_container_width=True, key='bulk_clr'):
                    st.session_state.bulk_xyz_offset = np.zeros(3, dtype=np.float32)
                    st.session_state.bulk_feat_offset = np.zeros(11, dtype=np.float32)
                    apply_bulk_offsets()
                    st.rerun()

            apply_bulk_offsets()
            xyz = st.session_state.xyz
            feats = st.session_state.features

            st.info(
                f'Current bulk offset: Δx={bulk_xyz[0]:+.2f}, '
                f'Δy={bulk_xyz[1]:+.2f}, Δz={bulk_xyz[2]:+.2f} km'
            )
        else:
            st.markdown('#### Station parameters')
            st.caption(
                f'Editing **{st.session_state.station_ids[i]}** — '
                '11 polarity/amplitude/SNR features + 3-D location (14-D token).'
            )

            new_sid = st.text_input('Station ID', value=st.session_state.station_ids[i])
            st.session_state.station_ids[i] = new_sid
            st.session_state.baseline['station_ids'][i] = new_sid

            st.markdown('**Location (relative to hypocenter)**')
            loc_cols = st.columns(3)
            for j, (name, help_txt) in enumerate(zip(config.LOCATION_NAMES, config.LOCATION_HELP)):
                lo, hi, step = config.LOCATION_SLIDER_RANGES[j]
                with loc_cols[j]:
                    xyz[i, j] = _num_or_slider(
                        name,
                        float(xyz[i, j]),
                        lo,
                        hi,
                        step,
                        help_txt=help_txt,
                        key=f'xyz_{eid}_{i}_{j}',
                        as_slider=use_sliders,
                    )

            st.markdown('**Polarity & amplitude features**')
            pol_choice = st.radio(
                config.FEATURE_NAMES[0],
                options=[1.0, -1.0],
                format_func=lambda v: '+1 compression' if v > 0 else '−1 dilatation',
                index=0 if feats[i, 0] >= 0 else 1,
                horizontal=True,
                help=config.FEATURE_HELP[0],
                key=f'pol_{eid}_{i}',
            )
            feats[i, 0] = float(pol_choice)

            amp_cols = st.columns(2)
            for j in range(1, 11):
                lo, hi, step = config.FEATURE_SLIDER_RANGES[j]
                with amp_cols[(j - 1) % 2]:
                    feats[i, j] = _num_or_slider(
                        config.FEATURE_NAMES[j],
                        float(feats[i, j]),
                        lo,
                        hi,
                        step,
                        help_txt=config.FEATURE_HELP[j],
                        key=f'feat_{eid}_{i}_{j}',
                        as_slider=use_sliders,
                    )

            st.session_state.features = feats
            st.session_state.xyz = xyz

    try:
        result = predict(st.session_state.features, st.session_state.xyz, model=model)
        err = None
    except Exception as exc:  # noqa: BLE001
        result = None
        err = str(exc)

    with top_mid:
        st.markdown('#### Predicted focal mechanism')
        if err:
            st.error(f'Prediction failed: {err}')
        elif result:
            fig = draw_beachball(
                result['strike'],
                result['dip'],
                result['rake'],
                'FocoNet prediction',
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            m1, m2, m3 = st.columns(3)
            m1.metric('Strike', f'{result["strike"]:.1f}°')
            m2.metric('Dip', f'{result["dip"]:.1f}°')
            m3.metric('Rake', f'{result["rake"]:.1f}°')
            st.caption(
                f'Auxiliary plane  ·  '
                f'strike {result["strike2"]:.1f}°  ·  '
                f'dip {result["dip2"]:.1f}°  ·  '
                f'rake {result["rake2"]:.1f}°'
            )
            st.info(f'Faulting style: **{result["faulting"]}**  ·  {result["n_stations"]} stations')

    with top_right:
        st.markdown('#### Axis details')
        if result:
            st.markdown(
                f'<div class="panel mono">'
                f'<b>P axis</b><br>'
                f'[{result["p_axis"][0]:+.4f}, {result["p_axis"][1]:+.4f}, {result["p_axis"][2]:+.4f}]<br><br>'
                f'<b>T axis</b><br>'
                f'[{result["t_axis"][0]:+.4f}, {result["t_axis"][1]:+.4f}, {result["t_axis"][2]:+.4f}]<br><br>'
                f'<b>B axis</b><br>'
                f'[{result["b_axis"][0]:+.4f}, {result["b_axis"][1]:+.4f}, {result["b_axis"][2]:+.4f}]'
                f'</div>',
                unsafe_allow_html=True,
            )

            cat = st.session_state.focalmech
            if np.any(np.abs(cat) > 1e-6):
                st.markdown('##### Catalog reference')
                st.write(
                    f'Strike {cat[0]:.1f}° · Dip {cat[1]:.1f}° · Rake {cat[2]:.1f}°'
                )
            else:
                st.caption('No catalog focal mechanism in this XQ entry (zeros).')

    st.divider()
    st.plotly_chart(
        station_map(st.session_state.xyz, st.session_state.features, i, height=520),
        use_container_width=True,
        config={'displayModeBar': False},
    )

    st.markdown('#### All stations')
    table = np.column_stack(
        [
            np.arange(1, len(st.session_state.station_ids) + 1),
            st.session_state.features,
            st.session_state.xyz,
        ]
    )
    colnames = ['#'] + config.FEATURE_NAMES + config.LOCATION_NAMES
    df = pd.DataFrame(table, columns=colnames)
    df.insert(1, 'Station', st.session_state.station_ids)
    st.dataframe(df, use_container_width=True, height=280)

    with st.expander('Feature vector reference'):
        st.markdown(
            """
Each station is a **14-D token**: 11 waveform features + relative *(x, y, z)*.

| Col | Name | Meaning |
|---|---|---|
| 1 | Polarity | P first-motion (+1 / −1) |
| 2 | ln(maxS/maxP) | Broadband S/P ratio |
| 3–8 | ln(S·/P·) | Channel-specific S/P ratios |
| 9–11 | ln(SNR) | Signal-to-noise (R, T, Z) |
| 12–14 | x, y, z | Relative station location (km); model scales x,y÷50 and z÷20 |

FocoNet pads unused slots to **32 stations** and masks them during attention pooling.
            """
        )
    return result


def render_regional_map(model, result, station_idx: int) -> None:
    st.markdown('#### XQ NAMASTE regional map')
    st.caption(
        'Auto-loads `XQ_inventory.xml` + `catalog_ISC_summary.csv`, overlays FocoNet beachballs, '
        'and highlights the **currently selected** event (red outline) with your live prediction.'
    )

    catalog = cached_catalog()
    stations = cached_stations()
    foco = cached_foco_map_df(model)

    c1, c2, c3 = st.columns(3)
    c1.metric('Catalog events', len(catalog))
    c2.metric('XQ stations', len(stations))
    c3.metric('FocoNet beachballs', len(foco))

    selected = None
    coords = lookup_event_coords(catalog, st.session_state.event_id)
    if result and coords:
        selected = {
            **coords,
            'strike': result['strike'],
            'dip': result['dip'],
            'rake': result['rake'],
            'magnitude': float(st.session_state.mag)
            if np.isfinite(st.session_state.mag)
            else coords.get('magnitude', 3.5),
        }
    elif not coords:
        st.warning(
            f'No ISC catalog match for `{st.session_state.event_id}` — '
            'map still shows inventory + catalog + other FocoNet solutions.'
        )

    with st.spinner('Rendering regional map…'):
        fig = plot_xq_map(
            catalog=catalog,
            stations=stations,
            foco=foco,
            selected=selected,
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.divider()
    st.markdown('#### Event station geometry')
    st.caption('Plan view of stations for the current event — relative to hypocenter.')
    st.plotly_chart(
        station_map(st.session_state.xyz, st.session_state.features, station_idx, height=480),
        use_container_width=True,
        config={'displayModeBar': False},
    )

    with st.expander('Matched FocoNet events on map'):
        show = foco[
            ['event_id', 'latitude', 'longitude', 'depth_km', 'magnitude', 'strike', 'dip', 'rake']
        ].copy()
        st.dataframe(show, use_container_width=True, height=260)


if __name__ == '__main__':
    main()
