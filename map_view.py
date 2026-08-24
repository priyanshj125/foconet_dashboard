"""XQ NAMASTE regional map — SRTM hillshade, GEM faults, FocoNet beachballs.

Matches the notebook cell:
  Beach Ball Map — Fixed Linking + Corrected Background/Legend
"""

from __future__ import annotations

import math
import os
import urllib.request
from io import BytesIO
from typing import Any

import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LightSource, LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from obspy import read_inventory
from obspy.imaging.beachball import beach
from PIL import Image

import config

# Digitized Himalayan thrusts (fallback when fault/ folder is absent)
MFT_LON = np.array([82.0, 82.8, 83.5, 84.0, 84.6, 85.2, 85.8, 86.4, 87.0, 87.8, 88.2])
MFT_LAT = np.array([27.8, 27.6, 27.4, 27.3, 27.2, 27.1, 27.0, 26.9, 26.8, 26.7, 26.5])
MBT_LON = np.array([82.0, 82.8, 83.5, 84.0, 84.5, 85.0, 85.5, 86.0, 86.5, 87.0, 87.5, 88.2])
MBT_LAT = np.array([28.2, 28.05, 27.9, 27.8, 27.7, 27.6, 27.5, 27.4, 27.3, 27.2, 27.05, 26.9])
MCT_LON = np.array([82.0, 82.8, 83.5, 84.0, 84.5, 85.0, 85.6, 86.2, 86.8, 87.5, 88.2])
MCT_LAT = np.array([28.8, 28.6, 28.4, 28.3, 28.2, 28.1, 28.0, 27.9, 27.8, 27.6, 27.4])

CITIES = {
    'Kathmandu': (85.316, 27.717),
    'Pokhara': (83.994, 28.209),
    'Gorkha': (84.630, 28.000),
}

GORKHA_LON, GORKHA_LAT = 84.730, 28.230

HILLSHADE_CACHE = os.path.join(config.BASE_DIR, 'nepal_hillshade_beachball.npy')
HILLSHADE_EXTENT = os.path.join(config.BASE_DIR, 'nepal_hillshade_beachball_extent.npy')
FAULT_DIR = os.path.join(config.BASE_DIR, 'fault')
GEM_URL = (
    'https://raw.githubusercontent.com/GEMScienceTools/'
    'gem-global-active-faults/master/geojson/gem_active_faults.geojson'
)


def load_catalog(path: str | None = None) -> pd.DataFrame:
    cat_path = path or config.CATALOG_PATH
    df = pd.read_csv(cat_path)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df['time_label'] = df['time'].dt.strftime('%Y%m%dT%H%M%S')
    df['depth_km'] = pd.to_numeric(df['depth_km'], errors='coerce')
    df['magnitude'] = pd.to_numeric(df['magnitude'], errors='coerce')
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    return df.dropna(subset=['latitude', 'longitude']).reset_index(drop=True)


def load_stations(path: str | None = None) -> pd.DataFrame:
    inv_path = path or config.INVENTORY_PATH
    inv = read_inventory(inv_path)
    rows = []
    for net in inv:
        for sta in net:
            rows.append(
                {
                    'network': net.code,
                    'code': sta.code,
                    'latitude': float(sta.latitude),
                    'longitude': float(sta.longitude),
                    'elevation': float(sta.elevation) if sta.elevation is not None else np.nan,
                }
            )
    return pd.DataFrame(rows).drop_duplicates(subset=['code']).reset_index(drop=True)


def match_predictions_to_catalog(
    predictions: pd.DataFrame,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Join FocoNet SDR rows to ISC catalog lat/lon/depth via time stamp in event id."""
    df = predictions.copy()

    # Support both Full (folder_name/event_id + strike/dip/rake)
    # and SP CSV (event + strike1/dip1/rake1) formats.
    if 'event' in df.columns and 'event_id' not in df.columns:
        df = df.rename(columns={'event': 'event_id'})
    if 'folder_name' in df.columns and 'event_id' not in df.columns:
        df = df.rename(columns={'folder_name': 'event_id'})
    if 'strike1' in df.columns and 'strike' not in df.columns:
        df['strike'] = df['strike1']
        df['dip'] = df['dip1']
        df['rake'] = df['rake1']

    df['time_label'] = df['event_id'].astype(str).str.extract(r'^(\d{8}T\d{6})')
    if 'magnitude' not in df.columns:
        df['magnitude'] = df['event_id'].astype(str).str.extract(r'_M([\d.]+)').astype(float)

    merged = df.merge(
        catalog[['time_label', 'latitude', 'longitude', 'depth_km', 'magnitude']].rename(
            columns={'magnitude': 'cat_magnitude'}
        ),
        on='time_label',
        how='left',
    )
    if 'magnitude' in merged.columns:
        merged['magnitude'] = merged['magnitude'].fillna(merged['cat_magnitude'])
    else:
        merged['magnitude'] = merged['cat_magnitude']
    merged['depth_km'] = merged['depth_km'].fillna(12.0)
    return merged.dropna(subset=['latitude', 'longitude', 'strike', 'dip', 'rake']).reset_index(drop=True)


def lookup_event_coords(catalog: pd.DataFrame, event_id: str) -> dict[str, float] | None:
    """Match XQ event id (YYYYMMDDTHHMMSS_M…) to catalog lat/lon/depth."""
    import re
    from datetime import datetime, timezone

    m = re.match(r'^(\d{8}T\d{6})', str(event_id))
    if not m:
        return None
    label = m.group(1)
    hit = catalog[catalog['time_label'] == label]
    if not hit.empty:
        row = hit.iloc[0]
    else:
        try:
            t = datetime.strptime(label, '%Y%m%dT%H%M%S').replace(tzinfo=timezone.utc)
            deltas = (catalog['time'] - t).abs()
            i = int(deltas.idxmin())
            if deltas.loc[i].total_seconds() > 120:
                return None
            row = catalog.loc[i]
        except Exception:
            return None
    return {
        'longitude': float(row['longitude']),
        'latitude': float(row['latitude']),
        'depth_km': float(row['depth_km']) if pd.notna(row['depth_km']) else 12.0,
        'magnitude': float(row['magnitude']) if pd.notna(row['magnitude']) else float('nan'),
    }


def lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    n = 2 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def tile_to_lonlat(x: int, y: int, zoom: int) -> tuple[float, float]:
    n = 2 ** zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    return lon, math.degrees(lat_rad)


def build_hillshade(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    zoom: int = 9,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Download Mapzen Terrarium elevation tiles and render a soft hillshade."""
    x0, y0 = lonlat_to_tile(lon_min, lat_max, zoom)
    x1, y1 = lonlat_to_tile(lon_max, lat_min, zoom)
    xs = range(min(x0, x1), max(x0, x1) + 1)
    ys = range(min(y0, y1), max(y0, y1) + 1)
    tile_size = 256
    mosaic = np.zeros((len(list(ys)) * tile_size, len(list(xs)) * tile_size), dtype=float)
    xs = list(xs)
    ys = list(ys)
    mosaic = np.zeros((len(ys) * tile_size, len(xs) * tile_size), dtype=float)

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            url = f'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{zoom}/{x}/{y}.png'
            try:
                r = requests.get(url, timeout=20)
                r.raise_for_status()
                img = np.array(Image.open(BytesIO(r.content)).convert('RGB'), dtype=float)
                elev = (img[:, :, 0] * 256.0 + img[:, :, 1] + img[:, :, 2] / 256.0) - 32768.0
                mosaic[j * tile_size : (j + 1) * tile_size, i * tile_size : (i + 1) * tile_size] = elev
            except Exception as exc:  # noqa: BLE001
                print(f'  tile ({x},{y}) failed: {exc}')

    lon_tl, lat_tl = tile_to_lonlat(min(xs), min(ys), zoom)
    lon_br, lat_br = tile_to_lonlat(max(xs) + 1, max(ys) + 1, zoom)
    ls = LightSource(azdeg=315, altdeg=45)
    cmap_terrain = LinearSegmentedColormap.from_list(
        'terrain_muted',
        ['#e8e4d8', '#d9d2b8', '#c4b896', '#a89a73', '#8f7d5a', '#6b5d45'],
        N=256,
    )
    rgb = ls.shade(mosaic, cmap=cmap_terrain, vert_exag=1.5, blend_mode='soft', dx=1, dy=1)
    return rgb, (lon_tl, lon_br, lat_br, lat_tl)


def get_hillshade(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    zoom: int = 9,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    try:
        rgb = np.load(HILLSHADE_CACHE)
        extent = tuple(np.load(HILLSHADE_EXTENT))
        return rgb, extent  # type: ignore[return-value]
    except Exception:
        pass

    print('Building SRTM hillshade for map extent …')
    rgb, extent = build_hillshade(lon_min, lon_max, lat_min, lat_max, zoom=zoom)
    np.save(HILLSHADE_CACHE, rgb)
    np.save(HILLSHADE_EXTENT, np.array(extent))
    print('Saved →', HILLSHADE_CACHE)
    return rgb, extent


def ensure_gem_faults() -> str | None:
    path = config.FAULTS_PATH
    if os.path.exists(path):
        return path
    try:
        print('Downloading GEM fault database …')
        urllib.request.urlretrieve(GEM_URL, path)
        return path
    except Exception as exc:  # noqa: BLE001
        print(f'GEM download failed: {exc}')
        return None


def load_fault_file(path: str) -> tuple[np.ndarray, np.ndarray]:
    lons, lats = [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().replace(',', ' ').split()
            if len(parts) >= 2:
                try:
                    lons.append(float(parts[0]))
                    lats.append(float(parts[1]))
                except ValueError:
                    pass
    return np.asarray(lons), np.asarray(lats)


def _plot_faults(ax, lon_min, lon_max, lat_min, lat_max) -> None:
    # GEM reverse faults
    gem_path = ensure_gem_faults()
    if gem_path:
        try:
            import geopandas as gpd
            from shapely.geometry import box

            faults = gpd.read_file(gem_path)
            bbox = box(lon_min, lat_min, lon_max, lat_max)
            local = faults[faults.geometry.intersects(bbox)]
            if 'slip_type' in local.columns:
                reverse = local[local['slip_type'] == 'Reverse']
                if len(reverse):
                    reverse.plot(ax=ax, color='#222222', linewidth=1.8, zorder=5, alpha=0.9)
                else:
                    local.plot(ax=ax, color='#222222', linewidth=1.4, zorder=5, alpha=0.85)
            else:
                local.plot(ax=ax, color='#222222', linewidth=1.4, zorder=5, alpha=0.85)
        except Exception as exc:  # noqa: BLE001
            print(f'GEM plot skipped: {exc}')

    # Digitized traces from fault/ if present, else built-in fallbacks
    pairs = [
        ('mft.dat', MFT_LON, MFT_LAT, '#111111', 2.2),
        ('MBT-new.txt', MBT_LON, MBT_LAT, '#555555', 1.6),
        ('MCT-new.txt', MCT_LON, MCT_LAT, '#111111', 1.2),
    ]
    for fname, fb_lon, fb_lat, color, lw in pairs:
        fpath = os.path.join(FAULT_DIR, fname)
        if os.path.exists(fpath):
            lon, lat = load_fault_file(fpath)
            if len(lon):
                ax.plot(lon, lat, color=color, linewidth=lw, zorder=5, alpha=0.95)
                continue
        ax.plot(fb_lon, fb_lat, color=color, linewidth=lw, zorder=5, alpha=0.95)

    label_kw = dict(
        fontsize=8.5,
        fontweight='bold',
        color='#111111',
        zorder=11,
        bbox=dict(facecolor='white', alpha=0.80, edgecolor='none', pad=2),
    )
    ax.text(85.8, 28.55, 'MCT', **label_kw)
    ax.text(85.5, 27.18, 'MFT', **label_kw)
    ax.text(85.9, 27.75, 'MBT', **label_kw)


def plot_xq_map(
    catalog: pd.DataFrame,
    stations: pd.DataFrame,
    foco: pd.DataFrame | None = None,
    selected: dict[str, Any] | None = None,
    title: str = 'XQ NAMASTE — Focal Mechanisms (FocoNet), Nepal Aftershock Sequence 2015–2016',
    figsize: tuple[float, float] = (13, 9),
) -> plt.Figure:
    """
    Draw regional map with:
      - cached SRTM terrarium hillshade
      - GEM reverse faults + digitized MCT/MBT/MFT
      - XQ inventory triangles
      - FocoNet beachballs coloured by depth
      - optional highlighted selected-event beachball
    """
    lon_c = float(catalog['longitude'].mean())
    lat_c = float(catalog['latitude'].mean())
    lon_min, lon_max = lon_c - 2.5, lon_c + 2.5
    lat_min, lat_max = lat_c - 1.5, lat_c + 1.5

    fig, ax = plt.subplots(figsize=figsize, dpi=120)

    try:
        rgb_hillshade, bg_extent = get_hillshade(lon_min, lon_max, lat_min, lat_max, zoom=9)
        ax.imshow(rgb_hillshade, extent=bg_extent, origin='upper', zorder=0, interpolation='bilinear')
    except Exception as exc:  # noqa: BLE001
        print(f'Hillshade unavailable ({exc}); using muted terrain wash.')
        ax.set_facecolor('#d9cbb8')

    _plot_faults(ax, lon_min, lon_max, lat_min, lat_max)

    cmap_bb = cm.RdYlBu_r
    norm_bb = Normalize(vmin=0, vmax=40)

    # Catalog epicentres (light background)
    mag_sizes = 5 * (3 ** (catalog['magnitude'].fillna(3.5) - 2.5))
    ax.scatter(
        catalog['longitude'],
        catalog['latitude'],
        c=catalog['depth_km'].fillna(12.0),
        cmap=cmap_bb,
        norm=norm_bb,
        s=mag_sizes,
        alpha=0.35,
        edgecolors='#333333',
        linewidths=0.2,
        zorder=4,
    )

    # FocoNet beachballs
    if foco is not None and len(foco):
        n_failed = 0
        for _, row in foco.iterrows():
            mag = float(row.get('magnitude', 3.5) or 3.5)
            depth = float(row.get('depth_km', 12.0) or 12.0)
            width = max(0.12, min(0.35, 0.025 * (3 ** (mag - 3.5))))
            color = cmap_bb(norm_bb(depth))
            try:
                bb = beach(
                    [float(row['strike']), float(row['dip']), float(row['rake'])],
                    xy=(float(row['longitude']), float(row['latitude'])),
                    width=width,
                    linewidth=0.4,
                    facecolor=color,
                    bgcolor='white',
                    edgecolor='black',
                    alpha=0.85,
                    zorder=7,
                )
                ax.add_collection(bb)
            except Exception:
                n_failed += 1
        if n_failed:
            print(f'Warning: {n_failed} beachballs failed to plot')

    # Selected live prediction
    if selected is not None:
        lon = float(selected['longitude'])
        lat = float(selected['latitude'])
        mag = float(selected.get('magnitude', 3.5) or 3.5)
        depth = float(selected.get('depth_km', 12.0) or 12.0)
        width = max(0.20, min(0.45, 0.035 * (3 ** (mag - 3.5))))
        try:
            bb = beach(
                [float(selected['strike']), float(selected['dip']), float(selected['rake'])],
                xy=(lon, lat),
                width=width,
                linewidth=1.1,
                facecolor=cmap_bb(norm_bb(depth)),
                bgcolor='white',
                edgecolor='#FF2200',
                alpha=1.0,
                zorder=12,
            )
            ax.add_collection(bb)
            ax.add_patch(
                Circle(
                    (lon, lat),
                    width * 0.72,
                    fill=False,
                    edgecolor='#FF2200',
                    linewidth=1.4,
                    linestyle='--',
                    zorder=13,
                )
            )
        except Exception:
            ax.plot(lon, lat, 'o', color='#FF2200', markersize=10, zorder=12)

    # Gorkha mainshock
    ax.plot(
        GORKHA_LON,
        GORKHA_LAT,
        marker='*',
        color='#FF2200',
        markersize=18,
        markeredgecolor='#880000',
        markeredgewidth=0.8,
        linestyle='None',
        zorder=10,
    )
    ax.annotate(
        'Mw 7.8  25 Apr 2015',
        xy=(GORKHA_LON, GORKHA_LAT),
        xytext=(GORKHA_LON + 0.18, GORKHA_LAT + 0.22),
        fontsize=8,
        fontweight='bold',
        color='#880000',
        zorder=11,
        arrowprops=dict(arrowstyle='->', color='#880000', lw=0.8),
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.85, edgecolor='#880000', linewidth=0.8),
    )

    # Stations
    ax.plot(
        stations['longitude'],
        stations['latitude'],
        marker='^',
        color='#0D3B66',
        markersize=9,
        markeredgecolor='white',
        linestyle='None',
        zorder=8,
    )

    for name, (lo, la) in CITIES.items():
        ax.plot(lo, la, marker='s', color='#222222', markersize=8, linestyle='None', zorder=9)
        ax.annotate(
            name,
            xy=(lo, la),
            xytext=(lo + 0.08, la - 0.14),
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=1, edgecolor='none'),
            zorder=9,
        )

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

    # Depth colorbar (inset, notebook style)
    cax = inset_axes(
        ax,
        width='2%',
        height='30%',
        loc='lower right',
        bbox_to_anchor=(0.02, 0.05, 1, 1),
        bbox_transform=ax.transAxes,
    )
    cb = ColorbarBase(cax, cmap=cmap_bb, norm=norm_bb, orientation='vertical')
    cb.set_label('Depth (km)', fontsize=8)

    # 100 km scale bar
    scale_km = 100.0
    scale_deg = scale_km / (111.32 * math.cos(math.radians(lat_c)))
    x0 = lon_max - scale_deg - 0.25
    y0 = lat_min + 0.25
    ax.plot([x0, x0 + scale_deg], [y0, y0], color='black', linewidth=2.5, zorder=10, solid_capstyle='butt')
    ax.plot([x0, x0], [y0 - 0.04, y0 + 0.04], color='black', linewidth=2.5, zorder=10)
    ax.plot([x0 + scale_deg] * 2, [y0 - 0.04, y0 + 0.04], color='black', linewidth=2.5, zorder=10)
    ax.text(
        x0 + scale_deg / 2,
        y0 + 0.10,
        f'{int(scale_km)} km',
        ha='center',
        va='bottom',
        fontsize=8,
        fontweight='bold',
        zorder=10,
        bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.8, edgecolor='none'),
    )

    legend_handles = [
        mlines.Line2D([], [], color='#222222', linewidth=1.8, label='Reverse Faults (GEM)'),
        mlines.Line2D(
            [],
            [],
            linestyle='None',
            marker='^',
            color='#0D3B66',
            markersize=7,
            markeredgecolor='white',
            label='XQ Stations',
        ),
        mlines.Line2D(
            [],
            [],
            linestyle='None',
            marker='o',
            color='#888888',
            markeredgecolor='black',
            markersize=8,
            label='Focal Mechanism',
        ),
        mlines.Line2D(
            [],
            [],
            linestyle='None',
            marker='*',
            color='#FF2200',
            markersize=12,
            markeredgecolor='#880000',
            label='Gorkha Mw 7.8',
        ),
        mlines.Line2D(
            [],
            [],
            linestyle='None',
            marker='o',
            markerfacecolor='none',
            markeredgecolor='#FF2200',
            markersize=10,
            label='Selected (live)',
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc='lower left',
        fontsize=8,
        framealpha=0.9,
        edgecolor='#AAAAAA',
    )

    fig.tight_layout()
    return fig
