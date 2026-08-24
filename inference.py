"""FocoNet inference helpers for the interactive dashboard."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from pyrocko import moment_tensor as pmt

import config
from model import FocoNet

_MODEL: FocoNet | None = None
_DEVICE: torch.device | None = None


def get_device() -> torch.device:
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return _DEVICE


def load_model(model_path: str | None = None) -> FocoNet:
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    path = model_path or config.MODEL_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f'Model not found: {path}')

    device = get_device()
    model = FocoNet().to(device)
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    _MODEL = model
    return _MODEL


def event_ids(dataset: dict[str, Any]) -> list[str]:
    ids = sorted(dataset.keys())
    # Keep the 1-station demo at the top of the picker
    if config.DEMO_EVENT_ID in ids:
        ids = [config.DEMO_EVENT_ID] + [e for e in ids if e != config.DEMO_EVENT_ID]
    return ids


def make_demo_one_station() -> dict[str, Any]:
    """Synthetic single-station case for sliding features and watching the beachball."""
    feats = np.array(
        [
            1.0,   # polarity (compression)
            1.20,  # ln(maxS/maxP)
            1.10,  # ln(Sr/Pr)
            0.85,  # ln(Sr/Pz)
            0.95,  # ln(St/Pr)
            0.70,  # ln(St/Pz)
            1.05,  # ln(Sz/Pr)
            0.80,  # ln(Sz/Pz)
            3.50,  # ln(SNR_R)
            3.20,  # ln(SNR_T)
            4.00,  # ln(SNR_Z)
        ],
        dtype=np.float32,
    )
    return {
        'polarities': feats.reshape(1, 11),
        'stationxyz': np.array([['DEMO1', 25.0, 15.0, 12.0]], dtype=object),
        'focalmech': np.array([100.0, 30.0, 90.0], dtype=float),
        'mag': 3.5,
    }


def with_demo_event(dataset: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of the dataset that includes the 1-station demo."""
    out = dict(dataset)
    out[config.DEMO_EVENT_ID] = make_demo_one_station()
    return out


def load_dataset(path: str | None = None) -> dict[str, Any]:
    data_path = path or config.DEV_DATA
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'Dataset not found: {data_path}')
    data = np.load(data_path, allow_pickle=True).item()
    return with_demo_event(data)


def parse_event(event: dict[str, Any]) -> dict[str, Any]:
    """Extract editable station table + metadata from one dataset event."""
    stationxyz = np.asarray(event['stationxyz'])
    polarities = np.asarray(event['polarities'], dtype=np.float32)

    n_sta = min(len(stationxyz), len(polarities), config.MAX_STATIONS)
    station_ids: list[str] = []
    xyz = np.zeros((n_sta, 3), dtype=np.float32)
    feats = np.zeros((n_sta, 11), dtype=np.float32)

    for i in range(n_sta):
        row = stationxyz[i]
        station_ids.append(str(row[0]))
        xyz[i] = np.asarray(row[1:4], dtype=np.float32)
        feats[i] = polarities[i, :11]

    focal = np.asarray(event.get('focalmech', [0.0, 0.0, 0.0]), dtype=float)
    mag = float(event.get('mag', np.nan))

    return {
        'station_ids': station_ids,
        'xyz': xyz,
        'features': feats,
        'focalmech': focal,
        'mag': mag,
        'n_stations': n_sta,
    }


def pack_tensors(
    features: np.ndarray,
    xyz: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad stations to MAX_STATIONS and apply FocoNet location scaling."""
    n_sta = int(features.shape[0])
    if n_sta < 1:
        raise ValueError('Need at least one station')
    if n_sta > config.MAX_STATIONS:
        raise ValueError(f'At most {config.MAX_STATIONS} stations supported')

    pol = np.zeros((1, config.MAX_STATIONS, 11), dtype=np.float32)
    loc = np.zeros((1, config.MAX_STATIONS, 3), dtype=np.float32)
    mask = np.zeros((1, config.MAX_STATIONS, 1), dtype=np.float32)

    pol[0, :n_sta] = features.astype(np.float32)
    loc[0, :n_sta] = xyz.astype(np.float32)
    loc[0, :n_sta, 0:2] /= config.XY_SCALE
    loc[0, :n_sta, 2] /= config.Z_SCALE
    loc[0, n_sta:, 2] -= 100.0
    mask[0, :n_sta, 0] = 1.0

    device = get_device()
    return (
        torch.tensor(pol, device=device),
        torch.tensor(loc, device=device),
        torch.tensor([[float(n_sta)]], device=device),
        torch.tensor(mask, device=device),
    )


def pt_to_sdr(p_axis: np.ndarray, t_axis: np.ndarray) -> tuple[float, float, float]:
    p = np.asarray(p_axis, dtype=float)
    t = np.asarray(t_axis, dtype=float)
    p = p / (np.linalg.norm(p) + 1e-12)
    t = t / (np.linalg.norm(t) + 1e-12)
    strike, dip, rake = pmt.pt_axes_to_strike_dip_rake(p, t)
    return float(strike), float(dip), float(rake)


def auxiliary_plane(strike: float, dip: float, rake: float) -> tuple[float, float, float]:
    mt = pmt.MomentTensor(strike=strike, dip=dip, rake=rake)
    s2, d2, r2 = mt.both_strike_dip_rake()[1]
    return float(s2), float(d2), float(r2)


def faulting_type(rake: float) -> str:
    r = ((rake + 180) % 360) - 180
    if abs(r) < 30 or abs(abs(r) - 180) < 30:
        return 'Strike-slip'
    if 30 <= r <= 150:
        return 'Reverse / Thrust'
    if -150 <= r <= -30:
        return 'Normal'
    return 'Oblique'


def predict(
    features: np.ndarray,
    xyz: np.ndarray,
    model: FocoNet | None = None,
) -> dict[str, Any]:
    """Run FocoNet and return strike/dip/rake + PTB axes."""
    net = model or load_model()
    wave, loc, sta_num, sta_mask = pack_tensors(features, xyz)

    with torch.no_grad():
        out = net(wave, loc, sta_num, sta_mask, mode='test')

    axes = out.detach().cpu().numpy()[0]  # (3, 3) P, T, B
    p_axis, t_axis, b_axis = axes[0], axes[1], axes[2]
    strike, dip, rake = pt_to_sdr(p_axis, t_axis)
    s2, d2, r2 = auxiliary_plane(strike, dip, rake)

    return {
        'strike': strike,
        'dip': dip,
        'rake': rake,
        'strike2': s2,
        'dip2': d2,
        'rake2': r2,
        'p_axis': p_axis,
        't_axis': t_axis,
        'b_axis': b_axis,
        'faulting': faulting_type(rake),
        'n_stations': int(features.shape[0]),
    }


def default_station_row(index: int = 0) -> tuple[str, np.ndarray, np.ndarray]:
    """Neutral starter values when the user adds a station."""
    sid = f'NEW{index + 1}'
    xyz = np.array([20.0, 0.0, 10.0], dtype=np.float32)
    feats = np.zeros(11, dtype=np.float32)
    feats[0] = 1.0
    return sid, xyz, feats


def predict_all_events(
    dataset: dict[str, Any] | None = None,
    model: FocoNet | None = None,
    cache_path: str | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Run FocoNet on every XQ event; cache CSV for the regional map."""
    import pandas as pd

    path = cache_path or config.PREDICTIONS_CACHE
    if not force and os.path.exists(path):
        return pd.read_csv(path).to_dict('records')

    data = dataset if dataset is not None else load_dataset()
    net = model or load_model()
    rows: list[dict[str, Any]] = []

    for eid in sorted(data.keys()):
        parsed = parse_event(data[eid])
        try:
            result = predict(parsed['features'], parsed['xyz'], model=net)
            rows.append(
                {
                    'event_id': eid,
                    'folder_name': eid,
                    'magnitude': parsed['mag'],
                    'n_stations': parsed['n_stations'],
                    'strike': result['strike'],
                    'dip': result['dip'],
                    'rake': result['rake'],
                }
            )
        except Exception as exc:  # noqa: BLE001
            print(f'skip {eid}: {exc}')

    pd.DataFrame(rows).to_csv(path, index=False)
    return rows
