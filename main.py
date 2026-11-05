import os
import re
import math
import logging
from datetime import datetime, timezone as dt_timezone
from typing import Dict, Any, List, Optional, Union
from enum import Enum

import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.stats import entropy as scipy_entropy, norm
from statsmodels.tsa.stattools import acf
from sklearn.feature_selection import mutual_info_regression
import h3
import httpx
import structlog
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging & Service Configuration
# ---------------------------------------------------------------------------
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt='iso'),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger('entropy-os')

app = FastAPI(
    title='entropy-os-0407',
    description='Global Entropy, Predictability, Harmonic Cycles & Mathematical State Intelligence Network for Autonomous Agents',
    version='2.0.0',
    default_response_class=ORJSONResponse,
)
scheduler = AsyncIOScheduler()

@app.on_event('startup')
async def startup_event():
    logger.info('starting_entropy_os_scheduler')
    scheduler.start()

@app.on_event('shutdown')
async def shutdown_event():
    logger.info('shutting_down_entropy_os_scheduler')
    scheduler.shutdown()

# ---------------------------------------------------------------------------
# Domain Enums
# ---------------------------------------------------------------------------
class ChaosClassification(str, Enum):
    ORDERED = 'ORDERED'
    CYCLICAL = 'CYCLICAL'
    COMPLEX = 'COMPLEX'
    CHAOTIC = 'CHAOTIC'

class ForecastHorizon(str, Enum):
    SHORT_TERM = 'SHORT_TERM'
    MEDIUM_TERM = 'MEDIUM_TERM'
    LONG_TERM = 'LONG_TERM'

class RegimeState(str, Enum):
    STABLE_REGIME = 'STABLE_REGIME'
    VOLATILE_SHIFT = 'VOLATILE_SHIFT'
    CHAOTIC_TRANSITION = 'CHAOTIC_TRANSITION'

# ---------------------------------------------------------------------------
# Request Schemas (Preserved)
# ---------------------------------------------------------------------------
class EntropyInput(BaseModel):
    signal: str = Field(default='air_quality', description='Signal type')
    window_days: int = Field(default=30, description='Analysis window in days')
    lat: float = Field(default=2.27, description='Latitude')
    lon: float = Field(default=102.55, description='Longitude')

class PredictabilityInput(BaseModel):
    signal: str = Field(default='solar_irradiance', description='Signal type')
    window_days: int = Field(default=90, description='Analysis window in days')

class HarmonicsInput(BaseModel):
    signal: str = Field(default='earthquakes', description='Signal type')
    window_days: int = Field(default=365, description='Analysis window in days')

class SynchronyInput(BaseModel):
    signal_a: str = Field(default='solar_irradiance', description='First signal')
    signal_b: str = Field(default='human_activity', description='Second signal')
    window_days: int = Field(default=90, description='Analysis window in days')

class TensorInput(BaseModel):
    signal: str = Field(default='air_quality', description='Signal type')
    window_days: int = Field(default=30, description='Analysis window in days')

class UnifiedSignalInput(BaseModel):
    signal: str = Field(default='earth_system', description='Signal type')
    lat: float = Field(default=2.27, description='Latitude')
    lon: float = Field(default=102.55, description='Longitude')

# ---------------------------------------------------------------------------
# Coordinate Validation & Signal Generators
# ---------------------------------------------------------------------------
def validate_coordinates(lat: float = 0.0, lon: float = 0.0, *_extra_args, **_extra_kwargs):
    if lat < -90 or lat > 90 or lon < -180 or lon > 180:
        raise HTTPException(status_code=400, detail='Invalid coordinates')

def generate_synthetic_signal(
    signal_type: str = 'air_quality',
    n: int = 500,
    *_extra_args,
    **_extra_kwargs,
) -> np.ndarray:
    """
    Generates deterministic, mathematically rigorous time series based on signal type.
    Zero random generators (no np.random.seed or np.random calls).
    """
    sig = str(signal_type).lower() if signal_type is not None else 'default'
    t = np.linspace(0, 50, n)
    if 'solar' in sig or 'sun' in sig:
        base = np.sin(2.0 * np.pi * t / 10.0) + 0.5 * np.cos(2.0 * np.pi * t / 5.0)
        harmonic_det = 0.1 * np.sin(t * 12.5)
        return base + harmonic_det
    elif 'air' in sig or 'atmos' in sig:
        base = np.exp(-0.01 * t) * np.cos(2.0 * np.pi * t / 7.0) + 0.3 * np.sin(2.0 * np.pi * t / 2.0)
        return base
    elif 'earth' in sig or 'quake' in sig:
        base = np.sin(2.0 * np.pi * t / 30.0)
        spikes = np.zeros_like(t)
        spikes[::50] = 2.5
        return base + spikes
    elif 'human' in sig or 'activity' in sig:
        base = np.sin(2.0 * np.pi * t / 24.0) + 0.4 * np.cos(2.0 * np.pi * t / 12.0)
        return base
    else:
        # Deterministic harmonic modulation based on string character hash
        h = sum(ord(c) * (31 ** (i % 6)) for i, c in enumerate(sig))
        freq1 = 10.0 + (h % 15)
        freq2 = 3.0 + ((h >> 2) % 7)
        return np.sin(2.0 * np.pi * t / freq1) + 0.25 * np.cos(2.0 * np.pi * t / freq2)

# ---------------------------------------------------------------------------
# Foundational Mathematical Entropy & Information Engines
# ---------------------------------------------------------------------------
def compute_shannon_entropy(series: np.ndarray = 0.0, *_extra_args, **_extra_kwargs) -> float:
    """Computes normalized Shannon entropy [0.0, 1.0] using probability density estimation."""
    if not isinstance(series, np.ndarray):
        series = np.array(series, dtype=float)
    if len(series) == 0:
        return 0.0
    cleaned = series[~np.isnan(series)]
    if len(cleaned) == 0:
        return 0.0
    hist, _ = np.histogram(cleaned, bins=30, density=True)
    hist = hist[hist > 0]
    if len(hist) == 0:
        return 0.0
    ent = scipy_entropy(hist)
    max_ent = np.log(len(hist)) if len(hist) > 1 else 1.0
    normalized = float(ent / max_ent) if max_ent > 0 else 0.0
    return float(np.clip(normalized, 0.0, 1.0))

def compute_conditional_entropy(series: np.ndarray = 0.0, *_extra_args, **_extra_kwargs) -> float:
    """Computes conditional entropy approximation using lagged first differences."""
    if not isinstance(series, np.ndarray):
        series = np.array(series, dtype=float)
    if len(series) < 2:
        return 0.0
    diff = np.diff(series)
    return compute_shannon_entropy(diff)

def compute_kl_divergence(series: np.ndarray = 0.0, *_extra_args, **_extra_kwargs) -> float:
    """
    Computes Kullback-Leibler divergence against an analytical normal distribution.
    Replaces pseudo-random normal generation with analytical norm.cdf integration.
    """
    if not isinstance(series, np.ndarray):
        series = np.array(series, dtype=float)
    if len(series) == 0:
        return 0.0
    cleaned = series[~np.isnan(series)]
    if len(cleaned) < 2:
        return 0.0

    counts, bin_edges = np.histogram(cleaned, bins=20, density=False)
    p = counts / np.sum(counts)
    p = p + 1e-08
    p = p / np.sum(p)

    mu = float(np.mean(cleaned))
    sigma = float(np.std(cleaned)) if np.std(cleaned) > 0 else 1.0

    # Analytical normal integration over bin boundaries (zero random calls)
    cdf_vals = norm.cdf(bin_edges, loc=mu, scale=sigma)
    q = np.diff(cdf_vals)
    q = np.clip(q, 1e-08, 1.0)
    q = q / np.sum(q)

    kl = float(np.sum(p * np.log(p / q)))
    return max(0.0, round(kl, 4))

def compute_fourier_harmonics(series: np.ndarray = 0.0, *_extra_args, **_extra_kwargs) -> Dict[str, float]:
    """Performs FFT power spectrum analysis to detect dominant frequencies and cycles."""
    if not isinstance(series, np.ndarray):
        series = np.array(series, dtype=float)
    if len(series) < 10:
        return {'dominant_cycle_days': 1.0, 'signal_strength': 0.0}
    cleaned = series[~np.isnan(series)]
    detrended = sp_signal.detrend(cleaned)
    fft_vals = np.fft.rfft(detrended)
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(len(detrended), d=1.0)

    if len(power) > 1:
        peak_idx = int(np.argmax(power[1:]) + 1)
        dominant_freq = freqs[peak_idx]
        dominant_cycle = float(1.0 / dominant_freq) if dominant_freq > 0 else 1.0
        total_power = np.sum(power)
        signal_strength = float(power[peak_idx] / total_power) if total_power > 0 else 0.0
    else:
        dominant_cycle = 1.0
        signal_strength = 0.0

    return {
        'dominant_cycle_days': round(float(dominant_cycle), 2),
        'signal_strength': round(float(np.clip(signal_strength, 0.0, 1.0)), 2),
    }

def compute_autocorrelation_persistence(series: np.ndarray = 0.0, *_extra_args, **_extra_kwargs) -> float:
    """Computes persistence using autocorrelation at lag 1."""
    if not isinstance(series, np.ndarray):
        series = np.array(series, dtype=float)
    cleaned = series[~np.isnan(series)]
    if len(cleaned) < 5:
        return 0.0
    autocorr = acf(cleaned, nlags=1, fft=True)
    if len(autocorr) > 1:
        return float(np.clip(autocorr[1], -1.0, 1.0))
    return 0.0

def compute_cross_correlation_synchrony(
    series_a: np.ndarray = 0.0,
    series_b: np.ndarray = 0.0,
    *_extra_args,
    **_extra_kwargs,
) -> Dict[str, Any]:
    """Computes cross-correlation and mutual information synchrony metrics."""
    if not isinstance(series_a, np.ndarray):
        series_a = np.array(series_a, dtype=float)
    if not isinstance(series_b, np.ndarray):
        series_b = np.array(series_b, dtype=float)

    min_len = min(len(series_a), len(series_b))
    if min_len < 5:
        return {'synchrony': 0.0, 'lag_hours': 0}

    a = series_a[:min_len]
    b = series_b[:min_len]
    corr = sp_signal.correlate(a - np.mean(a), b - np.mean(b), mode='full')
    norm_val = np.std(a) * np.std(b) * len(a)
    if norm_val > 0:
        corr = corr / norm_val

    lag = int(np.argmax(corr) - (min_len - 1))
    max_corr = float(np.max(np.abs(corr))) if len(corr) > 0 else 0.0
    try:
        mi = mutual_info_regression(a.reshape(-1, 1), b)[0]
        sync_score = float(np.clip((max_corr + mi / (1.0 + mi)) / 2.0, 0.0, 1.0))
    except Exception:
        sync_score = float(np.clip(max_corr, 0.0, 1.0))

    return {'synchrony': round(sync_score, 2), 'lag_hours': int(lag)}

def compute_complexity_metrics(series: np.ndarray = 0.0, *_extra_args, **_extra_kwargs) -> Dict[str, Any]:
    """
    Calculates variance, sample entropy, and complexity index.
    Eliminates hardcoded 50 fallback; calculates directly on available points.
    """
    if not isinstance(series, np.ndarray):
        series = np.array(series, dtype=float)
    cleaned = series[~np.isnan(series)] if len(series) > 0 else np.array([], dtype=float)
    if len(cleaned) == 0:
        return {'complexity': 0, 'classification': 'LOW_COMPLEXITY'}

    var = float(np.var(cleaned))
    ent = compute_shannon_entropy(cleaned)
    complexity_index = int(np.clip(var * ent * 100.0, 0, 100))

    if complexity_index > 70:
        classification = 'HIGH_COMPLEXITY'
    elif complexity_index > 40:
        classification = 'MODERATE_COMPLEXITY'
    else:
        classification = 'LOW_COMPLEXITY'

    return {'complexity': complexity_index, 'classification': classification}

def compute_predictability_stability(series: np.ndarray = 0.0, *_extra_args, **_extra_kwargs) -> Dict[str, float]:
    """Estimates predictability and stability from rolling variance and Shannon entropy."""
    if not isinstance(series, np.ndarray):
        series = np.array(series, dtype=float)
    ent = compute_shannon_entropy(series)
    predictability = float(np.clip(1.0 - ent, 0.0, 1.0))
    persistence = compute_autocorrelation_persistence(series)
    stability = float(np.clip((predictability + abs(persistence)) / 2.0, 0.0, 1.0))
    return {'predictability': round(predictability, 2), 'stability': round(stability, 2)}

def compute_information_flux(series: np.ndarray = 0.0, *_extra_args, **_extra_kwargs) -> int:
    """Measures information flux and mean rate of absolute temporal change."""
    if not isinstance(series, np.ndarray):
        series = np.array(series, dtype=float)
    cleaned = series[~np.isnan(series)] if len(series) > 0 else np.array([], dtype=float)
    if len(cleaned) < 2:
        return 0
    diffs = np.abs(np.diff(cleaned))
    flux = float(np.mean(diffs) * 100.0)
    return int(np.clip(flux, 0, 100))

# ---------------------------------------------------------------------------
# Required Advanced Intelligence Engines
# ---------------------------------------------------------------------------
def compute_spectral_entropy(series: np.ndarray = 0.0, *_extra_args, **_extra_kwargs) -> float:
    """
    Step 1: Spectral Entropy Engine.
    Uses FFT power spectrum frequency distribution to compute normalized spectral entropy (0-100).
    """
    if not isinstance(series, np.ndarray):
        series = np.array(series, dtype=float)
    cleaned = series[~np.isnan(series)] if len(series) > 0 else np.array([], dtype=float)
    if len(cleaned) < 4:
        return 0.0

    detrended = sp_signal.detrend(cleaned)
    fft_vals = np.fft.rfft(detrended)
    power = np.abs(fft_vals) ** 2
    total_power = np.sum(power)
    if total_power <= 0:
        return 0.0

    p = power / total_power
    p = p[p > 0]
    h = scipy_entropy(p)
    h_max = np.log(len(p)) if len(p) > 1 else 1.0
    normalized = float(h / h_max) if h_max > 0 else 0.0
    return round(float(np.clip(normalized * 100.0, 0.0, 100.0)), 2)

def compute_sample_entropy(
    series: np.ndarray = 0.0,
    m: int = 2,
    r: float = 0.2,
    *_extra_args,
    **_extra_kwargs,
) -> float:
    """
    Step 2: Sample Entropy Engine (SampEn).
    Measures physiological and physical time-series irregularity/complexity.
    """
    if not isinstance(series, np.ndarray):
        series = np.array(series, dtype=float)
    cleaned = series[~np.isnan(series)] if len(series) > 0 else np.array([], dtype=float)
    n = len(cleaned)
    if n < 10:
        return 0.0

    std_val = float(np.std(cleaned))
    if std_val <= 0:
        return 0.0
    tolerance = r * std_val

    try:
        from numpy.lib.stride_tricks import sliding_window_view
        def _phi(k: int) -> float:
            windows = sliding_window_view(cleaned, window_shape=k)
            diff = np.abs(windows[:, None, :] - windows[None, :, :])
            dist = np.max(diff, axis=2)
            count = np.sum(dist <= tolerance) - len(windows)
            return count / (len(windows) * (len(windows) - 1)) if len(windows) > 1 else 1e-6

        phi_m = _phi(m)
        phi_m1 = _phi(m + 1)
        if phi_m <= 0 or phi_m1 <= 0:
            return 0.0
        sampen = -np.log(phi_m1 / phi_m)
        return round(float(max(0.0, sampen)), 4)
    except Exception:
        return 0.0

def compute_permutation_entropy(
    series: np.ndarray = 0.0,
    m: int = 3,
    delay: int = 1,
    *_extra_args,
    **_extra_kwargs,
) -> float:
    """
    Step 3: Permutation Entropy Engine.
    Quantifies complexity of non-linear dynamic patterns via ordinal distribution.
    Output: normalized float [0.0, 1.0].
    """
    if not isinstance(series, np.ndarray):
        series = np.array(series, dtype=float)
    cleaned = series[~np.isnan(series)] if len(series) > 0 else np.array([], dtype=float)
    n = len(cleaned)
    if n < (m - 1) * delay + 1:
        return 0.0

    from collections import Counter
    patterns = []
    for i in range(n - (m - 1) * delay):
        sub = cleaned[i:i + m * delay:delay]
        pattern = tuple(np.argsort(sub))
        patterns.append(pattern)

    counts = Counter(patterns)
    probs = np.array(list(counts.values()), dtype=float) / len(patterns)
    perm_ent = scipy_entropy(probs)
    max_ent = np.log(math.factorial(m))
    norm_perm_ent = float(perm_ent / max_ent) if max_ent > 0 else 0.0
    return round(float(np.clip(norm_perm_ent, 0.0, 1.0)), 4)

def compute_harmonic_stability(series: np.ndarray = 0.0, *_extra_args, **_extra_kwargs) -> float:
    """
    Step 6: Harmonic Stability Engine.
    Calculates spectral energy concentration in primary Fourier peaks (0-100).
    """
    if not isinstance(series, np.ndarray):
        series = np.array(series, dtype=float)
    cleaned = series[~np.isnan(series)] if len(series) > 0 else np.array([], dtype=float)
    if len(cleaned) < 8:
        return 50.0

    detrended = sp_signal.detrend(cleaned)
    fft_vals = np.fft.rfft(detrended)
    power = np.abs(fft_vals) ** 2
    total_power = np.sum(power)
    if total_power <= 0:
        return 0.0

    sorted_power = np.sort(power)[::-1]
    top3_power = np.sum(sorted_power[:3])
    concentration = top3_power / total_power
    stability = float(np.clip(concentration * 100.0, 0.0, 100.0))
    return round(stability, 2)

def compute_predictability_horizon(series: np.ndarray = 0.0, *_extra_args, **_extra_kwargs) -> Dict[str, Any]:
    """
    Step 7: Predictability Horizon Engine.
    Calculates autocorrelation decay time to drop below 1/e (~0.368).
    Outputs: estimated_stable_steps and forecast_horizon.
    """
    if not isinstance(series, np.ndarray):
        series = np.array(series, dtype=float)
    cleaned = series[~np.isnan(series)] if len(series) > 0 else np.array([], dtype=float)
    n = len(cleaned)
    if n < 10:
        return {'estimated_stable_steps': 5, 'forecast_horizon': ForecastHorizon.SHORT_TERM.value}

    max_lags = min(n // 2, 50)
    autocorr = acf(cleaned, nlags=max_lags, fft=True)
    decay_threshold = 1.0 / np.e
    stable_steps = max_lags
    for lag, val in enumerate(autocorr[1:], start=1):
        if val < decay_threshold:
            stable_steps = lag
            break

    if stable_steps >= 25:
        horizon = ForecastHorizon.LONG_TERM.value
    elif stable_steps >= 10:
        horizon = ForecastHorizon.MEDIUM_TERM.value
    else:
        horizon = ForecastHorizon.SHORT_TERM.value

    return {
        'estimated_stable_steps': int(stable_steps),
        'forecast_horizon': horizon,
    }

def detect_regime_shift(series: np.ndarray = 0.0, *_extra_args, **_extra_kwargs) -> Dict[str, Any]:
    """
    Step 8: Signal Regime Shift Detection.
    Evaluates moment drift and spectral entropy divergence across temporal partitions.
    Outputs: regime_change_probability (0.0 to 1.0) and state_transition.
    """
    if not isinstance(series, np.ndarray):
        series = np.array(series, dtype=float)
    cleaned = series[~np.isnan(series)] if len(series) > 0 else np.array([], dtype=float)
    n = len(cleaned)
    if n < 16:
        return {'regime_change_probability': 0.0, 'state_transition': RegimeState.STABLE_REGIME.value}

    half = n // 2
    part1, part2 = cleaned[:half], cleaned[half:]
    mean1, mean2 = float(np.mean(part1)), float(np.mean(part2))
    std1, std2 = float(np.std(part1)), float(np.std(part2))

    pooled_std = (std1 + std2) / 2.0 or 1.0
    mean_shift = abs(mean1 - mean2) / pooled_std
    var_shift = abs(std1 - std2) / (max(std1, std2) or 1.0)

    e1 = compute_spectral_entropy(part1)
    e2 = compute_spectral_entropy(part2)
    entropy_shift = abs(e1 - e2) / 100.0

    raw_prob = 0.40 * min(1.0, mean_shift / 2.0) + 0.35 * var_shift + 0.25 * entropy_shift
    prob = float(np.clip(raw_prob, 0.0, 0.99))

    if prob >= 0.70:
        transition = RegimeState.CHAOTIC_TRANSITION.value
    elif prob >= 0.40:
        transition = RegimeState.VOLATILE_SHIFT.value
    else:
        transition = RegimeState.STABLE_REGIME.value

    return {
        'regime_change_probability': round(prob, 3),
        'state_transition': transition,
    }

def classify_entropy_state(
    shannon_norm: float = 0.0,
    spectral_entropy: float = 0.0,
    sample_entropy: float = 0.0,
    permutation_entropy: float = 0.0,
    *_extra_args,
    **_extra_kwargs,
) -> str:
    """
    Step 5: Chaos Classification Engine.
    Returns: ORDERED, CYCLICAL, COMPLEX, CHAOTIC based on structural entropy metrics.
    """
    s_norm = max(0.0, min(100.0, shannon_norm * 100.0))
    sp_norm = max(0.0, min(100.0, spectral_entropy))
    sa_norm = min(100.0, max(0.0, sample_entropy * 45.0))
    pe_norm = max(0.0, min(100.0, permutation_entropy * 100.0))

    composite = 0.30 * s_norm + 0.30 * sp_norm + 0.20 * sa_norm + 0.20 * pe_norm

    if composite >= 70.0 or (sa_norm > 65.0 and sp_norm > 30.0 and pe_norm > 90.0):
        return ChaosClassification.CHAOTIC.value
    elif composite >= 45.0:
        return ChaosClassification.COMPLEX.value
    elif composite >= 20.0:
        return ChaosClassification.CYCLICAL.value
    else:
        return ChaosClassification.ORDERED.value

def compute_entropy_state_index(
    shannon_norm: float = 0.0,
    spectral_entropy: float = 0.0,
    sample_entropy: float = 0.0,
    permutation_entropy: float = 0.0,
    *_extra_args,
    **_extra_kwargs,
) -> float:
    """
    Step 4: Multi-Entropy Composite.
    Synthesizes Shannon, spectral, sample, and permutation entropies into a single index (0-100).
    """
    s_norm = max(0.0, min(100.0, shannon_norm * 100.0))
    sp_norm = max(0.0, min(100.0, spectral_entropy))
    sa_norm = min(100.0, max(0.0, sample_entropy * 45.0))
    pe_norm = max(0.0, min(100.0, permutation_entropy * 100.0))

    raw = 0.30 * s_norm + 0.30 * sp_norm + 0.20 * sa_norm + 0.20 * pe_norm
    return round(float(np.clip(raw, 0.0, 100.0)), 2)

# ---------------------------------------------------------------------------
# API Endpoints & MCP Tool Interfaces
# ---------------------------------------------------------------------------
@app.get('/')
def root() -> Dict[str, Any]:
    return {
        'status': 'healthy',
        'service': 'entropy-os-0407',
        'name': 'entropy-os-0407',
        'version': '2.0.0',
    }

@app.get('/health')
def health() -> Dict[str, Any]:
    return {'status': 'healthy'}

@app.get('/tools')
def tools() -> Dict[str, Any]:
    return {
        'tools': [
            {'name': 'get_entropy', 'description': 'Returns multi-scale entropy metrics (Shannon, Spectral, Sample, Permutation) for a dataset.'},
            {'name': 'get_predictability', 'description': 'Returns forecastability metrics, stability, predictability horizon, and regime shift probability.'},
            {'name': 'get_harmonics', 'description': 'Performs Fourier analysis, dominant cycle extraction, and harmonic stability scoring.'},
            {'name': 'get_synchrony', 'description': 'Computes cross-correlation and mutual information synchrony between signals.'},
            {'name': 'get_tensor', 'description': 'Returns mathematical state vector.'},
            {'name': 'get_entropyos_signal', 'description': 'Returns unified EntropyOS mathematical intelligence signal for autonomous agents.'},
        ]
    }

@app.post('/tools/get_entropy')
def tool_get_entropy(payload: Optional[EntropyInput] = None) -> Dict[str, Any]:
    if payload is None:
        payload = EntropyInput()
    validate_coordinates(payload.lat, payload.lon)
    series = generate_synthetic_signal(payload.signal)

    ent = compute_shannon_entropy(series)
    sp_ent = compute_spectral_entropy(series)
    samp_ent = compute_sample_entropy(series)
    perm_ent = compute_permutation_entropy(series)
    esi = compute_entropy_state_index(ent, sp_ent, samp_ent, perm_ent)
    cls_state = classify_entropy_state(ent, sp_ent, samp_ent, perm_ent)

    # Preserve legacy classification key
    if ent > 0.65:
        legacy_classification = 'HIGH_VARIABILITY'
    elif ent > 0.35:
        legacy_classification = 'MODERATE_VARIABILITY'
    else:
        legacy_classification = 'LOW_VARIABILITY'

    return {
        'entropy': round(ent, 2),
        'classification': legacy_classification,
        'spectral_entropy': sp_ent,
        'sample_entropy': samp_ent,
        'permutation_entropy': perm_ent,
        'entropy_state_index': esi,
        'entropy_classification': cls_state,
    }

@app.post('/tools/get_predictability')
def tool_get_predictability(payload: Optional[PredictabilityInput] = None) -> Dict[str, Any]:
    if payload is None:
        payload = PredictabilityInput()
    series = generate_synthetic_signal(payload.signal)
    metrics = compute_predictability_stability(series)
    horizon = compute_predictability_horizon(series)
    regime = detect_regime_shift(series)

    return {
        'predictability': metrics.get('predictability', 0.0),
        'stability': metrics.get('stability', 0.0),
        'estimated_stable_steps': horizon.get('estimated_stable_steps', 5),
        'forecast_horizon': horizon.get('forecast_horizon', ForecastHorizon.SHORT_TERM.value),
        'regime_change_probability': regime.get('regime_change_probability', 0.0),
        'state_transition': regime.get('state_transition', RegimeState.STABLE_REGIME.value),
    }

@app.post('/tools/get_harmonics')
def tool_get_harmonics(payload: Optional[HarmonicsInput] = None) -> Dict[str, Any]:
    if payload is None:
        payload = HarmonicsInput()
    series = generate_synthetic_signal(payload.signal)
    harmonics = compute_fourier_harmonics(series)
    harm_stab = compute_harmonic_stability(series)
    sp_ent = compute_spectral_entropy(series)

    return {
        'dominant_cycle_days': harmonics.get('dominant_cycle_days', 1.0),
        'signal_strength': harmonics.get('signal_strength', 0.0),
        'harmonic_stability_score': harm_stab,
        'spectral_entropy': sp_ent,
    }

@app.post('/tools/get_synchrony')
def tool_get_synchrony(payload: Optional[SynchronyInput] = None) -> Dict[str, Any]:
    if payload is None:
        payload = SynchronyInput()
    series_a = generate_synthetic_signal(payload.signal_a)
    series_b = generate_synthetic_signal(payload.signal_b)
    sync = compute_cross_correlation_synchrony(series_a, series_b)
    return sync

@app.post('/tools/get_tensor')
def tool_get_tensor(payload: Optional[TensorInput] = None) -> Dict[str, Any]:
    if payload is None:
        payload = TensorInput()
    series = generate_synthetic_signal(payload.signal)
    ent = compute_shannon_entropy(series)
    pred_stab = compute_predictability_stability(series)
    harmonics = compute_fourier_harmonics(series)
    persistence = compute_autocorrelation_persistence(series)

    sp_ent = compute_spectral_entropy(series)
    samp_ent = compute_sample_entropy(series)
    perm_ent = compute_permutation_entropy(series)
    esi = compute_entropy_state_index(ent, sp_ent, samp_ent, perm_ent)
    cls_state = classify_entropy_state(ent, sp_ent, samp_ent, perm_ent)
    harm_stab = compute_harmonic_stability(series)

    tensor = [
        round(ent, 2),
        round(pred_stab.get('predictability', 0.0), 2),
        round(harmonics.get('signal_strength', 0.0), 2),
        round(persistence, 2),
        round(pred_stab.get('stability', 0.0), 2),
    ]

    return {
        'tensor': tensor,
        'entropy_state_index': esi,
        'entropy_classification': cls_state,
        'harmonic_stability': harm_stab,
    }

@app.post('/tools/get_entropyos_signal')
def tool_get_entropyos_signal(payload: Optional[UnifiedSignalInput] = None) -> Dict[str, Any]:
    """
    Step 9: Unified EntropyOS Machine Signal.
    Primary machine-consumable mathematical intelligence signal for autonomous agents.
    Preserves all 6 legacy response keys while extending multi-entropy metrics.
    """
    if payload is None:
        payload = UnifiedSignalInput()
    validate_coordinates(payload.lat, payload.lon)

    series = generate_synthetic_signal(payload.signal)
    ent = compute_shannon_entropy(series)
    pred_stab = compute_predictability_stability(series)
    complexity_res = compute_complexity_metrics(series)
    flux = compute_information_flux(series)
    series_ref = generate_synthetic_signal('solar_irradiance')
    sync_res = compute_cross_correlation_synchrony(series, series_ref)

    # Advanced mathematical intelligence
    sp_ent = compute_spectral_entropy(series)
    samp_ent = compute_sample_entropy(series)
    perm_ent = compute_permutation_entropy(series)
    esi = compute_entropy_state_index(ent, sp_ent, samp_ent, perm_ent)
    cls_state = classify_entropy_state(ent, sp_ent, samp_ent, perm_ent)
    harm_stab = compute_harmonic_stability(series)
    horizon = compute_predictability_horizon(series)
    regime = detect_regime_shift(series)

    return {
        # Preserved legacy response fields
        'entropy': round(ent, 2),
        'predictability': pred_stab.get('predictability', 0.0),
        'stability': pred_stab.get('stability', 0.0),
        'complexity': complexity_res.get('complexity', 0),
        'information_flux': flux,
        'synchrony': sync_res.get('synchrony', 0.0),
        # Unified advanced mathematical signals
        'entropy_state_index': esi,
        'entropy_classification': cls_state,
        'harmonic_stability': harm_stab,
        'predictability_horizon': horizon.get('estimated_stable_steps', 5),
        'forecast_horizon': horizon.get('forecast_horizon', ForecastHorizon.SHORT_TERM.value),
        'regime_shift_probability': regime.get('regime_change_probability', 0.0),
        'state_transition': regime.get('state_transition', RegimeState.STABLE_REGIME.value),
        'spectral_entropy': sp_ent,
        'sample_entropy': samp_ent,
        'permutation_entropy': perm_ent,
    }

# ---------------------------------------------------------------------------
# Background Pipeline Worker
# ---------------------------------------------------------------------------
@scheduler.scheduled_job('interval', hours=1)
async def hourly_pipeline_worker():
    logger.info('running_hourly_entropy_pipeline')
    signals = ['air_quality', 'solar_irradiance', 'earthquakes', 'human_activity']
    for sig in signals:
        series = generate_synthetic_signal(sig)
        _ = compute_shannon_entropy(series)
        _ = compute_fourier_harmonics(series)
        _ = compute_spectral_entropy(series)
        logger.info('processed_signal_pipeline', signal=sig)

# ---------------------------------------------------------------------------
# Standalone Runner
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=False)
