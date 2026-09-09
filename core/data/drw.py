from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.random import Generator

@dataclass
class LightCurve:
    t: np.ndarray        
    mag: np.ndarray     
    mag_err: np.ndarray 
    log_tau: float       
    log_sigma: float  

def sample_observation_times(
    seq_len: int,
    cadence_days: float,
    cadence_jitter: float,
    season_gap_frac: float,
    rng: Generator,
) -> np.ndarray:
    steps = np.clip(
        rng.normal(cadence_days, cadence_jitter, size=seq_len),
        0.5 * cadence_days,
        None,
    )
    t = np.cumsum(steps)
    year = 365.25
    phase = (t % year) / year
    in_gap = phase > (1.0 - season_gap_frac)
    t = np.where(in_gap, t + season_gap_frac * year, t)
    return np.sort(t)

def simulate_drw(
    seq_len: int,
    log_tau: float,
    log_sigma: float,
    cadence_days: float,
    cadence_jitter: float,
    season_gap_frac: float,
    phot_err: float,
    rng: Generator,
) -> LightCurve:
    tau = 10.0**log_tau
    sigma = 10.0**log_sigma
    t = sample_observation_times(seq_len, cadence_days, cadence_jitter, season_gap_frac, rng)
    dt = np.diff(t, prepend=t[0])
    x = np.empty(seq_len, dtype=np.float64)
    x[0] = rng.normal(0.0, sigma)  
    decay = np.exp(-dt / tau)
    var = sigma**2 * (1.0 - np.exp(-2.0 * dt / tau))
    noise = rng.normal(0.0, np.sqrt(np.maximum(var, 1e-12)))
    for i in range(1, seq_len):
        x[i] = x[i - 1] * decay[i] + noise[i]
    mag_err = np.full(seq_len, phot_err) * rng.uniform(0.8, 1.2, size=seq_len)
    mag = x + rng.normal(0.0, mag_err)
    return LightCurve(
        t=t.astype(np.float32),
        mag=mag.astype(np.float32),
        mag_err=mag_err.astype(np.float32),
        log_tau=float(log_tau),
        log_sigma=float(log_sigma),
    )

def sample_params(
    log_tau_range: tuple[float, float],
    log_sigma_range: tuple[float, float],
    rng: Generator,
) -> tuple[float, float]:
    log_tau = rng.uniform(*log_tau_range)
    log_sigma = rng.uniform(*log_sigma_range)
    return log_tau, log_sigma

def simulate_clagn(
    seq_len: int,
    log_tau: float,
    log_sigma: float,
    cadence_days: float,
    cadence_jitter: float,
    season_gap_frac: float,
    phot_err: float,
    rng: Generator,
    level_shift_sigma: tuple[float, float] = (2.0, 4.0),
    amp_factor_range: tuple[float, float] = (0.3, 2.5),
) -> LightCurve:
    lc = simulate_drw(
        seq_len, log_tau, log_sigma, cadence_days, cadence_jitter,
        season_gap_frac, phot_err, rng,
    )
    sigma = 10.0**log_sigma
    t_c_idx = int(rng.uniform(0.2, 0.8) * seq_len)
    sign = rng.choice([-1.0, 1.0])
    shift = sign * rng.uniform(*level_shift_sigma) * sigma
    amp = rng.uniform(*amp_factor_range)
    mag = lc.mag.copy()
    base = mag.mean()
    post = mag[t_c_idx:]
    mag[t_c_idx:] = base + amp * (post - base) + shift
    lc.mag = mag.astype(np.float32)
    return lc