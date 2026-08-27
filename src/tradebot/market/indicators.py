"""Technical indicators.

Pure functions over numpy arrays. No state, no I/O, no configuration — every one
of these is verified against hand-calculated values in
``tests/unit/test_indicators.py``.

Conventions:

* Input arrays are ordered oldest → newest.
* Output arrays are the same length as the input, with ``nan`` for the warm-up
  region. Callers must check for ``nan`` rather than assuming a value exists;
  ``last_valid`` is provided for that.
* Wilder's smoothing is used for RSI, ATR and ADX, matching the standard
  definitions used by charting platforms. Simple rolling means would give
  subtly different numbers and make comparisons with a chart impossible.
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _as_array(values: Array | list[float]) -> Array:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("indicator input must be one-dimensional")
    return arr


def last_valid(values: Array, default: float = float("nan")) -> float:
    """Most recent non-nan value, or ``default`` if there is none."""
    if values.size == 0:
        return default
    finite = values[np.isfinite(values)]
    return float(finite[-1]) if finite.size else default


def _empty_like(values: Array) -> Array:
    return np.full(values.shape, np.nan, dtype=np.float64)


def _wilder_smooth(values: Array, period: int) -> Array:
    """Wilder's smoothing: seed with a simple mean, then y[i] = y[i-1] + (x[i]-y[i-1])/n.

    Equivalent to an EMA with alpha = 1/period, which is what RSI/ATR/ADX use.
    """
    out = _empty_like(values)
    n = values.size
    if n < period or period < 1:
        return out
    seed = float(np.mean(values[:period]))
    out[period - 1] = seed
    prev = seed
    for i in range(period, n):
        prev = prev + (values[i] - prev) / period
        out[i] = prev
    return out


# --------------------------------------------------------------------------- #
# Moving averages
# --------------------------------------------------------------------------- #
def sma(values: Array | list[float], period: int) -> Array:
    """Simple moving average."""
    arr = _as_array(values)
    out = _empty_like(arr)
    if period < 1 or arr.size < period:
        return out
    cumsum = np.cumsum(np.insert(arr, 0, 0.0))
    out[period - 1 :] = (cumsum[period:] - cumsum[:-period]) / period
    return out


def ema(values: Array | list[float], period: int) -> Array:
    """Exponential moving average, seeded with the SMA of the first `period` bars."""
    arr = _as_array(values)
    out = _empty_like(arr)
    if period < 1 or arr.size < period:
        return out
    alpha = 2.0 / (period + 1.0)
    prev = float(np.mean(arr[:period]))
    out[period - 1] = prev
    for i in range(period, arr.size):
        prev = arr[i] * alpha + prev * (1 - alpha)
        out[i] = prev
    return out


def wma(values: Array | list[float], period: int) -> Array:
    """Linearly weighted moving average."""
    arr = _as_array(values)
    out = _empty_like(arr)
    if period < 1 or arr.size < period:
        return out
    weights = np.arange(1, period + 1, dtype=np.float64)
    denom = weights.sum()
    for i in range(period - 1, arr.size):
        out[i] = float(np.dot(arr[i - period + 1 : i + 1], weights) / denom)
    return out


# --------------------------------------------------------------------------- #
# Momentum / oscillators
# --------------------------------------------------------------------------- #
def rsi(values: Array | list[float], period: int = 14) -> Array:
    """Relative Strength Index using Wilder's smoothing.

    Returns 100 when there are no losses in the window (not nan), which is the
    conventional behaviour and avoids special-casing at call sites.
    """
    arr = _as_array(values)
    out = _empty_like(arr)
    if arr.size <= period or period < 1:
        return out

    delta = np.diff(arr)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))

    def _rsi_value(g: float, loss: float) -> float:
        if loss == 0:
            return 100.0 if g > 0 else 50.0
        rs = g / loss
        return 100.0 - (100.0 / (1.0 + rs))

    out[period] = _rsi_value(avg_gain, avg_loss)
    for i in range(period + 1, arr.size):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        out[i] = _rsi_value(avg_gain, avg_loss)
    return out


def roc(values: Array | list[float], period: int = 10) -> Array:
    """Rate of change as a fraction (not a percentage)."""
    arr = _as_array(values)
    out = _empty_like(arr)
    if period < 1 or arr.size <= period:
        return out
    prior = arr[:-period]
    with np.errstate(divide="ignore", invalid="ignore"):
        change = np.where(prior != 0, (arr[period:] - prior) / np.abs(prior), np.nan)
    out[period:] = change
    return out


def momentum(values: Array | list[float], period: int = 10) -> Array:
    """Absolute price change over `period` bars."""
    arr = _as_array(values)
    out = _empty_like(arr)
    if period < 1 or arr.size <= period:
        return out
    out[period:] = arr[period:] - arr[:-period]
    return out


def stochastic(
    high: Array, low: Array, close: Array, period: int = 14, smooth_k: int = 3, smooth_d: int = 3
) -> tuple[Array, Array]:
    """Stochastic oscillator, returning (%K, %D)."""
    h, l_, c = _as_array(high), _as_array(low), _as_array(close)
    raw = _empty_like(c)
    for i in range(period - 1, c.size):
        window_high = float(np.max(h[i - period + 1 : i + 1]))
        window_low = float(np.min(l_[i - period + 1 : i + 1]))
        span = window_high - window_low
        raw[i] = 50.0 if span == 0 else (c[i] - window_low) / span * 100.0
    k = sma(raw, smooth_k) if smooth_k > 1 else raw
    d = sma(k, smooth_d) if smooth_d > 1 else k
    return k, d


def macd(
    values: Array | list[float], fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[Array, Array, Array]:
    """MACD line, signal line and histogram."""
    arr = _as_array(values)
    macd_line = ema(arr, fast) - ema(arr, slow)
    # The signal EMA must ignore the leading nans, or it never starts.
    valid = np.isfinite(macd_line)
    signal_line = _empty_like(arr)
    if valid.any():
        first = int(np.argmax(valid))
        tail = ema(macd_line[first:], signal)
        signal_line[first:] = tail
    return macd_line, signal_line, macd_line - signal_line


# --------------------------------------------------------------------------- #
# Volatility
# --------------------------------------------------------------------------- #
def true_range(high: Array, low: Array, close: Array) -> Array:
    """True range. The first bar has no previous close, so it is high - low."""
    h, l_, c = _as_array(high), _as_array(low), _as_array(close)
    tr = _empty_like(h)
    if h.size == 0:
        return tr
    tr[0] = h[0] - l_[0]
    if h.size > 1:
        prev_close = c[:-1]
        tr[1:] = np.maximum.reduce(
            [
                h[1:] - l_[1:],
                np.abs(h[1:] - prev_close),
                np.abs(l_[1:] - prev_close),
            ]
        )
    return tr


def atr(high: Array, low: Array, close: Array, period: int = 14) -> Array:
    """Average True Range (Wilder)."""
    tr = true_range(high, low, close)
    return _wilder_smooth(tr, period)


def atr_percent(high: Array, low: Array, close: Array, period: int = 14) -> Array:
    """ATR expressed as a fraction of price — comparable across symbols.

    This is what the scanner and risk engine use; raw ATR is meaningless when
    comparing a 100k-dollar symbol against a 0.0001-dollar one.
    """
    c = _as_array(close)
    a = atr(high, low, close, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(c > 0, a / c, np.nan)


def bollinger(
    values: Array | list[float], period: int = 20, num_std: float = 2.0
) -> tuple[Array, Array, Array]:
    """Bollinger bands: (upper, middle, lower)."""
    arr = _as_array(values)
    middle = sma(arr, period)
    std = _empty_like(arr)
    for i in range(period - 1, arr.size):
        std[i] = float(np.std(arr[i - period + 1 : i + 1]))
    return middle + num_std * std, middle, middle - num_std * std


def bollinger_bandwidth(
    values: Array | list[float], period: int = 20, num_std: float = 2.0
) -> Array:
    """(upper - lower) / middle. Low values mean a squeeze."""
    upper, middle, lower = bollinger(values, period, num_std)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(middle != 0, (upper - lower) / middle, np.nan)


def realised_volatility(values: Array | list[float], period: int = 20) -> Array:
    """Standard deviation of log returns over a rolling window (per-bar, not annualised)."""
    arr = _as_array(values)
    out = _empty_like(arr)
    if arr.size < 2:
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret = np.diff(np.log(np.where(arr > 0, arr, np.nan)))
    for i in range(period, arr.size):
        window = log_ret[i - period : i]
        finite = window[np.isfinite(window)]
        if finite.size >= 2:
            out[i] = float(np.std(finite))
    return out


# --------------------------------------------------------------------------- #
# Trend strength
# --------------------------------------------------------------------------- #
def adx(high: Array, low: Array, close: Array, period: int = 14) -> tuple[Array, Array, Array]:
    """Average Directional Index. Returns (adx, +DI, -DI).

    ADX measures trend STRENGTH regardless of direction; +DI/-DI carry the
    direction. The regime detector uses ADX and the strategies use the DIs.
    """
    h, l_, c = _as_array(high), _as_array(low), _as_array(close)
    n = h.size
    adx_out, plus_di, minus_di = _empty_like(h), _empty_like(h), _empty_like(h)
    if n < period * 2 or period < 1:
        return adx_out, plus_di, minus_di

    up_move = h[1:] - h[:-1]
    down_move = l_[:-1] - l_[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = true_range(h, l_, c)[1:]

    smooth_tr = _wilder_smooth(tr, period)
    smooth_plus = _wilder_smooth(plus_dm, period)
    smooth_minus = _wilder_smooth(minus_dm, period)

    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = np.where(smooth_tr > 0, 100.0 * smooth_plus / smooth_tr, np.nan)
        mdi = np.where(smooth_tr > 0, 100.0 * smooth_minus / smooth_tr, np.nan)
        denom = pdi + mdi
        dx = np.where(denom > 0, 100.0 * np.abs(pdi - mdi) / denom, np.nan)

    plus_di[1:] = pdi
    minus_di[1:] = mdi

    # ADX is Wilder's smoothing of DX, which itself starts at index period-1.
    valid = np.isfinite(dx)
    if valid.any():
        first = int(np.argmax(valid))
        smoothed = _wilder_smooth(dx[first:], period)
        adx_out[first + 1 :] = smoothed
    return adx_out, plus_di, minus_di


def linear_slope(values: Array | list[float], period: int = 20) -> Array:
    """Slope of a least-squares fit over a rolling window, normalised by price.

    Positive means rising. Normalising by the window mean makes it comparable
    between symbols.
    """
    arr = _as_array(values)
    out = _empty_like(arr)
    if arr.size < period or period < 2:
        return out
    x = np.arange(period, dtype=np.float64)
    x_centred = x - x.mean()
    denom = float(np.dot(x_centred, x_centred))
    for i in range(period - 1, arr.size):
        window = arr[i - period + 1 : i + 1]
        mean = float(np.mean(window))
        if mean == 0:
            continue
        slope = float(np.dot(x_centred, window - mean) / denom)
        out[i] = slope / mean
    return out


# --------------------------------------------------------------------------- #
# Volume / price-volume
# --------------------------------------------------------------------------- #
def vwap(
    high: Array, low: Array, close: Array, volume: Array, reset_indices: Array | None = None
) -> Array:
    """Volume-weighted average price, optionally resetting at session boundaries.

    ``reset_indices`` is a boolean array marking bars where the session restarts.
    """
    h, l_, c, v = (_as_array(high), _as_array(low), _as_array(close), _as_array(volume))
    typical = (h + l_ + c) / 3.0
    out = _empty_like(c)
    cum_pv = 0.0
    cum_v = 0.0
    for i in range(c.size):
        if reset_indices is not None and bool(reset_indices[i]):
            cum_pv = 0.0
            cum_v = 0.0
        cum_pv += typical[i] * v[i]
        cum_v += v[i]
        out[i] = cum_pv / cum_v if cum_v > 0 else typical[i]
    return out


def vwap_bands(
    high: Array,
    low: Array,
    close: Array,
    volume: Array,
    num_std: float = 2.0,
    reset_indices: Array | None = None,
) -> tuple[Array, Array, Array]:
    """VWAP with volume-weighted standard-deviation bands: (upper, vwap, lower)."""
    h, l_, c, v = (_as_array(high), _as_array(low), _as_array(close), _as_array(volume))
    typical = (h + l_ + c) / 3.0
    centre = vwap(h, l_, c, v, reset_indices)
    upper, lower = _empty_like(c), _empty_like(c)
    cum_v = 0.0
    cum_pv2 = 0.0
    for i in range(c.size):
        if reset_indices is not None and bool(reset_indices[i]):
            cum_v = 0.0
            cum_pv2 = 0.0
        cum_v += v[i]
        cum_pv2 += v[i] * typical[i] ** 2
        if cum_v > 0:
            variance = max(0.0, cum_pv2 / cum_v - centre[i] ** 2)
            dev = float(np.sqrt(variance))
            upper[i] = centre[i] + num_std * dev
            lower[i] = centre[i] - num_std * dev
    return upper, centre, lower


def volume_ratio(volume: Array | list[float], period: int = 20) -> Array:
    """Current volume divided by the average of the PRECEDING `period` bars.

    The baseline excludes the current bar deliberately. Including it lets a
    spike dilute its own reference (a 5x bar against 19 normal bars measures as
    4.17x), which understates exactly the event the indicator exists to detect.
    """
    v = _as_array(volume)
    out = _empty_like(v)
    if period < 1 or v.size <= period:
        return out
    for i in range(period, v.size):
        baseline = float(np.mean(v[i - period : i]))
        if baseline > 0:
            out[i] = v[i] / baseline
    return out


def volume_zscore(volume: Array | list[float], period: int = 30) -> Array:
    """How unusual the current volume is, in standard deviations."""
    v = _as_array(volume)
    out = _empty_like(v)
    for i in range(period, v.size):
        window = v[i - period : i]
        std = float(np.std(window))
        if std > 0:
            out[i] = (v[i] - float(np.mean(window))) / std
    return out


def obv(close: Array, volume: Array) -> Array:
    """On-balance volume."""
    c, v = _as_array(close), _as_array(volume)
    out = np.zeros_like(c)
    for i in range(1, c.size):
        if c[i] > c[i - 1]:
            out[i] = out[i - 1] + v[i]
        elif c[i] < c[i - 1]:
            out[i] = out[i - 1] - v[i]
        else:
            out[i] = out[i - 1]
    return out


# --------------------------------------------------------------------------- #
# Market structure
# --------------------------------------------------------------------------- #
def swing_highs(high: Array, left: int = 2, right: int = 2) -> Array:
    """Boolean array marking pivot highs confirmed by `right` later bars."""
    h = _as_array(high)
    out = np.zeros(h.shape, dtype=bool)
    for i in range(left, h.size - right):
        window_max = float(np.max(h[i - left : i + right + 1]))
        if h[i] == window_max:
            out[i] = True
    return out


def swing_lows(low: Array, left: int = 2, right: int = 2) -> Array:
    """Boolean array marking pivot lows confirmed by `right` later bars."""
    l_ = _as_array(low)
    out = np.zeros(l_.shape, dtype=bool)
    for i in range(left, l_.size - right):
        window_min = float(np.min(l_[i - left : i + right + 1]))
        if l_[i] == window_min:
            out[i] = True
    return out


def donchian(high: Array, low: Array, period: int = 20) -> tuple[Array, Array]:
    """Rolling highest high and lowest low, EXCLUDING the current bar.

    Excluding the current bar matters: a breakout test that includes the current
    bar is trivially always true and is a classic look-ahead bug.
    """
    h, l_ = _as_array(high), _as_array(low)
    upper, lower = _empty_like(h), _empty_like(l_)
    for i in range(period, h.size):
        upper[i] = float(np.max(h[i - period : i]))
        lower[i] = float(np.min(l_[i - period : i]))
    return upper, lower


def price_levels(
    high: Array,
    low: Array,
    atr_value: float,
    lookback: int = 60,
    cluster_atr: float = 0.4,
    min_touches: int = 2,
    left: int = 2,
    right: int = 2,
) -> list[tuple[float, int]]:
    """Cluster recent swing points into support/resistance levels.

    Returns [(level_price, touch_count), ...] sorted by touch count descending.
    Pivots within ``cluster_atr * ATR`` of each other are treated as one level.
    """
    h, l_ = _as_array(high), _as_array(low)
    if h.size == 0 or atr_value <= 0:
        return []
    start = max(0, h.size - lookback)
    hs, ls = h[start:], l_[start:]

    pivots: list[float] = []
    pivots.extend(float(p) for p in hs[swing_highs(hs, left, right)])
    pivots.extend(float(p) for p in ls[swing_lows(ls, left, right)])
    if not pivots:
        return []

    tolerance = cluster_atr * atr_value
    pivots.sort()
    clusters: list[list[float]] = [[pivots[0]]]
    for price in pivots[1:]:
        if price - clusters[-1][-1] <= tolerance:
            clusters[-1].append(price)
        else:
            clusters.append([price])

    levels = [
        (float(np.mean(cluster)), len(cluster))
        for cluster in clusters
        if len(cluster) >= min_touches
    ]
    levels.sort(key=lambda item: item[1], reverse=True)
    return levels


def structure_score(close: Array, lookback: int = 50) -> float:
    """Higher-highs/higher-lows quality in [-1, 1].

    +1 is a clean uptrend structure, -1 a clean downtrend, 0 unstructured.
    """
    c = _as_array(close)
    if c.size < lookback or lookback < 4:
        return 0.0
    window = c[-lookback:]
    half = lookback // 2
    first_high, second_high = float(np.max(window[:half])), float(np.max(window[half:]))
    first_low, second_low = float(np.min(window[:half])), float(np.min(window[half:]))
    score = 0.0
    score += 0.5 if second_high > first_high else -0.5
    score += 0.5 if second_low > first_low else -0.5
    return score
