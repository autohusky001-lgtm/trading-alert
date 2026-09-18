#!/usr/bin/env python3
"""
alert.py — Trading Signal Bot
Model v2 (2025-2026) · XGBoost · H1 Candles
GitHub Actions compatible · Telegram Alert with Chart

Features (38): ret_1..10, body_ratio, upper/lower_wick,
               ema_21/50/100/200 dist+slope, rsi_7/14/21,
               macd, macd_hist, bb_20/50_pos, atr_14,
               stoch_14/21, dist_high/low_20/50,
               hour, hour_sin, hour_cos, dow,
               is_london, is_overlap, dir_lag_1/2/3
"""

import os
import io
import json
import pickle
import warnings
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
PAIRS_CONFIG = {
    'USDJPY': {'threshold': 0.60, 'enabled': True,  'wr': 60.53},
    'USDCHF': {'threshold': 0.65, 'enabled': True,  'wr': 59.48},
    'GBPUSD': {'threshold': 0.60, 'enabled': True,  'wr': 55.65},
    'EURUSD': {'threshold': 0.65, 'enabled': True,  'wr': 55.27},
    'NZDUSD': {'threshold': 0.65, 'enabled': False, 'wr': 51.37},  # ปิด — WR ต่ำ
}

YAHOO_TICKERS = {
    'USDJPY': 'USDJPY=X',
    'USDCHF': 'USDCHF=X',
    'GBPUSD': 'GBPUSD=X',
    'EURUSD': 'EURUSD=X',
    'NZDUSD': 'NZDUSD=X',
}

# Session filter: London session
SESSION_START_UTC = 7    # 07:00 UTC
SESSION_END_UTC   = 16   # 16:00 UTC
TRADING_WEEKDAYS  = [1, 2, 3]  # Tue=1, Wed=2, Thu=3 (Mon=0)

# Model files directory (same dir as this script, or override via env)
SCRIPT_DIR   = Path(__file__).parent
MODELS_DIR   = Path(os.environ.get('MODELS_DIR', SCRIPT_DIR))
SIGNALS_FILE = SCRIPT_DIR / 'signals.json'
MAX_SIGNALS_HISTORY = 500

# Telegram
TELEGRAM_TOKEN   = os.environ.get('TELEGRAM_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

# Chart palette
CHART_STYLE = {
    'bg':    '#0d1117',
    'sur':   '#161b22',
    'bor':   '#30363d',
    'txt':   '#e6edf3',
    'dim':   '#7d8590',
    'call':  '#3fb950',
    'put':   '#f85149',
    'ema21': '#f39c12',
    'ema50': '#2ecc71',
    'ema200':'#e74c3c',
    'up':    '#26a69a',
    'dn':    '#ef5350',
}

# ─────────────────────────────────────────────────────────────────────────────
# SESSION GATE
# ─────────────────────────────────────────────────────────────────────────────
def is_trading_session(now=None):
    """Return True only inside London session on Tue-Thu."""
    if now is None:
        now = datetime.now(timezone.utc)
    if now.weekday() not in TRADING_WEEKDAYS:
        log.info(f"Outside trading days (today={now.strftime('%A')}). Skip.")
        return False
    if not (SESSION_START_UTC <= now.hour < SESSION_END_UTC):
        log.info(f"Outside London session (UTC {now.hour:02d}:xx). Skip.")
        return False
    return True

# ─────────────────────────────────────────────────────────────────────────────
# TELEGRAM
# ─────────────────────────────────────────────────────────────────────────────
def send_telegram_text(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram credentials missing — skip send_text.")
        return False
    url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
    try:
        r = requests.post(url, json={
            'chat_id': TELEGRAM_CHAT_ID,
            'text': text,
            'parse_mode': 'HTML',
            'disable_web_page_preview': True,
        }, timeout=15)
        r.raise_for_status()
        log.info("Telegram text sent OK")
        return True
    except Exception as e:
        log.error(f"Telegram text failed: {e}")
        return False


def send_telegram_photo(img_bytes, caption):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram credentials missing — skip send_photo.")
        return False
    url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto'
    try:
        r = requests.post(
            url,
            data={'chat_id': TELEGRAM_CHAT_ID, 'caption': caption, 'parse_mode': 'HTML'},
            files={'photo': ('chart.png', img_bytes, 'image/png')},
            timeout=30,
        )
        r.raise_for_status()
        log.info("Telegram photo sent OK")
        return True
    except Exception as e:
        log.error(f"Telegram photo failed: {e}")
        return False

# ─────────────────────────────────────────────────────────────────────────────
# SIGNALS JSON
# ─────────────────────────────────────────────────────────────────────────────
def load_signals_json():
    if SIGNALS_FILE.exists():
        try:
            with open(SIGNALS_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {'signals': [], 'last_update': None, 'stats': {'total_signals': 0}}


def save_signals_json(data):
    data['signals'] = data['signals'][-MAX_SIGNALS_HISTORY:]
    data['last_update'] = datetime.now(timezone.utc).isoformat()
    data['stats']['total_signals'] = len(data['signals'])
    with open(SIGNALS_FILE, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    log.info(f"signals.json saved ({data['stats']['total_signals']} entries)")


def append_signal(store, pair, signal, score, price):
    entry = {
        'time':   datetime.now(timezone.utc).isoformat(),
        'pair':   pair,
        'signal': signal,
        'score':  round(score * 100, 2),
        'price':  price,
    }
    store['signals'].append(entry)
    log.info(f"  Signal recorded: {entry}")

# ─────────────────────────────────────────────────────────────────────────────
# PRICE FETCH
# ─────────────────────────────────────────────────────────────────────────────
def fetch_prices(pair):
    ticker = YAHOO_TICKERS.get(pair)
    if not ticker:
        log.error(f"No Yahoo ticker for {pair}")
        return None
    try:
        df = yf.download(ticker, period='30d', interval='1h',
                         auto_adjust=True, progress=False)
        if df is None or len(df) < 50:
            log.warning(f"{pair}: not enough data ({len(df) if df is not None else 0} rows)")
            return None
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df = df[['Open','High','Low','Close','Volume']].copy()
        df.columns = ['open','high','low','close','volume']
        df.dropna(subset=['open','high','low','close'], inplace=True)
        log.info(f"{pair}: {len(df)} H1 candles, last={df.index[-1]}")
        return df
    except Exception as e:
        log.error(f"{pair}: fetch_prices error: {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING  ← must match training pipeline exactly
# ─────────────────────────────────────────────────────────────────────────────
def _ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def _rsi(series, period):
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=period-1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period-1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _stoch(high, low, close, period):
    h = high.rolling(period).max()
    l = low.rolling(period).min()
    return ((close - l) / (h - l).replace(0, np.nan)) * 100

def _bb_pos(close, period):
    m   = close.rolling(period).mean()
    std = close.rolling(period).std()
    ub  = m + 2 * std
    lb  = m - 2 * std
    return (close - lb) / (ub - lb).replace(0, np.nan)

def _atr(high, low, close, period):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def build_features(df):
    """Build 38 features matching Model v2 training pipeline."""
    o, h, l, c = df['open'], df['high'], df['low'], df['close']
    feat = pd.DataFrame(index=df.index)

    # Price returns
    for n in [1, 2, 3, 5, 10]:
        feat[f'ret_{n}'] = c.pct_change(n)

    # Candle shape
    body = (c - o).abs()
    rng  = (h - l).replace(0, np.nan)
    feat['body_ratio']  = body / rng
    feat['upper_wick']  = (h - pd.concat([c, o], axis=1).max(axis=1)) / rng
    feat['lower_wick']  = (pd.concat([c, o], axis=1).min(axis=1) - l) / rng

    # EMA distances + slopes (21, 50, 100 have slope; 200 dist only)
    for p in [21, 50, 100, 200]:
        ema = _ema(c, p)
        feat[f'ema_{p}_dist'] = (c - ema) / c
    for p in [21, 50, 100]:
        ema = _ema(c, p)
        feat[f'ema_{p}_slope'] = ema.diff() / c

    # RSI (normalised 0-1)
    for p in [7, 14, 21]:
        feat[f'rsi_{p}'] = _rsi(c, p) / 100.0

    # MACD (12, 26, 9)
    ema12   = _ema(c, 12)
    ema26   = _ema(c, 26)
    macd_l  = ema12 - ema26
    sig_l   = _ema(macd_l, 9)
    feat['macd']      = macd_l / c
    feat['macd_hist'] = (macd_l - sig_l) / c

    # Bollinger Band position
    feat['bb_20_pos'] = _bb_pos(c, 20)
    feat['bb_50_pos'] = _bb_pos(c, 50)

    # ATR (normalised)
    feat['atr_14'] = _atr(h, l, c, 14) / c

    # Stochastic (normalised 0-1)
    feat['stoch_14'] = _stoch(h, l, c, 14) / 100.0
    feat['stoch_21'] = _stoch(h, l, c, 21) / 100.0

    # Distance from recent high/low
    feat['dist_high_20'] = (h.rolling(20).max() - c) / c
    feat['dist_low_20']  = (c - l.rolling(20).min()) / c
    feat['dist_high_50'] = (h.rolling(50).max() - c) / c
    feat['dist_low_50']  = (c - l.rolling(50).min()) / c

    # Time features
    idx  = pd.DatetimeIndex(df.index)
    hour = idx.hour
    feat['hour']     = hour
    feat['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    feat['dow']      = idx.dayofweek

    # Session flags
    feat['is_london']  = ((hour >= 7) & (hour < 16)).astype(float)
    feat['is_overlap'] = ((hour >= 12) & (hour < 16)).astype(float)

    # Directional lags
    direction = (c > o).astype(float)
    for lag in [1, 2, 3]:
        feat[f'dir_lag_{lag}'] = direction.shift(lag)

    feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    return feat

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────
def load_model(pair):
    model_path = MODELS_DIR / f'{pair}_model.pkl'
    feats_path = MODELS_DIR / f'{pair}_features.pkl'
    if not model_path.exists():
        log.error(f"Model not found: {model_path}")
        return None, None
    if not feats_path.exists():
        log.error(f"Features not found: {feats_path}")
        return None, None
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(feats_path, 'rb') as f:
            features = pickle.load(f)
        log.info(f"{pair}: model loaded ({len(features)} features)")
        return model, features
    except Exception as e:
        log.error(f"{pair}: load_model error: {e}")
        return None, None

# ─────────────────────────────────────────────────────────────────────────────
# CHART GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def generate_chart(df, pair, signal, score, price, n_candles=60):
    """Dark-theme candlestick chart with EMAs and signal arrow."""
    df_plot = df.tail(n_candles).copy()
    bg  = CHART_STYLE['bg']
    sur = CHART_STYLE['sur']
    bor = CHART_STYLE['bor']
    txt = CHART_STYLE['txt']
    dim = CHART_STYLE['dim']

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=bg)
    ax.set_facecolor(bg)

    x = np.arange(len(df_plot))

    # Candlesticks
    for i, (_, row) in enumerate(df_plot.iterrows()):
        o_p, h_p, l_p, c_p = row['open'], row['high'], row['low'], row['close']
        color    = CHART_STYLE['up'] if c_p >= o_p else CHART_STYLE['dn']
        body_bot = min(o_p, c_p)
        body_h   = max(abs(c_p - o_p), (h_p - l_p) * 0.001)
        ax.add_patch(plt.Rectangle((i - 0.3, body_bot), 0.6, body_h,
                                   color=color, zorder=3))
        ax.plot([i, i], [l_p, h_p], color=color, linewidth=0.8, zorder=2)

    # EMAs
    for period, color, lw in [(21, CHART_STYLE['ema21'], 1.5),
                               (50, CHART_STYLE['ema50'], 1.5),
                               (200, CHART_STYLE['ema200'], 1.0)]:
        ema_full = _ema(df['close'], period)
        ema_plot = ema_full.loc[df_plot.index].values
        ax.plot(x, ema_plot, color=color, linewidth=lw,
                label=f'EMA {period}', alpha=0.85, zorder=4)

    # Signal arrow on last candle
    last_i   = len(df_plot) - 1
    last_row = df_plot.iloc[-1]
    sig_color = CHART_STYLE['call'] if signal == 'CALL' else CHART_STYLE['put']
    rng_h = last_row['high'] - last_row['low']
    if signal == 'CALL':
        ax.annotate('▲ CALL',
                    xy=(last_i, last_row['low']),
                    xytext=(last_i, last_row['low'] - rng_h * 2.5),
                    color=sig_color, fontsize=12, fontweight='bold', ha='center',
                    arrowprops=dict(arrowstyle='->', color=sig_color, lw=2), zorder=6)
    else:
        ax.annotate('▼ PUT',
                    xy=(last_i, last_row['high']),
                    xytext=(last_i, last_row['high'] + rng_h * 2.5),
                    color=sig_color, fontsize=12, fontweight='bold', ha='center',
                    arrowprops=dict(arrowstyle='->', color=sig_color, lw=2), zorder=6)

    # Price dashed line
    ax.axhline(price, color=sig_color, linewidth=0.8, linestyle='--', alpha=0.5, zorder=1)

    # Axes
    ax.set_xlim(-1, len(df_plot) + 1)
    ax.tick_params(colors=dim, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(bor)
    ax.yaxis.set_tick_params(labelcolor=txt)

    # X labels
    step = max(1, len(df_plot) // 6)
    x_ticks = x[::step]
    x_labels = [df_plot.index[i].strftime('%d/%m %H:%M') for i in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=30, ha='right', fontsize=7, color=dim)

    # Title
    now_str   = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    score_pct = score * 100
    ax.set_title(
        f'{pair}  |  H1  |  {signal}  |  Score: {score_pct:.1f}%  |  Price: {price:.5f}  |  {now_str}',
        color=txt, fontsize=10, pad=10
    )

    # Legend
    handles = [
        mpatches.Patch(color=CHART_STYLE['ema21'],  label='EMA 21'),
        mpatches.Patch(color=CHART_STYLE['ema50'],  label='EMA 50'),
        mpatches.Patch(color=CHART_STYLE['ema200'], label='EMA 200'),
        mpatches.Patch(color=sig_color,             label=f'Signal: {signal}'),
    ]
    ax.legend(handles=handles, loc='upper left', fontsize=8,
              facecolor=sur, edgecolor=bor, labelcolor=txt)

    # Watermark
    ax.text(0.98, 0.02, '🤖 Model v2 (2025-2026)',
            transform=ax.transAxes, color=dim, fontsize=8,
            ha='right', va='bottom', alpha=0.7)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, facecolor=bg,
                bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# ─────────────────────────────────────────────────────────────────────────────
# TELEGRAM CAPTION
# ─────────────────────────────────────────────────────────────────────────────
def build_caption(pair, signal, score, price, cfg):
    sig_emoji = '🟢 CALL ▲' if signal == 'CALL' else '🔴 PUT ▼'
    score_pct = score * 100
    now_th    = datetime.now(timezone(timedelta(hours=7))).strftime('%d/%m/%Y %H:%M น.')
    now_utc   = datetime.now(timezone.utc).strftime('%H:%M UTC')
    expiry    = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime('%H:%M UTC')
    return '\n'.join([
        f"📊 <b>Trading Signal — Model v2</b>",
        f"",
        f"💱 <b>Pair:</b>   {pair}",
        f"📈 <b>Signal:</b> {sig_emoji}",
        f"💰 <b>Price:</b>  <code>{price:.5f}</code>",
        f"🎯 <b>Score:</b>  {score_pct:.1f}%  [Thr {cfg['threshold']:.2f} | WR {cfg['wr']:.1f}%]",
        f"⏱ <b>TF:</b>     H1  ·  Expiry ~{expiry}",
        f"",
        f"🕐 {now_th}  ({now_utc})",
        f"",
        f"⚠️ <i>ใช้เพื่อศึกษาเท่านั้น ไม่ใช่คำแนะนำการลงทุน</i>",
    ])

# ─────────────────────────────────────────────────────────────────────────────
# PROCESS ONE PAIR
# ─────────────────────────────────────────────────────────────────────────────
def process_pair(pair, cfg, store):
    log.info(f"── {pair} ─────────────────────────────")

    if not cfg.get('enabled', False):
        log.info(f"{pair}: disabled. Skip.")
        return False

    model, features = load_model(pair)
    if model is None:
        return False

    df = fetch_prices(pair)
    if df is None or len(df) < 60:
        return False

    feat_df = build_features(df)

    # Ensure columns match training order
    for col in features:
        if col not in feat_df.columns:
            log.warning(f"{pair}: missing feature '{col}' — filling 0")
            feat_df[col] = 0.0
    feat_df = feat_df[features]
    feat_df.dropna(inplace=True)

    if feat_df.empty:
        log.warning(f"{pair}: all rows NaN. Skip.")
        return False

    last_row = feat_df.iloc[[-2]]  # -2 = last CLOSED H1 bar (-1 may be forming)
    try:
        proba    = model.predict_proba(last_row)[0]  # [prob_DOWN, prob_UP]
        prob_call = float(proba[1])
        prob_put  = float(proba[0])
    except Exception as e:
        log.error(f"{pair}: predict_proba error: {e}")
        return False

    best_prob = max(prob_call, prob_put)
    signal    = 'CALL' if prob_call >= prob_put else 'PUT'
    price     = float(df['close'].iloc[-2])  # closed bar price (matches prediction bar)
    threshold = cfg['threshold']

    log.info(f"{pair}: CALL={prob_call:.3f}  PUT={prob_put:.3f}  → {signal} ({best_prob:.3f})  thr={threshold}")

    if best_prob < threshold:
        log.info(f"{pair}: below threshold ({best_prob:.3f} < {threshold:.2f}). No signal.")
        return False

    log.info(f"✅ {pair}: SIGNAL {signal}  score={best_prob:.3f}")
    append_signal(store, pair, signal, best_prob, price)

    try:
        img_bytes = generate_chart(df, pair, signal, best_prob, price)
    except Exception as e:
        log.error(f"{pair}: chart error: {e}")
        img_bytes = None

    caption = build_caption(pair, signal, best_prob, price, cfg)

    if img_bytes:
        send_telegram_photo(img_bytes, caption)
    else:
        send_telegram_text(caption)

    return True

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    now_utc = datetime.now(timezone.utc)
    log.info("=" * 60)
    log.info("Trading Signal Bot — Model v2 (2025-2026)")
    log.info(f"Run: {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    log.info("=" * 60)

    store = load_signals_json()

    if not is_trading_session(now_utc):
        log.info("Outside trading session — no alerts sent.")
        save_signals_json(store)
        return

    signal_count = 0
    for pair, cfg in PAIRS_CONFIG.items():
        try:
            if process_pair(pair, cfg, store):
                signal_count += 1
        except Exception as e:
            log.error(f"{pair}: unhandled error: {e}", exc_info=True)

    save_signals_json(store)
    log.info("=" * 60)
    enabled = sum(1 for c in PAIRS_CONFIG.values() if c['enabled'])
    log.info(f"Done. Signals: {signal_count}/{enabled}")
    log.info("=" * 60)


if __name__ == '__main__':
    main()
