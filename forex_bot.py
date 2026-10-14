"""
Forex bot v2 - backtest + segnali Telegram.

Strategia: incrocio EMA 20/50 filtrato dal trend EMA 200 e dall'RSI,
stop loss e take profit basati sull'ATR, operazioni long e short.

Uso:
    python forex_bot.py backtest                 # backtest su tutte le coppie
    python forex_bot.py backtest --interval 1h --period 730d
    python forex_bot.py signals --dry            # mostra i segnali senza inviarli
    python forex_bot.py signals                  # invia i segnali su Telegram
    python forex_bot.py paper                    # paper trading: operazioni simulate, soldi finti
    python forex_bot.py paper --notify           # ... e avvisa su Telegram di aperture/chiusure
    python forex_bot.py paper --report           # riepilogo del paper trading

Il token Telegram NON va scritto nel codice: usa le variabili d'ambiente
TELEGRAM_TOKEN e TELEGRAM_CHAT_ID (vedi env.example).

Questo NON e' un consiglio finanziario. Nessuna strategia garantisce profitti.
"""

import argparse
import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------- configurazione

# ticker Yahoo -> spread tipico in unita' di prezzo (da verificare col tuo broker)
PAIRS = {
    "EURUSD=X": 0.00012,
    "GBPUSD=X": 0.00015,
    "AUDUSD=X": 0.00015,
    "EURJPY=X": 0.015,
}

EMA_FAST, EMA_SLOW, EMA_TREND = 20, 50, 200
RSI_PERIOD, ATR_PERIOD = 14, 14
SL_ATR, TP_ATR = 1.5, 3.0      # stop 1.5 ATR, target 3 ATR (rapporto 1:2)
MAX_SPREAD_FRAC = 0.25         # non operare se lo spread supera il 25% della distanza di stop
RISK_PER_TRADE = 0.01          # 1% del capitale per operazione
START_EQUITY = 10_000.0
STATE_FILE = Path("last_signals.json")
PAPER_FILE = Path(__file__).with_name("paper_state.json")   # accanto allo script: funziona da qualsiasi cwd


# ---------------------------------------------------------------- dati e indicatori

def load_data(ticker, interval, period, synthetic=False):
    """Scarica le candele OHLC. Con synthetic=True genera dati finti (solo per test)."""
    if synthetic:
        rng = np.random.default_rng(zlib.crc32(ticker.encode()))
        n = 6000
        drift = np.repeat(rng.normal(0, 0.00008, n // 300 + 1), 300)[:n]
        ret = drift + rng.normal(0, 0.0005, n)
        close = 1.10 * np.exp(np.cumsum(ret))
        idx = pd.date_range("2024-01-01", periods=n, freq="h")
        open_ = np.concatenate([[close[0]], close[:-1]])
        spread = np.abs(rng.normal(0, 0.0003, n)) * close
        return pd.DataFrame({
            "Open": open_,
            "High": np.maximum(open_, close) + spread,
            "Low": np.minimum(open_, close) - spread,
            "Close": close,
        }, index=idx)

    import yfinance as yf
    df = yf.download(ticker, interval=interval, period=period,
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):      # yfinance recente: colonne MultiIndex
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    return df


def prepare(df):
    """Aggiunge EMA, RSI (Wilder), ATR (Wilder) e il segnale di ingresso."""
    df = df.copy()
    c = df["Close"]
    df["ema_fast"] = c.ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = c.ewm(span=EMA_SLOW, adjust=False).mean()
    df["ema_trend"] = c.ewm(span=EMA_TREND, adjust=False).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    prev_close = c.shift()
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - prev_close).abs(),
                    (df["Low"] - prev_close).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / ATR_PERIOD, adjust=False).mean()

    cross_up = (df["ema_fast"] > df["ema_slow"]) & (df["ema_fast"].shift() <= df["ema_slow"].shift())
    cross_dn = (df["ema_fast"] < df["ema_slow"]) & (df["ema_fast"].shift() >= df["ema_slow"].shift())
    long_ok = cross_up & (c > df["ema_trend"]) & (df["rsi"] < 70)
    short_ok = cross_dn & (c < df["ema_trend"]) & (df["rsi"] > 30)

    df["signal"] = 0
    df.loc[long_ok, "signal"] = 1
    df.loc[short_ok, "signal"] = -1
    df.iloc[:EMA_TREND] = df.iloc[:EMA_TREND].assign(signal=0)   # warm-up degli indicatori
    return df


# ---------------------------------------------------------------- backtest

def backtest(df, spread):
    """
    Simula le operazioni. Il segnale nasce sulla chiusura della candela i,
    l'ingresso avviene all'apertura della i+1 (nessun look-ahead).
    Se nella stessa candela si toccano sia SL che TP, si assume lo SL (ipotesi prudente).
    Il costo dello spread e' applicato meta' all'ingresso e meta' all'uscita.
    """
    o, h, l = df["Open"].values, df["High"].values, df["Low"].values
    sig, atr = df["signal"].values, df["atr"].values
    n = len(df)
    equity = START_EQUITY
    curve, trades = [equity], []
    i = 0
    while i < n - 1:
        d = sig[i]
        if d == 0 or np.isnan(atr[i]) or atr[i] <= 0:
            i += 1
            continue
        stop_dist, tp_dist = SL_ATR * atr[i], TP_ATR * atr[i]
        if spread > MAX_SPREAD_FRAC * stop_dist:   # costi sproporzionati rispetto allo stop: salta
            i += 1
            continue
        entry = o[i + 1] + d * spread / 2
        sl, tp = entry - d * stop_dist, entry + d * tp_dist
        units = equity * RISK_PER_TRADE / stop_dist

        exit_price, j = None, i + 1
        while j < n:
            hit_sl = l[j] <= sl if d == 1 else h[j] >= sl
            hit_tp = h[j] >= tp if d == 1 else l[j] <= tp
            if hit_sl:
                exit_price = sl
                break
            if hit_tp:
                exit_price = tp
                break
            j += 1
        if exit_price is None:        # operazione ancora aperta a fine dati: la ignoriamo
            break
        exit_price -= d * spread / 2
        pnl = units * (exit_price - entry) * d
        equity += pnl
        trades.append({"time": df.index[i + 1], "dir": d, "pnl": pnl, "r": pnl / (equity - pnl) / RISK_PER_TRADE})
        curve.append(equity)
        i = j
    return pd.DataFrame(trades), np.array(curve)


def stats(trades, curve):
    if trades.empty:
        return {"operazioni": 0}
    wins, losses = trades.loc[trades.pnl > 0, "pnl"], trades.loc[trades.pnl <= 0, "pnl"]
    peak = np.maximum.accumulate(curve)
    return {
        "operazioni": len(trades),
        "win rate %": round(100 * len(wins) / len(trades), 1),
        "profit factor": round(float(wins.sum() / abs(losses.sum())), 2) if len(losses) and losses.sum() != 0 else float("inf"),
        "R medio": round(float(trades["r"].mean()), 2),
        "rendimento %": round(float(100 * (curve[-1] / curve[0] - 1)), 1),
        "max drawdown %": round(float(100 * ((curve - peak) / peak).min()), 1),
    }


def run_backtest(args):
    print(f"Backtest | interval={args.interval} period={args.period} | rischio {RISK_PER_TRADE:.0%}/trade\n")
    for ticker, spread in PAIRS.items():
        df = load_data(ticker, args.interval, args.period, args.synthetic)
        if len(df) < EMA_TREND + 100:
            print(f"{ticker}: dati insufficienti ({len(df)} candele)\n")
            continue
        df = prepare(df)
        cut = int(len(df) * 0.7)       # 70% sviluppo, 30% test mai "visto" per scegliere i parametri
        print(f"== {ticker} ==")
        for label, part in (("in-sample 70%", df.iloc[:cut]), ("out-of-sample 30%", df.iloc[cut:])):
            t, c = backtest(part, spread)
            print(f"  {label:18s}", stats(t, c))
        print()
    print("Se l'out-of-sample e' molto peggiore dell'in-sample, la strategia non regge.")
    print("Il backtest non e' una garanzia: spread reali, slippage e gap peggiorano i risultati.")


# ---------------------------------------------------------------- segnali Telegram

def send_telegram(text):
    import requests
    token, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        sys.exit("Imposta TELEGRAM_TOKEN e TELEGRAM_CHAT_ID (vedi env.example).")
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data={"chat_id": chat, "text": text}, timeout=15)
    r.raise_for_status()


def run_signals(args):
    state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
    messages = []
    for ticker in PAIRS:
        try:
            df = prepare(load_data(ticker, args.interval, args.period, args.synthetic))
        except Exception as e:
            print(f"{ticker}: errore dati ({e})")
            continue
        bar = df.iloc[-2]                  # ultima candela CHIUSA (l'ultima riga e' in formazione)
        stamp = str(df.index[-2])
        if bar["signal"] == 0 or state.get(ticker) == stamp:
            continue
        d = int(bar["signal"])
        entry = float(df["Close"].iloc[-1])
        stop, tp = entry - d * SL_ATR * bar["atr"], entry + d * TP_ATR * bar["atr"]
        name = ticker.replace("=X", "")
        messages.append(
            f"{'BUY' if d == 1 else 'SELL'} {name}\n"
            f"Entry ~ {entry:.5f}\nStop loss: {stop:.5f}\nTake profit: {tp:.5f}\n"
            f"RSI {bar['rsi']:.0f} | candela chiusa {stamp}"
        )
        state[ticker] = stamp
    if not messages:
        print("Nessun nuovo segnale.")
        return
    text = "Segnali forex (non e' un consiglio finanziario)\n\n" + "\n\n".join(messages)
    print(text)
    if not args.dry:
        send_telegram(text)
        STATE_FILE.write_text(json.dumps(state))


# ---------------------------------------------------------------- paper trading

def run_paper(args):
    """
    Paper trading con le stesse regole del backtest, sui dati in tempo reale.
    Va lanciato a intervalli regolari (es. ogni ora): a ogni esecuzione controlla le candele
    chiuse da allora, chiude le posizioni che hanno toccato SL/TP e apre quelle nuove.
    Un segnale e' valido solo sull'ultima candela chiusa: se il PC e' spento, i segnali persi non si recuperano.
    """
    path = PAPER_FILE.with_name("paper_synthetic.json") if args.synthetic else PAPER_FILE
    if path.exists():
        st = json.loads(path.read_text())
    else:
        st = {"start": pd.Timestamp.now(tz="UTC").isoformat(timespec="minutes"),
              "equity": START_EQUITY, "open": {}, "closed": [], "last_bar": {}}
    if args.report:
        paper_report(st)
        return

    events = []
    for ticker, spread in PAIRS.items():
        try:
            df = prepare(load_data(ticker, args.interval, args.period, args.synthetic))
        except Exception as e:
            print(f"{ticker}: errore dati ({e})")
            continue
        name = ticker.replace("=X", "")
        closed_bars = df.iloc[:-1]               # l'ultima riga e' la candela in formazione

        pos = st["open"].get(ticker)
        if pos:
            d = pos["dir"]
            for t, bar in closed_bars[closed_bars.index >= pd.Timestamp(pos["time"])].iterrows():
                hit_sl = bar["Low"] <= pos["sl"] if d == 1 else bar["High"] >= pos["sl"]
                hit_tp = bar["High"] >= pos["tp"] if d == 1 else bar["Low"] <= pos["tp"]
                if hit_sl or hit_tp:             # entrambi nella stessa candela: si assume lo SL
                    exit_price = (pos["sl"] if hit_sl else pos["tp"]) - d * spread / 2
                    pnl = pos["units"] * (exit_price - pos["entry"]) * d
                    st["equity"] += pnl
                    st["closed"].append({**pos, "ticker": ticker, "exit": exit_price,
                                         "exit_time": str(t), "pnl": pnl, "r": pnl / pos["risk"]})
                    del st["open"][ticker]
                    events.append(f"CHIUSA {name} ({'stop loss' if hit_sl else 'take profit'}) "
                                  f"a {exit_price:.5f} | P&L {pnl:+.2f} | capitale {st['equity']:.2f}")
                    break

        bar, stamp = closed_bars.iloc[-1], str(closed_bars.index[-1])
        if ticker not in st["open"] and bar["signal"] != 0 and st["last_bar"].get(ticker) != stamp:
            d = int(bar["signal"])
            stop_dist, tp_dist = SL_ATR * float(bar["atr"]), TP_ATR * float(bar["atr"])
            if spread > MAX_SPREAD_FRAC * stop_dist:
                events.append(f"{name}: segnale saltato (spread troppo alto rispetto allo stop)")
            else:
                entry = float(df["Open"].iloc[-1]) + d * spread / 2    # ingresso all'apertura della candela successiva
                risk = st["equity"] * RISK_PER_TRADE
                st["open"][ticker] = {"dir": d, "time": str(df.index[-1]), "entry": entry,
                                      "sl": entry - d * stop_dist, "tp": entry + d * tp_dist,
                                      "units": risk / stop_dist, "risk": risk}
                events.append(f"APERTA {'BUY' if d == 1 else 'SELL'} {name} a {entry:.5f} | "
                              f"SL {entry - d * stop_dist:.5f} | TP {entry + d * tp_dist:.5f}")
        st["last_bar"][ticker] = stamp

    path.write_text(json.dumps(st, indent=1))
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    if not events:
        print(f"{now} nessuna novita' | capitale {st['equity']:.2f} | posizioni aperte {len(st['open'])}")
        return
    text = "Paper trading (soldi finti)\n\n" + "\n".join(events)
    print(f"{now}\n{text}")
    if args.notify:
        send_telegram(text)


def paper_report(st):
    print(f"Paper trading dal {st['start']} | capitale {st['equity']:.2f} (partenza {START_EQUITY:.0f})\n")
    for tk, p in st["open"].items():
        print(f"  aperta {'BUY' if p['dir'] == 1 else 'SELL'} {tk.replace('=X', '')} dal {p['time']} "
              f"a {p['entry']:.5f} (SL {p['sl']:.5f}, TP {p['tp']:.5f})")
    closed = pd.DataFrame(st["closed"])
    if closed.empty:
        print("  Nessuna operazione chiusa finora.")
        return
    curve = START_EQUITY + np.concatenate([[0.0], closed["pnl"].cumsum().values])
    print("\n ", stats(closed, curve))
    print("\nCon poche operazioni il risultato dipende molto dal caso: confrontalo col backtest, non giudicarlo da solo.")


# ---------------------------------------------------------------- main

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["backtest", "signals", "paper"])
    p.add_argument("--interval", default="1h")
    p.add_argument("--period", default="730d")
    p.add_argument("--dry", action="store_true", help="non inviare su Telegram")
    p.add_argument("--notify", action="store_true", help="paper: avvisa su Telegram di aperture/chiusure")
    p.add_argument("--report", action="store_true", help="paper: mostra solo il riepilogo")
    p.add_argument("--synthetic", action="store_true", help="dati finti, solo per testare il codice")
    a = p.parse_args()
    {"backtest": run_backtest, "signals": run_signals, "paper": run_paper}[a.mode](a)
