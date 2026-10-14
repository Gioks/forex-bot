"""
Trend following multi-strumento - backtest con criteri di successo FISSATI IN ANTICIPO.

Strategia (parametri decisi prima di guardare i dati, NON da ottimizzare):
  - Breakout Donchian: entra long se il prezzo supera il massimo di 55 giorni,
    short se rompe il minimo di 55 giorni. Esce quando rompe il minimo/massimo di 20 giorni.
  - Sizing a volatilita' costante: ogni strumento rischia circa lo stesso (vol target 10%).
  - Portafoglio: media equipesata di ~16 strumenti (indici, obbligazioni, materie prime, valute).
  - Il segnale nasce sulla chiusura del giorno t e si applica dal giorno t+1 (niente look-ahead).

Verdetto automatico su 5 criteri (tutti devono passare):
  1. almeno 300 operazioni totali
  2. profit factor >= 1.3 sulle operazioni out-of-sample (ultimo 30% del periodo)
  3. risultato positivo (out-of-sample) sulla maggioranza degli strumenti
  4. max drawdown del portafoglio < 20% (volatilita' del portafoglio normalizzata al 10%)
  5. ancora in utile con costi RADDOPPIATI

Uso:
    py trend_backtest.py
    py trend_backtest.py --period 20y
    py trend_backtest.py --synthetic     (solo per testare il codice, dati finti)

Limiti noti: gli indici sono "price index" senza dividendi, lo short sugli ETF ignora il costo
di prestito titoli, le valute ignorano il carry (interessi). Non e' un consiglio finanziario.
"""

import argparse
import sys
import zlib

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- configurazione (non ritoccare dopo aver visto i risultati)

# ticker Yahoo -> (nome, classe)
UNIVERSE = {
    "^GSPC": ("S&P 500", "indice"),
    "^NDX": ("Nasdaq 100", "indice"),
    "^GDAXI": ("DAX", "indice"),
    "^FTSE": ("FTSE 100", "indice"),
    "^N225": ("Nikkei 225", "indice"),
    "TLT": ("Treasury 20+ anni", "obbligaz"),
    "IEF": ("Treasury 7-10 anni", "obbligaz"),
    "GLD": ("Oro", "materie"),
    "SLV": ("Argento", "materie"),
    "DBC": ("Materie prime (paniere)", "materie"),
    "USO": ("Petrolio", "materie"),
    "EURUSD=X": ("EUR/USD", "valute"),
    "GBPUSD=X": ("GBP/USD", "valute"),
    "USDJPY=X": ("USD/JPY", "valute"),
    "AUDUSD=X": ("AUD/USD", "valute"),
    "USDCAD=X": ("USD/CAD", "valute"),
}

COST_BPS = {"indice": 2.0, "obbligaz": 2.0, "materie": 5.0, "valute": 1.5}  # costo per lato, in punti base

ENTRY_N, EXIT_N = 55, 20
VOL_SPAN = 60
INSTR_VOL_TARGET = 0.10
MAX_LEV = 3.0
PORT_VOL_TARGET = 0.10
OOS_FRAC = 0.30
MIN_ROWS = 1500

# criteri di successo
MIN_TRADES = 300
MIN_PF_OOS = 1.3
MAX_DD_ALLOWED = 0.20


# ---------------------------------------------------------------- dati

def load_close(ticker, period, synthetic=False):
    if synthetic:
        rng = np.random.default_rng(zlib.crc32(ticker.encode()))
        n = 5000
        drift = np.repeat(rng.normal(0, 0.0003, n // 250 + 1), 250)[:n]
        ret = drift + rng.normal(0, 0.01, n)
        idx = pd.bdate_range("2006-01-02", periods=n)
        return pd.Series(100 * np.exp(np.cumsum(ret)), index=idx, name=ticker)

    import yfinance as yf
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    s = df["Close"].dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def load_all(period, synthetic):
    closes = {}
    for tk, (name, _cls) in UNIVERSE.items():
        try:
            s = load_close(tk, period, synthetic)
        except Exception as e:
            print(f"  {tk:9s} {name}: errore dati ({e})")
            continue
        if s is None or len(s) < MIN_ROWS:
            n = 0 if s is None else len(s)
            print(f"  {tk:9s} {name}: scartato (solo {n} giorni di dati)")
            continue
        closes[tk] = s
        print(f"  {tk:9s} {name}: {len(s)} giorni")
    return closes


# ---------------------------------------------------------------- strategia

def donchian_position(close, entry_n, exit_n):
    """Ritorna la posizione desiderata a fine giorno (+1 long, -1 short, 0 flat)."""
    s = pd.Series(close)
    hh = s.rolling(entry_n).max().shift(1).values
    ll = s.rolling(entry_n).min().shift(1).values
    xh = s.rolling(exit_n).max().shift(1).values
    xl = s.rolling(exit_n).min().shift(1).values
    c = s.values
    pos = np.zeros(len(c))
    state = 0
    for t in range(len(c)):
        if np.isnan(hh[t]):
            continue
        if state == 0:
            if c[t] > hh[t]:
                state = 1
            elif c[t] < ll[t]:
                state = -1
        elif state == 1:
            if c[t] < xl[t]:
                state = -1 if c[t] < ll[t] else 0
        else:
            if c[t] > xh[t]:
                state = 1 if c[t] > hh[t] else 0
        pos[t] = state
    return pos


def extract_trades(net, sign, ticker):
    """Una operazione = un tratto continuo con la stessa direzione. P&L netto di costi."""
    seg = (sign != sign.shift()).cumsum()
    df = pd.DataFrame({"net": net, "sign": sign, "seg": seg})
    df = df[df["sign"] != 0].dropna(subset=["net"])
    if df.empty:
        return pd.DataFrame(columns=["ticker", "exit", "pnl"])
    g = df.groupby("seg")
    out = pd.DataFrame({"pnl": g["net"].sum(), "exit": g.apply(lambda x: x.index[-1])})
    out["ticker"] = ticker
    return out.reset_index(drop=True)


def run(closes, cost_mult=1.0, scale=None):
    raw_net, signs = {}, {}
    for tk, close in closes.items():
        cls = UNIVERSE[tk][1]
        cost = COST_BPS[cls] / 10000 * cost_mult
        ret = close.pct_change()
        vol = ret.ewm(span=VOL_SPAN, min_periods=VOL_SPAN).std() * np.sqrt(252)
        lev = (INSTR_VOL_TARGET / vol).clip(upper=MAX_LEV)
        pos = pd.Series(donchian_position(close.values, ENTRY_N, EXIT_N), index=close.index)
        w = (pos * lev).shift(1).fillna(0.0)          # posizione decisa ieri, applicata oggi
        turnover = w.diff().abs().fillna(w.abs())
        raw_net[tk] = w * ret - turnover * cost
        signs[tk] = np.sign(w)

    port_raw = pd.concat(raw_net, axis=1).mean(axis=1, skipna=True).dropna()
    cut = port_raw.index[int(len(port_raw) * (1 - OOS_FRAC))]
    if scale is None:
        is_vol = port_raw[port_raw.index < cut].std() * np.sqrt(252)
        scale = PORT_VOL_TARGET / is_vol if is_vol > 0 else 1.0

    nets = {tk: s * scale for tk, s in raw_net.items()}
    trades = pd.concat([extract_trades(nets[tk], signs[tk], tk) for tk in closes], ignore_index=True)
    return {"port": port_raw * scale, "nets": nets, "trades": trades, "scale": scale, "cut": cut}


# ---------------------------------------------------------------- metriche

def metrics(r):
    if len(r) < 30:
        return {}
    equity = (1 + r).cumprod()
    years = len(r) / 252
    peak = equity.cummax()
    return {
        "rendimento annuo %": round(float(equity.iloc[-1] ** (1 / years) - 1) * 100, 1),
        "volatilita' %": round(float(r.std() * np.sqrt(252)) * 100, 1),
        "sharpe": round(float(r.mean() / r.std() * np.sqrt(252)), 2) if r.std() > 0 else 0.0,
        "max drawdown %": round(float(((equity - peak) / peak).min()) * 100, 1),
    }


def profit_factor(pnl):
    wins, losses = pnl[pnl > 0].sum(), -pnl[pnl <= 0].sum()
    return float(wins / losses) if losses > 0 else float("inf")


def total_return(r):
    return float((1 + r).prod() - 1)


# ---------------------------------------------------------------- report e verdetto

def report(closes):
    base = run(closes)
    stress = run(closes, cost_mult=2.0, scale=base["scale"])
    port, cut, trades = base["port"], base["cut"], base["trades"]
    is_r, oos_r = port[port.index < cut], port[port.index >= cut]

    print(f"\nPeriodo: {port.index[0].date()} -> {port.index[-1].date()} | "
          f"sviluppo fino al {cut.date()}, test (mai usato per scegliere i parametri) dopo\n")
    for label, r in (("Sviluppo (70%)", is_r), ("Test out-of-sample (30%)", oos_r), ("Totale", port)):
        print(f"  {label:26s}", metrics(r))

    print("\nPer strumento (periodo di test, ogni strumento con volatilita' target 10% annua):")
    pos_count, tot_count = 0, 0
    for tk, net in base["nets"].items():
        o = (net / base["scale"])[net.index >= cut].dropna()
        if len(o) < 30:
            continue
        tot_count += 1
        res = total_return(o) * 100
        pos_count += res > 0
        n_tr = int((trades["ticker"] == tk).sum())
        print(f"  {tk:9s} {UNIVERSE[tk][0]:26s} operazioni totali {n_tr:3d} | test {res:+6.1f}%")

    oos_trades = trades[trades["exit"] >= cut]
    n_trades = len(trades)
    pf_oos = profit_factor(oos_trades["pnl"]) if len(oos_trades) else 0.0
    dd = abs(metrics(port).get("max drawdown %", 0)) / 100
    stress_oos = total_return(stress["port"][stress["port"].index >= cut])
    stress_all = total_return(stress["port"])
    majority = tot_count > 0 and pos_count > tot_count / 2

    checks = [
        (f"1. Operazioni totali >= {MIN_TRADES}", n_trades >= MIN_TRADES, f"{n_trades}"),
        (f"2. Profit factor out-of-sample >= {MIN_PF_OOS}", pf_oos >= MIN_PF_OOS,
         f"{pf_oos:.2f} su {len(oos_trades)} operazioni"),
        ("3. Positivo (test) sulla maggioranza degli strumenti", majority, f"{pos_count} su {tot_count}"),
        (f"4. Max drawdown < {MAX_DD_ALLOWED:.0%}", dd < MAX_DD_ALLOWED, f"{dd:.1%}"),
        ("5. In utile con costi raddoppiati (test e totale)", stress_oos > 0 and stress_all > 0,
         f"test {stress_oos:+.1%}, totale {stress_all:+.1%}"),
    ]
    print("\nCRITERI DI SUCCESSO (fissati in anticipo)")
    for text, ok, detail in checks:
        print(f"  [{'OK' if ok else 'NO'}] {text:56s} {detail}")

    passed = sum(ok for _, ok, _ in checks)
    print(f"\nVERDETTO: {passed}/5 criteri superati.")
    if passed == 5:
        print("Tutti i criteri sono superati: l'idea merita il passo successivo (paper trading su demo).")
        print("Non e' una garanzia: il test e' passato, ma il futuro puo' essere diverso.")
    else:
        print("Non tutti i criteri sono superati: l'idea NON passa. Niente ritocchi ai parametri per far tornare i numeri.")


# ---------------------------------------------------------------- main

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--period", default="20y")
    p.add_argument("--synthetic", action="store_true", help="dati finti, solo per testare il codice")
    a = p.parse_args()

    print(f"Scarico i dati ({'sintetici' if a.synthetic else a.period}):")
    closes = load_all(a.period, a.synthetic)
    if len(closes) < 8:
        sys.exit(f"\nTroppi pochi strumenti disponibili ({len(closes)}): controlla la connessione.")
    report(closes)
