"""
Mean reversion di breve sugli indici azionari - backtest con criteri di successo FISSATI IN ANTICIPO.

Idea (opposta al trend following gia' bocciato da trend_backtest.py): negli indici azionari
in trend rialzista, i cali bruschi di pochi giorni tendono a rientrare. Si compra la debolezza, solo long.

Regole (decise prima di guardare i dati, NON da ottimizzare):
  - entra long se la chiusura e' sopra la media mobile a 200 giorni E l'RSI a 2 giorni e' sotto 10
  - esci quando la chiusura supera la media mobile a 5 giorni
  - segnale sulla chiusura del giorno t, esecuzione all'APERTURA del giorno t+1 (niente look-ahead,
    e niente rimbalzo "regalato" tra chiusura e apertura)
  - sizing a volatilita' costante e portafoglio equipesato, come in trend_backtest.py

Verdetto sugli stessi 5 criteri di trend_backtest.py, con le stesse soglie.

Attenzione: e' la seconda idea testata sugli stessi dati storici. Piu' idee si provano, piu' e'
probabile che una passi per caso: un verdetto positivo qui vale meno di quanto sembri.

Uso:
    py meanrev_backtest.py
    py meanrev_backtest.py --period 20y
    py meanrev_backtest.py --synthetic     (solo per testare il codice, dati finti)

Limiti noti: indici "price index" senza dividendi (penalizza un po' il long), nessun costo di
finanziamento overnight dei CFD. Non e' un consiglio finanziario.
"""

import argparse
import sys
import zlib

import numpy as np
import pandas as pd

from trend_backtest import (MAX_DD_ALLOWED, MAX_LEV, MIN_PF_OOS, MIN_ROWS, MIN_TRADES, OOS_FRAC,
                            PORT_VOL_TARGET, INSTR_VOL_TARGET, VOL_SPAN,
                            extract_trades, metrics, profit_factor, total_return)

# ---------------------------------------------------------------- configurazione (non ritoccare dopo aver visto i risultati)

UNIVERSE = {
    "^GSPC": "S&P 500",
    "^NDX": "Nasdaq 100",
    "^DJI": "Dow Jones",
    "^RUT": "Russell 2000",
    "^GDAXI": "DAX",
    "^FTSE": "FTSE 100",
    "^FCHI": "CAC 40",
    "^STOXX50E": "Euro Stoxx 50",
    "^N225": "Nikkei 225",
    "^HSI": "Hang Seng",
    "^AXJO": "ASX 200",
    "^GSPTSE": "TSX Canada",
}

COST_BPS = 2.0                 # costo per lato, in punti base
SMA_TREND, SMA_EXIT = 200, 5
RSI_N, RSI_ENTRY = 2, 10
MAX_FLAT_OPEN = 0.05           # scarta lo strumento se Open == Close in oltre il 5% dei giorni (aperture inaffidabili)
MIN_INSTRUMENTS = 6


# ---------------------------------------------------------------- dati

def load_ohlc(ticker, period, synthetic=False):
    if synthetic:
        rng = np.random.default_rng(zlib.crc32(ticker.encode()))
        n = 5000
        drift = np.repeat(rng.normal(0.0002, 0.0003, n // 250 + 1), 250)[:n]
        close = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.01, n)))
        open_ = np.concatenate([[close[0]], close[:-1]]) * np.exp(rng.normal(0, 0.003, n))
        return pd.DataFrame({"Open": open_, "Close": close}, index=pd.bdate_range("2006-01-02", periods=n))

    import yfinance as yf
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "Close"]].dropna()
    df = df[(df["Open"] > 0) & (df["Close"] > 0)]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def load_all(period, synthetic):
    data = {}
    for tk, name in UNIVERSE.items():
        try:
            df = load_ohlc(tk, period, synthetic)
        except Exception as e:
            print(f"  {tk:10s} {name}: errore dati ({e})")
            continue
        if df is None or len(df) < MIN_ROWS:
            print(f"  {tk:10s} {name}: scartato (solo {0 if df is None else len(df)} giorni di dati)")
            continue
        flat = float((df["Open"] == df["Close"]).mean())
        if flat > MAX_FLAT_OPEN:
            print(f"  {tk:10s} {name}: scartato (apertura uguale alla chiusura nel {flat:.0%} dei giorni)")
            continue
        data[tk] = df
        print(f"  {tk:10s} {name}: {len(df)} giorni")
    return data


# ---------------------------------------------------------------- strategia

def rsi(close, n):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def mr_position(close):
    """Posizione desiderata a fine giorno (+1 long, 0 flat)."""
    sma_t = close.rolling(SMA_TREND).mean().values
    sma_e = close.rolling(SMA_EXIT).mean().values
    r = rsi(close, RSI_N).values
    c = close.values
    pos = np.zeros(len(c))
    state = 0
    for t in range(len(c)):
        if np.isnan(sma_t[t]):
            continue
        if state == 0:
            if c[t] > sma_t[t] and r[t] < RSI_ENTRY:
                state = 1
        elif c[t] > sma_e[t]:
            state = 0
        pos[t] = state
    return pos


def run(data, cost_mult=1.0, scale=None):
    raw_net, signs = {}, {}
    for tk, df in data.items():
        o, c = df["Open"], df["Close"]
        cost = COST_BPS / 10000 * cost_mult
        ret_oo = o.shift(-1) / o - 1                 # dall'apertura di t all'apertura di t+1
        vol = c.pct_change().ewm(span=VOL_SPAN, min_periods=VOL_SPAN).std() * np.sqrt(252)
        lev = (INSTR_VOL_TARGET / vol).clip(upper=MAX_LEV)
        pos = pd.Series(mr_position(c), index=c.index)
        w = (pos * lev).shift(1).fillna(0.0)          # deciso alla chiusura di ieri, in posizione dall'apertura di oggi
        turnover = w.diff().abs().fillna(w.abs())
        raw_net[tk] = w * ret_oo - turnover * cost
        signs[tk] = np.sign(w)

    port_raw = pd.concat(raw_net, axis=1, sort=True).mean(axis=1, skipna=True).dropna()
    cut = port_raw.index[int(len(port_raw) * (1 - OOS_FRAC))]
    if scale is None:
        is_vol = port_raw[port_raw.index < cut].std() * np.sqrt(252)
        scale = PORT_VOL_TARGET / is_vol if is_vol > 0 else 1.0

    nets = {tk: s * scale for tk, s in raw_net.items()}
    trades = pd.concat([extract_trades(nets[tk], signs[tk], tk) for tk in data], ignore_index=True)
    exposure = float(np.mean([(s != 0).mean() for s in signs.values()]))
    return {"port": port_raw * scale, "nets": nets, "trades": trades, "scale": scale, "cut": cut,
            "exposure": exposure}


# ---------------------------------------------------------------- report e verdetto

def report(data):
    base = run(data)
    stress = run(data, cost_mult=2.0, scale=base["scale"])
    port, cut, trades = base["port"], base["cut"], base["trades"]
    is_r, oos_r = port[port.index < cut], port[port.index >= cut]

    print(f"\nPeriodo: {port.index[0].date()} -> {port.index[-1].date()} | "
          f"sviluppo fino al {cut.date()}, test (mai usato per scegliere i parametri) dopo")
    print(f"Tempo medio in posizione per strumento: {base['exposure']:.0%} | "
          f"leva di portafoglio per volatilita' 10%: {base['scale']:.2f}x\n")
    for label, r in (("Sviluppo (70%)", is_r), ("Test out-of-sample (30%)", oos_r), ("Totale", port)):
        print(f"  {label:26s}", metrics(r))

    print("\nPer strumento (periodo di test):")
    pos_count, tot_count = 0, 0
    for tk, net in base["nets"].items():
        o = (net / base["scale"])[net.index >= cut].dropna()
        if len(o) < 30:
            continue
        tot_count += 1
        res = total_return(o) * 100
        pos_count += res > 0
        n_tr = int((trades["ticker"] == tk).sum())
        print(f"  {tk:10s} {UNIVERSE[tk]:16s} operazioni totali {n_tr:3d} | test {res:+6.1f}%")

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
        print("E' la seconda idea provata sugli stessi dati: il rischio che sia un caso e' piu' alto del solito.")
    else:
        print("Non tutti i criteri sono superati: l'idea NON passa. Niente ritocchi ai parametri per far tornare i numeri.")


# ---------------------------------------------------------------- main

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--period", default="20y")
    p.add_argument("--synthetic", action="store_true", help="dati finti, solo per testare il codice")
    a = p.parse_args()

    print(f"Scarico i dati ({'sintetici' if a.synthetic else a.period}):")
    data = load_all(a.period, a.synthetic)
    if len(data) < MIN_INSTRUMENTS:
        sys.exit(f"\nTroppi pochi strumenti disponibili ({len(data)}): controlla la connessione.")
    report(data)
