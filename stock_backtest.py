"""
Test storico della regola settimanale su azioni globali, con criteri FISSATI IN ANTICIPO.
Regola e universo: vedi stock_universe.py (momentum a 4 settimane, top 5, ribilanciamento settimanale, costo 5 bps).

Verdetto su 7 criteri (devono passare TUTTI), sul confronto col paniere equipesato di tutte le azioni:
  1. Test (ultimo 30%): rendimento annuo in eccesso > 0 e information ratio >= 0.3
  2. Sviluppo (primo 70%): stesso
  3. Batte il 95% di 1000 selezioni CASUALI di 5 azioni (periodo di test)
  4. Eccesso positivo in almeno il 60% degli anni del test
  5. Max drawdown non peggiore del paniere di piu' di 5 punti
  6. Con costi doppi l'eccesso nel test resta > 0
  7. Anche le finestre 8 e 13 settimane hanno eccesso > 0 nel test

LIMITE GRAVE: l'elenco contiene aziende grandi OGGI (survivorship bias): le societa' fallite o uscite dal listino
non ci sono. Questo gonfia i risultati. Se il test fallisce anche cosi', e' un segnale molto chiaro.
Non e' un consiglio finanziario.

Uso:
    py stock_backtest.py
    py stock_backtest.py --synthetic [--edge x]
"""

import argparse
import sys

import numpy as np
import pandas as pd

import stock_universe as su

COST = 5 / 10000
OOS_FRAC = 0.30
N_PLACEBO = 1000
SEED = 2026
MIN_ROWS = 400
PPY = 52
MIN_IR, MIN_YEARS_POSITIVE, MAX_DD_WORSE, PLACEBO_PCTL = 0.3, 0.60, 0.05, 0.95
ROBUST_LOOKBACKS = (40, 65)


def load_prices(period):
    import yfinance as yf
    tickers = list(su.UNIVERSE)
    df = yf.download(tickers, period=period, interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        sys.exit("Nessun dato ricevuto da Yahoo: controlla la connessione.")
    close = df["Close"]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    keep = [t for t in close.columns if close[t].notna().sum() >= MIN_ROWS]
    missing = [t for t in tickers if t not in keep]
    if missing:
        print(f"  scartati (pochi dati o ticker non trovato): {', '.join(missing)}")
    return close[keep]


def synthetic_prices(edge=0.0, seed=5):
    rng = np.random.default_rng(seed)
    n, k = 2600, len(su.UNIVERSE)
    idx = pd.bdate_range("2016-10-17", periods=n)
    common = rng.normal(0.0003, 0.010, n)
    idio = rng.normal(0, 0.016, (n, k))
    drift = np.zeros((n, k))
    for t in range(1, n):
        drift[t] = 0.95 * drift[t - 1] + rng.normal(0, 0.00025 * edge, k)
    ret = common[:, None] + idio + drift
    return pd.DataFrame(100 * np.exp(np.cumsum(ret, axis=0)), index=idx, columns=list(su.UNIVERSE))


def build(close, lookback):
    cl = close.ffill(limit=5)
    idx = cl.index
    last_of_week = idx.to_series().groupby(idx.to_period("W")).transform("max").values == idx.values
    pos = np.where(last_of_week)[0]
    pos = pos[pos >= lookback]
    pos = pos[pos + 1 < len(idx)]
    A = cl.values
    entry = pos + 1
    R = A[entry[1:]] / A[entry[:-1]] - 1
    sig = pos[:-1]
    S = A[sig] / A[sig - lookback] - 1
    return S, R, idx[entry[:-1]]


def top_n_weights(S, avail, n):
    Sm = np.where(avail, S, -np.inf)
    order = np.argsort(-Sm, axis=1)
    rank = np.empty_like(order)
    rows = np.arange(S.shape[0])[:, None]
    rank[rows, order] = np.arange(S.shape[1])[None, :]
    sel = (rank < n) & avail
    cnt = sel.sum(axis=1, keepdims=True)
    return np.where(cnt > 0, sel / np.maximum(cnt, 1), 0.0)


def net_returns(W, Rc, cost):
    gross = (W * Rc).sum(axis=1)
    turnover = np.abs(np.diff(W, axis=0, prepend=np.zeros((1, W.shape[1])))).sum(axis=1)
    return gross - turnover * cost


def ann_ret(r):
    return float(np.prod(1 + r) ** (PPY / len(r)) - 1)


def ann_vol(r):
    return float(np.std(r, ddof=1) * np.sqrt(PPY))


def sharpe(r):
    sd = np.std(r, ddof=1)
    return float(np.mean(r) / sd * np.sqrt(PPY)) if sd > 0 else 0.0


def max_dd(r):
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(eq)
    return float(((eq - peak) / peak).min())


def info_ratio(rs, rb):
    d = rs - rb
    sd = np.std(d, ddof=1)
    return float(np.mean(d) / sd * np.sqrt(PPY)) if sd > 0 else 0.0


def line(label, r):
    return (f"  {label:30s} rend. annuo {ann_ret(r):+6.1%} | volatilita' {ann_vol(r):5.1%} | "
            f"sharpe {sharpe(r):5.2f} | max drawdown {max_dd(r):6.1%}")


def report(close):
    close = close.drop(columns=[su.BENCH], errors="ignore")
    S, R, dates = build(close, su.LOOKBACK)
    M, K = R.shape
    avail = ~np.isnan(S) & ~np.isnan(R)
    Rc = np.where(np.isnan(R), 0.0, R)
    cut = int(M * (1 - OOS_FRAC))

    W = top_n_weights(S, avail, su.TOP_N)
    Wb = avail / np.maximum(avail.sum(axis=1, keepdims=True), 1)
    strat, bench = net_returns(W, Rc, COST), net_returns(Wb, Rc, COST)
    turn = float(np.abs(np.diff(W, axis=0)).sum(axis=1).mean())

    print(f"\nSettimane testate: {M} ({dates[0].date()} -> {dates[-1].date()}), sviluppo = prime {cut}, test = ultime {M - cut}")
    print(f"Azioni disponibili nell'ultima settimana: {int(avail[-1].sum())} | ricambio medio a settimana: {turn:.0%} del conto\n")
    for label, sl in (("SVILUPPO (primo 70%)", slice(0, cut)), ("TEST (ultimo 30%)", slice(cut, M))):
        print(label)
        print(line("Settimanale top-5 4 sett.", strat[sl]))
        print(line("Paniere equipesato (tutte)", bench[sl]))
        print(f"  {'Eccesso annuo / info ratio':30s} {ann_ret(strat[sl]) - ann_ret(bench[sl]):+6.1%} / "
              f"{info_ratio(strat[sl], bench[sl]):.2f}\n")

    ex_oos = ann_ret(strat[cut:]) - ann_ret(bench[cut:])
    ex_is = ann_ret(strat[:cut]) - ann_ret(bench[:cut])
    ir_oos, ir_is = info_ratio(strat[cut:], bench[cut:]), info_ratio(strat[:cut], bench[:cut])

    rng = np.random.default_rng(SEED)
    real_mean = float(np.mean(strat[cut:]))
    plac = np.array([np.mean(net_returns(top_n_weights(rng.random((M, K)), avail, su.TOP_N), Rc, COST)[cut:])
                     for _ in range(N_PLACEBO)])
    pctl = float(np.mean(plac < real_mean))

    years = np.array([d.year for d in dates[cut:]])
    pos_y = tot_y = 0
    for y in np.unique(years):
        m = years == y
        if m.sum() < 20:
            continue
        tot_y += 1
        pos_y += np.prod(1 + strat[cut:][m]) > np.prod(1 + bench[cut:][m])
    frac_y = pos_y / tot_y if tot_y else 0.0

    dd_s, dd_b = abs(max_dd(strat)), abs(max_dd(bench))
    ex_stress = ann_ret(net_returns(W, Rc, COST * 2)[cut:]) - ann_ret(net_returns(Wb, Rc, COST * 2)[cut:])

    rob = {}
    for L in ROBUST_LOOKBACKS:
        S2, R2, _ = build(close, L)
        S2, R2 = S2[-M:], R2[-M:]                      # stesse settimane del test principale
        av2 = ~np.isnan(S2) & ~np.isnan(R2)
        rc2 = np.where(np.isnan(R2), 0.0, R2)
        rl = net_returns(top_n_weights(S2, av2, su.TOP_N), rc2, COST)
        rob[L] = ann_ret(rl[cut:]) - ann_ret(bench[cut:])

    checks = [
        (f"1. Test: eccesso > 0 e info ratio >= {MIN_IR}", ex_oos > 0 and ir_oos >= MIN_IR, f"eccesso {ex_oos:+.1%}, IR {ir_oos:.2f}"),
        (f"2. Sviluppo: eccesso > 0 e info ratio >= {MIN_IR}", ex_is > 0 and ir_is >= MIN_IR, f"eccesso {ex_is:+.1%}, IR {ir_is:.2f}"),
        (f"3. Batte il {PLACEBO_PCTL:.0%} di {N_PLACEBO} selezioni casuali", pctl >= PLACEBO_PCTL, f"percentile {pctl:.0%}"),
        (f"4. Eccesso positivo in >= {MIN_YEARS_POSITIVE:.0%} degli anni di test", frac_y >= MIN_YEARS_POSITIVE, f"{pos_y} anni su {tot_y}"),
        (f"5. Drawdown non peggiore del paniere di >{MAX_DD_WORSE:.0%}", dd_s <= dd_b + MAX_DD_WORSE, f"regola {dd_s:.1%} contro paniere {dd_b:.1%}"),
        ("6. Eccesso di test > 0 con costi raddoppiati", ex_stress > 0, f"{ex_stress:+.1%}"),
        ("7. Finestre 8 e 13 settimane con eccesso di test > 0", all(v > 0 for v in rob.values()),
         ", ".join(f"{L // 5} sett.: {v:+.1%}" for L, v in rob.items())),
    ]
    print("CRITERI DI SUCCESSO (fissati in anticipo)")
    for text, ok, detail in checks:
        print(f"  [{'OK' if ok else 'NO'}] {text:58s} {detail}")
    passed = sum(ok for _, ok, _ in checks)
    print(f"\nVERDETTO: {passed}/7 criteri superati.")
    if passed == 7:
        print("Superati tutti i criteri. Ma l'elenco contiene solo aziende sopravvissute: il vero giudice e' il conto in tempo reale.")
    else:
        print("Non tutti i criteri sono superati: la regola NON passa il test storico. Niente ritocchi ai parametri.")
    print("Promemoria: l'elenco contiene aziende grandi OGGI (survivorship bias): il risultato storico e' favorevole per costruzione.")
    return passed


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--period", default="10y")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--edge", type=float, default=0.0)
    a = p.parse_args()
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    if a.synthetic:
        close = synthetic_prices(a.edge)
        print(f"Prezzi sintetici (momentum iniettato: {a.edge}), {close.shape[1]} azioni, {len(close)} giorni")
    else:
        print(f"Scarico i prezzi di {len(su.UNIVERSE)} azioni ({a.period})...")
        close = load_prices(a.period)
        print(f"  {close.shape[1]} azioni disponibili, {len(close)} giorni, {close.index[0].date()} -> {close.index[-1].date()}")
    report(close)
