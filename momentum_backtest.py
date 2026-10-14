"""
Rotazione sui mercati globali "piu' produttivi del momento" (momentum) - test con criteri FISSATI IN ANTICIPO.

Regola (valori standard della letteratura, decisi prima di guardare i dati, NON da ottimizzare):
  - Universo: 29 ETF di mercati azionari mondiali (20 paesi) e dei 9 settori USA, con dividendi reinvestiti.
    Si usano ETF e non singole azioni perche' le liste gratuite di azioni contengono solo le societa'
    sopravvissute fino a oggi, e questo gonfia i risultati del passato (survivorship bias).
  - L'ultimo giorno di ogni mese si classificano gli ETF per rendimento degli ultimi 12 mesi
    escludendo l'ultimo mese ("12-1", per evitare l'effetto di inversione di breve).
  - Si comprano i 5 migliori, pesi uguali, long-only, e li si tiene un mese. Ingresso il giorno
    DOPO il segnale (niente look-ahead). Costo 5 punti base per lato su ogni operazione.
  - Il confronto e' col paniere equipesato di TUTTI gli ETF: battere il mercato e' il vero test,
    non semplicemente guadagnare quando le borse salgono.

Verdetto automatico su 7 criteri (devono passare TUTTI):
  1. Test (ultimo 30%): rendimento annuo in eccesso sul paniere > 0 e information ratio >= 0.3
  2. Sviluppo (primo 70%): rendimento in eccesso > 0 e information ratio >= 0.3
  3. Batte il 95% di 1000 selezioni CASUALI di 5 ETF (stesso periodo di test): la classifica aggiunge informazione
  4. In eccesso positivo in almeno il 60% degli anni del test
  5. Max drawdown non peggiore di quello del paniere di piu' di 5 punti percentuali
  6. Con costi raddoppiati l'eccesso nel test resta > 0
  7. Anche le finestre 6-1 e 9-1 mesi hanno eccesso > 0 nel test (robustezza, non scelta)

Uso:
    py momentum_backtest.py
    py momentum_backtest.py --synthetic            (dati finti senza momentum: deve fallire)
    py momentum_backtest.py --synthetic --edge 0.5 (dati finti CON momentum: deve passare)

Limiti: il test e' sul passato; un mese di rotazione con ~65 mesi di test ha poca potenza
statistica; i dati Yahoo mancano degli ETF chiusi nel tempo. Non e' un consiglio finanziario.
"""

import argparse
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- configurazione (non ritoccare dopo aver visto i risultati)

UNIVERSE = {
    "SPY": "USA (S&P 500)", "EWJ": "Giappone", "EWG": "Germania", "EWU": "Regno Unito", "EWQ": "Francia",
    "EWI": "Italia", "EWP": "Spagna", "EWL": "Svizzera", "EWA": "Australia", "EWC": "Canada",
    "EWZ": "Brasile", "EWW": "Messico", "EWY": "Corea del Sud", "EWT": "Taiwan", "EWH": "Hong Kong",
    "EWS": "Singapore", "FXI": "Cina (large cap)", "INDA": "India", "EZA": "Sudafrica", "TUR": "Turchia",
    "XLK": "Tecnologia USA", "XLF": "Finanza USA", "XLE": "Energia USA", "XLV": "Sanita' USA",
    "XLY": "Consumi discrezionali USA", "XLP": "Beni di prima necessita' USA", "XLI": "Industriali USA",
    "XLB": "Materiali USA", "XLU": "Utility USA",
}

TOP_N = 5
LOOKBACK, SKIP = 252, 21            # 12-1 mesi, in giorni di borsa
ROBUST_LOOKBACKS = (126, 189)       # 6-1 e 9-1 mesi
COST = 5 / 10000                    # per lato, sul valore scambiato
OOS_FRAC = 0.30
N_PLACEBO = 1000
SEED = 2026
MIN_ROWS = 800

MIN_IR = 0.3
MIN_YEARS_POSITIVE = 0.60
MAX_DD_WORSE = 0.05
PLACEBO_PCTL = 0.95


# ---------------------------------------------------------------- dati

def load_prices(period):
    import yfinance as yf
    df = yf.download(list(UNIVERSE), period=period, interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        sys.exit("Nessun dato ricevuto da Yahoo: controlla la connessione.")
    close = df["Close"]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    keep, dropped = [], []
    for tk in close.columns:
        (keep if close[tk].notna().sum() >= MIN_ROWS else dropped).append(tk)
    for tk in dropped:
        print(f"  {tk}: scartato (pochi dati)")
    return close[keep]


def synthetic_prices(edge=0.0, seed=3):
    """Prezzi finti. edge=0: nessun momentum. edge>0: ogni ETF ha un trend lento e persistente."""
    rng = np.random.default_rng(seed)
    n, k = 5000, len(UNIVERSE)
    idx = pd.bdate_range("2006-10-16", periods=n)
    common = rng.normal(0.0002, 0.009, n)
    idio = rng.normal(0, 0.008, (n, k))
    drift = np.zeros((n, k))
    for t in range(1, n):
        drift[t] = 0.995 * drift[t - 1] + rng.normal(0, 0.00008 * edge, k)
    ret = common[:, None] + idio + drift
    return pd.DataFrame(100 * np.exp(np.cumsum(ret, axis=0)), index=idx, columns=list(UNIVERSE))


# ---------------------------------------------------------------- strategia

def build(close):
    cl = close.ffill(limit=5)
    idx = cl.index
    last_of_month = idx.to_series().groupby(idx.to_period("M")).transform("max").values == idx.values
    pos = np.where(last_of_month)[0]
    pos = pos[pos >= LOOKBACK]                 # servono 12 mesi di storia
    pos = pos[pos + 1 < len(idx)]              # serve il giorno dopo per entrare
    A = cl.values
    entry = pos + 1
    R = A[entry[1:]] / A[entry[:-1]] - 1       # rendimento di ogni mese tenuto
    sig = pos[:-1]
    dates = idx[entry[:-1]]
    return A, sig, R, dates


def scores(A, sig, lookback):
    return A[sig - SKIP] / A[sig - lookback] - 1


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


# ---------------------------------------------------------------- metriche (rendimenti mensili)

def ann_ret(r):
    return float(np.prod(1 + r) ** (12 / len(r)) - 1)


def ann_vol(r):
    return float(np.std(r, ddof=1) * np.sqrt(12))


def sharpe(r):
    sd = np.std(r, ddof=1)
    return float(np.mean(r) / sd * np.sqrt(12)) if sd > 0 else 0.0


def max_dd(r):
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(eq)
    return float(((eq - peak) / peak).min())


def info_ratio(rs, rb):
    d = rs - rb
    sd = np.std(d, ddof=1)
    return float(np.mean(d) / sd * np.sqrt(12)) if sd > 0 else 0.0


def line(label, r):
    return (f"  {label:30s} rend. annuo {ann_ret(r):+6.1%} | volatilita' {ann_vol(r):5.1%} | "
            f"sharpe {sharpe(r):5.2f} | max drawdown {max_dd(r):6.1%}")


# ---------------------------------------------------------------- report e verdetto

def report(close):
    A, sig, R, dates = build(close)
    M, K = R.shape
    S = scores(A, sig, LOOKBACK)
    avail = ~np.isnan(S) & ~np.isnan(R)
    Rc = np.where(np.isnan(R), 0.0, R)
    cut = int(M * (1 - OOS_FRAC))

    W = top_n_weights(S, avail, TOP_N)
    Wb = avail / np.maximum(avail.sum(axis=1, keepdims=True), 1)
    strat = net_returns(W, Rc, COST)
    bench = net_returns(Wb, Rc, COST)

    print(f"\nMesi di test del metodo: {M} ({dates[0].date()} -> {dates[-1].date()}), "
          f"sviluppo = primi {cut}, test = ultimi {M - cut}")
    print(f"Universo usato: {int(avail[-1].sum())} ETF disponibili nell'ultimo mese\n")
    for label, sl in (("SVILUPPO (primo 70%)", slice(0, cut)), ("TEST (ultimo 30%)", slice(cut, M))):
        print(label)
        print(line("Rotazione top-5 momentum", strat[sl]))
        print(line("Paniere equipesato (tutti)", bench[sl]))
        print(f"  {'Eccesso annuo / info ratio':30s} {ann_ret(strat[sl]) - ann_ret(bench[sl]):+6.1%} / "
              f"{info_ratio(strat[sl], bench[sl]):.2f}\n")

    ex_oos = ann_ret(strat[cut:]) - ann_ret(bench[cut:])
    ex_is = ann_ret(strat[:cut]) - ann_ret(bench[:cut])
    ir_oos, ir_is = info_ratio(strat[cut:], bench[cut:]), info_ratio(strat[:cut], bench[:cut])

    # 3. placebo: 1000 selezioni casuali di 5 ETF
    rng = np.random.default_rng(SEED)
    real_mean = float(np.mean(strat[cut:]))
    plac = np.empty(N_PLACEBO)
    for i in range(N_PLACEBO):
        Wr = top_n_weights(rng.random((M, K)), avail, TOP_N)
        plac[i] = np.mean(net_returns(Wr, Rc, COST)[cut:])
    pctl = float(np.mean(plac < real_mean))

    # 4. anni positivi nel test
    years = np.array([d.year for d in dates[cut:]])
    pos_years, tot_years = 0, 0
    for y in np.unique(years):
        m = years == y
        if m.sum() < 6:
            continue
        tot_years += 1
        pos_years += np.prod(1 + strat[cut:][m]) > np.prod(1 + bench[cut:][m])
    frac_years = pos_years / tot_years if tot_years else 0.0

    # 5. drawdown
    dd_s, dd_b = abs(max_dd(strat)), abs(max_dd(bench))

    # 6. costi doppi
    ex_stress = ann_ret(net_returns(W, Rc, COST * 2)[cut:]) - ann_ret(net_returns(Wb, Rc, COST * 2)[cut:])

    # 7. finestre alternative
    rob = {}
    for L in ROBUST_LOOKBACKS:
        Wl = top_n_weights(scores(A, sig, L), avail, TOP_N)
        rl = net_returns(Wl, Rc, COST)
        rob[L] = ann_ret(rl[cut:]) - ann_ret(bench[cut:])

    checks = [
        (f"1. Test: eccesso > 0 e info ratio >= {MIN_IR}", ex_oos > 0 and ir_oos >= MIN_IR,
         f"eccesso {ex_oos:+.1%}, IR {ir_oos:.2f}"),
        (f"2. Sviluppo: eccesso > 0 e info ratio >= {MIN_IR}", ex_is > 0 and ir_is >= MIN_IR,
         f"eccesso {ex_is:+.1%}, IR {ir_is:.2f}"),
        (f"3. Batte il {PLACEBO_PCTL:.0%} di {N_PLACEBO} selezioni casuali", pctl >= PLACEBO_PCTL,
         f"percentile {pctl:.0%}"),
        (f"4. Eccesso positivo in >= {MIN_YEARS_POSITIVE:.0%} degli anni di test", frac_years >= MIN_YEARS_POSITIVE,
         f"{pos_years} anni su {tot_years}"),
        (f"5. Drawdown non peggiore del paniere di >{MAX_DD_WORSE:.0%}", dd_s <= dd_b + MAX_DD_WORSE,
         f"rotazione {dd_s:.1%} contro paniere {dd_b:.1%}"),
        ("6. Eccesso di test > 0 con costi raddoppiati", ex_stress > 0, f"{ex_stress:+.1%}"),
        ("7. Finestre 6-1 e 9-1 con eccesso di test > 0", all(v > 0 for v in rob.values()),
         ", ".join(f"{L // 21}-1: {v:+.1%}" for L, v in rob.items())),
    ]
    print("CRITERI DI SUCCESSO (fissati in anticipo)")
    for text, ok, detail in checks:
        print(f"  [{'OK' if ok else 'NO'}] {text:58s} {detail}")
    passed = sum(ok for _, ok, _ in checks)
    print(f"\nVERDETTO: {passed}/7 criteri superati.")
    if passed == 7:
        print("Tutti i criteri sono superati: la regola merita di essere seguita in tempo reale (solo osservazione, poi demo).")
        print("Non e' una garanzia: nel test ci sono pochi anni e il futuro puo' essere diverso.")
    else:
        print("Non tutti i criteri sono superati: la regola NON passa. Niente ritocchi ai parametri per far tornare i numeri.")
    print(f"\nPotenza statistica: con {M - cut} mesi di test, anche un vantaggio vero puo' non risultare significativo.")
    return passed


def show_today(close):
    cl = close.ffill(limit=5)
    A = cl.values
    s = A[-1 - SKIP] / A[-1 - LOOKBACK] - 1
    order = np.argsort(-np.where(np.isnan(s), -np.inf, s))
    print(f"\nCLASSIFICA DI OGGI ({cl.index[-1].date()}): rendimento 12-1 mesi")
    print("(informativa: applica la stessa regola del test; il verdetto qui sopra dice quanta fiducia darle)")
    for rank, j in enumerate(order[:10], 1):
        if np.isnan(s[j]):
            continue
        tk = cl.columns[j]
        mark = "  <-- fra i 5 scelti dalla regola" if rank <= TOP_N else ""
        print(f"  {rank:2d}. {tk:5s} {UNIVERSE[tk]:28s} {s[j]:+7.1%}{mark}")


# ---------------------------------------------------------------- main

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--period", default="20y")
    p.add_argument("--synthetic", action="store_true", help="dati finti, solo per testare il codice")
    p.add_argument("--edge", type=float, default=0.0, help="con --synthetic: intensita' del momentum iniettato")
    a = p.parse_args()

    if a.synthetic:
        close = synthetic_prices(a.edge)
        print(f"Prezzi sintetici (momentum iniettato: {a.edge}), {close.shape[1]} ETF, {len(close)} giorni")
    else:
        print(f"Scarico i prezzi di {len(UNIVERSE)} ETF ({a.period})...")
        close = load_prices(a.period)
        print(f"  {close.shape[1]} ETF disponibili, {len(close)} giorni, "
              f"{close.index[0].date()} -> {close.index[-1].date()}")
    report(close)
    show_today(close)
