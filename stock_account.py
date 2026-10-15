"""
CONTO DEMO "VELOCE" SU AZIONI GLOBALI - simulato, nessun broker, nessun denaro vero.

Ogni settimana (primo giorno di borsa) compra le 5 azioni, fra ~79 grandi societa' di USA, Europa, Giappone, Hong Kong,
Australia, Canada, Corea, Taiwan, India, con il miglior rendimento delle ultime 4 settimane, in parti uguali (20% ciascuna),
e vende le altre. Segnale all'ultima chiusura della settimana prima, esecuzione alla chiusura del giorno dopo.
Costo simulato 5 punti base per lato. Solo denaro disponibile, niente leva. Se il conto perde il 25% dal massimo,
vende tutto e si ferma (comando resume per riattivare). I prezzi sono in valuta locale: il cambio non e' simulato.
Regola e universo: stock_universe.py (fissati in anticipo, non ritoccare).

Uso:
    py stock_account.py init --cash 10000 | update [--dry] [--notify] | report | resume
"""

import argparse
import copy
import json
import os
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import stock_universe as su
import virtual_account as va            # riuso: valore posizioni, operazioni, chiusura, Telegram

DATA_DIR = Path(os.environ.get("VA_DATA_DIR") or (Path.home() / "forex_data"))
VARIANT = "base"            # base = conto veloce originale; trend / riskw = conti sfidanti
VARIANTS = ("base", "trend", "riskw")
TREND_DAYS = 200            # trend: se l'S&P 500 e' sotto la sua media a 200 giorni, tutto in contanti
VOL_DAYS = 60               # riskw: peso inverso alla volatilita' degli ultimi 60 giorni
W_CAP = 0.30                # riskw: massimo 30% su un titolo


def state_path():
    return DATA_DIR / ("stock_account.json" if VARIANT == "base" else f"stock_account_{VARIANT}.json")


def load_state():
    p = state_path()
    if not p.exists():
        sys.exit("Il conto non esiste ancora. Crealo con:  py stock_account.py init --cash 10000")
    return json.loads(p.read_text(encoding="utf-8"))


def save_state(state):
    p = state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        shutil.copy2(p, p.with_suffix(".bak"))
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    tmp.replace(p)


def new_state(cash):
    return {"created": datetime.now().strftime("%Y-%m-%d %H:%M"), "start_cash": float(cash), "cash": float(cash),
            "positions": {}, "trades": [], "snapshots": [], "last_rebalance_week": None,
            "start_date": None, "peak": float(cash), "halted": False, "halt_reason": ""}


def load_prices(start):
    import time
    import yfinance as yf
    tickers = list(su.UNIVERSE) + [su.BENCH]
    df = None
    for attempt in range(4):
        try:
            df = yf.download(tickers, start=start, interval="1d", auto_adjust=True, progress=False)
            if df is not None and not df.empty and df["Close"].notna().sum().sum() > 0:
                break
        except Exception as e:
            print(f"Tentativo {attempt + 1} fallito: {type(e).__name__}")
        time.sleep(20 * (attempt + 1))
    if df is None or df.empty:
        raise RuntimeError("nessun dato ricevuto da Yahoo: controlla la connessione")
    close = df["Close"]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close.dropna(how="all").ffill(limit=5)
    ok = [t for t in close.columns if close[t].notna().sum() >= 100]
    return close[ok]


def rebalance(state, A, e, s, tag):
    date = A.index[e].strftime("%Y-%m-%d")
    equity, vals = va.equity_at(state, A, e)
    state["peak"] = max(state["peak"], equity)
    if equity < state["peak"] * (1 - va.MAX_DD_STOP):
        va.liquidate(state, A, e, f"blocco: perdita oltre {va.MAX_DD_STOP:.0%} dal massimo")
        state["last_rebalance_week"] = tag
        return
    cand = [c for c in A.columns if c != su.BENCH]
    if VARIANT == "trend":
        bench = A[su.BENCH]
        if s >= TREND_DAYS - 1 and bench.iloc[s] < bench.iloc[s - TREND_DAYS + 1:s + 1].mean():
            for tk, v in vals.items():
                va._trade(state, date, tk, -v, va.mb.COST * v, "filtro di tendenza: S&P 500 sotto la media a 200 giorni")
            state["positions"] = {}
            state["last_rebalance_week"] = tag
            return
    S = (A[cand].iloc[s] / A[cand].iloc[s - su.LOOKBACK] - 1)
    S = S[A[cand].iloc[e].notna()].dropna().sort_values(ascending=False)     # serve un prezzo valido anche all'esecuzione
    chosen = list(S.index[:su.TOP_N])
    if len(chosen) < su.TOP_N:
        return                                                               # dati insufficienti: non opera
    if VARIANT == "riskw":
        vol = A[chosen].iloc[s - VOL_DAYS:s + 1].pct_change().std().replace(0, np.nan).dropna()
        if len(vol) == len(chosen):
            w = (1 / vol) / (1 / vol).sum()
            w = w.clip(upper=W_CAP)
            target = {tk: equity * va.INVEST_FRAC * float(w[tk]) for tk in chosen}
        else:
            target = {tk: equity * va.INVEST_FRAC / len(chosen) for tk in chosen}
    else:
        target = {tk: equity * va.INVEST_FRAC / len(chosen) for tk in chosen}
    for tk in sorted(set(vals) | set(target)):
        delta = target.get(tk, 0.0) - vals.get(tk, 0.0)
        if abs(delta) < va.MIN_TRADE:
            if tk in target and tk in state["positions"]:
                state["positions"][tk] = {"value": vals[tk], "ref_date": date}
            continue
        va._trade(state, date, tk, delta, va.mb.COST * abs(delta),
                  "rotazione settimanale: " + (f"entra in top-{su.TOP_N}" if delta > 0 else "esce dai migliori"))
        if tk in target:
            state["positions"][tk] = {"value": target[tk], "ref_date": date}
        else:
            state["positions"].pop(tk, None)
    state["last_rebalance_week"] = tag


def wtag(p):
    return p.start_time.strftime("%Y-%m-%d")          # lunedi' della settimana


def wperiod(tag):
    return pd.Period(tag, freq="W-SUN")


def weeks_to_do(state, A):
    last = A.index[-1].to_period("W-SUN")
    if state["last_rebalance_week"] is None:
        return [wtag(last)]
    p = wperiod(state["last_rebalance_week"]) + 1
    out = []
    while p <= last:
        out.append(wtag(p))
        p += 1
    return out


def update(state, A):
    if len(A) < su.LOOKBACK + 10:
        raise RuntimeError("dati insufficienti per calcolare la regola")
    notes = []
    last = len(A) - 1
    if not state["halted"]:
        for tag in weeks_to_do(state, A):
            if state["start_date"] is None:
                e, s = last, last - 1
                state["start_date"] = A.index[e].strftime("%Y-%m-%d")
            else:
                inweek = np.where(A.index.to_period("W-SUN") == wperiod(tag))[0]
                if len(inweek) == 0:
                    continue
                e = int(inweek[0])
                s = e - 1
            n_before = len(state["trades"])
            rebalance(state, A, e, s, tag)
            notes.append(f"Ribilanciamento settimana dal {tag}: {len(state['trades']) - n_before} operazioni "
                         f"(eseguite alla chiusura del {A.index[e].date()})")
            if state["halted"]:
                break

    equity, _ = va.equity_at(state, A, last)
    state["peak"] = max(state["peak"], equity)
    if not state["halted"] and equity < state["peak"] * (1 - va.MAX_DD_STOP):
        va.liquidate(state, A, last, f"blocco: perdita oltre {va.MAX_DD_STOP:.0%} dal massimo")
        notes.append("BLOCCO ATTIVATO: tutto venduto. " + state["halt_reason"])
        equity, _ = va.equity_at(state, A, last)

    if state["start_date"] is not None:
        si = va.idx_of(A, state["start_date"])
        spy = state["start_cash"] * A[su.BENCH].iloc[last] / A[su.BENCH].iloc[si]
        cand = [c for c in A.columns if c != su.BENCH]
        rel = (A[cand].iloc[last] / A[cand].iloc[si]).dropna()
        ew = state["start_cash"] * float(rel.mean())
        snap = {"date": A.index[last].strftime("%Y-%m-%d"), "equity": round(equity, 2),
                "spy": round(float(spy), 2), "ew": round(ew, 2)}
        if state["snapshots"] and state["snapshots"][-1]["date"] == snap["date"]:
            state["snapshots"][-1] = snap
        else:
            state["snapshots"].append(snap)
    return notes


def build_report(state, A=None):
    L = [f"CONTO DEMO VELOCE - AZIONI GLOBALI [{VARIANT}] (simulato, non sono soldi veri)"]
    sc = state["start_cash"]
    if state["start_date"] is None:
        L.append(f"Saldo iniziale {sc:,.2f} USD. Nessuna operazione ancora.")
        return "\n".join(L)
    snap = state["snapshots"][-1]
    eq, spy, ew = snap["equity"], snap["spy"], snap["ew"]
    days = (pd.Timestamp(snap["date"]) - pd.Timestamp(state["start_date"])).days
    L.append(f"Dal {state['start_date']} al {snap['date']} ({days} giorni)")
    L.append(f"Saldo iniziale:   {sc:>12,.2f}")
    L.append(f"Saldo attuale:    {eq:>12,.2f}   ({eq / sc - 1:+.2%})")
    L.append(f"Riferimento S&P 500:        {spy:>12,.2f}   ({spy / sc - 1:+.2%})")
    L.append(f"Riferimento tutte le azioni:{ew:>12,.2f}   ({ew / sc - 1:+.2%})")
    L.append(f"Differenza rispetto all'S&P 500: {eq / sc - spy / sc:+.2%}")
    if state["halted"]:
        L.append(f"\n*** CONTO FERMO *** {state['halt_reason']}")
    L.append("\nPosizioni:")
    tot = eq
    for tk, p in sorted(state["positions"].items()):
        v = va.pos_value(A, tk, p, len(A) - 1) if A is not None else p["value"]
        L.append(f"  {tk:10s} {su.UNIVERSE.get(tk, ''):14s} {v:>10,.2f}  ({v / tot:5.1%})")
    L.append(f"  {'contanti':25s} {state['cash']:>10,.2f}  ({state['cash'] / tot:5.1%})")
    L.append(f"\nUltima settimana ribilanciata: {state['last_rebalance_week']} | operazioni totali: {len(state['trades'])}")
    for t in state["trades"][-10:]:
        L.append(f"  {t['date']}  {t['side']:6s} {t['ticker']:10s} {t['amount']:>10,.2f}  (costo {t['cost']:.2f})")
    if days < 90:
        L.append("\nNota: con meno di 3 mesi di vita il risultato e' quasi tutto rumore.")
    return "\n".join(L)


def fetch_start(state):
    s = datetime.now() - timedelta(days=420)
    if state.get("start_date"):
        s = min(s, pd.Timestamp(state["start_date"]).to_pydatetime() - timedelta(days=60))
    return s.strftime("%Y-%m-%d")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["init", "update", "report", "resume"])
    p.add_argument("--cash", type=float, default=10000.0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry", action="store_true")
    p.add_argument("--notify", action="store_true")
    p.add_argument("--variant", choices=VARIANTS, default="base")
    a = p.parse_args()
    VARIANT = a.variant

    if a.mode == "init":
        if state_path().exists() and not a.force:
            sys.exit("Il conto esiste gia'. Per ricominciare da zero aggiungi --force")
        save_state(new_state(a.cash))
        print(f"Conto veloce creato con {a.cash:,.2f} USD virtuali in {state_path()}")
        sys.exit(0)

    state = load_state()
    if a.mode == "resume":
        state["halted"], state["halt_reason"] = False, ""
        state["peak"] = state["snapshots"][-1]["equity"] if state["snapshots"] else state["cash"]
        save_state(state)
        print("Conto riattivato.")
        sys.exit(0)

    A = load_prices(fetch_start(state))
    if a.mode == "report":
        print(build_report(state, A))
        sys.exit(0)

    work = copy.deepcopy(state) if a.dry else state
    notes = update(work, A)
    for n in notes:
        print(n)
    if not notes:
        print("Nessun ribilanciamento dovuto oggi: aggiornato solo il valore del conto.")
    text = build_report(work, A)
    print("\n" + text)
    if a.dry:
        print("\n(SIMULAZIONE: non e' stato salvato nulla)")
    else:
        save_state(work)
        if a.notify and (notes or datetime.now().weekday() == 4):     # se succede qualcosa, o il venerdi'
            va.send_telegram(("\n".join(notes) + "\n\n" if notes else "") + text)
