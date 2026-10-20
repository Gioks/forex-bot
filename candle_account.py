"""
CONTO DEMO "TRE CANDELE" - simulato, nessun broker, nessun denaro vero.

Regola (la stessa di tre_candele_backtest.py, fissata in anticipo):
  - Candele giornaliere. Dopo 3 candele consecutive in rialzo (chiusura sopra apertura) si COMPRA all'apertura
    della candela successiva. Si tiene finche' sale. Dopo 3 candele consecutive in ribasso si VENDE all'apertura successiva.
  - Ogni azione ha la sua quota di capitale (saldo iniziale / numero di azioni); quando non e' in posizione la quota
    resta in contanti. Solo acquisti, niente leva, niente vendite allo scoperto. Costo 5 punti base per lato.
  - Riferimenti: S&P 500 e paniere equipesato di tutte le azioni.
Prezzi veri di mercato, ordini simulati. Il test storico e' tre_candele_backtest.py (da lanciare a parte).

Uso:  py candle_account.py init --cash 10000 | update [--dry] | report
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

DATA_DIR = Path(os.environ.get("VA_DATA_DIR") or (Path.home() / "forex_data"))
STATE_FILE = "candle_account.json"
N = 3
COST = 5 / 10000
HISTORY_DAYS = 45


def state_path():
    return DATA_DIR / STATE_FILE


def save_state(state):
    p = state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        shutil.copy2(p, p.with_suffix(".bak"))
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    tmp.replace(p)


def load_state():
    p = state_path()
    if not p.exists():
        sys.exit("Il conto non esiste ancora. Crealo con:  py candle_account.py init --cash 10000")
    return json.loads(p.read_text(encoding="utf-8"))


def new_state(cash):
    return {"created": datetime.now().strftime("%Y-%m-%d %H:%M"), "start_cash": float(cash), "sleeves": {},
            "last_date": None, "start_date": None, "trades": [], "snapshots": [], "p0": {}, "spy0": None}


def load_ohlc(start):
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
    O, C = df["Open"], df["Close"]
    for x in (O, C):
        x.index = pd.to_datetime(x.index).tz_localize(None)
    keep = [t for t in C.columns if C[t].notna().sum() >= 10]
    return O[keep].dropna(how="all"), C[keep].dropna(how="all")


def runs_at(O, C, tk, i):
    """True/False: le ultime N candele chiuse (fino a i) sono tutte in rialzo / tutte in ribasso."""
    if i < N - 1:
        return False, False
    o, c = O[tk].iloc[i - N + 1:i + 1].values, C[tk].iloc[i - N + 1:i + 1].values
    if np.isnan(o).any() or np.isnan(c).any():
        return False, False
    return bool((c > o).all()), bool((c < o).all())


def update_signal(sl, g, r):
    if not sl["in"] and sl["want"] == 0 and g:
        sl["want"] = 1
    elif (sl["in"] or sl["want"] == 1) and r:
        sl["want"] = 0


def equity(state):
    return sum(s["cash"] + s["pos"] for s in state["sleeves"].values())


def update(state, O, C):
    tks = [t for t in C.columns if t != su.BENCH]
    idx = C.index
    notes = []
    if state["last_date"] is None:                       # primo giorno: si crea il conto e si leggono i segnali di oggi
        i = len(idx) - 1
        valid = [t for t in tks if not np.isnan(C[t].iloc[i])]
        slot = state["start_cash"] / len(valid)
        for t in valid:
            sl = {"cash": slot, "pos": 0.0, "in": False, "want": 0}
            g, r = runs_at(O, C, t, i)
            update_signal(sl, g, r)
            state["sleeves"][t] = sl
        state["start_date"] = idx[i].strftime("%Y-%m-%d")
        state["last_date"] = state["start_date"]
        state["p0"] = {t: float(C[t].iloc[i]) for t in valid}
        state["spy0"] = float(C[su.BENCH].iloc[i])
        todo = []
        notes.append(f"Conto creato il {state['start_date']} con {len(valid)} quote da {slot:,.2f}. "
                     f"Segnali di acquisto pronti per domani: {sum(s['want'] for s in state['sleeves'].values())}")
        add_snapshot(state, C, i)
    else:
        todo = [k for k in range(len(idx)) if idx[k].strftime("%Y-%m-%d") > state["last_date"]]
    for i in todo:
        date = idx[i].strftime("%Y-%m-%d")
        for t, sl in state["sleeves"].items():
            if t not in C.columns:
                continue
            o, c, pc = O[t].iloc[i], C[t].iloc[i], C[t].iloc[i - 1] if i > 0 else np.nan
            if np.isnan(o) or np.isnan(c):
                continue
            if sl["in"] and not np.isnan(pc) and pc > 0:
                sl["pos"] *= o / pc                                   # notte: dalla chiusura precedente all'apertura
            if sl["want"] == 1 and not sl["in"]:                      # acquisto all'apertura
                inv = sl["cash"] / (1 + COST)
                cost = inv * COST
                sl["cash"], sl["pos"], sl["in"] = 0.0, inv, True
                state["trades"].append({"date": date, "ticker": t, "side": "COMPRA", "amount": round(inv, 2), "cost": round(cost, 2),
                                        "why": "3 candele in rialzo"})
            elif sl["want"] == 0 and sl["in"]:                        # vendita all'apertura
                proceeds = sl["pos"] * (1 - COST)
                state["trades"].append({"date": date, "ticker": t, "side": "VENDI", "amount": round(sl["pos"], 2),
                                        "cost": round(sl["pos"] * COST, 2), "why": "3 candele in ribasso"})
                sl["cash"], sl["pos"], sl["in"] = sl["cash"] + proceeds, 0.0, False
            if sl["in"]:
                sl["pos"] *= c / o                                    # giornata: dall'apertura alla chiusura
            g, r = runs_at(O, C, t, i)
            update_signal(sl, g, r)                                   # segnale a mercato chiuso: vale da domani
        state["last_date"] = date
        add_snapshot(state, C, i)
    return notes


def add_snapshot(state, C, i):
    eq = equity(state)
    spy = state["start_cash"] * C[su.BENCH].iloc[i] / state["spy0"]
    rel = [C[t].iloc[i] / p for t, p in state["p0"].items() if t in C.columns and not np.isnan(C[t].iloc[i])]
    ew = state["start_cash"] * float(np.mean(rel))
    snap = {"date": C.index[i].strftime("%Y-%m-%d"), "equity": round(eq, 2), "spy": round(float(spy), 2), "ew": round(ew, 2),
            "invested": sum(1 for s in state["sleeves"].values() if s["in"])}
    if state["snapshots"] and state["snapshots"][-1]["date"] == snap["date"]:
        state["snapshots"][-1] = snap
    else:
        state["snapshots"].append(snap)


def build_report(state):
    L = ["CONTO DEMO TRE CANDELE (simulato, non sono soldi veri)"]
    if not state["snapshots"]:
        return "\n".join(L + ["Nessun dato ancora."])
    s = state["snapshots"][-1]
    sc = state["start_cash"]
    L.append(f"Dal {state['start_date']} al {s['date']}")
    L.append(f"Saldo: {s['equity']:,.2f} ({s['equity'] / sc - 1:+.2%}) | S&P 500: {s['spy'] / sc - 1:+.2%} | paniere: {s['ew'] / sc - 1:+.2%}")
    held = sorted(t for t, x in state["sleeves"].items() if x["in"])
    L.append(f"In posizione: {len(held)} azioni su {len(state['sleeves'])}" + (f" ({', '.join(held[:12])}{'...' if len(held) > 12 else ''})" if held else ""))
    L.append(f"Operazioni totali: {len(state['trades'])}")
    return "\n".join(L)


def fetch_start(state):
    d = datetime.now() - timedelta(days=HISTORY_DAYS)
    if state.get("last_date"):
        d = min(d, pd.Timestamp(state["last_date"]).to_pydatetime() - timedelta(days=HISTORY_DAYS))
    return d.strftime("%Y-%m-%d")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["init", "update", "report"])
    p.add_argument("--cash", type=float, default=10000.0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry", action="store_true")
    a = p.parse_args()
    if a.mode == "init":
        if state_path().exists() and not a.force:
            sys.exit("Il conto esiste gia'. Per ricominciare da zero aggiungi --force")
        save_state(new_state(a.cash))
        print(f"Conto tre candele creato con {a.cash:,.2f} USD virtuali in {state_path()}")
        sys.exit(0)
    state = load_state()
    if a.mode == "report":
        print(build_report(state))
        sys.exit(0)
    O, C = load_ohlc(fetch_start(state))
    work = copy.deepcopy(state) if a.dry else state
    for n in update(work, O, C):
        print(n)
    print(build_report(work))
    if not a.dry:
        save_state(work)
