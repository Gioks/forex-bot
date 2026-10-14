"""
CONTO DEMO SIMULATO - nessun broker, nessun ordine reale, nessun denaro vero.

Il programma tiene un conto virtuale (un registro) con un saldo iniziale finto. Ogni mese applica
la regola testata in momentum_backtest.py: compra i 5 ETF con il miglior rendimento degli ultimi
12 mesi (escluso l'ultimo), in parti uguali, e vende gli altri. I prezzi sono quelli veri di mercato.
Costo simulato: 5 punti base per lato su ogni operazione (come nel test).

Regole di protezione, sempre attive:
  - solo denaro disponibile (niente leva, niente vendite allo scoperto);
  - al massimo il 20% del conto su un singolo ETF;
  - se il conto perde il 25% dal suo massimo, vende tutto, si ferma e aspetta il tuo OK (comando resume).

Il conto e' in dollari USA, perche' gli ETF sono quotati in dollari.
Il confronto e' sempre con due riferimenti: S&P 500 e il paniere di tutti i 29 ETF, con lo stesso saldo iniziale.

Uso:
    py virtual_account.py init --cash 10000     crea il conto (una volta sola)
    py virtual_account.py update                aggiorna il conto e, se e' il momento, ribilancia (anche piu' volte al giorno: e' sicuro)
    py virtual_account.py update --dry          mostra cosa farebbe senza salvare niente
    py virtual_account.py update --notify       come update, poi manda il resoconto su Telegram (se configurato)
    py virtual_account.py report                mostra lo stato senza modificare nulla
    py virtual_account.py resume                riattiva il conto dopo il blocco al -25%

Non e' un consiglio finanziario. Un conto simulato non garantisce nulla sul reale: non ha slippage vero,
non ha il rischio emotivo e, soprattutto, ha pochi mesi di storia: i primi mesi non dimostrano niente.
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

import momentum_backtest as mb      # stesso universo, stessa regola, stessi costi del test

DATA_DIR = Path(os.environ.get("VA_DATA_DIR") or (Path.home() / "forex_data"))
STATE_FILE = "virtual_account.json"
MAX_DD_STOP = 0.25
INVEST_FRAC = 0.99                  # 1% resta liquido per i costi
MIN_TRADE = 1.0                     # sotto 1 dollaro non si opera


# ---------------------------------------------------------------- stato

def state_path(data_dir=None):
    return Path(data_dir or DATA_DIR) / STATE_FILE


def load_state(data_dir=None):
    p = state_path(data_dir)
    if not p.exists():
        sys.exit("Il conto non esiste ancora. Crealo con:  py virtual_account.py init --cash 10000")
    return json.loads(p.read_text(encoding="utf-8"))


def save_state(state, data_dir=None):
    p = state_path(data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        shutil.copy2(p, p.with_suffix(".bak"))
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    tmp.replace(p)


def new_state(cash):
    return {"created": datetime.now().strftime("%Y-%m-%d %H:%M"), "start_cash": float(cash), "cash": float(cash),
            "positions": {}, "trades": [], "snapshots": [], "last_rebalance_month": None,
            "start_date": None, "peak": float(cash), "halted": False, "halt_reason": ""}


# ---------------------------------------------------------------- prezzi

def load_prices(start):
    import time
    import yfinance as yf
    df = None
    for attempt in range(4):                       # Yahoo a volte rifiuta: riprova con pause crescenti
        try:
            df = yf.download(list(mb.UNIVERSE), start=start, interval="1d", auto_adjust=True, progress=False)
            if df is not None and not df.empty and df["Close"].notna().sum().sum() > 0:
                break
        except Exception as e:
            print(f"Tentativo {attempt + 1} fallito: {type(e).__name__}")
        time.sleep(20 * (attempt + 1))
    if df is None or df.empty:
        raise RuntimeError("nessun dato ricevuto da Yahoo: controlla la connessione")
    close = df["Close"]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.dropna(how="all").ffill(limit=5)


def idx_of(A, date_str):
    i = A.index.get_indexer([pd.Timestamp(date_str)], method="pad")[0]
    if i < 0:
        raise RuntimeError(f"data {date_str} fuori dai dati scaricati")
    return int(i)


def pos_value(A, tk, pos, i_now):
    ref = idx_of(A, pos["ref_date"])
    a_now, a_ref = A[tk].iloc[i_now], A[tk].iloc[ref]
    if np.isnan(a_now) or np.isnan(a_ref) or a_ref == 0:
        return pos["value"]
    return pos["value"] * a_now / a_ref        # rendimento totale (dividendi inclusi) fra le due date


def equity_at(state, A, i):
    vals = {tk: pos_value(A, tk, p, i) for tk, p in state["positions"].items()}
    return state["cash"] + sum(vals.values()), vals


# ---------------------------------------------------------------- operazioni simulate

def _trade(state, date, tk, delta, cost, why):
    state["cash"] -= delta + cost
    state["trades"].append({"date": date, "ticker": tk, "side": "COMPRA" if delta > 0 else "VENDI",
                            "amount": round(abs(delta), 2), "cost": round(cost, 2), "why": why})


def liquidate(state, A, i, reason):
    date = A.index[i].strftime("%Y-%m-%d")
    _, vals = equity_at(state, A, i)
    for tk, v in vals.items():
        _trade(state, date, tk, -v, mb.COST * v, reason)
    state["positions"] = {}
    state["halted"] = True
    state["halt_reason"] = f"{date}: {reason}"


def rebalance(state, A, e, s, month_tag):
    """Segnale alla chiusura s, esecuzione alla chiusura e (il giorno dopo): niente look-ahead."""
    date = A.index[e].strftime("%Y-%m-%d")
    equity, vals = equity_at(state, A, e)
    state["peak"] = max(state["peak"], equity)
    if equity < state["peak"] * (1 - MAX_DD_STOP):
        liquidate(state, A, e, f"blocco: perdita oltre {MAX_DD_STOP:.0%} dal massimo")
        state["last_rebalance_month"] = month_tag
        return
    S = (A.iloc[s - mb.SKIP] / A.iloc[s - mb.LOOKBACK] - 1).dropna().sort_values(ascending=False)
    chosen = list(S.index[:mb.TOP_N])
    target = {tk: equity * INVEST_FRAC / len(chosen) for tk in chosen}
    for tk in sorted(set(vals) | set(target)):
        delta = target.get(tk, 0.0) - vals.get(tk, 0.0)
        if abs(delta) < MIN_TRADE:
            continue
        _trade(state, date, tk, delta, mb.COST * abs(delta),
               "rotazione mensile: " + (f"entra in top-{mb.TOP_N}" if delta > 0 else "esce dai migliori"))
        if tk in target:
            state["positions"][tk] = {"value": target[tk], "ref_date": date}
        else:
            state["positions"].pop(tk, None)
    state["last_rebalance_month"] = month_tag


def months_to_do(state, A):
    last_tag = A.index[-1].strftime("%Y-%m")
    if state["last_rebalance_month"] is None:
        return [last_tag]
    p = pd.Period(state["last_rebalance_month"], "M") + 1
    out = []
    while p <= pd.Period(last_tag, "M"):
        out.append(str(p))
        p += 1
    return out


def update(state, A):
    """Aggiorna il conto: ribilancia i mesi mancanti, controlla il blocco, registra lo stato di oggi."""
    if len(A) < mb.LOOKBACK + mb.SKIP + 5:
        raise RuntimeError("dati insufficienti per calcolare la regola")
    notes = []
    last = len(A) - 1
    if not state["halted"]:
        for tag in months_to_do(state, A):
            if state["start_date"] is None:                      # primo giorno: segnale ieri, esecuzione oggi
                e, s = last, last - 1
                state["start_date"] = A.index[e].strftime("%Y-%m-%d")
            else:
                pos_in_month = np.where(A.index.to_period("M") == pd.Period(tag, "M"))[0]
                if len(pos_in_month) == 0:
                    continue
                e = int(pos_in_month[0])
                s = e - 1
            n_before = len(state["trades"])
            rebalance(state, A, e, s, tag)
            notes.append(f"Ribilanciamento di {tag}: {len(state['trades']) - n_before} operazioni "
                         f"(eseguite alla chiusura del {A.index[e].date()})")
            if state["halted"]:
                break

    equity, _ = equity_at(state, A, last)
    state["peak"] = max(state["peak"], equity)
    if not state["halted"] and equity < state["peak"] * (1 - MAX_DD_STOP):
        liquidate(state, A, last, f"blocco: perdita oltre {MAX_DD_STOP:.0%} dal massimo")
        notes.append("BLOCCO ATTIVATO: tutto venduto. " + state["halt_reason"])
        equity, _ = equity_at(state, A, last)

    if state["start_date"] is not None:
        si = idx_of(A, state["start_date"])
        spy = state["start_cash"] * A["SPY"].iloc[last] / A["SPY"].iloc[si]
        rel = (A.iloc[last] / A.iloc[si]).dropna()
        ew = state["start_cash"] * float(rel.mean())
        snap = {"date": A.index[last].strftime("%Y-%m-%d"), "equity": round(equity, 2),
                "spy": round(float(spy), 2), "ew": round(ew, 2)}
        if state["snapshots"] and state["snapshots"][-1]["date"] == snap["date"]:
            state["snapshots"][-1] = snap
        else:
            state["snapshots"].append(snap)
    return notes


# ---------------------------------------------------------------- resoconto

def build_report(state, A=None):
    L = []
    sc = state["start_cash"]
    L.append("CONTO DEMO SIMULATO (non sono soldi veri)")
    if state["start_date"] is None:
        L.append(f"Saldo iniziale {sc:,.2f} USD. Nessuna operazione ancora: lancia  py virtual_account.py update")
        return "\n".join(L)
    snap = state["snapshots"][-1]
    eq, spy, ew = snap["equity"], snap["spy"], snap["ew"]
    days = (pd.Timestamp(snap["date"]) - pd.Timestamp(state["start_date"])).days
    L.append(f"Dal {state['start_date']} al {snap['date']} ({days} giorni)")
    L.append(f"Saldo iniziale:   {sc:>12,.2f} USD")
    L.append(f"Saldo attuale:    {eq:>12,.2f} USD   ({eq / sc - 1:+.2%})")
    L.append(f"Riferimento S&P 500:          {spy:>12,.2f} USD   ({spy / sc - 1:+.2%})")
    L.append(f"Riferimento paniere 29 ETF:   {ew:>12,.2f} USD   ({ew / sc - 1:+.2%})")
    L.append(f"Differenza rispetto all'S&P 500: {eq / sc - spy / sc:+.2%}")
    eqs = np.array([x["equity"] for x in state["snapshots"]])
    if len(eqs) > 1:
        dd = float(((eqs - np.maximum.accumulate(eqs)) / np.maximum.accumulate(eqs)).min())
        L.append(f"Peggior calo dal massimo (sui punti registrati): {dd:.1%}  | blocco automatico a -{MAX_DD_STOP:.0%}")
    if state["halted"]:
        L.append(f"\n*** CONTO FERMO *** {state['halt_reason']}  (per riattivarlo: py virtual_account.py resume)")
    L.append("\nPosizioni:")
    if not state["positions"]:
        L.append("  tutto in contanti: " + f"{state['cash']:,.2f} USD")
    else:
        tot = eq
        for tk, p in sorted(state["positions"].items()):
            v = p["value"]
            if A is not None:
                v = pos_value(A, tk, p, len(A) - 1)
            L.append(f"  {tk:5s} {mb.UNIVERSE.get(tk, ''):26s} {v:>11,.2f} USD  ({v / tot:5.1%})")
        L.append(f"  {'contanti':32s} {state['cash']:>11,.2f} USD  ({state['cash'] / tot:5.1%})")
    L.append(f"\nUltimo ribilanciamento: {state['last_rebalance_month']} | operazioni totali: {len(state['trades'])}")
    for t in state["trades"][-8:]:
        L.append(f"  {t['date']}  {t['side']:6s} {t['ticker']:5s} {t['amount']:>10,.2f} USD  (costo {t['cost']:.2f})")
    if days < 90:
        L.append("\nNota: con meno di 3 mesi di vita il risultato e' quasi tutto rumore: non dice se la regola funziona.")
    return "\n".join(L)


def send_telegram(text):
    import requests
    token, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("(Telegram non configurato: resoconto non inviato)")
        return
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data={"chat_id": chat, "text": text[:4000]}, timeout=20)
    print("Telegram:", "inviato" if r.ok else f"errore {r.status_code}")


# ---------------------------------------------------------------- main

def fetch_start(state):
    s = datetime.now() - timedelta(days=1100)
    if state.get("start_date"):
        s = min(s, pd.Timestamp(state["start_date"]).to_pydatetime() - timedelta(days=15))
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
    a = p.parse_args()

    if a.mode == "init":
        if state_path().exists() and not a.force:
            sys.exit("Il conto esiste gia'. Per ricominciare da zero (cancella lo storico): aggiungi --force")
        save_state(new_state(a.cash))
        print(f"Conto demo creato con {a.cash:,.2f} USD virtuali in {state_path()}")
        print("Ora lancia:  py virtual_account.py update")
        sys.exit(0)

    state = load_state()
    if a.mode == "resume":
        state["halted"] = False
        state["halt_reason"] = ""
        state["peak"] = state["snapshots"][-1]["equity"] if state["snapshots"] else state["cash"]
        save_state(state)
        print("Conto riattivato. Il massimo di riferimento riparte dal saldo attuale.")
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
    report_text = build_report(work, A)
    print("\n" + report_text)
    if a.dry:
        print("\n(SIMULAZIONE: non e' stato salvato nulla)")
    else:
        save_state(work)
        if a.notify and (notes or datetime.now().weekday() == 0):    # solo se succede qualcosa, o il lunedi'
            send_telegram(("\n".join(notes) + "\n\n" if notes else "") + report_text)
