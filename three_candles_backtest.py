"""
Tre candele consecutive, seguendo la direzione - backtest con regole FISSATE IN ANTICIPO.

Regole (decise prima di guardare i dati, NON da ottimizzare):
  - 3 candele orarie consecutive verdi (chiusura > apertura) -> compra
  - 3 candele orarie consecutive rosse (chiusura < apertura) -> vendi
  - stop 1.5 ATR, target 3 ATR, spread, rischio 1%, split 70/30: identici a forex_bot.py
    (si riusa la sua funzione backtest(): ingresso all'apertura della candela successiva)

Criterio di successo (fissato in anticipo): profit factor >= 1.3 nel periodo di test (30%)
sulla maggioranza delle coppie, e profit factor > 1 anche nel periodo di sviluppo (70%).

Uso:
    py three_candles_backtest.py
    py three_candles_backtest.py --synthetic     (solo per testare il codice, dati finti)

E' la quarta idea testata sugli stessi dati: un verdetto positivo vale meno di quanto sembri.
Non e' un consiglio finanziario.
"""

import argparse

import numpy as np

from forex_bot import EMA_TREND, PAIRS, backtest, load_data, prepare, stats

N_CANDLES = 3
MIN_PF_OOS = 1.3


def three_candles_signal(df):
    """+1 dopo N candele verdi consecutive, -1 dopo N rosse, 0 altrimenti."""
    green = (df["Close"] > df["Open"]).astype(int)
    red = (df["Close"] < df["Open"]).astype(int)
    up = green.rolling(N_CANDLES).sum() == N_CANDLES
    dn = red.rolling(N_CANDLES).sum() == N_CANDLES
    sig = np.where(up, 1, np.where(dn, -1, 0))
    sig[:EMA_TREND] = 0                     # warm-up dell'ATR, come in forex_bot.prepare()
    return sig


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--interval", default="1h")
    p.add_argument("--period", default="730d")
    p.add_argument("--synthetic", action="store_true", help="dati finti, solo per testare il codice")
    a = p.parse_args()

    print(f"Tre candele consecutive (segue la direzione) | interval={a.interval} period={a.period}\n")
    passed, total = 0, 0
    for ticker, spread in PAIRS.items():
        df = prepare(load_data(ticker, a.interval, a.period, a.synthetic))   # serve per l'ATR
        df["signal"] = three_candles_signal(df)
        cut = int(len(df) * 0.7)
        print(f"== {ticker} ==")
        res = {}
        for label, part in (("sviluppo 70%", df.iloc[:cut]), ("test 30%", df.iloc[cut:])):
            res[label] = stats(*backtest(part, spread))
            print(f"  {label:14s}", res[label])
        pf_is = res["sviluppo 70%"].get("profit factor", 0)
        pf_oos = res["test 30%"].get("profit factor", 0)
        ok = pf_oos >= MIN_PF_OOS and pf_is > 1
        total += 1
        passed += ok
        print(f"  -> {'OK' if ok else 'NO'}\n")

    print(f"VERDETTO: criterio superato su {passed} coppie su {total}.")
    if passed > total / 2:
        print("Supera il criterio: merita il paper trading. E' pero' la quarta idea provata: cautela.")
    else:
        print("Non supera il criterio: l'idea NON passa. Niente ritocchi ai parametri.")
