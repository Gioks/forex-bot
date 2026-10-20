"""Resoconto serale di tutti i conti demo (simulati). Invia un messaggio Telegram pronto da copiare su Instagram."""
import json
import os
import sys
from pathlib import Path

import virtual_account as va

D = Path(os.environ.get("VA_DATA_DIR") or "data")
ACCOUNTS = [("stock_account.json", "Veloce base"), ("stock_account_trend.json", "Veloce trend"),
            ("stock_account_riskw.json", "Veloce rischio"), ("stock_account_nostop.json", "Veloce senza stop"),
            ("candle_account.json", "Tre candele"), ("virtual_account.json", "ETF mensile")]


def last(path):
    p = D / path
    if not p.exists():
        return None
    s = json.loads(p.read_text(encoding="utf-8"))
    sn = s.get("snapshots") or []
    if not sn:
        return None
    start = s.get("start_cash") or s.get("initial_cash") or sn[0]["equity"]
    prev = sn[-2]["equity"] if len(sn) > 1 else sn[-1]["equity"]
    return sn[-1]["date"], sn[-1]["equity"], start, prev


def main():
    rows, date = [], ""
    for f, name in ACCOUNTS:
        try:
            r = last(f)
        except Exception:
            r = None
        if r:
            date = max(date, r[0])
            rows.append((name, r[1], r[1] / r[2] - 1, r[1] / r[3] - 1))
    if not rows:
        print("Nessun conto da riassumere")
        return
    L = [f"Resoconto di fine giornata {date} - CONTI DEMO (soldi finti)", ""]
    for n, eq, tot, day in rows:
        L.append(f"{n}: {eq:,.0f} $  oggi {day:+.2%}  da inizio {tot:+.2%}")
    L += ["", "Simulazione a scopo di studio, non e' un consiglio finanziario. #borsa #trading #demo"]
    txt = "\n".join(L)
    print(txt)
    va.send_telegram(txt)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    main()
