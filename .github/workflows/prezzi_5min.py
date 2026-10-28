"""
Raccolta dei prezzi a 5 MINUTI (apertura, massimo, minimo, chiusura, volume) di titoli, indici, materie prime e crypto.
Serve a costruire, in alcuni mesi, uno storico vero su cui provare la regola "in tempo reale".
Ogni esecuzione scarica gli ultimi 5 giorni e aggiunge solo le candele nuove a data/intraday/AAAA-MM.csv.gz
(senza duplicati). Nessun denaro, nessun ordine.
"""
import gzip
import io
import os
import sys
import time
from pathlib import Path

import pandas as pd

DATA_DIR = Path(os.environ.get("VA_DATA_DIR") or "data") / "intraday"

TICKERS = [
    # materie prime (futures)
    "NG=F", "CL=F", "BZ=F", "GC=F", "SI=F", "HG=F",
    # indici
    "^GSPC", "^NDX", "^GDAXI", "FTSEMIB.MI",
    # cambi (forex)
    "EURUSD=X", "EURNZD=X", "GBPUSD=X", "USDJPY=X", "EURGBP=X", "EURJPY=X", "AUDUSD=X", "USDCHF=X",
    "USDCAD=X", "NZDUSD=X", "EURCHF=X", "GBPJPY=X",
    # crypto (24 ore su 24)
        # cambi (forex)
    "EURUSD=X", "EURNZD=X", "GBPUSD=X", "USDJPY=X", "EURGBP=X", "EURJPY=X", "AUDUSD=X", "USDCHF=X",
    "USDCAD=X", "NZDUSD=X", "EURCHF=X", "GBPJPY=X",
    "BTC-USD", "ETH-USD",
    # azioni USA
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "XOM", "AMD", "NFLX", "BAC",
    "AVGO", "V", "MA", "UNH", "LLY", "WMT", "COST", "HD", "PG", "JNJ", "KO", "PEP", "MRK", "ORCL", "CRM",
    "CVX", "INTC", "BA", "DIS", "PFE", "PLTR", "COIN", "UBER",
    # azioni Europa
    "ENEL.MI", "ISP.MI", "RACE.MI", "ASML.AS", "SAP.DE", "MC.PA",
    "SIE.DE", "ALV.DE", "OR.PA", "TTE.PA", "AIR.PA", "SAN.PA", "NESN.SW", "NOVN.SW", "ENI.MI", "UCG.MI",
]
COLS = ["ticker", "ts_utc", "open", "high", "low", "close", "volume"]


def fetch(tickers, period="5d"):
    import yfinance as yf
    df = None
    for attempt in range(3):
        try:
            df = yf.download(tickers, period=period, interval="5m", auto_adjust=False, progress=False,
                             group_by="ticker", threads=True)
            if df is not None and not df.empty:
                break
        except Exception as e:
            print(f"tentativo {attempt + 1} fallito: {type(e).__name__}")
        time.sleep(15 * (attempt + 1))
    return df


def to_rows(df, tickers):
    out = []
    for t in tickers:
        try:
            x = df[t] if len(tickers) > 1 else df
            x = x.dropna(subset=["Close"]).copy()
        except Exception:
            continue
        if x.empty:
            continue
        idx = pd.to_datetime(x.index)
        idx = idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")
        out.append(pd.DataFrame({"ticker": t, "ts_utc": idx.strftime("%Y-%m-%d %H:%M"),
                                 "open": x["Open"].values, "high": x["High"].values, "low": x["Low"].values,
                                 "close": x["Close"].values, "volume": x["Volume"].values}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=COLS)


def save(rows):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    added = 0
    rows = rows.copy()
    rows["month"] = rows["ts_utc"].str[:7]
    for month, part in rows.groupby("month"):
        p = DATA_DIR / f"{month}.csv.gz"
        part = part[COLS]
        if p.exists():
            old = pd.read_csv(p)
            allr = pd.concat([old, part], ignore_index=True)
        else:
            allr = part
        before = len(old) if p.exists() else 0
        allr = allr.drop_duplicates(subset=["ticker", "ts_utc"], keep="last").sort_values(["ticker", "ts_utc"])
        buf = io.StringIO()
        allr.to_csv(buf, index=False, float_format="%.6g")
        tmp = p.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            f.write(buf.getvalue())
        tmp.replace(p)
        added += len(allr) - before
    return added


def known_tickers():
    seen = set()
    if DATA_DIR.exists():
        for p in DATA_DIR.glob("*.csv.gz"):
            try:
                seen |= set(pd.read_csv(p, usecols=["ticker"])["ticker"].unique())
            except Exception:
                pass
    return seen


def collect(fetch_fn=fetch):
    new = [t for t in TICKERS if t not in known_tickers()]
    total = 0
    if new:                                             # strumenti mai visti: si recuperano gli ultimi 60 giorni (il massimo di Yahoo)
        df = fetch_fn(new, "60d")
        if df is not None and not df.empty:
            rows = to_rows(df, new)
            total += save(rows) if len(rows) else 0
            print(f"Recupero storico 60 giorni: {rows['ticker'].nunique() if len(rows) else 0}/{len(new)} strumenti nuovi")
    df = fetch_fn(TICKERS)
    if df is None or df.empty:
        print("Nessun dato ricevuto da Yahoo")
        return total
    rows = to_rows(df, TICKERS)
    got = rows["ticker"].nunique() if len(rows) else 0
    n = save(rows) if len(rows) else 0
    print(f"Strumenti con dati: {got}/{len(TICKERS)} | candele nuove salvate: {n}")
    return total + n


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    collect()
