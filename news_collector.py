"""
Raccolta in tempo reale di titoli di notizie finanziarie e prezzi. NON opera, NON invia ordini.

Scopo: costruire uno storico VERO di titoli con l'orario in cui li abbiamo visti, per poter
testare fra qualche mese (con gli stessi criteri rigorosi dei backtest) se le notizie anticipano
i mercati. Gli storici gratuiti di titoli non esistono, quindi l'unico modo e' raccoglierli.

Cosa salva (in una cartella FUORI da OneDrive, di default C:\\Users\\<tu>\\forex_data):
  - headlines: fonte, titolo, riassunto, link, orario di pubblicazione e, soprattutto,
               fetched_utc = quando il nostro programma l'ha visto. Per i test useremo SEMPRE
               l'orario piu' tardivo fra i due, per non barare con il futuro.
  - prices:    chiusure giornaliere di 16 strumenti globali (si aggiornano a ogni esecuzione).
  - runs:      un registro di ogni esecuzione, per sapere dove ci sono buchi (PC spento, ecc.).

Uso:
    py news_collector.py                 raccoglie una volta e si ferma
    py news_collector.py loop            raccoglie ogni 60 minuti finche' la finestra resta aperta
    py news_collector.py status          mostra quanto e' stato raccolto e gli eventuali buchi
    py news_collector.py export --out titoli.csv    esporta i titoli in CSV

Non e' un consiglio finanziario.
"""

import argparse
import csv
import hashlib
import html
import re
import shutil
import sqlite3
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

# ---------------------------------------------------------------- configurazione

# Verificati da una lettura di prova: marketwatch_top, fed_press, ecb_press.
# Gli altri sono feed pubblici noti: se uno non risponde, il programma lo segnala e va avanti.
FEEDS = {
    "marketwatch_top": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "marketwatch_pulse": "https://feeds.content.dowjones.io/public/rss/mw_marketpulse",
    "wsj_markets": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "wsj_world": "https://feeds.a.dj.com/rss/RSSWorldNews.xml",
    "fed_press": "https://www.federalreserve.gov/feeds/press_all.xml",
    "ecb_press": "https://www.ecb.europa.eu/rss/press.html",
    "bbc_business": "http://feeds.bbci.co.uk/news/business/rss.xml",
    "bbc_world": "http://feeds.bbci.co.uk/news/world/rss.xml",
    "cnbc_top": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "guardian_business": "https://www.theguardian.com/uk/business/rss",
    "nyt_business": "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
}

TICKERS = [
    "^GSPC", "^NDX", "^GDAXI", "^FTSE", "^N225",
    "TLT", "IEF", "GLD", "SLV", "DBC", "USO",
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X",
]

DEFAULT_DIR = Path.home() / "forex_data"
USER_AGENT = "Mozilla/5.0 (compatible; personal-news-logger/1.0)"
TARGET_DAYS = 90          # obiettivo minimo di giorni di raccolta prima del test
GAP_HOURS = 6             # buco segnalato se fra due esecuzioni passano piu' di 6 ore
KEEP_BACKUPS = 14

SCHEMA = """
CREATE TABLE IF NOT EXISTS headlines(
    id INTEGER PRIMARY KEY, hash TEXT UNIQUE, source TEXT, title TEXT, summary TEXT, url TEXT,
    published_utc TEXT, fetched_utc TEXT);
CREATE TABLE IF NOT EXISTS prices(
    date TEXT, ticker TEXT, close REAL, PRIMARY KEY(date, ticker));
CREATE TABLE IF NOT EXISTS runs(
    id INTEGER PRIMARY KEY, run_utc TEXT, feeds_ok INTEGER, feeds_failed INTEGER,
    new_headlines INTEGER, price_rows INTEGER, failed_names TEXT);
"""


# ---------------------------------------------------------------- utilita'

def log(data_dir, msg):
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line)
    try:
        with open(Path(data_dir) / "collector.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clean(text):
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _parse_date(s):
    if not s or not s.strip():
        return None
    s = s.strip()
    try:
        d = parsedate_to_datetime(s)
    except Exception:
        try:
            d = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:
            return None
    if d is None:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.astimezone(timezone.utc).isoformat(timespec="seconds")


def _local(tag):
    return tag.rsplit("}", 1)[-1]


# ---------------------------------------------------------------- lettura dei feed

def fetch(url):
    import requests
    r = requests.get(url, headers={"User-Agent": USER_AGENT,
                                   "Accept": "application/rss+xml, application/xml, text/xml, */*"}, timeout=25)
    r.raise_for_status()
    return r.content


def parse_feed(content):
    """Legge un feed RSS 2.0 o Atom e restituisce una lista di dizionari."""
    root = ET.fromstring(content)
    out = []
    for el in root.iter():
        if _local(el.tag) not in ("item", "entry"):
            continue
        d, dates = {}, {}
        for ch in el:
            n = _local(ch.tag)
            if n == "title":
                d["title"] = _clean(ch.text)
            elif n in ("description", "summary") and "summary" not in d:
                d["summary"] = _clean(ch.text)
            elif n == "link" and "url" not in d:
                href = ch.attrib.get("href")
                if href and ch.attrib.get("rel", "alternate") == "alternate":
                    d["url"] = href.strip()
                elif not href and (ch.text or "").strip():
                    d["url"] = ch.text.strip()
            elif n in ("pubDate", "published", "date", "updated"):
                dates[n] = ch.text
        for key in ("published", "pubDate", "date", "updated"):
            if dates.get(key):
                d["published_utc"] = _parse_date(dates[key])
                break
        if d.get("title"):
            out.append(d)
    return out


# ---------------------------------------------------------------- prezzi

def fetch_prices():
    import yfinance as yf
    df = yf.download(TICKERS, period="10d", interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError("nessun dato di prezzo ricevuto")
    close = df["Close"]
    rows = []
    for tk in close.columns:
        for dt, val in close[tk].dropna().items():
            rows.append((dt.strftime("%Y-%m-%d"), str(tk), float(val)))
    return rows


# ---------------------------------------------------------------- database

def open_db(data_dir):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    db = data_dir / "news.db"
    _backup(db)
    con = sqlite3.connect(db, timeout=30)
    con.executescript(SCHEMA)
    return con


def _backup(db):
    if not db.exists():
        return
    bdir = db.parent / "backups"
    bdir.mkdir(exist_ok=True)
    target = bdir / f"news_{datetime.now().strftime('%Y%m%d')}.db"
    if not target.exists():
        try:
            shutil.copy2(db, target)
        except OSError:
            return
    for old in sorted(bdir.glob("news_*.db"))[:-KEEP_BACKUPS]:
        try:
            old.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------- raccolta

def collect(data_dir, fetch_fn=fetch, price_fn=fetch_prices):
    con = open_db(data_dir)
    run_ts = now_utc()
    ok, failed, new_total = 0, [], 0

    for name, url in FEEDS.items():
        try:
            items = parse_feed(fetch_fn(url))
            new = 0
            for it in items:
                h = hashlib.sha1(f"{name}|{it['title'].lower()}".encode("utf-8")).hexdigest()
                cur = con.execute(
                    "INSERT OR IGNORE INTO headlines(hash, source, title, summary, url, published_utc, fetched_utc) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (h, name, it["title"], it.get("summary", ""), it.get("url", ""),
                     it.get("published_utc"), run_ts))
                new += cur.rowcount
            con.commit()
            ok += 1
            new_total += new
            log(data_dir, f"  {name:18s} {len(items):3d} titoli, {new:3d} nuovi")
        except Exception as e:
            failed.append(name)
            log(data_dir, f"  {name:18s} NON disponibile ({type(e).__name__}: {str(e)[:80]})")

    price_rows = 0
    try:
        rows = price_fn()
        con.executemany("INSERT OR REPLACE INTO prices(date, ticker, close) VALUES (?,?,?)", rows)
        price_rows = len(rows)
        con.commit()
    except Exception as e:
        log(data_dir, f"  prezzi: NON disponibili ({type(e).__name__}: {str(e)[:80]})")

    con.execute("INSERT INTO runs(run_utc, feeds_ok, feeds_failed, new_headlines, price_rows, failed_names) "
                "VALUES (?,?,?,?,?,?)", (run_ts, ok, len(failed), new_total, price_rows, ",".join(failed)))
    con.commit()
    con.close()
    log(data_dir, f"Fatto: {new_total} titoli nuovi, {ok}/{len(FEEDS)} fonti raggiunte, {price_rows} prezzi aggiornati.")
    if ok == 0:
        log(data_dir, "ATTENZIONE: nessuna fonte raggiunta. Controlla la connessione a Internet.")
    return new_total, ok, failed


def run_loop(data_dir, every_min):
    print(f"Raccolta ogni {every_min} minuti. Lascia questa finestra aperta (Ctrl+C per fermare).")
    while True:
        try:
            collect(data_dir)
        except Exception as e:
            log(data_dir, f"Errore nell'esecuzione: {type(e).__name__}: {e}")
        try:
            time.sleep(every_min * 60)
        except KeyboardInterrupt:
            print("\nFermato.")
            return


# ---------------------------------------------------------------- stato ed esportazione

def status(data_dir):
    db = Path(data_dir) / "news.db"
    if not db.exists():
        print(f"Nessun dato ancora in {data_dir}. Lancia prima:  py news_collector.py")
        return
    con = sqlite3.connect(db, timeout=30)
    total = con.execute("SELECT COUNT(*) FROM headlines").fetchone()[0]
    first, last = con.execute("SELECT MIN(fetched_utc), MAX(fetched_utc) FROM headlines").fetchone()
    days = con.execute("SELECT COUNT(DISTINCT substr(fetched_utc,1,10)) FROM headlines").fetchone()[0]
    runs = con.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    print(f"Cartella dati: {data_dir}")
    print(f"Titoli raccolti: {total} | esecuzioni: {runs} | giorni con dati: {days} (obiettivo minimo {TARGET_DAYS})")
    if first:
        print(f"Primo titolo visto: {first} | ultimo: {last} (UTC)")
    print("\nPer fonte:")
    for src, n, l in con.execute("SELECT source, COUNT(*), MAX(fetched_utc) FROM headlines "
                                 "GROUP BY source ORDER BY COUNT(*) DESC"):
        print(f"  {src:18s} {n:6d} titoli | ultimo {l}")
    seen = {r[0] for r in con.execute("SELECT DISTINCT source FROM headlines")}
    missing = [n for n in FEEDS if n not in seen]
    if missing:
        print(f"\nFonti che non hanno mai dato titoli: {', '.join(missing)}")
    pr = con.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM prices").fetchone()
    print(f"\nPrezzi: {pr[0]} righe, dal {pr[1]} al {pr[2]}")

    stamps = [datetime.fromisoformat(r[0]) for r in con.execute("SELECT run_utc FROM runs ORDER BY run_utc")]
    gaps = [(a, b) for a, b in zip(stamps, stamps[1:]) if b - a > timedelta(hours=GAP_HOURS)]
    if gaps:
        print(f"\nBuchi nella raccolta (piu' di {GAP_HOURS} ore senza esecuzioni), ultimi 5:")
        for a, b in gaps[-5:]:
            print(f"  dal {a:%Y-%m-%d %H:%M} al {b:%Y-%m-%d %H:%M} UTC ({(b - a).total_seconds() / 3600:.0f} ore)")
    else:
        print("\nNessun buco significativo nella raccolta.")
    con.close()


def export(data_dir, out):
    con = sqlite3.connect(Path(data_dir) / "news.db", timeout=30)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["fetched_utc", "published_utc", "source", "title", "url"])
        w.writerows(con.execute("SELECT fetched_utc, published_utc, source, title, url "
                                "FROM headlines ORDER BY fetched_utc"))
    con.close()
    print(f"Esportato in {out}")


# ---------------------------------------------------------------- main

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", nargs="?", default="collect", choices=["collect", "loop", "status", "export"])
    p.add_argument("--data-dir", default=str(DEFAULT_DIR))
    p.add_argument("--every", type=int, default=60, help="minuti fra le raccolte in modalita' loop")
    p.add_argument("--out", default="titoli.csv")
    a = p.parse_args()

    Path(a.data_dir).mkdir(parents=True, exist_ok=True)
    if a.mode == "collect":
        log(a.data_dir, "Raccolta in corso...")
        collect(a.data_dir)
    elif a.mode == "loop":
        run_loop(a.data_dir, a.every)
    elif a.mode == "status":
        status(a.data_dir)
    else:
        export(a.data_dir, a.out)
