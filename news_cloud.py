"""
Raccolta automatica dei titoli di giornale per il server (GitHub Actions). NON opera, NON decide niente:
salva soltanto i titoli, per poterli provare fra 3-6 mesi con criteri fissati in anticipo.

I titoli vanno in data/news/AAAA-MM.jsonl (un titolo per riga, senza doppioni) e l'esito dell'ultima raccolta
in data/news/status.json. Usa le stesse fonti di news_collector.py.

Uso:  python news_cloud.py
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import news_collector as nc

DATA_DIR = Path(os.environ.get("VA_DATA_DIR") or "data") / "news"


def load_hashes(months):
    seen = set()
    for m in months:
        p = DATA_DIR / f"{m}.jsonl"
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                try:
                    seen.add(json.loads(line)["hash"])
                except Exception:
                    pass
    return seen


def prev_month(dt):
    y, m = (dt.year, dt.month - 1) if dt.month > 1 else (dt.year - 1, 12)
    return f"{y:04d}-{m:02d}"


def collect(fetch_fn=nc.fetch):
    now = datetime.now(timezone.utc)
    month = now.strftime("%Y-%m")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    seen = load_hashes([month, prev_month(now)])
    path = DATA_DIR / f"{month}.jsonl"
    fetched = now.isoformat(timespec="seconds")
    status = {"run_utc": fetched, "feeds": {}}
    total_new = 0
    with path.open("a", encoding="utf-8") as f:
        for name, url in nc.FEEDS.items():
            try:
                items = nc.parse_feed(fetch_fn(url))
                new = 0
                for it in items:
                    h = hashlib.sha1(f"{name}|{it['title'].lower()}".encode("utf-8")).hexdigest()
                    if h in seen:
                        continue
                    seen.add(h)
                    f.write(json.dumps({"hash": h, "source": name, "title": it["title"],
                                        "summary": (it.get("summary") or "")[:300], "url": it.get("url", ""),
                                        "published_utc": it.get("published_utc"), "fetched_utc": fetched},
                                       ensure_ascii=False) + "\n")
                    new += 1
                status["feeds"][name] = {"ok": True, "items": len(items), "new": new}
                total_new += new
                print(f"  {name:18s} {len(items):3d} titoli, {new:3d} nuovi")
            except Exception as e:
                status["feeds"][name] = {"ok": False, "error": f"{type(e).__name__}"}
                print(f"  {name:18s} NON disponibile ({type(e).__name__})")
    status["new_total"] = total_new
    status["feeds_ok"] = sum(1 for v in status["feeds"].values() if v["ok"])
    (DATA_DIR / "status.json").write_text(json.dumps(status, indent=1), encoding="utf-8")
    print(f"Fatto: {total_new} titoli nuovi, {status['feeds_ok']}/{len(nc.FEEDS)} fonti raggiunte.")
    return total_new, status["feeds_ok"]


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    collect()
