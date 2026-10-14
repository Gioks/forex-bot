# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Panoramica

Due script Python autonomi per strategie forex/multi-asset, con dati da Yahoo Finance (`yfinance`). Non ci sono pacchetti, test, linter né repository git. Codice, commenti e output sono in italiano: mantenere la stessa lingua.

- `forex_bot.py`: strategia su 4 coppie forex (incrocio EMA 20/50, filtro trend EMA 200 e RSI, SL/TP calcolati sull'ATR). Ha due modalità: `backtest`, e `signals`, che invia i segnali su Telegram.
- `trend_backtest.py`: breakout Donchian 55/20 su circa 16 strumenti (indici, obbligazioni, materie prime, valute), con sizing a volatilità costante e un verdetto automatico su 5 criteri di successo.
- `meanrev_backtest.py`: mean reversion di breve, solo long, su 12 indici azionari (RSI(2) < 10 sopra la SMA200, uscita sopra la SMA5, esecuzione all'apertura successiva). Importa costanti, criteri e metriche da `trend_backtest.py`, quindi modificare quel file cambia anche questo.

- `live_bot.py`: esecuzione autonoma della strategia di `meanrev_backtest.py` su conto **DEMO** IG (REST API, `IG_BASE` fissato su demo-api), lanciata ogni giorno feriale da `.github/workflows/live_bot.yml` dopo la chiusura USA. Le notizie possono solo bloccare un ingresso: prima il calendario economico (ForexFactory, eventi "High" sulla valuta dell'indice nelle 24 ore successive), poi Claude (`claude-opus-5-5`, structured output `{veto, motivo}`) sui titoli di Google News. Ogni decisione viene scritta in `live_log.jsonl`, che il workflow committa nel repo; con la variabile `LIVE_LOG` si cambia percorso. `EPICS` va compilato con `--find`. Provare con `--dry` (eventualmente con `--force <ticker>`), che non tocca IG.

Esiti sui dati reali (ottobre 2026), tutti bocciati: `forex_bot.py` ha profit factor < 1 su quasi tutte le coppie; `trend_backtest.py` supera 1 criterio su 5; `meanrev_backtest.py` 4 su 5 (bocciato sul max drawdown, 37%); `three_candles_backtest.py` (3 candele orarie consecutive, segue la direzione, riusa `backtest()` di `forex_bot.py`) 0 coppie su 4, con profit factor 0,7-0,9 e perdite del 21-57%. Ogni nuova idea testata sugli stessi dati aumenta il rischio di un falso positivo.

## Comandi

```bash
pip install -r requirements.txt

python forex_bot.py backtest [--interval 1h --period 730d] [--synthetic]
python forex_bot.py signals --dry        # stampa i segnali senza inviarli
python forex_bot.py signals              # invia su Telegram e aggiorna last_signals.json
python forex_bot.py paper [--notify]     # paper trading (soldi finti), da lanciare ogni ora
python forex_bot.py paper --report       # riepilogo del paper trading

py trend_backtest.py [--period 20y] [--synthetic]
py meanrev_backtest.py [--period 20y] [--synthetic]
py live_bot.py --dry [--force ^GSPC]     # analisi + notizie senza toccare IG
py live_bot.py --sim                     # conto virtuale da 30.000 senza IG (stato in sim_state.json)
py live_bot.py --report                  # riepilogo di sim_state.json e live_log.jsonl
```

`--synthetic` genera dati finti deterministici (seed = crc32 del ticker), senza usare la rete. Serve a verificare che il codice giri dopo una modifica e sostituisce i test.

## Architettura e invarianti

- **Nessun look-ahead**: il segnale nasce sulla chiusura della candela/giorno t e si esegue a t+1. In `forex_bot.py` l'ingresso avviene su `Open[i+1]`; in modalità `signals` si usa `df.iloc[-2]`, perché l'ultima riga è la candela ancora in formazione. In `trend_backtest.py` si usa `(pos * lev).shift(1)` e i canali Donchian sono calcolati con `.shift(1)`. Ogni modifica deve preservare questa regola.
- **Split in-sample/out-of-sample 70/30** in entrambi gli script. In `trend_backtest.py` la scala di volatilità del portafoglio si calcola solo sull'in-sample e viene riusata nello stress test con costi raddoppiati (`run(..., scale=base["scale"])`).
- **I parametri sono fissati in anticipo**: le costanti in testa a `trend_backtest.py` ("non ritoccare dopo aver visto i risultati") e i criteri di successo non vanno ottimizzati per far passare il verdetto. Se una modifica cambia parametri o criteri, segnalarlo esplicitamente.
- **Scelte prudenti** in `forex_bot.py`: se una candela tocca sia SL che TP si assume lo SL; lo spread si paga metà all'ingresso e metà all'uscita; un trade viene saltato se lo spread supera il 25% della distanza di stop; le operazioni ancora aperte a fine dati vengono ignorate.
- **Modalità `paper`** (`run_paper`): replica le regole di `backtest()` in tempo reale. Una posizione al massimo per coppia, ingresso all'`Open` della candela in formazione, SL/TP controllati sulle candele chiuse dall'apertura. Lo stato (capitale, posizioni aperte e chiuse, ultima candela vista per ticker) si salva in `paper_state.json` accanto allo script; con `--synthetic` si usa invece `paper_synthetic.json`. Se si cambiano le regole del backtest, vanno allineate anche qui.
- Gli spread in `PAIRS` e i costi in `COST_BPS` sono stime da verificare col broker.
- `yfinance` recente restituisce colonne `MultiIndex`, che entrambi i loader appiattiscono. Mantenere questa gestione.

## Configurazione

`TELEGRAM_TOKEN` e `TELEGRAM_CHAT_ID` si leggono dalle variabili d'ambiente o da `.env` (caricato da `python-dotenv`, se installato). Il template è `env.example`. `last_signals.json` (nella cwd) memorizza l'ultimo timestamp inviato per ticker, per evitare duplicati; viene scritto solo senza `--dry`.
