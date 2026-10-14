"""
Universo e regola del conto "veloce" su AZIONI SINGOLE dei mercati globali.
Valori fissati in anticipo: non vanno ritoccati dopo aver visto i risultati.

Regola (settimanale):
  - L'ultimo giorno di borsa di ogni settimana si classificano le azioni per rendimento delle ultime 4 settimane
    (20 giorni di borsa).
  - Si comprano le 5 migliori, pesi uguali (20% ciascuna), long-only; si entra alla chiusura del giorno dopo il
    segnale (niente look-ahead). Costo 5 punti base per lato.
  - Riferimenti: S&P 500 (SPY) e paniere equipesato di tutte le azioni dell'universo.

I prezzi sono nella valuta locale: il rendimento e' calcolato in valuta locale e il cambio non e' simulato.
Solo grandi societa' liquide. ATTENZIONE: la lista contiene aziende grandi OGGI: nei test sul passato questo
gonfia i risultati (survivorship bias). Un test positivo sul passato quindi vale poco; la prova vera e' il conto in tempo reale.
"""

BENCH = "SPY"
LOOKBACK = 20            # 4 settimane di borsa
TOP_N = 5

UNIVERSE = {
    # USA
    "AAPL": "USA", "MSFT": "USA", "NVDA": "USA", "AMZN": "USA", "GOOGL": "USA", "META": "USA", "TSLA": "USA",
    "AVGO": "USA", "JPM": "USA", "V": "USA", "MA": "USA", "UNH": "USA", "XOM": "USA", "LLY": "USA",
    "WMT": "USA", "COST": "USA", "HD": "USA", "PG": "USA", "JNJ": "USA", "ABBV": "USA", "KO": "USA",
    "PEP": "USA", "MRK": "USA", "ORCL": "USA", "NFLX": "USA", "CRM": "USA", "AMD": "USA", "BAC": "USA",
    "CVX": "USA", "TMO": "USA",
    # Europa
    "ASML.AS": "Olanda", "SAP.DE": "Germania", "SIE.DE": "Germania", "ALV.DE": "Germania",
    "MC.PA": "Francia", "OR.PA": "Francia", "TTE.PA": "Francia", "SAN.PA": "Francia", "AIR.PA": "Francia",
    "NESN.SW": "Svizzera", "NOVN.SW": "Svizzera", "ROG.SW": "Svizzera",
    "AZN.L": "Regno Unito", "SHEL.L": "Regno Unito", "HSBA.L": "Regno Unito", "ULVR.L": "Regno Unito",
    "ENEL.MI": "Italia", "ISP.MI": "Italia", "RACE.MI": "Italia", "IBE.MC": "Spagna",
    # Giappone
    "7203.T": "Giappone", "6758.T": "Giappone", "6861.T": "Giappone", "8306.T": "Giappone",
    "9984.T": "Giappone", "9983.T": "Giappone", "6098.T": "Giappone", "8035.T": "Giappone",
    # Hong Kong / Cina
    "0700.HK": "Hong Kong", "9988.HK": "Hong Kong", "1299.HK": "Hong Kong", "0939.HK": "Hong Kong",
    "3690.HK": "Hong Kong", "1810.HK": "Hong Kong",
    # Australia
    "BHP.AX": "Australia", "CBA.AX": "Australia", "CSL.AX": "Australia", "NAB.AX": "Australia",
    "WES.AX": "Australia", "RIO.AX": "Australia",
    # Canada
    "RY.TO": "Canada", "SHOP.TO": "Canada", "TD.TO": "Canada", "ENB.TO": "Canada", "CNQ.TO": "Canada",
    "CP.TO": "Canada",
    # Asia
    "005930.KS": "Corea del Sud", "2330.TW": "Taiwan", "INFY": "India (ADR)",
}
