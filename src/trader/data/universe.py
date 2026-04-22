from __future__ import annotations

# ---------------------------------------------------------------------------
# NIFTY 50 constituents (reference date: 2024-01-01)
# ---------------------------------------------------------------------------

NIFTY_50: list[str] = [
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "APOLLOHOSP.NS",
    "ASIANPAINT.NS",
    "AXISBANK.NS",
    "BAJAJ-AUTO.NS",
    "BAJAJFINSV.NS",
    "BAJFINANCE.NS",
    "BHARTIARTL.NS",
    "BPCL.NS",
    "BRITANNIA.NS",
    "CIPLA.NS",
    "COALINDIA.NS",
    "DIVISLAB.NS",
    "DRREDDY.NS",
    "EICHERMOT.NS",
    "GRASIM.NS",
    "HCLTECH.NS",
    "HDFCBANK.NS",
    "HDFCLIFE.NS",
    "HEROMOTOCO.NS",
    "HINDALCO.NS",
    "HINDUNILVR.NS",
    "ICICIBANK.NS",
    "INDUSINDBK.NS",
    "INFY.NS",
    "ITC.NS",
    "JSWSTEEL.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "M&M.NS",
    "MARUTI.NS",
    "NESTLEIND.NS",
    "NTPC.NS",
    "ONGC.NS",
    "POWERGRID.NS",
    "RELIANCE.NS",
    "SBILIFE.NS",
    "SBIN.NS",
    "SUNPHARMA.NS",
    "TATAMOTORS.NS",
    "TATASTEEL.NS",
    "TCS.NS",
    "TECHM.NS",
    "TITAN.NS",
    "ULTRACEMCO.NS",
    "UPL.NS",
    "WIPRO.NS",
    "SHREECEM.NS",
    "LTIM.NS",
]

# ---------------------------------------------------------------------------
# Sectoral extension lists (top ~20 per sector, including NIFTY 50 overlap)
# Overlap is deduplicated at runtime by all_tickers().
# ---------------------------------------------------------------------------

SECTOR_MAP: dict[str, list[str]] = {
    "banking": [
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "KOTAKBANK.NS",
        "AXISBANK.NS",
        "SBIN.NS",
        "INDUSINDBK.NS",
        "BANKBARODA.NS",
        "PNB.NS",
        "FEDERALBNK.NS",
        "IDFCFIRSTB.NS",
        "AUBANK.NS",
        "BANDHANBNK.NS",
        "CANARABANK.NS",
        "RBLBANK.NS",
        "YESBANK.NS",
        "BAJFINANCE.NS",
        "BAJAJFINSV.NS",
        "HDFCLIFE.NS",
        "SBILIFE.NS",
        "ICICIGI.NS",
    ],
    "it": [
        "TCS.NS",
        "INFY.NS",
        "HCLTECH.NS",
        "WIPRO.NS",
        "TECHM.NS",
        "LTIM.NS",
        "MPHASIS.NS",
        "PERSISTENT.NS",
        "COFORGE.NS",
        "LTTS.NS",
        "KPITTECH.NS",
        "TATAELXSI.NS",
        "MASTEK.NS",
        "CYIENT.NS",
        "BIRLASOFT.NS",
        "SONATSOFTW.NS",
        "OFSS.NS",
        "HEXAWARE.NS",
        "NIIT.NS",
        "ZENSAR.NS",
    ],
    "pharma": [
        "SUNPHARMA.NS",
        "DRREDDY.NS",
        "CIPLA.NS",
        "DIVISLAB.NS",
        "APOLLOHOSP.NS",
        "LUPIN.NS",
        "TORNTPHARM.NS",
        "ALKEM.NS",
        "ABBOTINDIA.NS",
        "IPCALAB.NS",
        "LAURUSLABS.NS",
        "GRANULES.NS",
        "AUROPHARMA.NS",
        "PFIZER.NS",
        "GLENMARK.NS",
        "NATCOPHARM.NS",
        "ZYDUSLIFE.NS",
        "MANKIND.NS",
        "MAXHEALTH.NS",
        "FORTIS.NS",
    ],
    "auto": [
        "MARUTI.NS",
        "M&M.NS",
        "TATAMOTORS.NS",
        "BAJAJ-AUTO.NS",
        "HEROMOTOCO.NS",
        "EICHERMOT.NS",
        "TVSMOTOR.NS",
        "ASHOKLEY.NS",
        "MOTHERSON.NS",
        "BOSCHLTD.NS",
        "BALKRISIND.NS",
        "APOLLOTYRE.NS",
        "MRF.NS",
        "TIINDIA.NS",
        "BHARATFORG.NS",
        "ENDURANCE.NS",
        "AMARARAJA.NS",
        "EXIDEIND.NS",
        "OLECTRA.NS",
        "SUNDARMFIN.NS",
    ],
    "oil_gas_power": [
        "RELIANCE.NS",
        "ONGC.NS",
        "BPCL.NS",
        "NTPC.NS",
        "POWERGRID.NS",
        "COALINDIA.NS",
        "IOC.NS",
        "GAIL.NS",
        "HINDPETRO.NS",
        "ADANIGREEN.NS",
        "ADANIPOWER.NS",
        "TATAPOWER.NS",
        "CESC.NS",
        "TORNTPOWER.NS",
        "NHPC.NS",
        "SJVN.NS",
        "RECLTD.NS",
        "PFC.NS",
        "PETRONET.NS",
        "OIL.NS",
    ],
    "defence": [
        "LT.NS",
        "HAL.NS",
        "BEL.NS",
        "BHEL.NS",
        "COCHINSHIP.NS",
        "MAZDOCK.NS",
        "GRSE.NS",
        "BEML.NS",
        "ABB.NS",
        "SIEMENS.NS",
        "THERMAX.NS",
        "CUMMINSIND.NS",
        "AIAENG.NS",
        "CGPOWER.NS",
        "ENGINERSIN.NS",
        "RVNL.NS",
        "IRCON.NS",
        "NBCC.NS",
        "RAILVIKAS.NS",
        "TITAGARH.NS",
    ],
    "fmcg": [
        "HINDUNILVR.NS",
        "ITC.NS",
        "NESTLEIND.NS",
        "BRITANNIA.NS",
        "DABUR.NS",
        "MARICO.NS",
        "GODREJCP.NS",
        "COLPAL.NS",
        "EMAMILTD.NS",
        "TATACONSUM.NS",
        "RADICO.NS",
        "VARUNBEV.NS",
        "UBL.NS",
        "MCDOWELL-N.NS",
        "GILLETTE.NS",
        "PGHH.NS",
        "ZOMATO.NS",
        "NYKAA.NS",
        "DEVYANI.NS",
        "WESTLIFE.NS",
    ],
    "metals": [
        "JSWSTEEL.NS",
        "TATASTEEL.NS",
        "HINDALCO.NS",
        "COALINDIA.NS",
        "SAIL.NS",
        "VEDL.NS",
        "NMDC.NS",
        "NATIONALUM.NS",
        "HINDZINC.NS",
        "MOIL.NS",
        "APLAPOLLO.NS",
        "RATNAMANI.NS",
        "JINDALSTEL.NS",
        "JSPL.NS",
        "WELSPUNLIV.NS",
        "GPPL.NS",
        "ADANIENT.NS",
        "STLTECH.NS",
        "SHYAMMETL.NS",
        "HGINFRA.NS",
    ],
}

# Numeric ID assigned to each sector (0 = unknown/unclassified)
SECTOR_IDS: dict[str, int] = {
    "banking": 1,
    "it": 2,
    "pharma": 3,
    "auto": 4,
    "oil_gas_power": 5,
    "defence": 6,
    "fmcg": 7,
    "metals": 8,
}

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def all_tickers() -> list[str]:
    """Return deduped union of NIFTY 50 + all sectoral lists, NIFTY 50 first."""
    seen: set[str] = set()
    result: list[str] = []
    for t in NIFTY_50:
        if t not in seen:
            seen.add(t)
            result.append(t)
    for tickers in SECTOR_MAP.values():
        for t in tickers:
            if t not in seen:
                seen.add(t)
                result.append(t)
    return result


def sector_of(ticker: str) -> str | None:
    """Return the primary sector name for a ticker, or None if unclassified."""
    for sector, tickers in SECTOR_MAP.items():
        if ticker in tickers:
            return sector
    return None


def sector_id_of(ticker: str) -> int:
    """Return the numeric sector ID for a ticker (0 = unclassified)."""
    sector = sector_of(ticker)
    if sector is None:
        return 0
    return SECTOR_IDS.get(sector, 0)
