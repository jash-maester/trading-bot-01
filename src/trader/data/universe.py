from __future__ import annotations

# ---------------------------------------------------------------------------
# Permanently blacklisted tickers — never fetched, never traded.
# ---------------------------------------------------------------------------
# Empty as of 2026-09-04.  TATAMOTORS.NS used to sit here ("demerger; yfinance
# 404s on historical data"), but that was a symbol-resolution bug, not bad data:
# the 2025 demerger split Tata Motors into a passenger-vehicle and a commercial-
# vehicle arm, and NSE now lists the PV arm as TMPV — which is in `auto` below,
# with 5,358 clean bars from 2005 and exactly one >20% move in 21 years.
BLACKLISTED_TICKERS: frozenset[str] = frozenset()
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
    # banking — 121 names
    "banking": [
        "360ONE.NS", "AADHARHFC.NS", "AAVAS.NS", "ABCAPITAL.NS", "ABSLAMC.NS", "AIIL.NS",
        "ANANDRATHI.NS", "ANGELONE.NS", "APTUS.NS", "AUBANK.NS", "AXISBANK.NS",
        "BAJAJFINSV.NS", "BAJAJHFL.NS", "BAJAJHLDNG.NS", "BAJFINANCE.NS", "BANDHANBNK.NS",
        "BANKBARODA.NS", "BANKINDIA.NS", "BSE.NS", "CAMS.NS", "CANBK.NS", "CANFINHOME.NS",
        "CANHLIFE.NS", "CCAVENUE.NS", "CDSL.NS", "CENTRALBK.NS", "CGCL.NS", "CHOICEIN.NS",
        "CHOLAFIN.NS", "CHOLAHLDNG.NS", "CRAMC.NS", "CREDITACC.NS", "CRISIL.NS", "CSBBANK.NS",
        "CUB.NS", "DCBBANK.NS", "EDELWEISS.NS", "EQUITASBNK.NS", "FEDERALBNK.NS", "FEDFINA.NS",
        "FIVESTAR.NS", "GICRE.NS", "GODIGIT.NS", "GROWW.NS", "HDBFS.NS", "HDFCAMC.NS",
        "HDFCBANK.NS", "HDFCLIFE.NS", "HOMEFIRST.NS", "HUDCO.NS", "ICICIAMC.NS",
        "ICICIBANK.NS", "ICICIGI.NS", "ICICIPRULI.NS", "IDBI.NS", "IDFCFIRSTB.NS", "IEX.NS",
        "IFCI.NS", "IIFL.NS", "IIFLCAPS.NS", "INDIANB.NS", "INDIASHLTR.NS", "INDUSINDBK.NS",
        "IOB.NS", "IREDA.NS", "IRFC.NS", "J&KBANK.NS", "JIOFIN.NS", "JMFINANCIL.NS", "JSFB.NS",
        "KARURVYSYA.NS", "KFINTECH.NS", "KOTAKBANK.NS", "KTKBANK.NS", "LICHSGFIN.NS",
        "LICI.NS", "LTF.NS", "M&MFIN.NS", "MAHABANK.NS", "MAHSCOOTER.NS", "MANAPPURAM.NS",
        "MCX.NS", "MFSL.NS", "MOTILALOFS.NS", "MUTHOOTFIN.NS", "NAM-INDIA.NS", "NIACL.NS",
        "NIVABUPA.NS", "NUVAMA.NS", "PAYTM.NS", "PFC.NS", "PINELABS.NS", "PIRAMALFIN.NS",
        "PNB.NS", "PNBHOUSING.NS", "POLICYBZR.NS", "POONAWALLA.NS", "PRUDENT.NS", "RBLBANK.NS",
        "RECLTD.NS", "RELIGARE.NS", "SAMMAANCAP.NS", "SBFC.NS", "SBICARD.NS", "SBILIFE.NS",
        "SBIN.NS", "SHAREINDIA.NS", "SHRIRAMFIN.NS", "SOUTHBANK.NS", "STARHEALTH.NS",
        "STYL.NS", "SUNDARMFIN.NS", "TATACAP.NS", "TATAINVEST.NS", "TMB.NS", "TSFINV.NS",
        "UCOBANK.NS", "UJJIVANSFB.NS", "UNIONBANK.NS", "UTIAMC.NS", "YESBANK.NS",
    ],
    # it — 36 names
    "it": [
        "AFFLE.NS", "AURIONPRO.NS", "BBOX.NS", "BSOFT.NS", "CAPILLARY.NS", "COFORGE.NS",
        "CYIENT.NS", "DATAMATICS.NS", "HAPPSTMNDS.NS", "HCLTECH.NS", "HEXT.NS", "IKS.NS",
        "INFY.NS", "INTELLECT.NS", "KPITTECH.NS", "LATENTVIEW.NS", "LTM.NS", "LTTS.NS",
        "MAPMYINDIA.NS", "MASTEK.NS", "MPHASIS.NS", "NETWEB.NS", "NEWGEN.NS", "OFSS.NS",
        "PERSISTENT.NS", "RATEGAIN.NS", "SAGILITY.NS", "SONATSOFTW.NS", "TANLA.NS",
        "TATAELXSI.NS", "TATATECH.NS", "TCS.NS", "TECHM.NS", "WIPRO.NS", "ZAGGLE.NS",
        "ZENSARTECH.NS",
    ],
    # pharma — 70 names
    "pharma": [
        "AARTIDRUGS.NS", "AARTIPHARM.NS", "ABBOTINDIA.NS", "ACUTAAS.NS", "ADVENZYMES.NS",
        "AGARWALEYE.NS", "AJANTPHARM.NS", "AKUMS.NS", "ALIVUS.NS", "ALKEM.NS", "ANTHEM.NS",
        "APLLTD.NS", "APOLLOHOSP.NS", "ASTERDM.NS", "AUROPHARMA.NS", "BIOCON.NS", "BLUEJET.NS",
        "CAPLIPOINT.NS", "CIPLA.NS", "COHANCE.NS", "CONCORDBIO.NS", "CORONA.NS", "DIVISLAB.NS",
        "DRREDDY.NS", "EMCURE.NS", "ERIS.NS", "FORTIS.NS", "GLAND.NS", "GLAXO.NS",
        "GLENMARK.NS", "GRANULES.NS", "HCG.NS", "INDGN.NS", "IPCALAB.NS", "JLHL.NS",
        "JUBLPHARMA.NS", "KIMS.NS", "LALPATHLAB.NS", "LAURUSLABS.NS", "LUPIN.NS", "MANKIND.NS",
        "MARKSANS.NS", "MAXHEALTH.NS", "MEDANTA.NS", "METROPOLIS.NS", "NATCOPHARM.NS",
        "NEULANDLAB.NS", "NH.NS", "ONESOURCE.NS", "PARKHOSPS.NS", "PFIZER.NS", "POLYMED.NS",
        "PPLPHARMA.NS", "RAINBOW.NS", "RUBICON.NS", "SAILIFE.NS", "SANOFICONR.NS",
        "SHILPAMED.NS", "SPARC.NS", "STAR.NS", "SUNPHARMA.NS", "SUPRIYA.NS", "SYNGENE.NS",
        "THYROCARE.NS", "TORNTPHARM.NS", "VIJAYA.NS", "VIYASH.NS", "WOCKPHARMA.NS",
        "YATHARTH.NS", "ZYDUSLIFE.NS",
    ],
    # auto — 48 names
    "auto": [
        "APOLLOTYRE.NS", "ARE&M.NS", "ASAHIINDIA.NS", "ASKAUTOLTD.NS", "ATHERENERG.NS",
        "BAJAJ-AUTO.NS", "BALKRISIND.NS", "BANCOINDIA.NS", "BELRISE.NS", "BHARATFORG.NS",
        "BOSCHLTD.NS", "CEATLTD.NS", "CIEINDIA.NS", "CRAFTSMAN.NS", "EICHERMOT.NS",
        "ENDURANCE.NS", "EXIDEIND.NS", "FIEMIND.NS", "FORCEMOT.NS", "GABRIEL.NS",
        "HEROMOTOCO.NS", "HYUNDAI.NS", "JAMNAAUTO.NS", "JBMA.NS", "JKTYRE.NS", "LUMAXTECH.NS",
        "M&M.NS", "MARUTI.NS", "MINDACORP.NS", "MOTHERSON.NS", "MRF.NS", "MSUMI.NS",
        "OLAELEC.NS", "OLECTRA.NS", "PRICOLLTD.NS", "RKFORGE.NS", "SANSERA.NS",
        "SCHAEFFLER.NS", "SHRIPISTON.NS", "SKFINDIA.NS", "SONACOMS.NS", "TENNIND.NS",
        "TIINDIA.NS", "TMPV.NS", "TVSMOTOR.NS", "UNOMINDA.NS", "VARROC.NS", "ZFCVINDIA.NS",
    ],
    # oil_gas_power — 42 names
    "oil_gas_power": [
        "ACMESOLAR.NS", "ADANIENSOL.NS", "ADANIGREEN.NS", "ADANIPOWER.NS", "AEGISLOG.NS",
        "AEGISVOPAK.NS", "ATGL.NS", "BPCL.NS", "CASTROLIND.NS", "CESC.NS", "CHENNPETRO.NS",
        "COALINDIA.NS", "EIEL.NS", "GAIL.NS", "GMRP&UI.NS", "HINDPETRO.NS", "IGL.NS", "IOC.NS",
        "IONEXCHANG.NS", "JPPOWER.NS", "JSWENERGY.NS", "KPIGREEN.NS", "MGL.NS", "MRPL.NS",
        "NAVA.NS", "NHPC.NS", "NLCINDIA.NS", "NTPC.NS", "NTPCGREEN.NS", "OIL.NS", "ONGC.NS",
        "PETRONET.NS", "POWERGRID.NS", "PTC.NS", "REFEX.NS", "RELIANCE.NS", "RPOWER.NS",
        "RTNPOWER.NS", "SJVN.NS", "TATAPOWER.NS", "TORNTPOWER.NS", "WABAG.NS",
    ],
    # defence — 19 names
    "defence": [
        "ABB.NS", "AIAENG.NS", "BEL.NS", "BEML.NS", "BHEL.NS", "CGPOWER.NS", "COCHINSHIP.NS",
        "CUMMINSIND.NS", "ENGINERSIN.NS", "GRSE.NS", "HAL.NS", "IRCON.NS", "LT.NS",
        "MAZDOCK.NS", "NBCC.NS", "RVNL.NS", "SIEMENS.NS", "THERMAX.NS", "TITAGARH.NS",
    ],
    # fmcg — 45 names
    "fmcg": [
        "ABDL.NS", "AVANTIFEED.NS", "AWL.NS", "BALRAMCHIN.NS", "BBTC.NS", "BECTORFOOD.NS",
        "BIKAJI.NS", "BRITANNIA.NS", "CCL.NS", "COLPAL.NS", "CUPID.NS", "DABUR.NS", "DOMS.NS",
        "EIDPARRY.NS", "EMAMILTD.NS", "GAEL.NS", "GILLETTE.NS", "GODFRYPHLP.NS",
        "GODREJAGRO.NS", "GODREJCP.NS", "GOKULAGRO.NS", "HERITGFOOD.NS", "HINDUNILVR.NS",
        "HONASA.NS", "INDIAGLYCO.NS", "ITC.NS", "JYOTHYLAB.NS", "KRBL.NS", "KSCL.NS",
        "LTFOODS.NS", "MANORAMA.NS", "MARICO.NS", "NESTLEIND.NS", "ORKLAINDIA.NS",
        "PATANJALI.NS", "PICCADIL.NS", "RADICO.NS", "RENUKA.NS", "TATACONSUM.NS", "TI.NS",
        "TRIVENI.NS", "UBL.NS", "UNITDSPR.NS", "VBL.NS", "ZYDUSWELL.NS",
    ],
    # metals — 24 names
    "metals": [
        "ADANIENT.NS", "ASHAPURMIN.NS", "GMDCLTD.NS", "GRAVITA.NS", "HINDALCO.NS",
        "HINDCOPPER.NS", "HINDZINC.NS", "IMFA.NS", "JAIBALAJI.NS", "JAINREC.NS",
        "JINDALSTEL.NS", "JSL.NS", "JSWSTEEL.NS", "LLOYDSENT.NS", "LLOYDSME.NS", "MOIL.NS",
        "NATIONALUM.NS", "NMDC.NS", "NSLNISP.NS", "SAIL.NS", "SANDUMA.NS", "SARDAEN.NS",
        "TATASTEEL.NS", "VEDL.NS",
    ],
    # capital_goods — 99 names
    "capital_goods": [
        "ACE.NS", "AEQUS.NS", "ANUP.NS", "APARINDS.NS", "APLAPOLLO.NS", "APOLLO.NS",
        "ASHOKLEY.NS", "ASTRAL.NS", "ASTRAMICRO.NS", "ATLANTAELE.NS", "AVALON.NS",
        "AXISCADES.NS", "AZAD.NS", "BALUFORGE.NS", "BDL.NS", "BORORENEW.NS", "CARBORUNIV.NS",
        "CPPLUS.NS", "DATAPATTNS.NS", "DIACABS.NS", "DYNAMATECH.NS", "ELECON.NS",
        "ELECTCAST.NS", "ELGIEQUIP.NS", "EMMVEE.NS", "ENRIN.NS", "EPL.NS", "ESCORTS.NS",
        "FINCABLES.NS", "FINPIPE.NS", "GALLANTT.NS", "GMMPFAUDLR.NS", "GPIL.NS", "GRAPHITE.NS",
        "GREAVESCOT.NS", "GRINDWELL.NS", "GRWRHITECH.NS", "GVT&D.NS", "HBLENGINE.NS", "HEG.NS",
        "HONAUT.NS", "INOXINDIA.NS", "INOXWIND.NS", "JAYNECOIND.NS", "JINDALSAW.NS", "JWL.NS",
        "JYOTICNC.NS", "KAYNES.NS", "KEI.NS", "KIRLOSBROS.NS", "KIRLOSENG.NS", "KIRLPNU.NS",
        "KRN.NS", "KSB.NS", "LLOYDSENGG.NS", "MAHSEAMLES.NS", "MIDHANI.NS", "MTARTECH.NS",
        "OSWALPUMPS.NS", "PARAS.NS", "POLYCAB.NS", "POWERINDIA.NS", "PRAJIND.NS",
        "PREMIERENE.NS", "PTCIL.NS", "QPOWER.NS", "RATNAMANI.NS", "RHIM.NS", "RRKABEL.NS",
        "SAATVIKGL.NS", "SCHNEIDER.NS", "SHAKTIPUMP.NS", "SHYAMMETL.NS", "SKFINDUS.NS",
        "SKIPPER.NS", "SMLMAH.NS", "SUBROS.NS", "SUPREMEIND.NS", "SURYAROSNI.NS", "SUZLON.NS",
        "SYRMA.NS", "TARIL.NS", "TDPOWERSYS.NS", "TEGA.NS", "TEXRAIL.NS", "TIMETECHNO.NS",
        "TIMKEN.NS", "TMCV.NS", "TRANSRAILL.NS", "TRITURBINE.NS", "USHAMART.NS", "UTLSOLAR.NS",
        "VIKRAMSOLR.NS", "VOLTAMP.NS", "WAAREEENER.NS", "WAAREERTL.NS", "WEBELSOLAR.NS",
        "WELCORP.NS", "ZENTEC.NS",
    ],
    # chemicals — 45 names
    "chemicals": [
        "AARTIIND.NS", "ACI.NS", "AETHER.NS", "ALKYLAMINE.NS", "ANURAS.NS", "ATUL.NS",
        "BALAMINES.NS", "BAYERCROP.NS", "CHAMBLFERT.NS", "CLEAN.NS", "COROMANDEL.NS",
        "DEEPAKFERT.NS", "DEEPAKNTR.NS", "ELLEN.NS", "FACT.NS", "FLUOROCHEM.NS", "GHCL.NS",
        "GNFC.NS", "GSFC.NS", "HSCL.NS", "JUBLINGREA.NS", "LINDEINDIA.NS", "LXCHEM.NS",
        "NAVINFLUOR.NS", "NEOGEN.NS", "NFL.NS", "PARADEEP.NS", "PCBL.NS", "PIDILITIND.NS",
        "PIIND.NS", "PRIVISCL.NS", "RAIN.NS", "RALLIS.NS", "RCF.NS", "SHARDACROP.NS",
        "SOLARINDS.NS", "SPLPETRO.NS", "SRF.NS", "STYRENIX.NS", "SUDARSCHEM.NS",
        "SUDEEPPHRM.NS", "SUMICHEM.NS", "SWANCORP.NS", "TATACHEM.NS", "UPL.NS",
    ],
    # consumer_durables — 41 names
    "consumer_durables": [
        "AMBER.NS", "ASIANPAINT.NS", "BAJAJELEC.NS", "BATAINDIA.NS", "BERGEPAINT.NS",
        "BLUESTARCO.NS", "BLUESTONE.NS", "CAMPUS.NS", "CELLO.NS", "CENTURYPLY.NS", "CERA.NS",
        "CROMPTON.NS", "DIXON.NS", "ETHOSLTD.NS", "EUREKAFORB.NS", "HAVELLS.NS", "IFBIND.NS",
        "INDIGOPNTS.NS", "JSWDULUX.NS", "KAJARIACER.NS", "KALYANKJIL.NS", "KANSAINER.NS",
        "LGEINDIA.NS", "PCJEWELLER.NS", "PGEL.NS", "PNGJL.NS", "REDTAPE.NS", "RELAXO.NS",
        "SAFARI.NS", "SENCO.NS", "SFL.NS", "SHAILY.NS", "SKYGOLD.NS", "THANGAMAYL.NS",
        "TITAN.NS", "VAIBHAVGBL.NS", "VGUARD.NS", "VIPIND.NS", "VOLTAS.NS", "WAKEFIT.NS",
        "WHIRLPOOL.NS",
    ],
    # construction_materials — 16 names
    "construction_materials": [
        "ACC.NS", "AMBUJACEM.NS", "BIRLACORPN.NS", "DALBHARAT.NS", "GRASIM.NS", "INDIACEM.NS",
        "JKCEMENT.NS", "JKLAKSHMI.NS", "JSWCEMENT.NS", "NUVOCO.NS", "ORIENTCEM.NS",
        "PRSMJOHNSN.NS", "RAMCOCEM.NS", "SHREECEM.NS", "STARCEMENT.NS", "ULTRACEMCO.NS",
    ],
    # services — 26 names
    "services": [
        "ADANIPORTS.NS", "AWFIS.NS", "BLACKBUCK.NS", "BLUEDART.NS", "CMSINFO.NS", "CONCOR.NS",
        "DELHIVERY.NS", "ECLERX.NS", "FSL.NS", "GESHIP.NS", "GMRAIRPORT.NS", "GPPL.NS",
        "HEMIPROP.NS", "IGIL.NS", "INDIGO.NS", "INOXGREEN.NS", "JSWINFRA.NS", "MMTC.NS",
        "MSTCLTD.NS", "NESCO.NS", "QUESS.NS", "REDINGTON.NS", "SCI.NS", "SMARTWORKS.NS",
        "TVSSCS.NS", "WEWORK.NS",
    ],
    # telecom — 13 names
    "telecom": [
        "BHARTIARTL.NS", "BHARTIHEXA.NS", "HFCL.NS", "IDEA.NS", "INDUSTOWER.NS", "ITI.NS",
        "OPTIEMUS.NS", "RAILTEL.NS", "ROUTE.NS", "STLTECH.NS", "TATACOMM.NS", "TEJASNET.NS",
        "TTML.NS",
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
    "capital_goods": 9,
    "chemicals": 10,
    "consumer_durables": 11,
    "construction_materials": 12,
    "services": 13,
    "telecom": 14,
}

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Active vs. fetched universe
# ---------------------------------------------------------------------------
# The data pipeline fetches and panels EVERY sector in SECTOR_MAP, but training
# runs on a subset. Keeping the two separate means widening the traded universe
# later costs a config change, not a re-download and a panel rebuild — the bars
# and the panel columns are already on disk.
#
# The five sectors below were added on 2026-09-04 to recover eight NIFTY 50
# names the original 8-sector taxonomy excluded (ASIANPAINT, TITAN, BHARTIARTL,
# ULTRACEMCO, SHREECEM, GRASIM, ADANIPORTS, UPL). Their data is fetched and
# panelled; they are held out of training for now purely to keep the observation
# width — and therefore the wall-clock cost per run — at 504 rather than 645.
# Empty this set to trade the full 645.
INACTIVE_SECTORS: frozenset[str] = frozenset(
    {
        "chemicals",
        "consumer_durables",
        "construction_materials",
        "services",
        "telecom",
    }
)


def active_tickers() -> list[str]:
    """Tickers the model actually trades — :func:`all_tickers` minus INACTIVE_SECTORS.

    Use this for anything that builds an observation space (training, walk-forward,
    evaluation, paper trading). Use :func:`all_tickers` for fetching and panel
    construction, so the wider data is on disk and ready.
    """
    inactive = {t for s in INACTIVE_SECTORS for t in SECTOR_MAP.get(s, [])}
    return [t for t in all_tickers() if t not in inactive]


def all_tickers() -> list[str]:
    """Return every ticker that has a sector, NIFTY 50 first.

    Previously this returned the union of NIFTY_50 and SECTOR_MAP, which let
    NIFTY 50 members with no sectoral entry through with ``sector_id_of() == 0``
    — an id absent from SECTOR_IDS. For a model whose graph is built from sector
    membership that is not a harmless default: those stocks became a phantom
    ninth sector node. Membership in SECTOR_MAP is now the sole criterion, so an
    unsectored ticker is impossible by construction rather than by discipline.

    Names dropped by this rule are reported by :func:`unsectored_nifty50` so the
    loss is visible instead of silent.
    """
    seen: set[str] = set()
    result: list[str] = []
    sectored = {t for tickers in SECTOR_MAP.values() for t in tickers}
    for t in NIFTY_50:
        if t in sectored and t not in seen and t not in BLACKLISTED_TICKERS:
            seen.add(t)
            result.append(t)
    for tickers in SECTOR_MAP.values():
        for t in tickers:
            if t not in seen and t not in BLACKLISTED_TICKERS:
                seen.add(t)
                result.append(t)
    return result


def unsectored_nifty50() -> list[str]:
    """NIFTY 50 members that no sector in SECTOR_MAP claims.

    These are excluded from the traded universe. They are real, liquid large
    caps — NSE simply classifies them into industries this project's taxonomy
    does not cover (Consumer Durables, Telecommunication, Construction
    Materials, Services, Chemicals). Widening the taxonomy would recover them.
    """
    sectored = {t for tickers in SECTOR_MAP.values() for t in tickers}
    return [t for t in NIFTY_50 if t not in sectored]


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
