"""
ner_entity_linker.py
====================
NER + lightweight knowledge graph for the FinSent semantic relevance layer.

Replaces spaCy (broken in this env) with a pure-regex / dictionary NER
approach that is specifically tuned to financial news text and the 10-stock
basket.  This is appropriate for an intern-scoped production prototype:
the KG schema is identical to what a full spaCy + DBpedia pipeline would
produce, so the downstream consumers (bi-encoder, cross-encoder) are
unaffected when a full NLP library is swapped in later.

What this module does
---------------------
1.  ENTITY CATALOGUE
    For every ticker in the basket, maintains:
      - canonical company name
      - all known surface forms (aliases, abbreviations, common misspellings)
      - GICS sector code + sector name
      - region
      - a small set of related entities (subsidiaries, parent, key executives,
        major suppliers / customers) that imply relevance

2.  ENTITY DETECTION  (NER substitute)
    Scans article text for surface-form matches using pre-compiled regexes
    (word-boundary anchored, case-insensitive).  Returns a list of
    EntityMention objects with ticker, matched_text, span, and confidence.

3.  ENTITY LINKING  (KG resolution)
    Maps each mention to its canonical ticker.  Resolves ambiguity
    (e.g. "ICICI" → ICICI Bank vs ICICI Pru) by scoring context window
    around the mention for disambiguating tokens.

4.  RELEVANCE SCORE
    Returns a float in [0, 1] representing how strongly an article is linked
    to a target ticker, based on:
      - direct mention count (primary company name / ticker symbol)
      - related-entity mention count (subsidiaries, executives, suppliers)
      - title-mention bonus (title mentions carry higher weight)
      - sector-mention bonus (sector keywords without specific entity)

    This score is used as an *additional signal* alongside the cosine
    similarity from the bi-encoder — it feeds into the cross-encoder
    candidate ranking and the GICS broadcast logic.

5.  GICS BROADCAST
    Given an article that is highly relevant to ticker A, returns a dict of
    {ticker: broadcast_weight} for all tickers in the same GICS leaf node.
    Used by the analytics layer to propagate macro-sector news.

Usage
-----
    from ner_entity_linker import EntityLinker

    linker  = EntityLinker()
    result  = linker.score_article(ticker_display="JPM",
                                   title="JPMorgan raises Q3 guidance",
                                   content="The bank's CEO Jamie Dimon said...")
    # result.score          → float [0, 1]
    # result.mentions       → list[EntityMention]
    # result.broadcast      → {ticker: weight} for same-sector tickers

Design notes
------------
- All regex patterns are pre-compiled at import time (fast at inference).
- The KG is a plain Python dict — no external DB dependency.
- Thread-safe for read operations (patterns/KG are immutable after init).
- To extend: add entries to ENTITY_CATALOGUE.  The rest is automatic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Knowledge Graph  — entity catalogue
# ---------------------------------------------------------------------------
# Schema per entry:
#   "display"    : str       — ticker key used throughout the pipeline
#   "symbol"     : str       — exchange symbol
#   "canonical"  : str       — official company name
#   "aliases"    : list[str] — all surface forms to match in text
#   "title_kws"  : list[str] — extra high-confidence keywords (ticker-specific)
#   "related"    : list[str] — subsidiaries / executives / suppliers / customers
#                              (these give a lower-weight mention)
#   "sector_kws" : list[str] — sector-level keywords (lowest weight)
#   "gics_code"  : str       — 8-digit GICS code
#   "gics_name"  : str       — GICS sub-industry name
#   "region"     : str

ENTITY_CATALOGUE: Dict[str, dict] = {

    # ── US ──────────────────────────────────────────────────────────────────
    "JPM": {
        "display":   "JPM",
        "symbol":    "JPM",
        "canonical": "JPMorgan Chase",
        "aliases":   [
            "JPMorgan Chase", "JPMorgan", "JP Morgan", "J.P. Morgan",
            "Chase Bank", "JPM",
        ],
        "title_kws": ["jpmorgan", "jpm", "chase bank"],
        "related":   [
            "Jamie Dimon", "Jeremy Barnum",           # CEO / CFO
            "Chase Sapphire", "Chase Freedom",         # products
            "Bear Stearns", "Washington Mutual",       # historical acquisitions
            "First Republic",                          # 2023 acquisition
        ],
        "sector_kws": [
            "investment bank", "commercial bank", "credit card",
            "wealth management", "asset management", "Fed stress test",
            "Basel III", "CET1", "DFAST",
        ],
        "gics_code": "40101010",
        "gics_name": "Diversified Banks",
        "region":    "US",
    },

    "AVGO": {
        "display":   "AVGO",
        "symbol":    "AVGO",
        "canonical": "Broadcom",
        "aliases":   [
            "Broadcom", "Broadcom Inc", "Broadcom Limited", "AVGO",
        ],
        "title_kws": ["broadcom", "avgo"],
        "related":   [
            "Hock Tan",                                # CEO
            "VMware",                                  # 2023 acquisition
            "CA Technologies", "Brocade",             # prior acquisitions
            "ASIC", "custom silicon", "networking chip",
            "Wi-Fi chip", "Bluetooth chip",
        ],
        "sector_kws": [
            "semiconductor", "chip", "AI accelerator", "networking",
            "data center", "cloud infrastructure", "custom ASIC",
        ],
        "gics_code": "45301010",
        "gics_name": "Semiconductors",
        "region":    "US",
    },

    "TSLA": {
        "display":   "TSLA",
        "symbol":    "TSLA",
        "canonical": "Tesla",
        "aliases":   [
            "Tesla", "Tesla Inc", "Tesla Motors", "TSLA",
        ],
        "title_kws": ["tesla", "tsla"],
        "related":   [
            "Elon Musk", "Vaibhav Taneja",            # CEO / CFO
            "Model S", "Model 3", "Model X", "Model Y", "Cybertruck",
            "Megapack", "Powerwall", "Supercharger",
            "Gigafactory", "Fremont", "Austin",
            "Dojo", "FSD", "Full Self-Driving", "Autopilot",
        ],
        "sector_kws": [
            "electric vehicle", "EV", "battery", "autonomous driving",
            "energy storage", "solar", "charging network",
        ],
        "gics_code": "25102010",
        "gics_name": "Automobile Manufacturers",
        "region":    "US",
    },

    # ── India ────────────────────────────────────────────────────────────────
    "HDFCBANK": {
        "display":   "HDFCBANK",
        "symbol":    "HDFCBANK.NS",
        "canonical": "HDFC Bank",
        "aliases":   [
            "HDFC Bank", "HDFC Bank Ltd", "HDFCBANK",
            # Disambiguation: NOT "HDFC Life", "HDFC Ltd", "HDFC Securities"
            # Those are explicitly excluded by the context-window logic below
        ],
        "title_kws": ["hdfc bank", "hdfcbank"],
        "related":   [
            "Sashidhar Jagdishan",                     # MD & CEO
            "Srinivasaraghavan Vaidyanathan",          # CFO
            "eNetkar", "PayZapp",                      # products
            "HDFC merger",                             # 2023 merger event
            "NPA", "GNPA", "NNPA",                    # bank-specific KPIs
        ],
        "sector_kws": [
            "Indian bank", "PSL", "priority sector lending", "RBI",
            "CASA ratio", "net interest margin", "NIM", "credit growth",
            "retail loan", "home loan India",
        ],
        "gics_code": "40101010",
        "gics_name": "Diversified Banks",
        "region":    "India",
        # Disambiguation tokens: if these appear near "HDFC", it is NOT HDFC Bank
        "_disambig_exclude": ["life insurance", "life insur", "HDFC Life",
                               "HDFC Ltd", "housing finance", "HDFC Securities",
                               "HDFC AMC", "mutual fund"],
    },

    "RELIANCE": {
        "display":   "RELIANCE",
        "symbol":    "RELIANCE.NS",
        "canonical": "Reliance Industries",
        "aliases":   [
            "Reliance Industries", "Reliance", "RIL",
            "Reliance Industries Limited",
        ],
        "title_kws": ["reliance industries", "ril"],
        "related":   [
            "Mukesh Ambani", "Isha Ambani", "Akash Ambani",
            "Jio", "Reliance Jio", "JioMart",
            "Reliance Retail", "Reliance Fresh",
            "Navi Mumbai refinery", "Jamnagar",
        ],
        "sector_kws": [
            "petrochemical", "refinery", "telecom India", "retail India",
            "green energy India", "O2C", "oil-to-chemicals",
        ],
        "gics_code": "10102010",
        "gics_name": "Integrated Oil & Gas",
        "region":    "India",
    },

    "TCS": {
        "display":   "TCS",
        "symbol":    "TCS.NS",
        "canonical": "Tata Consultancy Services",
        "aliases":   [
            "Tata Consultancy Services", "TCS",
            "Tata Consultancy", "TCS Limited",
        ],
        "title_kws": ["tata consultancy", "tcs"],
        "related":   [
            "K Krithivasan", "N Chandrasekaran",       # CEO / Tata Sons chair
            "Contextus", "Quartz", "ignio",           # TCS platforms
            "Tata Group",
        ],
        "sector_kws": [
            "IT services", "outsourcing", "software services India",
            "digital transformation", "cloud migration",
        ],
        "gics_code": "45102010",
        "gics_name": "IT Consulting & Other Services",
        "region":    "India",
    },

    "INFY": {
        "display":   "INFY",
        "symbol":    "INFY.NS",
        "canonical": "Infosys",
        "aliases":   [
            "Infosys", "Infosys Limited", "Infosys Ltd", "INFY",
            "Infosys BPM",
        ],
        "title_kws": ["infosys", "infy"],
        "related":   [
            "Salil Parekh",                            # CEO
            "Nilekani",                                # Nandan Nilekani (founder)
            "Cobalt", "Nia",                          # Infosys platforms
            "EdgeVerve",
        ],
        "sector_kws": [
            "IT services", "outsourcing", "software services India",
            "attrition IT", "deal wins IT",
        ],
        "gics_code": "45102010",
        "gics_name": "IT Consulting & Other Services",
        "region":    "India",
    },

    # ── Japan ────────────────────────────────────────────────────────────────
    "Toyota": {
        "display":   "Toyota",
        "symbol":    "7203.T",
        "canonical": "Toyota Motor",
        "aliases":   [
            "Toyota Motor", "Toyota", "Toyota Motor Corporation",
            "7203.T", "TM",                            # US ADR symbol
        ],
        "title_kws": ["toyota motor", "toyota"],
        "related":   [
            "Koji Sato", "Akio Toyoda",               # current / former CEO
            "Lexus", "Daihatsu",                      # brands / subsidiaries
            "Woven City", "Woven Planet",
            "Aisin", "Denso", "JTEKT",                # key suppliers (Toyota group)
            "GR86", "Land Cruiser", "Prius", "Camry", "RAV4",
            "bZ4X",                                   # EV model
        ],
        "sector_kws": [
            "automobile", "auto Japan", "vehicle production", "EV Japan",
            "hybrid vehicle", "TPS", "Toyota Production System",
            "JAMA",                                   # Japan Automobile Manufacturers Assoc
        ],
        "gics_code": "25102010",
        "gics_name": "Automobile Manufacturers",
        "region":    "Japan",
    },

    "Sony": {
        "display":   "Sony",
        "symbol":    "6758.T",
        "canonical": "Sony Group",
        "aliases":   [
            "Sony Group", "Sony", "Sony Corporation", "Sony Group Corporation",
            "6758.T", "SONY",                         # US ADR symbol
        ],
        "title_kws": ["sony group", "sony"],
        "related":   [
            "Kenichiro Yoshida", "Hiroki Totoki",     # CEO / CFO
            "PlayStation", "PS5", "PS4",
            "Sony Pictures", "Sony Music",
            "Sony Semiconductor", "Sony Sensors",
            "Alpha camera", "WH-1000XM",
        ],
        "sector_kws": [
            "gaming", "console", "entertainment Japan", "image sensor",
            "CMOS sensor", "music streaming", "film studio",
        ],
        "gics_code": "25201040",
        "gics_name": "Consumer Electronics",
        "region":    "Japan",
    },

    "SoftBank": {
        "display":   "SoftBank",
        "symbol":    "9984.T",
        "canonical": "SoftBank Group",
        "aliases":   [
            "SoftBank Group", "SoftBank", "Softbank Group Corp",
            "9984.T",
        ],
        "title_kws": ["softbank group", "softbank"],
        "related":   [
            "Masayoshi Son", "Masa Son",              # founder / CEO
            "Vision Fund", "SoftBank Vision Fund",
            "ARM", "Arm Holdings",                    # key investee
            "T-Mobile",                               # former investee
            "ByteDance", "Grab", "DoorDash",         # portfolio companies
        ],
        "sector_kws": [
            "venture investment Japan", "tech investment", "ARM chips",
            "AI investment Japan", "SPAC Japan",
        ],
        "gics_code": "50202020",
        "gics_name": "Diversified Telecommunication Services",
        "region":    "Japan",
    },
}

# ---------------------------------------------------------------------------
# GICS broadcast map  (gics_code → list of tickers sharing that sector)
# Built automatically from ENTITY_CATALOGUE at import time.
# ---------------------------------------------------------------------------
_GICS_MAP: Dict[str, List[str]] = {}
for _ticker, _info in ENTITY_CATALOGUE.items():
    _code = _info["gics_code"]
    _GICS_MAP.setdefault(_code, []).append(_ticker)


# ---------------------------------------------------------------------------
# Pre-compile regex patterns for every alias
# ---------------------------------------------------------------------------
# Pattern: \b{alias}\b, case-insensitive, compiled once.
# We store (ticker, alias_text, pattern, weight) tuples.
# Weight:
#   2.0  — ticker symbol or canonical name
#   1.5  — aliases
#   1.0  — related entities
#   0.5  — sector keywords

@dataclass
class _Pattern:
    ticker:  str
    surface: str
    regex:   re.Pattern
    weight:  float


def _compile_patterns() -> List[_Pattern]:
    patterns = []
    for ticker, info in ENTITY_CATALOGUE.items():
        # Canonical + aliases
        for alias in [info["canonical"]] + info["aliases"]:
            if not alias.strip():
                continue
            try:
                pat = re.compile(r"\b" + re.escape(alias) + r"\b", re.IGNORECASE)
                # Higher weight for exact ticker symbol
                w = 2.0 if alias.upper() == ticker.upper() else 1.5
                patterns.append(_Pattern(ticker, alias, pat, w))
            except re.error:
                pass

        # Related entities
        for rel in info.get("related", []):
            try:
                pat = re.compile(r"\b" + re.escape(rel) + r"\b", re.IGNORECASE)
                patterns.append(_Pattern(ticker, rel, pat, 1.0))
            except re.error:
                pass

        # Sector keywords
        for kw in info.get("sector_kws", []):
            try:
                pat = re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
                patterns.append(_Pattern(ticker, kw, pat, 0.5))
            except re.error:
                pass

    return patterns


_ALL_PATTERNS: List[_Pattern] = _compile_patterns()


# ---------------------------------------------------------------------------
# Data classes for results
# ---------------------------------------------------------------------------

@dataclass
class EntityMention:
    ticker:       str    # which ticker was matched
    matched_text: str    # exact text that matched
    start:        int    # char offset in the full text
    end:          int
    weight:       float  # alias-level weight (2.0 / 1.5 / 1.0 / 0.5)
    in_title:     bool   # True if the match is in the title portion


@dataclass
class ArticleRelevanceResult:
    ticker:       str
    score:        float               # final NER-based relevance in [0, 1]
    mentions:     List[EntityMention]
    broadcast:    Dict[str, float]    # {other_ticker: weight} via GICS


# ---------------------------------------------------------------------------
# Disambiguation: context-window check
# ---------------------------------------------------------------------------

_CONTEXT_PRE  = 10   # chars BEFORE the match start to look for exclude tokens
_CONTEXT_POST = 60   # chars AFTER the match end

def _is_ambiguous_mention_excluded(
    text: str,
    match: re.Match,
    ticker: str,
) -> bool:
    """
    Returns True if the mention should be *excluded* due to disambiguation
    context (e.g. "HDFC Life Insurance" → the exclude token 'Life Insurance'
    follows the match within 60 chars, so exclude it).

    We look mostly FORWARD from the match end (60 chars) and only a tiny
    amount BACKWARD (10 chars for prefixes like "non-HDFC Bank").
    This prevents a disambiguation token that appears in a *later sentence*
    from incorrectly excluding an earlier, unambiguous mention.
    """
    info = ENTITY_CATALOGUE.get(ticker, {})
    exclude_tokens = info.get("_disambig_exclude", [])
    if not exclude_tokens:
        return False

    # Window: small look-back + moderate look-ahead
    start = max(0, match.start() - _CONTEXT_PRE)
    end   = min(len(text), match.end() + _CONTEXT_POST)
    ctx   = text[start:end].lower()

    for tok in exclude_tokens:
        if tok.lower() in ctx:
            return True
    return False


# ---------------------------------------------------------------------------
# Main EntityLinker class
# ---------------------------------------------------------------------------

class EntityLinker:
    """
    Lightweight NER + KG linker for financial news articles.

    Parameters
    ----------
    title_weight_multiplier : float
        Multiplier applied to mention weights when the match is in the title.
        Default 2.0 (title mentions are treated as twice as informative).
    max_score_cap : float
        Maximum raw score before sigmoid normalisation.  Prevents very long
        articles with many mentions from swamping the [0,1] scale.
    """

    def __init__(
        self,
        title_weight_multiplier: float = 2.0,
        max_score_cap: float = 10.0,
    ):
        self.title_weight_multiplier = title_weight_multiplier
        self.max_score_cap = max_score_cap

    # ------------------------------------------------------------------
    def score_article(
        self,
        ticker_display: str,
        title: str,
        content: str,
    ) -> ArticleRelevanceResult:
        """
        Score how relevant (title + content) is to ticker_display.

        Returns
        -------
        ArticleRelevanceResult with .score in [0, 1].
        """
        title   = title   or ""
        content = content or ""
        full_text = f"{title}\n{content}"
        title_end = len(title) + 1   # +1 for the \n

        mentions: List[EntityMention] = []

        for pat_obj in _ALL_PATTERNS:
            if pat_obj.ticker != ticker_display:
                continue
            for m in pat_obj.regex.finditer(full_text):
                if _is_ambiguous_mention_excluded(full_text, m, ticker_display):
                    continue
                in_title = m.start() < title_end
                mentions.append(EntityMention(
                    ticker       = ticker_display,
                    matched_text = m.group(),
                    start        = m.start(),
                    end          = m.end(),
                    weight       = pat_obj.weight,
                    in_title     = in_title,
                ))

        # Raw score accumulation
        raw_score = 0.0
        for ment in mentions:
            w = ment.weight
            if ment.in_title:
                w *= self.title_weight_multiplier
            raw_score += w

        # Normalise: sigmoid-like mapping to [0, 1]
        capped = min(raw_score, self.max_score_cap)
        normalised = capped / self.max_score_cap   # linear [0,1] — simple and interpretable

        # GICS broadcast
        gics_code = ENTITY_CATALOGUE.get(ticker_display, {}).get("gics_code", "")
        same_sector = _GICS_MAP.get(gics_code, [])
        broadcast: Dict[str, float] = {}
        if normalised > 0.1:   # only broadcast if article has meaningful primary relevance
            for other in same_sector:
                if other != ticker_display:
                    # Broadcast weight is 30% of the primary relevance score
                    broadcast[other] = round(normalised * 0.30, 4)

        return ArticleRelevanceResult(
            ticker    = ticker_display,
            score     = round(normalised, 4),
            mentions  = mentions,
            broadcast = broadcast,
        )

    # ------------------------------------------------------------------
    def score_all_tickers(
        self,
        title: str,
        content: str,
        tickers: Optional[List[str]] = None,
    ) -> Dict[str, ArticleRelevanceResult]:
        """
        Score one article against all tickers simultaneously.
        Useful for the portfolio-level analytics layer.

        Parameters
        ----------
        tickers : list of display names to score; defaults to all in catalogue.
        """
        if tickers is None:
            tickers = list(ENTITY_CATALOGUE.keys())
        return {t: self.score_article(t, title, content) for t in tickers}

    # ------------------------------------------------------------------
    @staticmethod
    def gics_name(ticker_display: str) -> str:
        return ENTITY_CATALOGUE.get(ticker_display, {}).get("gics_name", "Unknown")

    @staticmethod
    def region(ticker_display: str) -> str:
        return ENTITY_CATALOGUE.get(ticker_display, {}).get("region", "Unknown")


# ---------------------------------------------------------------------------
# Convenience function for use in semantic_relevance.py
# ---------------------------------------------------------------------------

_linker_singleton: Optional[EntityLinker] = None

def get_entity_linker() -> EntityLinker:
    global _linker_singleton
    if _linker_singleton is None:
        _linker_singleton = EntityLinker()
    return _linker_singleton


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    linker = EntityLinker()

    tests = [
        ("JPM",      "JPMorgan Chase raises net interest income guidance",
                     "CEO Jamie Dimon said the bank expects credit card revenue to grow."),
        ("HDFCBANK", "HDFC Bank reports record NIM for Q2",
                     "HDFC Bank's CASA ratio improved to 42%. HDFC Life Insurance also posted gains."),
        ("Toyota",   "Toyota Motor cuts EV production target citing battery shortage",
                     "The Aisin supply disruption is expected to affect Toyota's Q4 output."),
        ("AVGO",     "Broadcom custom ASIC wins major hyperscaler contract",
                     "Hock Tan confirmed that VMware integration is on track."),
        ("Sony",     "PlayStation 5 sales hit 50 million units",
                     "Sony Group's gaming segment drove record operating income."),
    ]

    print("\n=== EntityLinker smoke test ===\n")
    for ticker, title, content in tests:
        r = linker.score_article(ticker, title, content)
        print(f"  {ticker:<10} score={r.score:.4f}  "
              f"mentions={len(r.mentions)}  "
              f"broadcast={r.broadcast}")