# Runs the full pipeline for one ticker + week: retrieves news (skipping the
# fetch if the CSV is already saved) then scores it with FinBERT, printing
# the final relevance-weighted sentiment score.
#
# To run a different ticker/week, change ACTIVE_TICKER / ACTIVE_WEEK below -
# both must be keys that exist in PIPELINE_CONFIG.

from content_retriever import ContentRetriever
from finbert_reviewer import FinBertReviewer

PIPELINE_CONFIG = {
    "SFTBY": {
        "company_name": "SoftBank Group",
        "aliases": {"SFTBY.US"},
        "weeks": {
            "26Jun": {"from": "2026-06-17", "to": "2026-06-26"},
        },
    },
    "9988.HK": {
        "company_name": "Alibaba Group",
        "aliases": {"Alibaba"},
        "weeks": {
            "15May": {"from": "2026-05-06", "to": "2026-05-15"},
        },
    },
    "MU": {
        "company_name": "Micron Technology",
        "aliases": {"Micron"},
        "weeks": {
            "26Jun": {"from": "2026-06-17", "to": "2026-06-26"},
        },
    },
    "TM": {
        "company_name": "Toyota",
        "aliases": {"TM"},
        "weeks": {
            "08May": {"from": "2026-05-01", "to": "2026-05-08"}
        },
    },
    "SONY": {
        "company_name": "Sony Group",
        "aliases": {"SONY"},
        "weeks": {"13May": {"from": "2026-05-06", "to": "2026-05-13"}},
    },
    "DELL": {
        "company_name": "Dell Technologies",
        "aliases": {"DELL"},
        "weeks": {"29May": {"from": "2026-05-25", "to": "2026-05-29"}},
    },
    "HMC": {
        "company_name": "Honda Motor",
        "aliases": {"HMC", "Honda"},
        "weeks": {
            "13May": {"from": "2026-05-06", "to": "2026-05-13"},
        },
    },
    "NOVO-B.CO": {
        "company_name": "Novo Nordisk",
        "aliases": {"NOVO-B.CO", "Novo Nordisk"},
        "weeks": {
            "22Jun": {"from": "2026-06-15", "to": "2026-06-22"},
        },
    },
    "SAP.XETRA": {
        "company_name": "SAP SE",
        "aliases": {"SAP.XETRA", "SAP"},
        "weeks": {
            "24Apr": {"from": "2026-04-17", "to": "2026-04-24"},
        },
    },
    "FDX": {
        "company_name": "FedEx",
        "aliases": {"FDX"},
        "weeks": {
            "23Jun": {"from": "2026-06-16", "to": "2026-06-23"},
        },
    },
    "9888.HK": {
        "company_name": "Baidu",
        "aliases": {"Baidu", "BIDU"},
        "weeks": {
            "08Jul": {"from": "2026-07-01", "to": "2026-07-08"},
        },
    },
    "NMR": {
        "company_name": "Nomura Holdings",
        "aliases": {"NMR", "Nomura"},
        "weeks": {
            "01May": {"from": "2026-04-22", "to": "2026-05-01"},
        },
    },
    "AAPL": {
        "company_name": "Apple",
        "aliases": {"AAPL", "Apple"},
        "weeks": {
            "10Jun": {"from": "2026-06-01", "to": "2026-06-10"},
        },
    },
    "ASML": {
        "company_name": "ASML Holding",
        "aliases": {"ASML"},
        "weeks": {
            "22Apr": {"from": "2026-04-13", "to": "2026-04-22"},
        },
    },
}

ACTIVE_TICKER = "SFTBY"
ACTIVE_WEEK = "26Jun"


def run_pipeline(ticker, week):
    config = PIPELINE_CONFIG[ticker]
    window = config["weeks"][week]

    retriever = ContentRetriever(
        ticker=ticker,
        company=config["company_name"],
        date_from=window["from"],
        date_to=window["to"],
        label=week,
    )

    if retriever.already_retrieved():
        print(f"[retrieve] {retriever.output_path} already exists, skipping fetch")
    else:
        retriever.retrieve()

    reviewer = FinBertReviewer(
        ticker=ticker,
        company_name=config["company_name"],
        aliases=config["aliases"],
    )
    output_path = f"ahaan-code/results/{ticker}_US_data_finbert_{week}.csv"
    return reviewer.score(retriever.output_path, output_path)


if __name__ == "__main__":
    rw_sentiment = run_pipeline(ACTIVE_TICKER, ACTIVE_WEEK)
    print(f"\n=== FINAL SCORE ===")
    print(f"{ACTIVE_TICKER} ({ACTIVE_WEEK}): relevance-weighted sentiment = {rw_sentiment:.4f}")
