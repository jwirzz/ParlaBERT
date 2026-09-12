import html
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SPEECHES_FILE = "data/raw/speeches_official.parquet"
HISTORY_FILE = "data/raw/member_party_history.parquet"
RAPPORTEURS_FILE = "data/raw/rapporteurs.parquet"
SUBJECT_BUSINESS_FILE = "data/raw/subject_business.parquet"
OUTPUT_DIR = Path("data/processed")
PARQUET_FILE = OUTPUT_DIR / "parlabert.parquet"
CSV_FILE = OUTPUT_DIR / "parlabert.csv"

OUTPUT_COLUMNS = {
    "ID": "speech_id",
    "IdSubject": "debate_id",
    "PersonNumber": "person_number",
    "SpeakerFullName": "speaker",
    "MeetingDate": "date",
    "LanguageOfText": "language",
    "text": "text",
    "party_canonical": "party",
    "party_source": "party_source",
    "scraped_at": "scraped_at",
}


TRAINING_PARTIES = [
    "SP",
    "Die Mitte",
    "FDP-Liberale",
    "SVP",
    "GRÜNE",
    "glp",
    "EVP",
]

CANONICAL_PARTY = {
    "FDP": "FDP-Liberale",
    "LPS": "FDP-Liberale",
    "CVP": "Die Mitte",
    "BDP": "Die Mitte",
    "CVPO": "Die Mitte",
    "GPS": "GRÜNE",
    "GB": "GRÜNE",
    "GLiZ": "glp",
    "MCR": "MCG",
}

GOVERNMENT_COUNCIL_IDS = [98, 99]
GOVERNMENT_FUNCTIONS = r"^(BR|BPR|VPBR|BK)-"
PRESIDING_FUNCTIONS = r"^(P|1VP|2VP|AP)-"
MIN_WORDS = 50
FAR_FUTURE = pd.Timestamp("2099-12-31")


def clean_text(raw_text: pd.Series) -> pd.Series:
    return (
        raw_text.fillna("")
        # removes anything that looks like an HTML/XML tag
        .str.replace(r"<[^>]+>", " ", regex=True)
        # turns HTML entities back into plain characters ("&amp;" -> "&")
        .map(html.unescape)
        # removes bracketed transcript markers
        .str.replace(r"\[(VS|GZ|NB|NAM|PAGE[^\]]*)\]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def drop_government_speeches(speeches: pd.DataFrame) -> pd.DataFrame:
    from_government_council = speeches["CouncilId"].isin(GOVERNMENT_COUNCIL_IDS)
    has_government_function = (
        speeches["SpeakerFunction"].fillna("").str.match(GOVERNMENT_FUNCTIONS)
    )

    is_government = from_government_council | has_government_function

    print(f"Dropping {is_government.sum():,} speeches held for the government")

    return speeches[~is_government]


def drop_presiding_speeches(speeches: pd.DataFrame) -> pd.DataFrame:
    is_presiding = speeches["SpeakerFunction"].fillna("").str.match(PRESIDING_FUNCTIONS)

    print(f"Dropping {is_presiding.sum():,} speeches held while presiding")

    return speeches[~is_presiding]


def drop_rapporteur_speeches(
    speeches: pd.DataFrame,
    rapporteurs: pd.DataFrame,
    subject_business: pd.DataFrame,
) -> pd.DataFrame:
    reporting_pairs = (
        rapporteurs.merge(subject_business, on="BusinessNumber")[
            ["IdSubject", "MemberCouncilNumber"]
        ]
        .drop_duplicates()
        .rename(columns={"MemberCouncilNumber": "PersonNumber"})
        .assign(is_rapporteur=True)
    )

    marked = speeches.merge(
        reporting_pairs,
        on=["IdSubject", "PersonNumber"],
        how="left",
    )
    is_rapporteur = marked["is_rapporteur"].fillna(False).astype(bool).to_numpy()

    print(f"Dropping {is_rapporteur.sum():,} speeches held as rapporteur")

    return speeches[~is_rapporteur]


def drop_short_speeches(speeches: pd.DataFrame) -> pd.DataFrame:
    is_too_short = speeches["text"].str.split().str.len() < MIN_WORDS

    print(f"Dropping {is_too_short.sum():,} speeches under {MIN_WORDS} words")

    return speeches[~is_too_short]


def drop_speeches_with_missing_language(speeches: pd.DataFrame) -> pd.DataFrame:
    is_missing_language = speeches["LanguageOfText"].isna()

    print(f"Dropping {is_missing_language.sum():,} speeches with missing language")

    return speeches[~is_missing_language]


def group_periods_by_person(history: pd.DataFrame) -> dict[int, list]:
    history = history.copy()
    history["date_leaving"] = history["date_leaving"].fillna(FAR_FUTURE)

    periods_by_person: dict[int, list] = {}

    for period in history.itertuples():
        periods_by_person.setdefault(period.person_number, []).append(period)

    return periods_by_person


def days_between_period_and_date(period, date: pd.Timestamp) -> pd.Timedelta:
    if date < period.date_joining:
        return period.date_joining - date

    return date - period.date_leaving


def find_period(periods: list, date: pd.Timestamp) -> tuple:
    matching = [
        period
        for period in periods
        if period.date_joining <= date <= period.date_leaving
    ]

    if matching:
        return max(matching, key=lambda period: period.date_joining), "exact"

    nearest = min(
        periods, key=lambda period: days_between_period_and_date(period, date)
    )

    return nearest, "nearest"


def add_party_at_speech(
    speeches: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    periods_by_person = group_periods_by_person(history)

    abbreviations = []
    names = []
    sources = []

    for person_number, date in zip(
        speeches["PersonNumber"],
        speeches["MeetingDate"],
    ):
        periods = periods_by_person.get(person_number)

        if not periods:
            abbreviations.append(None)
            names.append(None)
            sources.append("none")
            continue

        period, source = find_period(periods, date)

        abbreviations.append(period.party_abbreviation)
        names.append(period.party_name)
        sources.append(source)

    speeches = speeches.copy()
    speeches["party_at_speech"] = abbreviations
    speeches["party_at_speech_name"] = names
    speeches["party_canonical"] = speeches["party_at_speech"].replace(CANONICAL_PARTY)
    speeches["party_source"] = sources

    return speeches


def keep_training_parties(speeches: pd.DataFrame) -> pd.DataFrame:
    is_training_party = speeches["party_canonical"].isin(TRAINING_PARTIES)

    print(
        f"Dropping {(~is_training_party).sum():,} speeches outside the "
        f"{len(TRAINING_PARTIES)} training parties"
    )

    return speeches[is_training_party]


def drop_duplicate_speeches(speeches: pd.DataFrame) -> pd.DataFrame:
    is_duplicate = speeches.duplicated(
        subset=["text", "PersonNumber", "party_canonical"]
    )

    print(f"Dropping {is_duplicate.sum():,} duplicate speeches")

    return speeches[~is_duplicate]


def add_scraped_at(speeches: pd.DataFrame) -> pd.DataFrame:

    if "scraped_at" in speeches.columns:
        return speeches

    scraped_at = pd.Timestamp(
        datetime.fromtimestamp(Path(SPEECHES_FILE).stat().st_mtime, UTC)
    )

    print(
        f"No 'scraped_at' in the raw file - using its file time {scraped_at:%Y-%m-%d %H:%M}"
    )

    return speeches.assign(scraped_at=scraped_at)


def select_output_columns(speeches: pd.DataFrame) -> pd.DataFrame:
    dropped = len(speeches.columns) - len(OUTPUT_COLUMNS)

    print(f"Dropping {dropped} columns that training does not need")

    return speeches[list(OUTPUT_COLUMNS)].rename(columns=OUTPUT_COLUMNS)


def main() -> None:
    speeches = pd.read_parquet(SPEECHES_FILE)
    party_history = pd.read_parquet(HISTORY_FILE)
    rapporteurs = pd.read_parquet(RAPPORTEURS_FILE)
    subject_business = pd.read_parquet(SUBJECT_BUSINESS_FILE)

    print(f"Loaded {len(speeches):,} speeches")

    speeches["text"] = clean_text(speeches["Text"])
    speeches = add_scraped_at(speeches)

    speeches = drop_government_speeches(speeches)
    speeches = drop_presiding_speeches(speeches)
    speeches = drop_rapporteur_speeches(speeches, rapporteurs, subject_business)
    speeches = drop_short_speeches(speeches)
    speeches = add_party_at_speech(speeches, party_history)
    speeches = keep_training_parties(speeches)
    speeches = drop_speeches_with_missing_language(speeches)
    speeches = drop_duplicate_speeches(speeches)
    speeches = select_output_columns(speeches)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    speeches.to_parquet(PARQUET_FILE, index=False)
    speeches.to_csv(CSV_FILE, index=False)

    print(f"Finished: {len(speeches):,} rows x {len(speeches.columns)} columns")
    print(f"  {PARQUET_FILE}")
    print(f"  {CSV_FILE}")


if __name__ == "__main__":
    main()
