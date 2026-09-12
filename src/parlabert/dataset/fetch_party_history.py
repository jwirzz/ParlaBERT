import pandas as pd

from parlabert.dataset.parlament_api import (
    DATA_DIR,
    SCRAPED_AT_COLUMN,
    download_pages,
    parse_odata_date,
    scraped_now,
)

URL = "https://ws.parlament.ch/odata.svc/MemberPartyHistory"

# Filter for German language only.
# The API returns the same data in all three languages, so we only need one.
ODATA_FILTER = "Language eq 'DE'"

FIELDS = [
    "PersonNumber",
    "FirstName",
    "LastName",
    "PartyNumber",
    "PartyName",
    "PartyAbbreviation",
    "DateJoining",
    "DateLeaving",
]

COLUMN_NAMES = {
    "PersonNumber": "person_number",
    "FirstName": "firstname",
    "LastName": "lastname",
    "PartyNumber": "party_number",
    "PartyName": "party_name",
    "PartyAbbreviation": "party_abbreviation",
    "DateJoining": "date_joining",
    "DateLeaving": "date_leaving",
}


def fetch_party_history() -> pd.DataFrame:
    rows: list[dict] = []

    for page in download_pages(URL, FIELDS, ODATA_FILTER):
        rows.extend(page)
        print(f"Fetched {len(rows):,} rows ...")

    history = pd.DataFrame(rows, columns=FIELDS).rename(columns=COLUMN_NAMES)

    history["person_number"] = pd.to_numeric(
        history["person_number"],
        errors="coerce",
    ).astype("Int64")

    history["date_joining"] = history["date_joining"].map(parse_odata_date)

    history["date_leaving"] = history["date_leaving"].map(parse_odata_date)

    history[SCRAPED_AT_COLUMN] = scraped_now()

    return history


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading MemberPartyHistory ...")
    history = fetch_party_history()

    destination = DATA_DIR / "member_party_history.parquet"
    history.to_parquet(destination, index=False)

    print(f"Finished: {len(history):,} rows -> {destination}")
    print(f"Persons: {history['person_number'].nunique():,}")
    print(f"Open periods: {history['date_leaving'].isna().sum():,}")


if __name__ == "__main__":
    main()
