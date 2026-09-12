import pandas as pd

from parlabert.dataset.parlament_api import (
    DATA_DIR,
    SCRAPED_AT_COLUMN,
    download_pages,
    scraped_now,
)

RAPPORTEUR_URL = "https://ws.parlament.ch/odata.svc/Rapporteur"
SUBJECT_BUSINESS_URL = "https://ws.parlament.ch/odata.svc/SubjectBusiness"

# Filter for German language only.
# The API returns the same data in all three languages, so we only need one.
ODATA_FILTER = "Language eq 'DE'"

RAPPORTEUR_FIELDS = ["BusinessNumber", "MemberCouncilNumber", "LastName", "FirstName"]
SUBJECT_BUSINESS_FIELDS = ["IdSubject", "BusinessNumber"]

NUMERIC_SUFFIXES = ("Number", "IdSubject")


def download_table(url: str, fields: list[str], label: str) -> pd.DataFrame:
    rows: list[dict] = []

    for page in download_pages(url, fields, ODATA_FILTER):
        rows.extend(page)
        print(f"{label}: {len(rows):,} rows")

    table = pd.DataFrame(rows, columns=fields)

    for column in table.columns:
        if column.endswith(NUMERIC_SUFFIXES):
            table[column] = pd.to_numeric(
                table[column],
                errors="coerce",
            ).astype("Int64")

    table[SCRAPED_AT_COLUMN] = scraped_now()

    return table


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rapporteurs = download_table(RAPPORTEUR_URL, RAPPORTEUR_FIELDS, "Rapporteur")
    rapporteurs.to_parquet(DATA_DIR / "rapporteurs.parquet", index=False)

    links = download_table(
        SUBJECT_BUSINESS_URL,
        SUBJECT_BUSINESS_FIELDS,
        "SubjectBusiness",
    )
    links.to_parquet(DATA_DIR / "subject_business.parquet", index=False)

    print(
        f"Finished: {len(rapporteurs):,} rapporteur roles, "
        f"{len(links):,} subject-business links"
    )


if __name__ == "__main__":
    main()
