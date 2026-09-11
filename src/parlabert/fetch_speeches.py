from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from parlabert.parlament_api import DATA_DIR, download_pages, parse_odata_date

URL = "https://ws.parlament.ch/odata.svc/Transcript"

# Filter for German language only and Type eq 1 (official transcripts).
# The API returns the same data in all three languages, so we only need one.
ODATA_FILTER = "Language eq 'DE' and Type eq 1"

NUMBER_COLUMNS = ["ID", "IdSubject", "IdSession", "PersonNumber", "CouncilId"]

TEXT_COLUMNS = [
    "Text",
    "LanguageOfText",
    "MeetingCouncilAbbreviation",
    "CouncilName",
    "SpeakerFirstName",
    "SpeakerLastName",
    "SpeakerFullName",
    "SpeakerFunction",
    "ParlGroupName",
    "ParlGroupAbbreviation",
    "CantonName",
    "CantonAbbreviation",
]

TIMESTAMP_COLUMNS = ["Start", "End", "Modified"]

FIELDS = NUMBER_COLUMNS + TEXT_COLUMNS + TIMESTAMP_COLUMNS + ["MeetingDate"]

SCRAPED_AT_COLUMN = "scraped_at"


def clean_page(page: list[dict], scraped_at: pd.Timestamp) -> pd.DataFrame:
    df = pd.DataFrame(page, columns=FIELDS)

    for column in NUMBER_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")

    for column in TEXT_COLUMNS:
        df[column] = df[column].astype("string")

    for column in TIMESTAMP_COLUMNS:
        df[column] = df[column].map(parse_odata_date)

    df["MeetingDate"] = pd.to_datetime(
        df["MeetingDate"],
        format="%Y%m%d",
        errors="coerce",
    )

    df[SCRAPED_AT_COLUMN] = scraped_at

    return df


SPEECHES_FILE = DATA_DIR / "speeches_official.parquet"


def fetch_speeches(destination: Path) -> int:
    scraped_at = pd.Timestamp(datetime.now(UTC))
    writer = None
    total = 0

    try:
        for page in download_pages(URL, FIELDS, ODATA_FILTER):
            table = pa.Table.from_pandas(
                clean_page(page, scraped_at),
                preserve_index=False,
            )

            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema)

            writer.write_table(table)

            total += len(page)
            print(f"Rows: {total:,}")
    finally:
        if writer is not None:
            writer.close()

    return total


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    destination = SPEECHES_FILE

    print("Downloading transcripts from ws.parlament.ch ...")
    total = fetch_speeches(destination)

    print(f"Finished: {total:,} rows -> {destination}")


if __name__ == "__main__":
    main()
