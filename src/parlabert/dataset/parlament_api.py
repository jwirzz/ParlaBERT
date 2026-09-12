import re
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path("data/raw")

SCRAPED_AT_COLUMN = "scraped_at"


def scraped_now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(UTC))


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# The OData API returns a maximum of 1000 rows per page.
# You can request more, but the API will ignore it and return only 1000 rows anyway.
PAGE_SIZE = 1000


def parse_odata_date(value: object) -> pd.Timestamp | None:
    if not isinstance(value, str):
        return None

    match = re.search(r"-?\d+", value)

    if match is None:
        return None
    return pd.to_datetime(int(match.group()) / 1000, unit="s")


def download_pages(
    url: str,
    fields: list[str],
    odata_filter: str,
) -> Iterator[list[dict]]:
    skip = 0

    while True:
        response = requests.get(
            url,
            params={
                "$filter": odata_filter,
                "$select": ",".join(fields),
                "$format": "json",
                "$top": PAGE_SIZE,
                "$skip": skip,
            },
            headers=HEADERS,
            timeout=180,
        )
        response.raise_for_status()

        page = response.json()["d"]

        if not page:
            return

        yield page

        if len(page) < PAGE_SIZE:
            return

        skip += PAGE_SIZE
