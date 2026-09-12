from pathlib import Path

import pandas as pd

from parlabert.dataset.fetch_speeches import (
    FIELDS,
    ODATA_FILTER,
    SPEECHES_FILE,
    URL,
    clean_page,
)
from parlabert.dataset.parlament_api import download_pages, scraped_now

FALLBACK_MARGIN = pd.Timedelta(days=1)


def find_cutoff(speeches: pd.DataFrame, path: Path) -> pd.Timestamp:
    if "Modified" in speeches.columns and speeches["Modified"].notna().any():
        return speeches["Modified"].max()

    cutoff = pd.Timestamp(path.stat().st_mtime, unit="s") - FALLBACK_MARGIN

    print(
        "No 'Modified' column in the existing file - "
        f"falling back to its file time minus {FALLBACK_MARGIN.days} day"
    )

    return cutoff


def download_changes(cutoff: pd.Timestamp) -> pd.DataFrame:
    odata_filter = (
        f"{ODATA_FILTER} and Modified gt datetime'{cutoff:%Y-%m-%dT%H:%M:%S}'"
    )

    scraped_at = scraped_now()
    pages = []

    for page in download_pages(URL, FIELDS, odata_filter):
        pages.append(clean_page(page, scraped_at))
        print(f"Rows: {sum(len(p) for p in pages):,}")

    if not pages:
        return pd.DataFrame(columns=FIELDS)

    return pd.concat(pages, ignore_index=True)


def merge_changes(
    speeches: pd.DataFrame,
    changes: pd.DataFrame,
) -> pd.DataFrame:
    is_known = changes["ID"].isin(speeches["ID"])

    print(f"  {(~is_known).sum():,} new, {is_known.sum():,} corrected")

    kept = speeches[~speeches["ID"].isin(changes["ID"])]

    return pd.concat([kept, changes], ignore_index=True).sort_values("ID")


def update(path: Path = SPEECHES_FILE) -> bool:

    speeches = pd.read_parquet(path)
    cutoff = find_cutoff(speeches, path)

    print(f"Loaded {len(speeches):,} speeches, changed since {cutoff:%Y-%m-%d %H:%M}")

    changes = download_changes(cutoff)

    if changes.empty:
        print("Already up to date.")
        return False

    speeches = merge_changes(speeches, changes)
    speeches.to_parquet(path, index=False)

    print(f"Finished: {len(speeches):,} rows -> {path}")

    return True


def main() -> None:
    if not SPEECHES_FILE.exists():
        raise SystemExit(
            f"{SPEECHES_FILE} not found - run fetch_speeches once to start."
        )

    update()


if __name__ == "__main__":
    main()
