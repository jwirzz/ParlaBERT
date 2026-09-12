import os
from functools import partial
from pathlib import Path

import pandas as pd

from parlabert.dataset.fetch_party_history import fetch_party_history
from parlabert.dataset.fetch_rapporteurs import (
    RAPPORTEUR_FIELDS,
    RAPPORTEUR_URL,
    SUBJECT_BUSINESS_FIELDS,
    SUBJECT_BUSINESS_URL,
    download_table,
)
from parlabert.dataset.fetch_speeches import SPEECHES_FILE, fetch_speeches
from parlabert.dataset.parlament_api import DATA_DIR, SCRAPED_AT_COLUMN
from parlabert.dataset.update_speeches import update as update_speeches

TABLES = {
    "member_party_history.parquet": fetch_party_history,
    "rapporteurs.parquet": partial(
        download_table,
        RAPPORTEUR_URL,
        RAPPORTEUR_FIELDS,
        "Rapporteur",
    ),
    "subject_business.parquet": partial(
        download_table,
        SUBJECT_BUSINESS_URL,
        SUBJECT_BUSINESS_FIELDS,
        "SubjectBusiness",
    ),
}


def sorted_rows(table: pd.DataFrame) -> pd.DataFrame:
    return table.sort_values(list(table.columns)).reset_index(drop=True)


def same_data(fresh: pd.DataFrame, known: pd.DataFrame) -> bool:

    if SCRAPED_AT_COLUMN not in known.columns:
        return False

    return fresh.drop(columns=SCRAPED_AT_COLUMN).equals(
        known.drop(columns=SCRAPED_AT_COLUMN)
    )


def update_table(name: str, fetch) -> bool:
    destination = DATA_DIR / name

    candidate = destination.with_suffix(".new.parquet")
    sorted_rows(fetch()).to_parquet(candidate, index=False)

    fresh = pd.read_parquet(candidate)

    if destination.exists():
        known = sorted_rows(pd.read_parquet(destination))

        if same_data(fresh, known):
            candidate.unlink()

            print(f"{name}: unchanged, {len(fresh):,} rows")

            return False

        print(f"{name}: {len(fresh) - len(known):+,} rows")

    candidate.replace(destination)

    print(f"{name}: updated to {len(fresh):,} rows")

    return True


def update_speeches_file() -> bool:
    if not SPEECHES_FILE.exists():
        print(f"{SPEECHES_FILE} not found - downloading every speech once.")
        fetch_speeches(SPEECHES_FILE)

        return True

    return update_speeches()


def report(changed: bool) -> None:
    github_env = os.environ.get("GITHUB_ENV")

    if not github_env:
        return

    with Path(github_env).open("a") as env:
        env.write(f"changed={str(changed).lower()}\n")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    speeches_changed = update_speeches_file()
    tables_changed = [update_table(name, fetch) for name, fetch in TABLES.items()]

    changed = speeches_changed or any(tables_changed)

    print("Raw data updated." if changed else "Raw data already up to date.")

    report(changed)


if __name__ == "__main__":
    main()
