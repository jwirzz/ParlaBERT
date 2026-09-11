from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path("data/raw")

FILES = [
    {
        "filename": "speeches_CHE.ndjson.gz",
        "url": "https://files.openparldata.ch/exports/speeches/speeches_CHE.ndjson.gz",
    },
    {
        "filename": "persons.ndjson.gz",
        "url": "https://files.openparldata.ch/exports/persons.ndjson.gz",
    },
]


def download_file(url: str, destination: Path) -> None:
    print(f"Downloading {destination.name} ...")

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()

        with destination.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)

    print(f"Downloaded: {destination}")


def convert_raw_data_to_parquet_and_csv():
    for file in DATA_DIR.glob("*.ndjson.gz"):
        source_file = DATA_DIR / (file.stem + ".gz")
        target_file_parquet = DATA_DIR.parent / (
            file.stem.replace(".ndjson", "") + ".parquet"
        )
        target_file_csv = DATA_DIR.parent / (file.stem.replace(".ndjson", "") + ".csv")

        df = pd.read_json(path_or_buf=source_file, lines=True, compression="gzip")
        df.to_parquet(target_file_parquet, index=False)
        df.to_csv(target_file_csv, index=False)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for file in FILES:
        download_file(file["url"], DATA_DIR / file["filename"])

    print("Converting raw data to Parquet and CSV formats...")
    convert_raw_data_to_parquet_and_csv()
    print("Finished!")


if __name__ == "__main__":
    main()
