"""Publishes the prepared dataset as a new version on Hugging Face."""

import os
from datetime import UTC, datetime
from pathlib import Path

from parlabert.dataset.hf_hub import DATASET_REPO, api
from parlabert.dataset.prepare_dataset import PARQUET_FILE


def build_tag(client, repo_id: str, day: datetime) -> str:
    version = os.environ.get("DATASET_VERSION")

    if version:
        return version

    taken = {
        ref.name for ref in client.list_repo_refs(repo_id, repo_type="dataset").tags
    }

    base = f"v{day:%Y.%m.%d}"
    tag = base
    attempt = 1

    while tag in taken:
        attempt += 1
        tag = f"{base}-{attempt}"

    return tag


def push_dataset(client, day: datetime) -> str:
    client.create_repo(DATASET_REPO, repo_type="dataset", exist_ok=True)

    client.upload_file(
        path_or_fileobj=PARQUET_FILE,
        path_in_repo=PARQUET_FILE.name,
        repo_id=DATASET_REPO,
        repo_type="dataset",
        commit_message=f"Weekly update of {day:%Y-%m-%d}",
    )

    tag = build_tag(client, DATASET_REPO, day)
    client.create_tag(DATASET_REPO, tag=tag, repo_type="dataset")

    return tag


def report(tag: str) -> None:
    summary = os.environ.get("GITHUB_STEP_SUMMARY")

    if not summary:
        return

    with Path(summary).open("a") as report_file:
        report_file.write(
            f"Published `{tag}` to "
            f"[{DATASET_REPO}](https://huggingface.co/datasets/{DATASET_REPO})\n"
        )


def main() -> None:
    if not PARQUET_FILE.exists():
        raise SystemExit(f"{PARQUET_FILE} not found - run prepare_dataset first.")

    client = api()
    day = datetime.now(UTC)

    tag = push_dataset(client, day)

    print(f"  {PARQUET_FILE.name} ({PARQUET_FILE.stat().st_size / 1e6:,.0f} MB)")
    print(f"Finished: {DATASET_REPO} tagged as {tag}")

    report(tag)


if __name__ == "__main__":
    main()
