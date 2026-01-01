from __future__ import annotations

import os
import hashlib
from typing import List

import orjson

from src.ingest.chunk import Chunk, chunk_text
from src.ingest.html_fetch import fetch_html, html_to_sections

OUT_PATH = "data/processed/chunks.jsonl"

URLS = [
  "https://support.atlassian.com/confluence-cloud/docs/manage-permissions-on-the-page-level/",
  "https://support.atlassian.com/cloud-automation/docs/create-and-edit-jira-automation-rules/",
  "https://support.atlassian.com/cloud-automation/docs/use-jira-automation-with-confluence/",
  "https://support.atlassian.com/confluence-cloud/docs/set-up-and-manage-public-links/",
  "https://www.atlassian.com/software/confluence/resources/guides/get-started/manage-permissions",

  "https://slack.com/help/articles/360035692513-Guide-to-Slack-Workflow-Builder",
  "https://slack.com/help/articles/17542172840595-Build-a-workflow--Create-a-workflow-in-Slack",
  "https://slack.com/help/articles/360041352714-Build-a-workflow--Create-a-workflow-that-starts-outside-of-Slack",
  "https://slack.com/help/articles/26800170438419-FAQ--Slack-Workflow-Builder",

  "https://www.notion.com/help/sharing-and-permissions",
]


def stable_doc_id(url: str) -> str:
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    return f"url_{h}"

def main():
    if not URLS:
        raise SystemExit("URLS list is empty. Add 3–10 public doc URLs in src/ingest/ingest_urls.py")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    total = 0
    with open(OUT_PATH, "wb") as out:
        for url in URLS:
            html = fetch_html(url)
            doc_id = stable_doc_id(url)
            sections = html_to_sections(html)

            for s_idx, sec in enumerate(sections):
                for c_idx, (a, b, ctext) in enumerate(chunk_text(sec.text)):
                    chunk_id = f"{doc_id}#s{s_idx}#c{c_idx}"
                    ch = Chunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        text=ctext,
                        start_char=a,
                        end_char=b,
                        header_path=sec.header_path,
                        source_type="html",
                    )
                    out.write(orjson.dumps(ch.__dict__) + b"\n")
                    total += 1

    print(f"Wrote {total} chunks to {OUT_PATH}")

if __name__ == "__main__":
    main()