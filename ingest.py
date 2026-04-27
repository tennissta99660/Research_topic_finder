# ingest.py — arXiv paper fetching + storage for CARTOGRAPH
import json
import time
import logging

import arxiv

from config import ARXIV_MAX_RESULTS
from db import insert_paper

logger = logging.getLogger(__name__)


def fetch_and_store_papers(
    topic: str,
    topic_id: int,
    max_results: int = ARXIV_MAX_RESULTS,
    max_retries: int = 3,
) -> int:
    """Fetch papers from arXiv for the given topic and store in SQLite.

    Retries on HTTP 429/503 with exponential backoff.
    If a failure occurs mid-fetch, keeps papers already stored.
    Returns the number of new papers stored.
    """
    for attempt in range(1, max_retries + 1):
        try:
            client = arxiv.Client(
                page_size=20,
                delay_seconds=3.0,
                num_retries=3,
            )
            search = arxiv.Search(
                query=topic,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )

            count = 0
            for result in client.results(search):
                raw_id = result.entry_id.split("/")[-1]
                paper_id = raw_id.split("v")[0]

                paper = {
                    "id": paper_id,
                    "topic_id": topic_id,
                    "title": result.title,
                    "abstract": result.summary,
                    "authors": json.dumps([a.name for a in result.authors]),
                    "published": str(result.published),
                    "url": result.entry_id,
                }
                insert_paper(paper)
                count += 1
                if count % 50 == 0:
                    logger.info(f"Ingested {count} papers so far...")

            logger.info(f"Finished ingesting {count} papers for topic '{topic}'")
            return count

        except Exception as e:
            error_str = str(e).lower()
            if any(code in error_str for code in ["429", "503", "rate"]):
                # Check how many papers we already stored
                from db import get_papers_by_topic
                existing = len(get_papers_by_topic(topic_id))

                if existing >= 20:
                    logger.warning(
                        f"arXiv error (attempt {attempt}): {e}. "
                        f"Continuing with {existing} papers already stored."
                    )
                    return existing

                wait = 10 * attempt
                logger.warning(
                    f"arXiv error (attempt {attempt}/{max_retries}). "
                    f"Waiting {wait}s... ({existing} papers stored so far)"
                )
                time.sleep(wait)
            else:
                raise

    # If all retries fail, return whatever we have
    from db import get_papers_by_topic
    existing = len(get_papers_by_topic(topic_id))
    if existing > 0:
        logger.warning(f"arXiv fetch incomplete after {max_retries} retries. Using {existing} papers.")
        return existing
    raise RuntimeError(f"arXiv fetch failed after {max_retries} retries")
