"""
Module for customizing Indeed job search parameters.
"""

import asyncio
from random import shuffle
from typing import Any, List, Union
from urllib.parse import quote_plus

from playwright.sync_api import Page

from config.app_config import EASY_APPLY_ONLY_MODE
from config.logger_config import logger
from src.job_manager.indeed import indeed_base_url, indeed_domain, indeed_search_radius
from src.job_manager.search_customizer import BaseSearchCustomizer
from src.utils.utils import async_pause

# Indeed experience level mapping
_EXPERIENCE_LEVEL_MAP = {
    "internship": "internship",
    "entry": "entry_level",
    "entry_level": "entry_level",
    "associate": "associate",
    "mid_senior_level": "mid_level",
    "director": "director",
    "executive": "executive",
}

# Indeed job type mapping
_JOB_TYPE_MAP = {
    "full_time": "fulltime",
    "part_time": "parttime",
    "contract": "contract",
    "temporary": "temporary",
    "internship": "internship",
}

# Indeed date posted mapping (supports both test keys and SearchConfig keys)
_DATE_POSTED_MAP = {
    "past_24_hours": "1",
    "day_24_hours": "1",
    "24_hours": "1",
    "past_week": "7",
    "week": "7",
    "past_month": "14",
    "month": "14",
    "any_time": "",
    "all_time": "",
}

# Indeed remote/work location mapping
_REMOTE_MAP = {
    "remote": "remote",
    "hybrid": "hybrid",
    "onsite": "onsite",
}


class IndeedSearchCustomizer(BaseSearchCustomizer):
    def __init__(self, page: Union[Page, Any]):
        super().__init__(page)
        self.search_urls: List[str] = []
        self.search_terms_for_urls: List[str] = []
        self._search_url_index = 0
        self.sort_by_date = False
        self.easy_apply_only = False
        logger.info(f"IndeedSearchCustomizer initialized for {indeed_domain()}")

    def _build_query(self) -> str:
        """Combine every configured position into one Indeed boolean query"""
        cleaned_positions = [
            position.strip() for position in self.positions if position and position.strip()
        ]
        if len(cleaned_positions) == 1:
            return cleaned_positions[0]
        return " OR ".join(f'"{position}"' for position in cleaned_positions)

    def _build_search_url(self, position: str, location: str = "") -> str:
        """Build an Indeed search URL for a given position and location"""
        params = [f"q={quote_plus(position)}"]

        if location:
            params.append(f"l={quote_plus(location)}")

        radius = indeed_search_radius()
        if radius is not None:
            params.append(f"radius={radius}")

        # Work arrangement: skip filter if all 3 are enabled (user wants all work styles)
        work_arrangements = []
        if self.remote:
            work_arrangements.append(_REMOTE_MAP["remote"])
        if self.hybrid:
            work_arrangements.append(_REMOTE_MAP["hybrid"])
        if self.onsite:
            work_arrangements.append(_REMOTE_MAP["onsite"])
        if work_arrangements and len(work_arrangements) < 3:
            params.append(f"sc=0kf%3Aattr({','.join(work_arrangements)})")

        # Job type
        active_job_types = [
            _JOB_TYPE_MAP[k] for k, v in self.job_types.items() if v and k in _JOB_TYPE_MAP
        ]
        if active_job_types:
            params.append(f"jt={active_job_types[0]}")

        # Date posted
        for key, value in self.date_posted.items():
            if value and key in _DATE_POSTED_MAP and _DATE_POSTED_MAP[key]:
                params.append(f"fromage={_DATE_POSTED_MAP[key]}")
                break

        if self.sort_by_date:
            params.append("sort=date")

        if self.easy_apply_only:
            params.append("iafilter=1")

        return f"{indeed_base_url()}/jobs?{'&'.join(params)}"

    def build_search_urls(self) -> List[str]:
        """One search URL per configured location, all positions in a single query"""
        query = self._build_query()
        if not query:
            return []
        locations = self.locations or [""]
        return [self._build_search_url(query, location) for location in locations]

    def build_per_term_search_urls(self, randomize: bool = False) -> List[str]:
        """One search URL per (position, location) pair, matching Linkedin_Test search loop."""
        cleaned = [p.strip() for p in self.positions if p and p.strip()]
        if not cleaned:
            return []
        if randomize:
            shuffle(cleaned)
        locations = self.locations or [""]
        urls: List[str] = []
        terms: List[str] = []
        for pos in cleaned:
            for loc in locations:
                urls.append(self._build_search_url(pos, loc))
                terms.append(pos)
        self.search_terms_for_urls = terms
        return urls

    @property
    def current_search_term(self) -> str:
        if 0 <= self._search_url_index < len(self.search_terms_for_urls):
            return self.search_terms_for_urls[self._search_url_index]
        return self._build_query()

    async def _goto_search_url(self, url: str, attempts: int = 3) -> None:
        logger.info(f"Navigating to Indeed search: {url}")
        for i in range(1, attempts + 1):
            try:
                await self.page.goto(url, wait_until="domcontentloaded")
                await async_pause(2, 3)
                return
            except Exception as e:
                msg = str(e)
                if i < attempts and any(k in msg for k in ("ERR_NAME_NOT_RESOLVED", "ERR_INTERNET_DISCONNECTED", "ERR_CONNECTION", "Timeout")):
                    logger.warning(f"Indeed search navigation failed (attempt {i}/{attempts}): {msg[:150]}. Retrying...")
                    await asyncio.sleep(i * 2)
                else:
                    raise

    async def set_search_params(
        self, *, per_term: bool = False, randomize: bool = False
    ) -> None:
        """Navigate to the first Indeed search URL."""
        if not self.positions:
            logger.warning("No positions configured for Indeed search")
            return

        if per_term:
            self.sort_by_date = True
            self.easy_apply_only = bool(EASY_APPLY_ONLY_MODE)
            self.search_urls = self.build_per_term_search_urls(randomize=randomize)
        else:
            self.search_urls = self.build_search_urls()
            self.search_terms_for_urls = [self._build_query()] * len(self.search_urls)

        if not self.search_urls:
            logger.warning("No positions configured for Indeed search")
            return

        self._search_url_index = 0
        logger.info(f"Built {len(self.search_urls)} Indeed search(es)")
        await self._goto_search_url(self.search_urls[0])

    async def go_to_next_search(self) -> bool:
        """Navigate to the next configured search. False when all are exhausted."""
        while self._search_url_index + 1 < len(self.search_urls):
            self._search_url_index += 1
            try:
                await self._goto_search_url(self.search_urls[self._search_url_index])
                return True
            except Exception as e:
                logger.warning(
                    f"Could not open search URL for '{self.current_search_term}': {e}. Skipping to next..."
                )
                continue
        return False
