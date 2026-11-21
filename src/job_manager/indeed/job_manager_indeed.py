import asyncio
import time
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any, List, Tuple

from playwright.sync_api import Page

from config.app_config import (
    COLLECT_INFO_MODE,
    EASY_APPLY_ONLY_MODE,
    MAX_APPLIES_NUM,
    MINIMUM_WAIT_TIME_SEC,
    MONKEY_MODE,
    TEST_MODE,
)
from config.constants import COVER_LETTER_DIR, OUTPUT_DIR_INDEED, RESUME_DIR, SEARCH_CONFIG_FILE
from config.logger_config import logger
from src.dashboard.runtime import StopRequested, emit_event
from src.job_manager.indeed import indeed_base_url
from src.job_manager.indeed.authenticator_indeed import IndeedAuthenticator
from src.job_manager.indeed.easy_applier_indeed import IndeedEasyApplier
from src.job_manager.indeed.linkedin_bridge import (
    ai_runtime,
    apply_resume_score_gate,
    capture_indeed_screenshot,
    check_jd_eligibility,
    estimate_date_posted,
    extract_indeed_jk,
    extract_years_of_experience,
    get_applied_job_ids,
    get_default_resume_path,
    load_linkedin_configs,
    maybe_send_jd_outreach,
    prepare_tailored_resume_pdf,
    record_applied_job,
    record_failed_job,
)
from src.job_manager.job_manager import BaseJobManager
from src.pydantic_models.job_models import Job
from src.utils.bot_control import should_stop
from src.utils.browser_utils import (
    debug_capture,
    find_element_safely,
    find_elements_safely,
    safe_click,
)
from src.utils.runtime_control import ShutdownState, runtime_controller
from src.utils.utils import async_pause, load_yaml_file, sanitize_text

search_config = load_yaml_file(SEARCH_CONFIG_FILE)
logger.info(f"Maximum allowed number of applications: {MAX_APPLIES_NUM}")

INDEED_JOB_CARD_SELECTOR = "div.job_seen_beacon, div[data-testid='jobcard-wrapper']"
INDEED_JOB_TITLE_SELECTOR = "a.jcs-JobTitle, h2.jobTitle a, [data-testid='jobTitle'] a"
INDEED_COMPANY_SELECTOR = "[data-testid='company-name'], .companyName"
INDEED_LOCATION_SELECTOR = "[data-testid='text-location'], .companyLocation"
INDEED_EASY_APPLY_BADGE = "span.iaLabel, [data-testid='ia-badge'], [data-testid='indeedApply']"
INDEED_APPLY_BUTTON = (
    "a[data-testid='viewjob-indeed-apply'], "
    "button[data-testid='viewjob-indeed-apply'], "
    "span.indeed-apply-status-not-applied button, "
    "button[aria-label*='Apply with Indeed'], "
    "button[aria-label*='Indeed Apply'], "
    "#indeedApplyButton, "
    "[data-testid='indeedApplyButton-test'], "
    ".jobsearch-IndeedApplyButton-buttonWrapper"
)
INDEED_NEXT_PAGE_SELECTOR = (
    "a[data-testid='pagination-page-next'], nav[role='navigation'] a[aria-label='Next Page']"
)


class IndeedJobManager(BaseJobManager):
    """Class for searching and sending applications to employers on Indeed"""

    def __init__(
        self, page: Page, linkedin_email: str, resume_anonymizer: Any, search_component: Any
    ):
        logger.info("Initializing IndeedJobManager")
        self.page = page
        self.email = linkedin_email  # reuses the same parameter name for interface compatibility
        self.resume_anonymizer = resume_anonymizer
        self.search_component = search_component
        self.llm_answerer_component = None
        self.llm_agent_component = None
        self.resume_generator_manager = None
        self.pause_checker = None
        self.jobs_no_info: List[Any] = []
        self.job_key_skills: List[str] = []
        self.interesting_jobs: List[Any] = []
        self.page_num = 0
        self.total_discovered_jobs = 0
        self.error_num = 0
        self.total_applies_num = 0
        self.applies_num = 0
        self.success_applies_num = 0
        self.resume_recommendations = ""
        self.submitted_resume_path = None
        self.applied_csv_ids: set[str] = get_applied_job_ids()
        self._card_meta: dict[str, tuple[str, bool]] = {}
        self._current_pre_resolved_resume: str | None = None
        self._last_encountered_questions: list[Any] = []
        self._use_linkedin_parity: bool = True
        self.rejected_jobs: set[str] = set()

        logger.info("IndeedJobManager successfully initialized")

    @staticmethod
    def _is_real_page(page: Any) -> bool:
        mod = getattr(type(page), "__module__", "") or ""
        return mod.startswith(("playwright", "patchright"))

    # ------------------------------------------------------------------
    # Interface methods (same signatures as LinkedIn LinkedInJobManager)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Main application loop
    # ------------------------------------------------------------------

    async def start_applying(self) -> None:
        """Main loop: iterate over all search URLs and apply to jobs"""
        # define the start time of the search
        if self.cache.last_run:
            last_run = self.cache.get_last_run_datetime()
            # if this is not the first launch - increase the time of the last search by 24 hours
            # and write it as the last search (to avoid the drift of the start time of the program)
            self.cache.last_run = (last_run + timedelta(hours=24)).isoformat()
        else:
            self.cache.update_last_run()
        result = ""
        # write recommendations for improving the resume
        self.resume_improvement_recommendations()

        # for url in search_urls:
        self.page_num = 0
        await async_pause(1, 2)

        # continue until the maximum number of applications is reached
        term_processed = 0
        switch_number = int(load_linkedin_configs().get("search", {}).get("switch_number", 50) or 50)
        while self.success_applies_num < self.max_applies_num and self.applies_num < 400:
            # Check if execution is paused
            if self.pause_checker:
                await self.pause_checker()

            # go through all pages until they are finished
            vacancies = await self.get_vacancies_from_page()
            if len(vacancies) == 0:
                if self.page_num == 0:
                    logger.warning("No vacancies found for the search query")
                # An empty location must not end the run while other configured
                # locations are still unsearched
                if await self._go_to_next_search():
                    term_processed = 0
                    continue
                break
            switched_term = False
            for vacancy in vacancies:
                if should_stop():
                    raise StopRequested("Stop requested before next job")
                # Check if execution is paused before processing each job
                if self.pause_checker:
                    await self.pause_checker()

                # Stop starting new jobs once a shutdown has been requested
                if runtime_controller.shutdown_state == ShutdownState.DRAINING:
                    logger.info("Shutdown requested — stopping before the next job")
                    result = "Shutdown"
                    break

                try:
                    result = await self.apply_job(vacancy)
                    term_processed += 1
                    if result == "Limit":
                        logger.warning("Maximum number of applications reached")
                        break
                except StopRequested:
                    raise
                except InterruptedError:
                    if should_stop():
                        raise StopRequested("Stop requested during job processing")
                    raise
                except Exception:
                    tb_str = traceback.format_exc()
                    logger.error(f"Unknown error on the page: {self.page.url}\n{tb_str}")
                    await debug_capture(self.page, "apply_loop_error")
                    # counter of repeated errors, if too many errors in a row -
                    # exit the program and send a notification
                    if self.error_num == MAX_APPLIES_NUM:
                        logger.error(f"Critical number of consecutive errors {MAX_APPLIES_NUM}")
                        result = "Error"
                        break
                    else:
                        self.error_num += 1
                    continue
                else:
                    self.error_num = 0

                if term_processed >= switch_number and self.success_applies_num < self.max_applies_num:
                    if await self._go_to_next_search():
                        logger.info(f"Switched search term after {term_processed} jobs (switch_number={switch_number})")
                        term_processed = 0
                        switched_term = True
                        break
            if switched_term:
                continue
            # break the search for vacancies if the limit is reached
            if result == "Limit" or result == "Error" or result == "Shutdown":
                break
            # go to the next page; when the pages run out, move to the next
            # configured search (one per location) before giving up
            if not await self._go_to_next_page():
                if not await self._go_to_next_search():
                    break
                term_processed = 0

        # if result == "Limit" or result == "Error":
        #     break

        logger.info(f"Applications sent: {self.success_applies_num}")
        logger.info("Ending the work.")
        await self.send_report(result)

    async def _go_to_next_search(self) -> bool:
        """Move to the next configured search URL. False when all are exhausted."""
        go_to_next_search = getattr(self.search_component, "go_to_next_search", None)
        if go_to_next_search is None:
            return False
        try:
            if not await go_to_next_search():
                return False
        except Exception as e:
            logger.warning(f"Could not start the next search: {e}")
            await debug_capture(self.page, "next_search_error")
            return False
        self.page_num = 0
        emit_event("page_changed", "Starting the next configured search", page_num=self.page_num)
        return True

    async def get_vacancies_from_page(self) -> List[Any]:
        """Return all job card elements on the current page"""
        try:
            await self.page.wait_for_selector(INDEED_JOB_CARD_SELECTOR, timeout=15000)
            await self._scroll_left_panel()
            cards = await find_elements_safely(self.page, INDEED_JOB_CARD_SELECTOR)
            if not cards:
                return []
            # Indeed's slider DOM can render multiple div.job_seen_beacon elements for
            # the same job (compact card + pre-fetched detail view). Deduplicate by
            # data-jk so each job is only processed once.
            seen_jks: set = set()
            unique_cards = []
            for card in cards:
                title_el = await find_element_safely(card, INDEED_JOB_TITLE_SELECTOR, timeout=500)
                jk = (await title_el.get_attribute("data-jk") or "") if title_el else ""
                if jk:
                    if jk in seen_jks:
                        continue
                    seen_jks.add(jk)
                unique_cards.append(card)
            self.total_discovered_jobs += len(unique_cards)
            emit_event(
                "jobs_discovered",
                f"Found {len(unique_cards)} jobs on page {self.page_num}",
                count=len(unique_cards),
                total_discovered=self.total_discovered_jobs,
                page_num=self.page_num,
            )
            return unique_cards
        except Exception as e:
            logger.warning(f"Could not find job cards: {e}")
            await debug_capture(self.page, "vacancies_not_found")
            return []

    async def _scroll_left_panel(self) -> None:
        """Scroll the full page to trigger lazy-loading of job cards"""
        await self.page.evaluate(
            """
            () => new Promise((resolve) => {
                const distance = document.body.scrollHeight;
                const durationMs = 2000;
                const startTime = performance.now();
                function step(now) {
                    const progress = Math.min((now - startTime) / durationMs, 1);
                    window.scrollTo(0, distance * progress);
                    if (progress < 1) requestAnimationFrame(step);
                    else resolve();
                }
                requestAnimationFrame(step);
            })
            """
        )
        await async_pause(1, 2)
        await self.page.evaluate("() => window.scrollTo(0, 0)")

    async def _right_pane_matches_job(self, job: Job, job_jk: str | None) -> bool:
        """Check whether the embedded desktop right pane corresponds to ``job``."""
        if not self._is_real_page(self.page):
            return True
        try:
            return bool(
                await self.page.evaluate(
                    """([jk, title]) => {
                        const url = window.location.href || '';
                        const vjkMatch = url.match(/[?&]vjk=([^&#]+)/);
                        if (jk && vjkMatch) return vjkMatch[1] === jk;
                        if (jk && window._initialData?.autoOpenTwoPaneJobKey) {
                            return window._initialData.autoOpenTwoPaneJobKey === jk;
                        }
                        const hdr = document.querySelector(
                            "h2[data-testid='jobsearch-JobInfoHeader-title'], h2.jobsearch-JobInfoHeader-title, [data-testid='simpler-jobTitle']"
                        );
                        if (hdr && title) {
                            const norm = s => s.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
                            const h = norm(hdr.innerText || hdr.textContent || '');
                            const t = norm(title);
                            if (t && h.includes(t)) return true;
                        }
                        return false;
                    }""",
                    [job_jk or "", job.job_title or ""],
                )
            )
        except Exception:
            return False

    async def _focus_card_in_list_pane(self, card: Any, job: Job, job_jk: str | None) -> bool:
        """Select a job card on the search list page so its JD renders in the right pane."""
        if not self._is_real_page(self.page):
            return False
        try:
            prev_jd = await self._extract_job_description_from_page(self.page)
            if prev_jd and len(prev_jd.strip()) >= 20 and await self._right_pane_matches_job(job, job_jk):
                return True

            title_el = await find_element_safely(card, INDEED_JOB_TITLE_SELECTOR, timeout=1500)
            if not title_el:
                return False
            await card.evaluate("el => el.scrollIntoView({block: 'center'})")
            await title_el.evaluate("el => el.click()")

            for tick in range(18):
                if should_stop():
                    raise StopRequested("Stop requested while loading job preview")
                await asyncio.sleep(0.25)
                if "/viewjob" in (self.page.url or ""):
                    # Single-pane fallback: restore search list and let caller open new_page
                    await self.page.go_back(wait_until="domcontentloaded")
                    return False
                jd_now = await self._extract_job_description_from_page(self.page)
                if not jd_now or len(jd_now.strip()) < 20:
                    continue
                if await self._right_pane_matches_job(job, job_jk):
                    if not prev_jd or jd_now != prev_jd:
                        return True
                    # After 2.5s with matching vjk, allow identical JD copy (e.g. reposted role)
                    if tick >= 10 and job_jk and f"vjk={job_jk}" in (self.page.url or ""):
                        return True
        except StopRequested:
            raise
        except Exception as e:
            logger.debug(f"Could not focus job card in list pane: {e}")
        return False

    async def _record_condition_skip(
        self,
        job: Job,
        job_jk: str | None,
        reason: str,
        detail: str,
        posted_text: str,
        evaluation: dict[str, Any],
        is_real_session: bool,
    ) -> str:
        """Record an eligibility/score mismatch in memory (session only, never skipped.yaml)."""
        self.rejected_jobs.add(job_jk or job.url)
        if is_real_session and not TEST_MODE:
            record_failed_job(
                job_id=job_jk or job.job_title,
                job_link=job.url,
                resume_tried=get_default_resume_path(),
                date_listed=estimate_date_posted(posted_text),
                reason=detail or reason,
                screenshot_name="Not Available",
            )
        await self._handle_apply_result(
            ("Skip", reason), job, evaluation=evaluation, persist_skip=False
        )
        return "Skip"

    async def apply_job(self, vacancy: Any) -> str:
        """Process a single Indeed job card"""
        evaluation = {"interest_score": None, "interest_reason": None, "skills": None}
        job = await self._extract_job_from_card(vacancy)
        if job is None:
            return "Skip"

        job_jk = extract_indeed_jk(job.url)
        if (job_jk or job.url) in getattr(self, "rejected_jobs", ()):
            logger.info(
                f"Skipping session-rejected job: {job.job_title} at {job.company_name}"
            )
            return "Skip"

        already_seen, reason = self._job_is_already_seen(job)
        if already_seen:
            logger.info(
                f"Skipping already seen job: {job.job_title} at {job.company_name} ({reason})"
            )
            await self._handle_apply_result(("Skip", reason), job, evaluation=evaluation)
            return "Skip"

        card_meta = getattr(self, "_card_meta", {}).get(job.url, ("", False))
        posted_text, has_applied_badge = card_meta
        if has_applied_badge or (job_jk and job_jk in getattr(self, "applied_csv_ids", set())):
            logger.info(
                f"Skipping already applied job (CSV/Badge): {job.job_title} at {job.company_name}"
            )
            await self._handle_apply_result(("Skip", "Already applied"), job, evaluation=evaluation)
            return "Skip"

        if self.search_component.is_job_blacklisted(job.job_title, job.company_name, job.location):
            logger.info(f"Skipping blacklisted job: {job.job_title} at {job.company_name}")
            await self._handle_apply_result(
                ("Skip", "Vacancy in the blacklist"), job, evaluation=evaluation
            )
            return "Skip"

        if job.apply_method != "easy_apply" and EASY_APPLY_ONLY_MODE and not COLLECT_INFO_MODE:
            logger.info(f"Skipping external apply job: {job.job_title} at {job.company_name}")
            await self._handle_apply_result(
                ("Skip", "External apply not allowed"),
                job,
                evaluation=evaluation,
                persist_skip=False,
            )
            return "Skip"

        minimum_job_time = time.time() + MINIMUM_WAIT_TIME_SEC
        new_page = None
        is_real = False
        is_real_search = self._is_real_page(self.page)

        async def _open_job_page() -> Any:
            nonlocal new_page, is_real
            if new_page is not None:
                return new_page
            new_page = await self.page.context.new_page()
            is_real = self._is_real_page(new_page)
            try:
                await new_page.goto(job.url, wait_until="domcontentloaded")
            except Exception as e:
                logger.warning(f"Could not open job page for {job.url}: {e}")
                try:
                    await new_page.close()
                except Exception:
                    pass
                new_page = None
                return None
            await async_pause(1, 2)
            if is_real and await IndeedAuthenticator.is_security_challenge(new_page):
                cleared = await IndeedAuthenticator(new_page).wait_for_manual_challenge_if_present()
                if not cleared:
                    return None
            return new_page

        try:
            # Prefer scraping JD & posted age from the desktop search list right pane
            # before opening any new browser tab.
            if is_real_search:
                await self._focus_card_in_list_pane(vacancy, job, job_jk)
            job.job_description = await self._extract_job_description_from_page(self.page)
            job.company_description = await self._extract_company_description(self.page)
            if is_real_search and not posted_text:
                posted_text = await self._extract_posted_text_from_page(self.page)

            # Fallback to opening job URL if embedded pane did not expose a JD
            if not job.job_description:
                opened = await _open_job_page()
                if opened is None:
                    return "Skip"
                job.job_description = await self._extract_job_description_from_page(opened)
                job.company_description = await self._extract_company_description(opened)
                if is_real and not posted_text:
                    posted_text = await self._extract_posted_text_from_page(opened)

            is_real_session = is_real_search or is_real

            if getattr(self, "_use_linkedin_parity", True):
                eligible, elig_reason, elig_detail = check_jd_eligibility(
                    job.job_description, job.company_description, posted_text
                )
                if not eligible:
                    logger.info(
                        f"Skipping ineligible job '{job.job_title}' at '{job.company_name}': "
                        f"{elig_reason} ({elig_detail})"
                    )
                    detail_msg = (
                        f"{elig_reason}: {elig_detail}"
                        if elig_detail
                        else (elig_reason or "Ineligible JD")
                    )
                    return await self._record_condition_skip(
                        job,
                        job_jk,
                        elig_reason or "Ineligible JD",
                        detail_msg,
                        posted_text,
                        evaluation,
                        is_real_session,
                    )

            if MONKEY_MODE is True and COLLECT_INFO_MODE is False:
                job_is_interesting = True
                score, reasoning = 0, "Monkey mode"
                logger.info(
                    "Monkey mode is enabled and Collect info mode is disabled, applying to all vacancies"
                )
            else:
                gate_ok, gate_score, gate_reason, gate_msg = (
                    apply_resume_score_gate(job.job_description)
                    if getattr(self, "_use_linkedin_parity", True)
                    else (True, None, None, None)
                )
                if not gate_ok:
                    score = gate_score or 0
                    reasoning = gate_msg or gate_reason or "Low resume score"
                    evaluation["interest_score"] = score
                    evaluation["interest_reason"] = reasoning
                    logger.info(
                        f"Skipping job below resume score threshold: {job.job_title} ({reasoning})"
                    )
                    return await self._record_condition_skip(
                        job,
                        job_jk,
                        gate_reason or "Low resume score",
                        reasoning,
                        posted_text,
                        evaluation,
                        is_real_session,
                    )
                if gate_score is not None:
                    job_is_interesting = True
                    score, reasoning = gate_score, f"Resume Score: {gate_score}/100"
                elif is_real_session and not ai_runtime.is_ai_enabled():
                    job_is_interesting = True
                    score, reasoning = 0, "AI disabled (passed gate by default)"
                else:
                    interest_result = self.llm_answerer_component.job_is_interesting(job.model_dump())
                    if interest_result is None:
                        apply_result = ("Error", "Error while determining if job is interesting")
                        await self._handle_apply_result(apply_result, job, evaluation=evaluation)
                        return "Error"
                    job_is_interesting, score, reasoning = interest_result

            evaluation["interest_score"] = int(score) if str(score).isdigit() else 0
            evaluation["interest_reason"] = reasoning

            if not job_is_interesting:
                logger.info(f"Skipping uninteresting job: {job.job_title} at {job.company_name}")
                return await self._record_condition_skip(
                    job, job_jk, reasoning, reasoning, posted_text, evaluation, is_real_session
                )
            # update the list of required skills for the vacancy and save job info to file
            # only if the vacancy was scored and considered interesting
            if int(score) > 0:
                extract_skills = bool(
                    load_linkedin_configs().get("secrets", {}).get("extract_job_skills_with_ai", False)
                )
                if COLLECT_INFO_MODE is True or extract_skills:
                    job.skills = self._extract_skills_from_vacancy(job)
                    self._update_skill_stat(self.job_key_skills)
                    evaluation["skills"] = self.job_key_skills
                if COLLECT_INFO_MODE is True:
                    self._save_interesting_job(job, score, reasoning)

            if COLLECT_INFO_MODE:
                logger.info(
                    "We are in the mode of collecting skill statistics or searching for "
                    "interesting jobs - do not apply to the vacancy"
                )
                return "Ok"

            self.submitted_resume_path = None
            self._last_encountered_questions = []
            self._current_pre_resolved_resume = None

            if not EASY_APPLY_ONLY_MODE and job.apply_method == "external" and TEST_MODE:
                apply_result = ("Skip", "Test mode")
            else:
                if TEST_MODE or not ai_runtime.is_ai_enabled():
                    self.llm_answerer_component.set_job(job.model_dump(), is_test=True)
                else:
                    try:
                        self.llm_answerer_component.set_job(job.model_dump())
                    except Exception as exc:
                        if ai_runtime.is_key_failover_error(exc):
                            ai_runtime.disable_ai(exc, source="Indeed JD summary")
                        else:
                            logger.warning(
                                f"Falling back to non-AI job context for {job.job_title}: {exc}"
                            )
                        self.llm_answerer_component.set_job(job.model_dump(), is_test=True)

                opened = await _open_job_page()
                if opened is None:
                    return "Skip"

                if (is_real or is_real_search) and (
                    EASY_APPLY_ONLY_MODE or job.apply_method == "easy_apply"
                ):
                    self._current_pre_resolved_resume = await asyncio.to_thread(
                        prepare_tailored_resume_pdf,
                        job.job_title,
                        job.company_name,
                        job.location,
                        job.job_description,
                        cancel_check=should_stop,
                    )
                    logger.info(
                        f"Tailored resume ready for '{job.job_title}': {self._current_pre_resolved_resume}. Starting application..."
                    )
                    if not TEST_MODE:
                        maybe_send_jd_outreach(
                            job.job_description, self._current_pre_resolved_resume
                        )

                if not EASY_APPLY_ONLY_MODE and job.apply_method == "external":
                    apply_url = await self._get_button_link(opened)
                    apply_result = await self.llm_agent_component.apply_to_job(apply_url)
                else:
                    apply_result = await self.easy_apply(job, opened, skip_navigation=True)

            if self.submitted_resume_path:
                evaluation["submitted_resume_path"] = self.submitted_resume_path
            result, reason = apply_result
            if result == "Skip" and reason.startswith("Could not"):
                self._collect_job_info(job.job_title, job.company_name, job.url, reason)
            await self._handle_apply_result(apply_result, job, evaluation=evaluation)

            if is_real and not TEST_MODE and new_page is not None:
                date_posted = estimate_date_posted(posted_text)
                exp_req = extract_years_of_experience(job.job_description or "")
                res_used = (
                    self.submitted_resume_path
                    or self._current_pre_resolved_resume
                    or get_default_resume_path()
                )
                if result == "Success":
                    if job_jk:
                        self.applied_csv_ids.add(job_jk)
                    record_applied_job(
                        job_id=job_jk or sanitize_text(job.job_title),
                        title=job.job_title,
                        company=job.company_name,
                        work_location=job.location,
                        work_style="Indeed",
                        description=job.job_description,
                        experience_required=exp_req,
                        skills=job.skills or "Unknown",
                        resume_path=str(res_used),
                        date_posted=date_posted,
                        job_link=job.url,
                        application_link="Indeed Easy Apply" if job.apply_method == "easy_apply" else "External",
                        questions_found=getattr(self, "_last_encountered_questions", []),
                    )
                else:
                    shot = await capture_indeed_screenshot(new_page, job_jk or "fail", result)
                    record_failed_job(
                        job_id=job_jk or sanitize_text(job.job_title),
                        job_link=job.url,
                        resume_tried=str(res_used),
                        date_listed=date_posted,
                        reason=reason or result,
                        screenshot_name=shot,
                    )

            if self.success_applies_num >= self.max_applies_num:
                logger.info(
                    f"The maximum number of applications has been reached: "
                    f"{self.success_applies_num}/{self.max_applies_num}"
                )
                return "Limit"
            return result
        finally:
            self._current_pre_resolved_resume = None
            if new_page is not None:
                # if the page was processed faster than the minimum time -
                # wait until this time is over
                time_left = int(minimum_job_time - time.time())
                if time_left > 0 and not should_stop():
                    wait_sec = min(time_left, 4.5)
                    await async_pause(max(0.5, wait_sec - 1.0), wait_sec)
                for cleanup_fn in (new_page.close, self.page.bring_to_front):
                    try:
                        await cleanup_fn()
                    except Exception:
                        pass

    async def easy_apply(
        self, job: Job, page: Any = None, skip_navigation: bool = False
    ) -> Tuple[str, str]:
        """Delegate application to IndeedEasyApplier"""
        easy_applier_component = IndeedEasyApplier(
            page=page,
            gpt_answerer=self.llm_answerer_component,
            resume_anonymizer=self.resume_anonymizer,
            resume_generator_manager=self.resume_generator_manager,
            pause_checker=self.pause_checker,
            answers_file=Path(OUTPUT_DIR_INDEED) / "answers.yaml",
            resume_dir=Path(RESUME_DIR),
            cover_letter_dir=Path(COVER_LETTER_DIR),
            test_mode=TEST_MODE,
            use_linkedin_bridge=getattr(self, "_use_linkedin_parity", True),
            pre_resolved_resume_path=getattr(self, "_current_pre_resolved_resume", None),
        )
        if skip_navigation:
            apply_result, self.submitted_resume_path = await easy_applier_component.apply_to_job(
                job, skip_navigation=True
            )
        else:
            apply_result, self.submitted_resume_path = await easy_applier_component.apply_to_job(job)
        self._last_encountered_questions = getattr(easy_applier_component, "encountered_questions", [])
        return apply_result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _extract_job_from_card(self, card: Any) -> Job | None:
        """Extract job data from an Indeed job card element"""
        try:
            title_el = await find_element_safely(card, INDEED_JOB_TITLE_SELECTOR, timeout=3000)
            if not title_el:
                return None
            title = await title_el.text_content() or ""
            jk = await title_el.get_attribute("data-jk") or ""
            if "0123456789abcdef" in "".join(sorted(jk)):
                return None
            if jk:
                # Use the canonical viewjob URL so the same job always maps to the
                # same URL regardless of which slider item or tracking URL was found.
                job_url = f"{indeed_base_url()}/viewjob?jk={jk}"
            else:
                job_url_path = await title_el.get_attribute("href") or ""
                job_url = (
                    job_url_path
                    if job_url_path.startswith("http")
                    else f"{indeed_base_url()}{job_url_path}"
                )

            company_el = await find_element_safely(card, INDEED_COMPANY_SELECTOR, timeout=3000)
            company = (await company_el.text_content() or "") if company_el else ""

            location_el = await find_element_safely(card, INDEED_LOCATION_SELECTOR, timeout=3000)
            location = (await location_el.text_content() or "") if location_el else ""

            logger.debug(
                f"Extracting job card: title={title!r}, company={company!r}, url={job_url!r}"
            )

            # Check for easy apply badge on card first (avoid slow click-and-wait during card extraction)
            easy_apply_badge = await find_element_safely(
                card, INDEED_EASY_APPLY_BADGE, timeout=500
            )
            has_easy_text = False
            if not easy_apply_badge and self._is_real_page(card):
                try:
                    ctext = await card.inner_text()
                    if isinstance(ctext, str) and "easily apply" in ctext.lower():
                        has_easy_text = True
                except Exception:
                    pass

            if easy_apply_badge or has_easy_text or EASY_APPLY_ONLY_MODE:
                apply_method = "easy_apply"
            else:
                apply_method = "external"

            posted_text, has_applied_badge = await self._extract_card_extra(card)
            job = Job(
                job_title=sanitize_text(title),
                company_name=sanitize_text(company),
                location=sanitize_text(location),
                url=job_url,
                apply_method=apply_method,
            )
            self._card_meta = getattr(self, "_card_meta", {})
            self._card_meta[job.url] = (posted_text, has_applied_badge)
            return job

        except Exception as e:
            if should_stop() or "has been closed" in str(e):
                return None
            logger.warning(
                f"Failed to extract job from card: title={title!r}, company={company!r}, url={job_url!r} — {e}"
                if "title" in dir()
                else f"Failed to extract job from card (title not yet parsed): {e}"
            )
            await debug_capture(self.page, "extract_job_error")
            return None

    async def _extract_job_description_from_page(self, page: Any) -> str:
        """Extract job description text from an Indeed job detail page"""
        selectors = [
            "#jobDescriptionText",
            "[data-testid='jobsearch-jobDescriptionText']",
            ".jobsearch-jobDescriptionText",
            "#job-description",
            "[data-testid='viewjob-job-content']",
        ]
        try:
            el = await find_element_safely(page, ", ".join(selectors), timeout=3000)
            if el:
                text = None
                if self._is_real_page(page):
                    try:
                        text = await el.inner_text()
                    except Exception:
                        text = None
                if not text:
                    text = await el.text_content()
                if text:
                    return text.strip()
        except Exception:
            pass
        logger.debug("Could not extract job description from Indeed page")
        return ""

    async def _extract_company_description(self, page: Any) -> str:
        """Extract company description from the 'About the company' section of an Indeed job page"""
        selectors = [
            "[data-testid='jobsearch-CompanyInfoContainer']",
            "#companyInfo",
            ".jobsearch-CompanyInfoWithoutHeaderImage",
            ".jobsearch-CompanyInfoContainer",
            "[data-testid='companyInfo']",
            "[data-testid='inlineHeader-companyName']",
        ]
        try:
            el = await find_element_safely(page, ", ".join(selectors), timeout=2000)
            if el:
                text = await el.text_content()
                if text:
                    text = text.strip()
                    if len(text) > 20:
                        logger.debug("Found company description on Indeed page")
                        return text
        except Exception:
            pass
        logger.debug("Could not extract company description from Indeed page")
        return ""

    async def _extract_card_extra(self, card: Any) -> tuple[str, bool]:
        """Safely read posted-age label and Applied badge from a job card without disturbing mocks."""
        posted_text = ""
        already_applied = False
        if not self._is_real_page(card):
            return posted_text, already_applied
        try:
            loc_fn = getattr(card, "locator", None)
            if not callable(loc_fn):
                return posted_text, already_applied
            date_loc = loc_fn("span[data-testid='myJobsStateDate'], span.date, [data-testid='job-age']")
            cnt = await date_loc.count()
            if isinstance(cnt, int) and cnt > 0:
                raw = await date_loc.first.text_content()
                if isinstance(raw, str):
                    posted_text = raw.strip()
            app_loc = loc_fn("span.indeed-apply-status-applied, [data-testid='applied-snippet'], :text-is('Applied')")
            acnt = await app_loc.count()
            if isinstance(acnt, int) and acnt > 0:
                already_applied = True
        except Exception:
            pass
        return posted_text, already_applied

    async def _extract_posted_text_from_page(self, page: Any) -> str:
        for sel in ("[data-testid='myJobsStateDate']", "[data-testid='job-age']", ".jobsearch-HiringInsights-entry--text"):
            try:
                el = await find_element_safely(page, sel, timeout=800)
                if el:
                    txt = await el.text_content()
                    if isinstance(txt, str) and txt.strip():
                        return txt.strip()
            except Exception:
                continue
        if self._is_real_page(page):
            try:
                age = await page.evaluate(
                    "() => window._rootProps?.preloadedVJData?.hiringInsightsModel?.age || ''"
                )
                if isinstance(age, str) and age.strip():
                    return age.strip()
            except Exception:
                pass
        return ""

    async def _get_button_link(self, page: Any) -> str:
        """Click the external apply button and return the URL it opens in a new tab."""
        apply_selectors = [
            "a[data-testid='viewjob-apply']",
            "button[data-testid='viewjob-apply']",
            "a[href*='/applystart']",
            "a[data-testid='applyButton']",
            "a[aria-label*='Apply on']",
            ".ia-BasePage-applyButton",
            "a[data-jk][href*='apply']",
        ]
        for selector in apply_selectors:
            try:
                elements = await find_elements_safely(page, selector)
                for button in elements:
                    if not (await button.is_visible() and await button.is_enabled()):
                        continue
                    async with page.context.expect_page() as new_page_info:
                        await button.click(timeout=3000)
                    new_page = await new_page_info.value
                    await async_pause()
                    link = new_page.url
                    await new_page.close()
                    logger.debug(f"External apply link obtained: {link}")
                    return link
            except Exception as e:
                logger.debug(f"Failed to get external apply link with selector {selector!r}: {e}")
                continue
        logger.warning("Could not find external apply button on Indeed job page")
        return ""

    async def _dismiss_overlays(self) -> None:
        """Dismiss Indeed overlay portals that intercept clicks"""
        for selector in ["ifl-portal", "div.gnav-hovbc7"]:
            try:
                count = await self.page.locator(selector).count()
                if count > 0:
                    await self.page.evaluate(
                        f"document.querySelectorAll('{selector}').forEach(el => el.remove())"
                    )
                    logger.debug(f"Dismissed {count} overlay(s) matching '{selector}'")
            except Exception:
                pass

    async def _current_page_signature(self) -> str:
        """URL plus the first job card id, to tell one result page from the next"""
        try:
            first_card = await find_element_safely(
                self.page, INDEED_JOB_TITLE_SELECTOR, timeout=5000
            )
            first_jk = (await first_card.get_attribute("data-jk") or "") if first_card else ""
        except Exception as e:
            logger.debug(f"Could not read the first job card of the page: {e}")
            first_jk = ""
        return f"{self.page.url}|{first_jk}"

    async def _go_to_next_page(self) -> bool:
        """Click next page button. Returns True if navigated, False if last page reached."""
        try:
            next_btn = await find_element_safely(self.page, INDEED_NEXT_PAGE_SELECTOR, timeout=5000)
            if not next_btn:
                logger.info("No next page button found - reached last page")
                return False
            signature_before = await self._current_page_signature()
            await self._dismiss_overlays()
            clicked = await safe_click(self.page, INDEED_NEXT_PAGE_SELECTOR, timeout=5000)
            if not clicked:
                logger.debug("Normal click failed, retrying with force")
                await next_btn.click(force=True, timeout=5000)
            try:
                await self.page.wait_for_load_state("domcontentloaded")
            except Exception as e:
                # The click often advances the page without the load state ever
                # settling; the page content itself decides whether it worked
                logger.debug(f"Waiting for the next page to load timed out: {e}")
            await async_pause(1, 2)
            if await self._current_page_signature() == signature_before:
                logger.warning("Next page button was clicked but the results did not change")
                await debug_capture(self.page, "next_page_not_changed")
                return False
            self.page_num += 1
            logger.info(f"Moved to page {self.page_num + 1}")
            emit_event(
                "page_changed", f"Moving to page {self.page_num + 1}", page_num=self.page_num
            )
            return True
        except Exception as e:
            logger.warning(f"Could not navigate to next page: {e}")
            await debug_capture(self.page, "next_page_error")
            return False
