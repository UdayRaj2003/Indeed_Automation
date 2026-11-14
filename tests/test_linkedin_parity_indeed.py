"""End-to-end feature parity tests for Indeed bot integration with Linkedin_Test."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from config.app_config import INDEED_DOMAIN, INDEED_USE_PERSISTENT_PROFILE, JOB_SITE, LLM_MODEL_TYPE
from config.constants import INDEED_CHROME_PROFILE_DIR
from src.job_manager.easy_applier import NoInfoException
from src.job_manager.indeed.authenticator_indeed import IndeedAuthenticator
from src.job_manager.indeed.easy_applier_indeed import IndeedEasyApplier
from src.job_manager.indeed.linkedin_bridge import (
    APPLIED_CSV_FIELDS,
    FAILED_CSV_FIELDS,
    AIKeyRuntime,
    answer_from_linkedin_data,
    apply_resume_score_gate,
    check_jd_eligibility,
    convert_to_lakhs,
    estimate_date_posted,
    extract_indeed_jk,
    extract_years_of_experience,
    get_applied_job_ids,
    get_saved_answer,
    load_linkedin_configs,
    maybe_send_jd_outreach,
    parse_job_age_minutes,
    prepare_tailored_resume_pdf,
    record_applied_job,
    record_failed_job,
    upsert_saved_answers,
)
from src.job_manager.indeed.search_customizer_indeed import IndeedSearchCustomizer


class TestLinkedInConfigAndSavedData:
    def test_loads_real_linkedin_configs_and_saved_answers(self):
        cfgs = load_linkedin_configs()
        assert cfgs["personals"]["current_city"] == "Indore"
        assert cfgs["personals"]["zipcode"] == "452003"
        assert len(cfgs["secrets"]["llm_api_keys"]) >= 1
        assert 1 <= int(cfgs["settings"]["resume_score_threshold"]) <= 100
        assert cfgs["settings"]["recent_job_max_age_minutes"] == 4320
        assert len(cfgs["search"]["search_terms"]) >= 1

        # Read-only lookup against shared Linkedin_Test/config/saved_answers.json
        assert get_saved_answer("city") is not None

    def test_app_config_uses_in_indeed_and_persistent_profile(self):
        assert JOB_SITE == "indeed"
        assert INDEED_DOMAIN == "in.indeed.com"
        assert INDEED_USE_PERSISTENT_PROFILE is True
        assert INDEED_CHROME_PROFILE_DIR == "browser_session/indeed_chrome_profile"
        assert LLM_MODEL_TYPE in ("openrouter", "openai_compatible")

    def test_saved_answers_upsert_and_guards(self, tmp_path):
        store = tmp_path / "saved_answers.json"
        written = upsert_saved_answers(
            [
                {"label": "Phone Country Code", "answer": "91", "type": "select"},
                {"label": "Expected CTC", "answer": "Select an option", "type": "select"},
                {"label": "Month of From", "answer": "Yes", "type": "select"},
                {"label": "Custom Question?", "answer": "42", "type": "text"},
            ],
            force_overwrite=True,
            path=store,
        )
        assert written == 1
        assert get_saved_answer("Custom Question", path=store) == "42"
        # Existing answer is not clobbered when force_overwrite=False
        assert (
            upsert_saved_answers(
                [{"label": "Custom Question", "answer": "99", "type": "text"}],
                force_overwrite=False,
                path=store,
            )
            == 0
        )
        assert get_saved_answer("Custom Question", path=store) == "42"
        # Fuzzy similarity match (>= 0.7 overlap)
        assert get_saved_answer("custom questions", path=store) == "42"

    def test_deterministic_answers_from_linkedin_data(self, tmp_path):
        store = tmp_path / "empty_saved.json"
        assert answer_from_linkedin_data("Expected salary per annum", saved_answers_file=store) == "600000"
        assert answer_from_linkedin_data("Monthly salary expectation", saved_answers_file=store) == "50000.0"
        assert answer_from_linkedin_data("Expected CTC in Lakhs", saved_answers_file=store) == "6.0"
        assert answer_from_linkedin_data("Notice period in days", saved_answers_file=store) == "0"
        assert answer_from_linkedin_data("Current City", saved_answers_file=store) == "Indore"
        assert answer_from_linkedin_data("Postal / Zip Code", saved_answers_file=store) == "452003"
        assert answer_from_linkedin_data(
            "Will you now or in the future require visa sponsorship?",
            options=["Yes", "No"],
            question_type="radio",
            saved_answers_file=store,
        ) == "No"
        assert convert_to_lakhs("600000") == "6.00"


class TestAIKeyRuntimeAndGates:
    def test_key_failover_and_independent_resume_engine(self):
        rt = AIKeyRuntime()
        rt.set_keys(["k1", "k2"])
        assert rt.current_api_key() == "k1"
        assert rt.is_ai_enabled() is True
        assert rt.is_resume_engine_enabled() is True

        assert rt.try_rotate_or_disable("Error 429 Too Many Requests") is True
        assert rt.current_api_key() == "k2"
        assert rt.is_ai_enabled() is True

        # Last key hits 429 -> Answer AI disables, Resume Engine stays enabled
        assert rt.try_rotate_or_disable("429 quota exceeded") is False
        assert rt.is_ai_enabled() is False
        assert rt.is_resume_engine_enabled() is True

        # Independent Resume Engine disablement
        rt.reload()
        assert rt.disable_resume_engine("429 rate limit") is True
        assert rt.is_resume_engine_enabled() is False
        assert rt.is_ai_enabled() is True

    def test_recent_job_age_and_jd_eligibility(self):
        assert parse_job_age_minutes("Just posted") == 0
        assert parse_job_age_minutes("Today") == 0
        assert parse_job_age_minutes("2 days ago") == 2880
        assert parse_job_age_minutes("30+ days ago") == 31 * 1440
        assert extract_years_of_experience("Minimum 3+ years of Python experience") == 3

        # Fresh JD with matching experience passes
        ok, reason, _ = check_jd_eligibility(
            "Looking for a Python Backend Developer with 1 year experience.",
            company_description="Product company",
            time_posted_text="Just posted",
        )
        assert ok is True and reason is None

        # Stale job > 4320 min rejected
        ok, reason, _ = check_jd_eligibility(
            "Python Developer", time_posted_text="10 days ago"
        )
        assert ok is False and "recent_job_max_age_minutes" in reason

        # High experience rejected (current_experience=1)
        ok, reason, _ = check_jd_eligibility(
            "Requires 5 years of experience in backend systems.",
            time_posted_text="Today",
        )
        assert ok is False and "Required experience" in reason

        # Security clearance (polygraph) rejected
        ok, reason, _ = check_jd_eligibility(
            "Requires polygraph examination before start.",
            time_posted_text="Today",
        )
        assert ok is False and "Security clearance" in reason

    def test_resume_score_gate_threshold_and_fail_open(self):
        # Short JD passes by default without calling scorer
        called = {"n": 0}

        def fail_if_called(_):
            called["n"] += 1
            return '{"score": 10}'

        ok, score, _, _ = apply_resume_score_gate("Too short", scorer_fn=fail_if_called)
        assert ok is True and score is None and called["n"] == 0

        threshold = int(load_linkedin_configs()["settings"].get("resume_score_threshold", 80))
        low_score = max(0, threshold - 15)
        high_score = min(100, threshold + 15)
        long_jd = "Python backend developer working on FastAPI REST APIs SQL PostgreSQL Docker Git " * 3
        ok, score, reason, _ = apply_resume_score_gate(
            long_jd, scorer_fn=lambda _: json.dumps({"score": low_score})
        )
        assert ok is False and score == low_score and reason == "Low resume score"

        ok, score, reason, _ = apply_resume_score_gate(
            long_jd, scorer_fn=lambda _: json.dumps({"score": high_score})
        )
        assert ok is True and score == high_score and reason is None

        # Unparseable or scorer error passes by default
        ok, score, _, _ = apply_resume_score_gate(long_jd, scorer_fn=lambda _: "cannot parse")
        assert ok is True and score is None

        def boom(_):
            raise RuntimeError("transient error")

        ok, score, _, _ = apply_resume_score_gate(long_jd, scorer_fn=boom)
        assert ok is True and score is None


class TestResumeOutreachAndCsvHistory:
    def test_prepare_tailored_resume_success_and_fallback(self, tmp_path):
        fake_pdf = tmp_path / "tailored.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4\n")
        long_jd = "Python Developer with Flask FastAPI SQL Docker REST APIs experience " * 4

        class _Proc:
            def __init__(self, json_out: Path):
                self.stdin = MagicMock()
                self.stdout = MagicMock()
                self.stderr = MagicMock()
                self._json_out = json_out

            def poll(self):
                self._json_out.write_text(
                    json.dumps({"ok": True, "success": True, "pdf_path": str(fake_pdf)}),
                    encoding="utf-8",
                )
                return 0

        def fake_popen(cmd, **_kwargs):
            return _Proc(Path(cmd[5]))

        with patch("src.job_manager.indeed.linkedin_bridge.subprocess.Popen", side_effect=fake_popen):
            resolved = prepare_tailored_resume_pdf("Python Dev", "Acme", "Indore", long_jd)
        assert Path(resolved).resolve() == fake_pdf.resolve()

        # Short JD falls back to configured default_resume_path
        default_pdf = prepare_tailored_resume_pdf("Python Dev", "Acme", "Indore", "short")
        assert default_pdf.lower().endswith(".pdf")

    def test_prepare_tailored_resume_streams_worker_progress(self, tmp_path):
        import io
        from src.job_manager.indeed import linkedin_bridge as lb

        fake_pdf = tmp_path / "tailored.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4")
        long_jd = "Python backend microservices Docker Kubernetes SQL testing APIs " * 5

        class _Proc:
            def __init__(self, json_out: Path):
                self._json_out = json_out
                self.stdin = io.StringIO()
                self.stdout = io.StringIO("Stage 1: tailoring\nStage 2: compiling PDF\n")
                self.stderr = io.StringIO("")

            def poll(self):
                self._json_out.write_text(
                    json.dumps({"ok": True, "success": True, "pdf_path": str(fake_pdf)}),
                    encoding="utf-8",
                )
                return 0

        logged: list[str] = []
        with (
            patch("src.job_manager.indeed.linkedin_bridge.subprocess.Popen", side_effect=lambda cmd, **_kw: _Proc(Path(cmd[5]))),
            patch.object(lb.logger, "info", side_effect=lambda msg, *_a, **_kw: logged.append(str(msg))),
        ):
            prepare_tailored_resume_pdf("Python Dev", "Acme", "Indore", long_jd)

        assert any("[RenderCV] Stage 1: tailoring" in m for m in logged)
        assert any("[RenderCV] Stage 2: compiling PDF" in m for m in logged)

    def test_maybe_send_jd_outreach_triggers_on_email(self, tmp_path):
        mock_mod = MagicMock()
        mock_gen = MagicMock()
        mock_gen.emails_in_text.return_value = ["hr@example.com"]
        pdf = tmp_path / "r.pdf"
        pdf.write_bytes(b"%PDF")

        class _ImmediateThread:
            def __init__(self, target=None, **_kwargs):
                self._target = target

            def start(self):
                self._target()

        with (
            patch.dict(sys.modules, {"ai_outreach": mock_mod, "ai_outreach.generator": mock_gen}),
            patch("src.job_manager.indeed.linkedin_bridge.threading.Thread", _ImmediateThread),
        ):
            maybe_send_jd_outreach("Send resume to hr@example.com ASAP", str(pdf))
        mock_mod.send_outreach.assert_called_once()

    def test_csv_history_schema_and_deduplication(self, tmp_path):
        applied = tmp_path / "applied.csv"
        failed = tmp_path / "failed.csv"
        record_applied_job(
            job_id="jk123",
            title="Python Developer",
            company="TechCo",
            work_location="Indore",
            work_style="Indeed",
            description="Backend role",
            experience_required=1,
            skills=["Python"],
            resume_path="C:/resumes/test_resume.pdf",
            date_posted=estimate_date_posted("1 day ago"),
            job_link="https://in.indeed.com/viewjob?jk=jk123",
            csv_path=applied,
        )
        assert get_applied_job_ids(applied) == {"jk123"}
        assert extract_indeed_jk("https://in.indeed.com/viewjob?jk=jk123&from=serp") == "jk123"
        with applied.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        assert list(rows[0].keys()) == APPLIED_CSV_FIELDS

        record_failed_job(
            job_id="jk999",
            job_link="https://in.indeed.com/viewjob?jk=jk999",
            resume_tried="C:/resumes/test_resume.pdf",
            date_listed="Unknown",
            reason="Low resume score",
            screenshot_name="jk999.png",
            csv_path=failed,
        )
        with failed.open("r", encoding="utf-8", newline="") as f:
            frows = list(csv.DictReader(f))
        assert list(frows[0].keys()) == FAILED_CSV_FIELDS


class TestSearchCustomizerAuthEasyApplierAndUI:
    def test_per_term_search_urls_24h_sort_date(self):
        sc = IndeedSearchCustomizer(MagicMock())
        sc.set_advanced_search_params(
            {
                "positions": ["Python Developer", "SDET"],
                "locations": ["India"],
                "remote": True,
                "hybrid": True,
                "onsite": True,
                "date": {"all_time": False, "month": False, "week": False, "24_hours": True},
            }
        )
        sc.sort_by_date = True
        urls = sc.build_per_term_search_urls(randomize=False)
        assert len(urls) == 2
        assert all(u.startswith("https://in.indeed.com/jobs?") for u in urls)
        assert all("fromage=1" in u and "sort=date" in u for u in urls)
        assert all("sc=0kf" not in u for u in urls)  # all 3 work styles enabled -> no narrowing

    @pytest.mark.asyncio
    async def test_authenticator_detects_security_challenge(self):
        page = AsyncMock()
        page.url = "https://in.indeed.com/challenge?ts=123"
        assert await IndeedAuthenticator.is_security_challenge(page) is True

    @pytest.mark.asyncio
    async def test_authenticator_aborts_immediately_on_stop(self):
        from src.dashboard.runtime import StopRequested

        page = AsyncMock()
        page.url = "https://secure.indeed.com/auth"
        auth = IndeedAuthenticator(page=page)
        with (
            patch("src.job_manager.indeed.authenticator_indeed.should_stop", return_value=True),
            pytest.raises(StopRequested),
        ):
            await auth.check_login_success(max_attempts=10)

    @pytest.mark.asyncio
    async def test_authenticator_checks_auth_url_errors(self):
        page = AsyncMock()
        page.url = "https://secure.indeed.com/auth"
        err_loc = AsyncMock()
        err_loc.count.return_value = 1
        err_loc.first.text_content.return_value = "Invalid verification code"
        page.locator = MagicMock(return_value=err_loc)
        auth = IndeedAuthenticator(page=page)
        with patch("src.job_manager.indeed.authenticator_indeed.should_stop", return_value=False):
            assert await auth.check_login_success(max_attempts=2) is False

    def test_selectors_support_2026_in_indeed_dom(self):
        from src.job_manager.indeed.easy_applier_indeed import INDEED_APPLY_BUTTON_SELECTOR
        from src.job_manager.indeed.job_manager_indeed import (
            INDEED_APPLY_BUTTON,
            INDEED_JOB_TITLE_SELECTOR,
        )

        assert "a.jcs-JobTitle" in INDEED_JOB_TITLE_SELECTOR
        assert "viewjob-indeed-apply" in INDEED_APPLY_BUTTON
        assert "viewjob-indeed-apply" in INDEED_APPLY_BUTTON_SELECTOR

    def test_indeed_email_override_used_in_answers_and_resume(self, tmp_path, monkeypatch):
        import os
        from src.job_manager.indeed.linkedin_bridge import (
            build_resume_from_linkedin,
            get_indeed_email,
        )

        test_email = "candidate@example.com"
        monkeypatch.setenv("indeed_email", test_email)
        monkeypatch.setattr("src.job_manager.indeed.linkedin_bridge.get_indeed_email", lambda *args, **kwargs: test_email)
        assert get_indeed_email() == test_email
        ans = answer_from_linkedin_data("Email Address", saved_answers_file=tmp_path / "s.json")
        assert ans == test_email
        _, structured = build_resume_from_linkedin()
        assert structured["personal_information"]["email"] == test_email

    @pytest.mark.asyncio
    async def test_browser_utils_uses_native_viewport_and_sandbox_for_indeed(self, tmp_path):
        from src.utils import browser_utils

        captured = {}
        fake_page = MagicMock()
        fake_ctx = MagicMock()
        fake_ctx.pages = [fake_page]
        fake_ctx.on = MagicMock()

        async def fake_launch_persistent(**kwargs):
            captured.update(kwargs)
            return fake_ctx

        fake_pw = MagicMock()
        fake_pw.chromium.launch_persistent_context = AsyncMock(side_effect=fake_launch_persistent)

        fake_cm = MagicMock()
        fake_cm.start = AsyncMock(return_value=fake_pw)

        with (
            patch.object(browser_utils, "JOB_SITE", "indeed"),
            patch.object(browser_utils, "async_playwright", return_value=fake_cm),
            patch.object(browser_utils, "INDEED_CHROME_PROFILE_DIR", str(tmp_path / "prof")),
            patch.object(browser_utils, "ensure_playwright_profile"),
        ):
            browser, ctx, page = await browser_utils.create_playwright_browser()

        assert browser is None and ctx is fake_ctx and page is fake_page
        assert captured.get("no_viewport") is True
        assert captured.get("chromium_sandbox") is True
        assert "--start-maximized" in captured.get("args", [])
        assert "viewport" not in captured
        assert not any("--window-size=1920,1080" in a for a in captured.get("args", []))

    @pytest.mark.asyncio
    async def test_easy_applier_bridge_answers_and_saves_new_ai_answer(self, tmp_path):
        store = tmp_path / "saved.json"
        applier = IndeedEasyApplier(
            page=MagicMock(),
            gpt_answerer=MagicMock(),
            resume_anonymizer=MagicMock(deanonymize_text=lambda s: s),
            resume_generator_manager=MagicMock(),
            pause_checker=None,
            answers_file=tmp_path / "answers.yaml",
            resume_dir=tmp_path,
            cover_letter_dir=tmp_path,
            test_mode=True,
            use_linkedin_bridge=True,
            saved_answers_path=store,
        )
        applier.gpt_answerer.answer_question_numeric = MagicMock(return_value="999")
        # City resolves deterministically from LinkedIn config -> GPTAnswerer not called
        ans_city = await applier._resolve_question_answer(
            "Current City", "text", lambda: applier.gpt_answerer.answer_question_numeric()
        )
        assert ans_city == "Indore"
        applier.gpt_answerer.answer_question_numeric.assert_not_called()

        # Unknown question calls AI and saves into saved_answers.json
        applier.gpt_answerer.answer_question_textual_wide_range = MagicMock(
            return_value="Yes, comfortable"
        )
        ans_new = await applier._resolve_question_answer(
            "Comfortable with on-call rotation?",
            "text",
            lambda: applier.gpt_answerer.answer_question_textual_wide_range(),
        )
        assert ans_new == "Yes, comfortable"
        assert get_saved_answer("Comfortable with on-call rotation?", path=store) == "Yes, comfortable"

        # AI failure raises NoInfoException cleanly in bridge mode
        def fail_ai():
            raise RuntimeError("429 exhausted")

        with pytest.raises(NoInfoException):
            await applier._resolve_question_answer("Totally unknown question?", "text", fail_ai)

    def test_applied_jobs_web_ui_endpoints(self, tmp_path, monkeypatch):
        import app as web_app

        csv_file = tmp_path / "applied.csv"
        record_applied_job(
            job_id="ui123",
            title="Python Engineer",
            company="OpenTest",
            work_location="Indore",
            work_style="Indeed",
            description="JD",
            experience_required=1,
            skills=["Python"],
            resume_path="C:/resumes/test_resume.pdf",
            date_posted="2026-11-11 10:00:00",
            job_link="https://in.indeed.com/viewjob?jk=ui123",
            csv_path=csv_file,
        )
        monkeypatch.setattr(web_app, "CSV_PATH", csv_file)

        client = TestClient(web_app.app)
        home_resp = client.get("/")
        assert home_resp.status_code == 200
        assert "Indeed Applied Jobs History" in home_resp.text

        list_resp = client.get("/applied-jobs")
        assert list_resp.status_code == 200
        data = list_resp.json()
        assert len(data) == 1 and data[0]["Job_ID"] == "ui123"

        put_resp = client.put("/applied-jobs/ui123")
        assert put_resp.status_code == 200

    @pytest.mark.asyncio
    async def test_security_challenge_ignores_invisible_smartapply_recaptcha_but_flags_bframe(self):
        class _FakeFrameEl:
            def __init__(self, visible: bool, box: dict | None, src: str):
                self._visible = visible
                self._box = box
                self._src = src

            async def is_visible(self):
                return self._visible

            async def bounding_box(self):
                return self._box

            async def get_attribute(self, name: str):
                return self._src if name == "src" else None

        class _FakeLoc:
            def __init__(self, items):
                self._items = items

            async def count(self):
                return len(self._items)

            def nth(self, idx: int):
                return self._items[idx]

        class _FakePlaywrightPage:
            __module__ = "playwright.async_api._generated"
            url = "https://smartapply.indeed.com/beta/indeedapply/form/resume"

            def __init__(self, frames_by_sel: dict[str, list[_FakeFrameEl]]):
                self._frames = frames_by_sel

            async def title(self):
                return "Indeed Apply"

            def locator(self, sel: str):
                return _FakeLoc(self._frames.get(sel, []))

        passive_page = _FakePlaywrightPage(
            {
                "iframe[src*='recaptcha']": [
                    _FakeFrameEl(
                        True,
                        {"x": 0, "y": 0, "width": 256, "height": 60},
                        "https://www.google.com/recaptcha/api2/anchor?size=invisible&k=123",
                    )
                ],
                "iframe[src*='hcaptcha']": [
                    _FakeFrameEl(
                        False,
                        None,
                        "https://newassets.hcaptcha.com/captcha/v1/hcaptcha.html#size=invisible",
                    )
                ],
            }
        )
        assert await IndeedAuthenticator.is_security_challenge(passive_page) is False

        bframe_page = _FakePlaywrightPage(
            {
                "iframe[src*='recaptcha']": [
                    _FakeFrameEl(
                        True,
                        {"x": 100, "y": 100, "width": 400, "height": 580},
                        "https://www.google.com/recaptcha/api2/bframe?hl=en&k=123",
                    )
                ]
            }
        )
        assert await IndeedAuthenticator.is_security_challenge(bframe_page) is True

    @pytest.mark.asyncio
    async def test_extract_job_description_uses_inner_text_on_real_page(self):
        from src.job_manager.indeed.job_manager_indeed import IndeedJobManager

        class _RealLikePage:
            __module__ = "patchright.async_api._generated"

        el = AsyncMock()
        el.inner_text = AsyncMock(return_value="  Python Developer clean text  ")
        el.text_content = AsyncMock(return_value="@layer htmlContent {} Python Developer")
        jm = IndeedJobManager.__new__(IndeedJobManager)
        with patch(
            "src.job_manager.indeed.job_manager_indeed.find_element_safely",
            new_callable=AsyncMock,
            return_value=el,
        ):
            jd = await jm._extract_job_description_from_page(_RealLikePage())
        assert jd == "Python Developer clean text"
        el.inner_text.assert_called_once()
        el.text_content.assert_not_called()

    @pytest.mark.asyncio
    async def test_per_term_search_adds_iafilter_when_easy_apply_only(self):
        page = AsyncMock()
        sc = IndeedSearchCustomizer(page)
        sc.set_advanced_search_params(
            {
                "positions": ["Python Developer"],
                "locations": ["India"],
                "remote": True,
                "hybrid": True,
                "onsite": True,
                "date": {"all_time": False, "month": False, "week": False, "24_hours": True},
            }
        )
        with (
            patch("src.job_manager.indeed.search_customizer_indeed.EASY_APPLY_ONLY_MODE", True),
            patch("src.job_manager.indeed.search_customizer_indeed.async_pause", new_callable=AsyncMock),
        ):
            await sc.set_search_params(per_term=True)
        assert "iafilter=1" in sc.search_urls[0]

    @pytest.mark.asyncio
    async def test_submit_application_aborts_on_stop_during_captcha(self):
        from src.dashboard.runtime import StopRequested

        applier = IndeedEasyApplier(
            page=MagicMock(),
            gpt_answerer=MagicMock(),
            resume_anonymizer=MagicMock(deanonymize_text=lambda s: s),
            resume_generator_manager=MagicMock(),
            pause_checker=None,
            answers_file=Path("nonexistent_answers.yaml"),
            resume_dir=Path("."),
            cover_letter_dir=Path("."),
            test_mode=False,
            use_linkedin_bridge=True,
        )
        with (
            patch(
                "src.job_manager.indeed.easy_applier_indeed.find_element_safely",
                new_callable=AsyncMock,
                return_value=AsyncMock(),
            ),
            patch("src.job_manager.indeed.easy_applier_indeed.should_stop", return_value=True),
            pytest.raises(StopRequested),
        ):
            await applier._submit_application()

    @pytest.mark.asyncio
    async def test_date_question_handles_ddmmyyyy_and_start_date(self, tmp_path):
        from datetime import datetime, timedelta

        store = tmp_path / "saved.json"
        ans = answer_from_linkedin_data(
            "If offered the position with Careway, what is your desired start date? *",
            question_type="date",
            saved_answers_file=store,
        )
        assert ans == datetime.now().strftime("%Y-%m-%d")

        date_input = AsyncMock()
        date_input.get_attribute = AsyncMock(
            side_effect=lambda name: "DD/MM/YYYY" if name == "placeholder" else "text"
        )
        applier = IndeedEasyApplier(
            page=MagicMock(),
            gpt_answerer=MagicMock(),
            resume_anonymizer=MagicMock(deanonymize_text=lambda s: s),
            resume_generator_manager=MagicMock(),
            pause_checker=None,
            answers_file=tmp_path / "answers.yaml",
            resume_dir=tmp_path,
            cover_letter_dir=tmp_path,
            test_mode=True,
            use_linkedin_bridge=True,
            saved_answers_path=store,
        )
        with (
            patch(
                "src.job_manager.indeed.easy_applier_indeed.find_element_safely",
                new_callable=AsyncMock,
                return_value=date_input,
            ),
            patch(
                "src.job_manager.indeed.easy_applier_indeed.get_clean_text",
                new_callable=AsyncMock,
                return_value="What is your desired start date?",
            ),
        ):
            ok = await applier._find_and_handle_date_question(MagicMock())
        assert ok is True
        expected_ddmm = (datetime.now() + timedelta(days=0)).strftime("%d/%m/%Y")
        date_input.fill.assert_awaited_once_with(expected_ddmm)

    @pytest.mark.asyncio
    async def test_fill_textbox_question_errors_skips_radio_and_checkbox(self, tmp_path):
        radio_el = AsyncMock()
        radio_el.get_attribute = AsyncMock(
            side_effect=lambda name: "radio" if name == "type" else ""
        )
        applier = IndeedEasyApplier(
            page=MagicMock(),
            gpt_answerer=MagicMock(),
            resume_anonymizer=MagicMock(deanonymize_text=lambda s: s),
            resume_generator_manager=MagicMock(),
            pause_checker=None,
            answers_file=tmp_path / "answers.yaml",
            resume_dir=tmp_path,
            cover_letter_dir=tmp_path,
            test_mode=True,
            use_linkedin_bridge=True,
        )
        with patch(
            "src.job_manager.indeed.easy_applier_indeed.find_elements_safely",
            new_callable=AsyncMock,
            return_value=[radio_el],
        ):
            fixed = await applier._fill_textbox_question_errors()
        assert fixed is False
        radio_el.fill.assert_not_called()

    def test_strip_form_noise_removes_validation_suffixes(self):
        cleaned = IndeedEasyApplier._strip_form_noise(
            "Are you eligible? * Choose an option to continue."
        )
        assert cleaned == "Are you eligible? *"

    @pytest.mark.asyncio
    async def test_external_apply_link_and_preloaded_posted_age_2026_dom(self):
        from src.job_manager.indeed.job_manager_indeed import IndeedJobManager

        jm = IndeedJobManager.__new__(IndeedJobManager)

        class _RealLikePage:
            __module__ = "patchright.async_api._generated"

            def __init__(self):
                self.evaluate = AsyncMock(return_value="Just posted")

        real_page = _RealLikePage()
        with patch(
            "src.job_manager.indeed.job_manager_indeed.find_element_safely",
            new_callable=AsyncMock,
            return_value=None,
        ):
            posted = await jm._extract_posted_text_from_page(real_page)
        assert posted == "Just posted"

        ext_btn = AsyncMock()
        ext_btn.is_visible = AsyncMock(return_value=True)
        ext_btn.is_enabled = AsyncMock(return_value=True)
        new_tab = AsyncMock()
        new_tab.url = "https://careers.ciklum.example/apply"

        ctx_mgr = MagicMock()
        ctx_mgr.__aenter__ = AsyncMock(return_value=MagicMock(value=AsyncMock(return_value=new_tab)()))
        ctx_mgr.__aexit__ = AsyncMock(return_value=False)

        page = MagicMock()
        page.context.expect_page = MagicMock(return_value=ctx_mgr)

        async def _fake_find_elements(_p, sel):
            return [ext_btn] if sel == "a[data-testid='viewjob-apply']" else []

        with (
            patch(
                "src.job_manager.indeed.job_manager_indeed.find_elements_safely",
                side_effect=_fake_find_elements,
            ),
            patch("src.job_manager.indeed.job_manager_indeed.async_pause", new_callable=AsyncMock),
        ):
            ext_url = await jm._get_button_link(page)
        assert ext_url == "https://careers.ciklum.example/apply"
