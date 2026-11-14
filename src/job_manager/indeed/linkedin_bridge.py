"""
Bridge between Indeed_Bot_Repo and Linkedin_Test saved data, configs, AI keys,
RenderCv resume engine, AI_Outreach, and Indeed CSV history.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

from config.logger_config import logger

csv.field_size_limit(1000000)

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_LINKEDIN_DIR = WORKSPACE_ROOT / "Linkedin_Test"

APPLIED_CSV_FIELDS = [
    "Job ID",
    "Title",
    "Company",
    "Work Location",
    "Work Style",
    "About Job",
    "Experience required",
    "Skills required",
    "HR Name",
    "HR Link",
    "Resume",
    "Re-posted",
    "Date Posted",
    "Date Applied",
    "Job Link",
    "External Job link",
    "Questions Found",
    "Connect Request",
]

FAILED_CSV_FIELDS = [
    "Job ID",
    "Job Link",
    "Resume Tried",
    "Date listed",
    "Date Tried",
    "Assumed Reason",
    "Stack Trace",
    "External Job link",
    "Screenshot Name",
]

_RATE_LIMIT_PATTERNS = (
    r"\b429\b",
    r"too many requests",
    r"rate[_ ]?limit",
    r"free-models-per-day",
    r"free_tier_daily",
    r"openrouter_free_tier",
    r"quota.?exceeded",
    r"insufficient.?quota",
    r"x-ratelimit-remaining['\"]?\s*[:=]\s*['\"]?0\b",
)

_KEY_FAILOVER_PATTERNS = _RATE_LIMIT_PATTERNS + (
    r"\b401\b",
    r"\b403\b",
    r"invalid.?api.?key",
    r"incorrect.?api.?key",
    r"unauthorized",
    r"user not found",
    r"key not found",
    r"authentication",
)

_RE_EXPERIENCE = re.compile(r"[(]?\s*(\d+)\s*[)]?\s*[-to]*\s*\d*[+]*\s*year[s]?", re.IGNORECASE)

_MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

_RESUME_SCORE_PROMPT = """You are a strict resume-to-job-description matcher.

Score how well the CANDIDATE RESUME fits this JOB DESCRIPTION on a scale of 0 to 100.
- 0-40: poor fit (wrong domain, missing core required skills)
- 41-69: partial fit (some overlap, notable gaps)
- 70-100: strong fit (core skills and experience align)

Use only information in the resume. Do not invent experience.
Return ONLY a JSON object with no extra text:
{"score": <integer 0-100>}

CANDIDATE RESUME:
<<<RESUME>>>

JOB DESCRIPTION:
<<<JD>>>
"""


def get_linkedin_bot_dir() -> Path:
    env_dir = os.environ.get("LINKEDIN_BOT_DIR", "").strip()
    if env_dir:
        return Path(env_dir).resolve()
    return DEFAULT_LINKEDIN_DIR.resolve()


def get_indeed_email(default: str = "") -> str:
    """Resolve Indeed email, preferring Indeed .env override over Linkedin_Test username."""
    env_val = (os.environ.get("indeed_email") or os.environ.get("INDEED_EMAIL") or "").strip()
    if env_val:
        return env_val
    dotenv_path = REPO_ROOT / ".env"
    if dotenv_path.is_file():
        try:
            import dotenv

            val = str(dotenv.dotenv_values(dotenv_path).get("indeed_email") or "").strip()
            if val:
                return val
        except Exception:
            pass
    return default.strip()


def _load_python_module(name: str, file_path: Path) -> Any:
    if not file_path.is_file():
        return None
    spec = importlib.util.spec_from_file_location(name, str(file_path))
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_linkedin_configs() -> dict[str, dict[str, Any]]:
    """Load config/{personals,questions,search,secrets,settings}.py from Linkedin_Test."""
    li_dir = get_linkedin_bot_dir()
    cfg_dir = li_dir / "config"
    out: dict[str, dict[str, Any]] = {}
    for mod_name in ("personals", "questions", "search", "secrets", "settings"):
        mod = _load_python_module(f"_li_cfg_{mod_name}", cfg_dir / f"{mod_name}.py")
        if mod is None:
            out[mod_name] = {}
        else:
            out[mod_name] = {k: v for k, v in vars(mod).items() if not k.startswith("_")}
    return out


# ---------------------------------------------------------------------------
# AI Key Pool & Runtime Rate-Limit Switches (LinkedIn parity)
# ---------------------------------------------------------------------------

class AIKeyRuntime:
    """Manages rotating LLM API keys and independent AI vs Resume Engine switches."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._keys: list[str] = []
        self._index = 0
        self._ai_disabled = False
        self._resume_disabled = False
        self._ai_disable_reason = ""
        self._resume_disable_reason = ""
        self._config_use_ai = True
        self._config_use_resume_engine = True
        self.reload()

    def reload(self) -> None:
        cfgs = load_linkedin_configs()
        sec = cfgs.get("secrets", {})
        sett = cfgs.get("settings", {})
        self._config_use_ai = bool(sec.get("use_AI", True))
        self._config_use_resume_engine = bool(sett.get("use_resume_engine", True))

        collected: list[str] = []
        primary = str(sec.get("llm_api_key") or os.environ.get("llm_api_key") or "").strip()
        if primary:
            collected.append(primary)
        extra = sec.get("llm_api_keys") or []
        if isinstance(extra, str):
            extra = [extra]
        for raw in extra:
            key = str(raw or "").strip()
            if key and key not in collected:
                collected.append(key)

        with self._lock:
            self._keys = collected
            self._index = 0
            self._ai_disabled = False
            self._resume_disabled = False
            self._ai_disable_reason = ""
            self._resume_disable_reason = ""

    def set_keys(self, keys: list[str]) -> None:
        with self._lock:
            self._keys = [str(k).strip() for k in keys if str(k or "").strip()]
            self._index = 0
            self._ai_disabled = False

    def api_keys(self) -> list[str]:
        with self._lock:
            return list(self._keys)

    def key_count(self) -> int:
        with self._lock:
            return len(self._keys)

    def current_api_key(self) -> str:
        with self._lock:
            if not self._keys:
                return ""
            return self._keys[min(self._index, len(self._keys) - 1)]

    def current_key_label(self) -> str:
        with self._lock:
            if not self._keys:
                return "none"
            return f"{self._index + 1}/{len(self._keys)}"

    @staticmethod
    def is_rate_limit_error(exc_or_text: Any) -> bool:
        text = str(exc_or_text or "")
        return bool(text) and any(
            re.search(p, text, flags=re.IGNORECASE) for p in _RATE_LIMIT_PATTERNS
        )

    @staticmethod
    def is_key_failover_error(exc_or_text: Any) -> bool:
        text = str(exc_or_text or "")
        return bool(text) and any(
            re.search(p, text, flags=re.IGNORECASE) for p in _KEY_FAILOVER_PATTERNS
        )

    def rotate_api_key(self, exc_or_text: Any = None) -> bool:
        with self._lock:
            if not self._keys or self._index + 1 >= len(self._keys):
                return False
            self._index += 1
            label = f"{self._index + 1}/{len(self._keys)}"
        logger.warning(
            f"Switching Indeed AI API key to {label} after error: {str(exc_or_text)[:160]}"
        )
        return True

    def try_rotate_or_disable(self, exc_or_text: Any, *, source: str = "Indeed AI") -> bool:
        if not self.is_key_failover_error(exc_or_text):
            return False
        if self.rotate_api_key(exc_or_text):
            return True
        self.disable_ai(exc_or_text, source=source)
        return False

    def disable_ai(self, exc_or_text: Any, *, source: str = "Indeed AI") -> bool:
        if not self.is_key_failover_error(exc_or_text):
            return False
        with self._lock:
            if self._ai_disabled:
                return True
            self._ai_disabled = True
            self._ai_disable_reason = str(exc_or_text)[:300]
        logger.warning(
            f"All AI API keys exhausted ({source}); disabling answer AI for this run. "
            "Resume Engine keeps running and Indeed Apply continues with saved/heuristic answers."
        )
        return True

    def disable_resume_engine(self, exc_or_text: Any, *, source: str = "Resume Engine") -> bool:
        if not self.is_rate_limit_error(exc_or_text):
            return False
        with self._lock:
            if self._resume_disabled:
                return True
            self._resume_disabled = True
            self._resume_disable_reason = str(exc_or_text)[:300]
        logger.warning(
            f"Rate limit hit ({source}); disabling Resume Engine for this run and using default PDF."
        )
        return True

    def is_ai_enabled(self) -> bool:
        return self._config_use_ai and not self._ai_disabled

    def is_resume_engine_enabled(self) -> bool:
        return self._config_use_resume_engine and not self._resume_disabled


ai_runtime = AIKeyRuntime()


# ---------------------------------------------------------------------------
# Saved Answers Store (reads/writes Linkedin_Test/config/saved_answers.json)
# ---------------------------------------------------------------------------

def normalize_question_label(label: str) -> str:
    if not label:
        return ""
    cleaned = re.sub(r"\s*\[.*?\]\s*$", "", str(label), flags=re.DOTALL)
    cleaned = cleaned.lower()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.strip(" :?-*")


def get_saved_answers_path() -> Path:
    cfgs = load_linkedin_configs()
    rel = str(cfgs.get("questions", {}).get("saved_answers_path") or "config/saved_answers.json")
    p = Path(rel)
    if p.is_absolute():
        return p
    return (get_linkedin_bot_dir() / p).resolve()


def load_saved_answers(path: Path | None = None) -> dict[str, Any]:
    target = path or get_saved_answers_path()
    try:
        if not target.is_file():
            return {}
        data = json.loads(target.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning(f"Failed to load saved answers from {target}: {e}")
        return {}


def _answer_from_entry(entry: Any) -> str | None:
    if not entry:
        return None
    if isinstance(entry, dict):
        val = entry.get("answer")
        return str(val) if val is not None and str(val).strip() != "" else None
    if isinstance(entry, str) and entry.strip():
        return entry
    return None


def get_saved_answer(label: str, path: Path | None = None) -> str | None:
    cfgs = load_linkedin_configs()
    if not cfgs.get("questions", {}).get("save_and_reuse_answers", True):
        return None
    key = normalize_question_label(label)
    if not key:
        return None
    data = load_saved_answers(path)

    found = _answer_from_entry(data.get(key))
    if found is not None:
        return found

    for saved_key, entry in data.items():
        if not saved_key or saved_key == "phone country code" or len(saved_key) < 8:
            continue
        shorter, longer = (saved_key, key) if len(saved_key) <= len(key) else (key, saved_key)
        if shorter in longer and len(shorter) / max(len(longer), 1) >= 0.7:
            found = _answer_from_entry(entry)
            if found is not None:
                return found

    aliases = {
        "month of from": ["from month", "start month", "month of start"],
        "year of from": ["from year", "start year", "year of start"],
        "month of to": ["to month", "end month", "month of end"],
        "year of to": ["to year", "end year", "year of end"],
        "description": ["work experience description", "role description", "job description"],
        "city": ["work city", "location city", "job city"],
        "company": ["company name", "employer", "organization"],
        "your title": ["title", "job title", "role title", "position"],
    }
    for canonical, alts in aliases.items():
        keys = [canonical] + alts
        if any(a in key or key in a for a in keys):
            for candidate in keys:
                found = _answer_from_entry(data.get(candidate))
                if found is not None:
                    return found
    return None


def upsert_saved_answers(
    entries: list[dict[str, Any]], force_overwrite: bool = True, path: Path | None = None
) -> int:
    cfgs = load_linkedin_configs()
    if not cfgs.get("questions", {}).get("save_and_reuse_answers", True):
        return 0
    target = path or get_saved_answers_path()
    data = load_saved_answers(target)
    updated = 0
    now = datetime.now().isoformat(timespec="seconds")
    for entry in entries:
        label = entry.get("label", "")
        answer = entry.get("answer", "")
        q_type = entry.get("type", "text")
        key = normalize_question_label(label)
        if not key or answer is None:
            continue
        if key == "phone country code" and str(answer).isdigit():
            continue
        answer_str = str(answer).strip()
        if not answer_str or answer_str.lower() == "select an option":
            continue
        if answer_str.lower() in ("yes", "no") and q_type == "select" and any(
            m in key for m in ("month", "year", "date", "from", "to")
        ):
            continue
        if key in data and not force_overwrite:
            continue
        data[key] = {
            "answer": answer_str,
            "type": q_type,
            "updated_at": now,
            "source": "user" if force_overwrite else "bot",
        }
        updated += 1
    if updated:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return updated


# ---------------------------------------------------------------------------
# Deterministic Question Answering (Linkedin_Test parity)
# ---------------------------------------------------------------------------

def convert_to_lakhs(value: str) -> str:
    value = str(value).strip()
    l = len(value)
    if l > 5:
        return value[: l - 5] + "." + value[l - 5 : l - 3]
    if l > 0:
        return "0." + "0" * (5 - l) + value[:2]
    return value


def _match_option(desired: str, options: list[str]) -> str | None:
    if not desired or not options:
        return None
    desired_l = desired.strip().lower()
    for opt in options:
        if opt.strip().lower() == desired_l:
            return opt
    phrases = (
        ["decline", "not wish", "don't wish", "prefer not", "not want"]
        if desired_l == "decline"
        else [desired_l]
    )
    for p in phrases:
        for opt in options:
            ol = opt.strip().lower()
            if ol and ol not in ("select an option", "select", "choose an option"):
                if p in ol or ol in p:
                    return opt
    return None


def answer_from_linkedin_data(
    question_text: str,
    options: list[str] | None = None,
    question_type: str = "text",
    work_location: str = "",
    saved_answers_file: Path | None = None,
) -> str | None:
    """Resolve an Indeed form question using Linkedin_Test saved_answers.json and config."""
    if not question_text or not question_text.strip():
        return None

    cfgs = load_linkedin_configs()
    p = cfgs.get("personals", {})
    q = cfgs.get("questions", {})
    sec = cfgs.get("secrets", {})
    label = normalize_question_label(question_text)
    indeed_email = get_indeed_email(str(sec.get("username", "")).strip())
    if (
        indeed_email
        and not options
        and (
            label in ("email", "email address", "e-mail", "e-mail address", "your email", "contact email")
            or ("email" in label and not any(w in label for w in ("update", "alert", "notify", "marketing", "consent")))
        )
    ):
        return indeed_email

    # 1. Saved answers memory first
    saved = get_saved_answer(question_text, path=saved_answers_file)
    if saved is not None:
        if options:
            matched = _match_option(saved, options)
            if matched:
                return matched
        else:
            return saved

    if not p and not q:
        return None

    first = str(p.get("first_name", "")).strip()
    middle = str(p.get("middle_name", "")).strip()
    last = str(p.get("last_name", "")).strip()
    full_name = f"{first} {middle} {last}".strip() if middle else f"{first} {last}".strip()
    email = indeed_email
    phone = str(p.get("phone_number", "")).strip()
    city = str(p.get("current_city", "")).strip() or work_location
    street = str(p.get("street", "")).strip()
    state = str(p.get("state", "")).strip()
    zipcode = str(p.get("zipcode", "")).strip()
    country = str(p.get("country", "")).strip()
    yoe = str(q.get("years_of_experience", "1")).strip()
    require_visa = str(q.get("require_visa", "No")).strip()
    website = str(q.get("website", "")).strip()
    linkedin_url = str(q.get("linkedIn", "")).strip()
    citizenship = str(q.get("us_citizenship", "")).strip()
    recent_employer = str(q.get("recent_employer", "")).strip()
    confidence = str(q.get("confidence_level", "10")).strip()
    cover_letter = str(q.get("cover_letter", "")).strip()
    summary = str(q.get("linkedin_summary", "")).strip()
    headline = str(q.get("linkedin_headline", "")).strip()
    gender = str(p.get("gender", "")).strip()
    ethnicity = str(p.get("ethnicity", "")).strip()
    disability = str(p.get("disability_status", "")).strip()
    veteran = str(p.get("veteran_status", "")).strip()

    desired_sal = int(q.get("desired_salary", 600000) or 600000)
    current_sal = int(q.get("current_ctc", 450000) or 450000)
    notice_days = int(q.get("notice_period", 0) or 0)

    raw_answer: str | None = None
    if "sponsorship" in label or "visa" in label:
        raw_answer = require_visa
    elif "phone country code" in label or ("country code" in label and "phone" in label):
        raw_answer = f"{country} (+91)" if country.lower() == "india" else "+91"
    elif any(k in label for k in ("phone", "mobile", "telephone", "contact number")):
        raw_answer = phone
    elif "email" in label and email:
        raw_answer = email
    elif "street" in label or "address" in label and "email" not in label:
        raw_answer = street or city
    elif "city" in label:
        raw_answer = city
    elif "state" in label or "province" in label:
        raw_answer = state
    elif "zip" in label or "postal" in label:
        raw_answer = zipcode
    elif "country" in label:
        raw_answer = country
    elif "signature" in label or "full name" in label:
        raw_answer = full_name
    elif "first" in label and "name" in label and "last" not in label:
        raw_answer = first
    elif "middle" in label and "name" in label and "last" not in label:
        raw_answer = middle
    elif "last" in label and "name" in label and "first" not in label:
        raw_answer = last
    elif "notice" in label:
        if "month" in label:
            raw_answer = str(notice_days // 30)
        elif "week" in label:
            raw_answer = str(notice_days // 7)
        else:
            raw_answer = str(notice_days)
    elif any(k in label for k in ("salary", "compensation", "ctc", "pay", "package")):
        is_current = "current" in label or "present" in label
        base = current_sal if is_current else desired_sal
        if "month" in label:
            raw_answer = str(round(base / 12, 2))
        elif "lakh" in label or "lac" in label:
            raw_answer = str(round(base / 100000, 2))
        else:
            raw_answer = str(base)
    elif "linkedin" in label:
        raw_answer = linkedin_url
    elif any(k in label for k in ("website", "github", "portfolio", "blog")):
        raw_answer = website
    elif "gender" in label or "sex" in label:
        raw_answer = gender
    elif "ethnicity" in label or "race" in label:
        raw_answer = ethnicity
    elif "disability" in label or "handicapped" in label:
        raw_answer = disability
    elif "veteran" in label or "protected" in label:
        raw_answer = veteran
    elif "citizenship" in label or "employment eligibility" in label:
        raw_answer = citizenship
    elif "scale of 1-10" in label or "confidence" in label:
        raw_answer = confidence
    elif "headline" in label:
        raw_answer = headline
    elif "cover" in label and question_type in ("text", "textarea"):
        raw_answer = cover_letter
    elif "summary" in label and question_type in ("text", "textarea"):
        raw_answer = summary
    elif any(k in label for k in ("employer", "company name", "organization")):
        raw_answer = recent_employer
    elif question_type == "date" or any(
        k in label for k in ("start date", "joining date", "available date", "earliest start")
    ):
        raw_answer = (datetime.now() + timedelta(days=max(0, notice_days))).strftime("%Y-%m-%d")
    elif any(k in label for k in ("years of experience", "how many years", "total experience")):
        raw_answer = yoe

    if not raw_answer:
        return None
    if options:
        return _match_option(raw_answer, options)
    return raw_answer


# ---------------------------------------------------------------------------
# Build Structured Resume & Text from Linkedin_Test
# ---------------------------------------------------------------------------

def build_resume_from_linkedin() -> tuple[str, dict[str, Any]]:
    """Build (resume_text, ResumeStructure dict) from Linkedin_Test configs."""
    cfgs = load_linkedin_configs()
    p = cfgs.get("personals", {})
    q = cfgs.get("questions", {})
    sec = cfgs.get("secrets", {})

    first = str(p.get("first_name", "") or "Candidate").strip()
    last = str(p.get("last_name", "") or "User").strip()
    gender = str(p.get("gender", "") or "male").strip().lower()
    if gender not in ("male", "female"):
        gender = "male"
    email = get_indeed_email(str(sec.get("username", "") or "").strip())
    phone = str(p.get("phone_number", "") or "").strip()
    city = str(p.get("current_city", "") or "").strip()
    state = str(p.get("state", "") or "").strip()
    zipcode = str(p.get("zipcode", "") or "").strip()
    country = str(p.get("country", "") or "India").strip()
    address = str(p.get("street", "") or "").strip()
    website = str(q.get("website", "") or "").strip()
    linkedin_url = str(q.get("linkedIn", "") or "").strip()
    recent_employer = str(q.get("recent_employer", "") or "Dice Enterprises").strip()
    user_info = str(q.get("user_information_all", "") or "").strip()
    notice = str(q.get("notice_period", "0"))
    desired_sal = str(q.get("desired_salary", "600000"))
    require_visa = str(q.get("require_visa", "No"))

    structured: dict[str, Any] = {
        "personal_information": {
            "first_name": first,
            "last_name": last,
            "gender": gender,
            "date_of_birth": "No info",
            "country": country or "India",
            "city": city or "Indore",
            "state_area_region": state or "Madhya Pradesh",
            "address": address or "No info",
            "zip_code": zipcode or "No info",
            "phone": phone or "No info",
            "phone_code": "+91" if (country or "").lower() == "india" else "No info",
            "email": email or "No info",
            "github": website if "github.com" in website.lower() else "No info",
            "linkedin": linkedin_url or "No info",
        },
        "self_identification": {
            "gender": gender.capitalize(),
            "pronouns": "He/Him" if gender == "male" else "She/Her",
            "veteran": str(p.get("veteran_status", "No") or "No"),
            "disability": str(p.get("disability_status", "No") or "No"),
            "ethnicity": str(p.get("ethnicity", "Asian") or "Asian"),
        },
        "legal_authorization": {
            "eu_work_authorization": "No",
            "us_work_authorization": "No",
            "requires_us_visa": require_visa,
            "requires_us_sponsorship": require_visa,
            "requires_eu_visa": require_visa,
            "legally_allowed_to_work_in_eu": "No",
            "legally_allowed_to_work_in_us": "No",
        },
        "work_preferences": {
            "remote_work": "Yes",
            "in_person_work": "Yes",
            "open_to_relocation": "Yes",
            "willing_to_complete_assessments": "Yes",
            "willing_to_undergo_drug_tests": "Yes",
            "willing_to_undergo_background_checks": "Yes",
        },
        "education_details": [
            {
                "education_level": "Bachelor of Technology",
                "institution": "SGSITS, Indore",
                "field_of_study": "Electronics & Telecommunication Engineering",
                "final_evaluation_grade": "7.8",
                "start_date": "2022",
                "year_of_completion": "2026",
            }
        ],
        "experience_details": [
            {
                "position": "QA Analyst",
                "company": recent_employer,
                "employment_period": "December 2025 - March 2026",
                "location": "Pune",
                "industry": "SaaS",
                "key_responsibilities": [
                    "Executed functional, regression, integration, E2E, UAT, and REST API testing"
                ],
                "skills_acquired": ["Python", "Pytest", "Selenium", "Postman", "Jira"],
            },
            {
                "position": "Software Developer Intern",
                "company": "Sherwa.Tech",
                "employment_period": "May 2024 - July 2024",
                "location": "Remote",
                "industry": "Software",
                "key_responsibilities": [
                    "Built React applications and Node.js/Express/MongoDB REST APIs"
                ],
                "skills_acquired": ["React.js", "Node.js", "Express.js", "MongoDB"],
            },
        ],
        "projects": [],
        "achievements": [],
        "certifications": [],
        "languages": [{"language": "English", "proficiency": "Professional"}],
        "interests": [],
        "availability": {"notice_period": f"{notice} days"},
        "salary_expectations": {"salary_range_usd": desired_sal},
        "skills": [
            "Python",
            "C++",
            "JavaScript",
            "Node.js",
            "React.js",
            "SQL",
            "MongoDB",
            "Playwright",
            "Selenium",
            "Pytest",
        ],
        "about_me": str(q.get("linkedin_summary", "") or user_info[:400]).strip(),
    }

    resume_text = user_info or (
        f"{first} {last}\nEmail: {email}\nPhone: {phone}\nLocation: {city}, {country}"
    )
    return resume_text, structured


def sync_resume_files_from_linkedin(resume_dir: Path) -> tuple[str, dict[str, Any]]:
    """Ensure resume_text.txt and structured_resume.yaml exist under resume_dir."""
    import yaml

    resume_dir.mkdir(parents=True, exist_ok=True)
    text_file = resume_dir / "resume_text.txt"
    struct_file = resume_dir / "structured_resume.yaml"
    resume_text, structured = build_resume_from_linkedin()
    if not text_file.is_file() or not text_file.read_text(encoding="utf-8").strip():
        text_file.write_text(resume_text, encoding="utf-8")
    if not struct_file.is_file():
        struct_file.write_text(
            yaml.safe_dump(structured, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    else:
        try:
            existing = yaml.safe_load(struct_file.read_text(encoding="utf-8")) or {}
            desired_email = get_indeed_email()
            if (
                desired_email
                and isinstance(existing, dict)
                and isinstance(existing.get("personal_information"), dict)
                and existing["personal_information"].get("email") != desired_email
            ):
                existing["personal_information"]["email"] = desired_email
                struct_file.write_text(
                    yaml.safe_dump(existing, allow_unicode=True, sort_keys=False),
                    encoding="utf-8",
                )
        except Exception:
            pass
    return resume_text, structured


def sync_search_config_from_linkedin(yaml_path: Path) -> None:
    """Keep Indeed search_config.yaml synced with Linkedin_Test/config/search.py."""
    import yaml

    cfgs = load_linkedin_configs()
    li_search = cfgs.get("search", {})
    if not li_search:
        return
    data: dict[str, Any] = {}
    if yaml_path.is_file():
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}

    if li_search.get("search_terms"):
        data["positions"] = list(li_search["search_terms"])
    if li_search.get("search_location"):
        data["locations"] = [str(li_search["search_location"])]
    elif li_search.get("country"):
        data["locations"] = [str(li_search["country"])]
    data["company_blacklist"] = list(li_search.get("company_blacklist") or [])
    data["title_blacklist"] = list(li_search.get("title_blacklist") or [])
    data["location_blacklist"] = list(li_search.get("location_blacklist") or [])
    data["apply_once_at_company"] = bool(li_search.get("apply_once_at_company", False))

    date_posted = str(li_search.get("date_posted", "") or "").lower()
    if "24" in date_posted:
        data["date"] = {"all_time": False, "month": False, "week": False, "24_hours": True}
    elif "week" in date_posted:
        data["date"] = {"all_time": False, "month": False, "week": True, "24_hours": False}
    elif "month" in date_posted:
        data["date"] = {"all_time": False, "month": True, "week": False, "24_hours": False}

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Eligibility Filters & Recent-Job Age Gate (Linkedin_Test parity)
# ---------------------------------------------------------------------------

def parse_job_age_minutes(time_posted_text: str | None) -> int | None:
    """Convert LinkedIn/Indeed posted-time text into age in minutes."""
    if time_posted_text is None:
        return None
    text = str(time_posted_text).strip()
    if not text:
        return None
    low_raw = re.sub(r"\s+", " ", text.lower()).strip()
    if low_raw in ("just now", "now", "just posted", "posted today", "today") or low_raw.startswith(
        ("just now", "just posted", "posted today")
    ):
        return 0
    text = re.sub(r"(?i)\b(reposted|posted|active)\b", "", text).strip()
    text = re.sub(r"\s+", " ", text)
    low = text.lower()

    if low in ("just now", "now", "just posted", "just", "today") or low.startswith(
        ("just now", "just posted")
    ):
        return 0
    if "30+ days ago" in low:
        return 31 * 1440
    if re.search(r"\ba\s+few\s+seconds?\s+ago\b", low):
        return 0
    if re.search(r"\ba\s+minute\s+ago\b", low):
        return 1
    if re.search(r"\ban?\s+hour\s+ago\b", low):
        return 60
    if re.search(r"\byesterday\b", low):
        return 1440

    match = re.search(
        r"(\d+)\+?\s*(second|minute|hour|day|week|month|year)s?\s+ago",
        low,
        re.IGNORECASE,
    )
    if not match:
        return None
    value = int(match.group(1))
    unit = match.group(2).lower()
    multipliers = {
        "second": 0,
        "minute": 1,
        "hour": 60,
        "day": 1440,
        "week": 10080,
        "month": 43200,
        "year": 525600,
    }
    if unit == "second":
        return max(0, value // 60)
    return value * multipliers.get(unit, 1)


def is_within_recent_job_window(
    job_age_minutes: int, *, enabled: bool, max_age_minutes: int
) -> bool:
    if not enabled:
        return True
    try:
        return int(job_age_minutes) <= int(max_age_minutes)
    except (TypeError, ValueError):
        return True


def extract_years_of_experience(text: str) -> int:
    matches = _RE_EXPERIENCE.findall(text or "")
    valid = [int(m) for m in matches if int(m) <= 12]
    return max(valid, default=0)


def check_jd_eligibility(
    job_description: str,
    company_description: str = "",
    time_posted_text: str | None = None,
) -> tuple[bool, str | None, str | None]:
    """Run all deterministic Linkedin_Test eligibility gates before AI scoring."""
    cfgs = load_linkedin_configs()
    search_cfg = cfgs.get("search", {})
    settings_cfg = cfgs.get("settings", {})

    # 1. Recent-job age gate
    recent_enabled = bool(settings_cfg.get("recent_job_feature_enabled", True))
    max_age = int(settings_cfg.get("recent_job_max_age_minutes", 4320))
    default_age = int(settings_cfg.get("recent_job_default_age_minutes", 0))
    parsed_age = parse_job_age_minutes(time_posted_text)
    age_minutes = default_age if parsed_age is None else parsed_age
    if not is_within_recent_job_window(age_minutes, enabled=recent_enabled, max_age_minutes=max_age):
        return (
            False,
            "Job older than recent_job_max_age_minutes",
            f"job_age_minutes={age_minutes} > limit={max_age}",
        )

    # 2. About Company bad/good words
    if company_description:
        about_low = company_description.lower()
        good_words = [str(w).lower() for w in (search_cfg.get("about_company_good_words") or [])]
        bad_company = search_cfg.get("about_company_bad_words") or []
        if not any(g in about_low for g in good_words):
            for word in bad_company:
                if str(word).lower() in about_low:
                    return False, "Found Blacklisted words in About Company", f'Company contains "{word}"'

    # 3. JD bad words
    jd_low = (job_description or "").lower()
    for word in search_cfg.get("bad_words") or []:
        if str(word).lower() in jd_low:
            return False, "Found a Bad Word in About Job", f'Contains bad word "{word}"'

    # 4. Security clearance
    security_clearance = bool(search_cfg.get("security_clearance", False))
    if not security_clearance and any(
        k in jd_low for k in ("polygraph", "clearance", "secret")
    ):
        return False, "Asking for Security clearance", 'Found "Clearance" or "Polygraph"'

    # 5. Experience required vs current_experience
    current_exp = int(search_cfg.get("current_experience", -1))
    did_masters = bool(search_cfg.get("did_masters", False))
    bonus = 2 if (did_masters and "master" in jd_low) else 0
    exp_req = extract_years_of_experience(job_description or "")
    if current_exp > -1 and exp_req > current_exp + bonus:
        return (
            False,
            "Required experience is high",
            f"Experience required {exp_req} > Current Experience {current_exp + bonus}",
        )

    return True, None, None


# ---------------------------------------------------------------------------
# Resume Score Gate (Linkedin_Test parity with multi-key failover)
# ---------------------------------------------------------------------------

def parse_resume_score(result: Any) -> int | None:
    if result is None:
        return None
    if isinstance(result, (int, float)):
        return max(0, min(100, int(result)))
    if isinstance(result, dict):
        if "score" in result:
            val = parse_resume_score(result.get("score"))
            if val is not None:
                return val
        nested = result.get("data")
        if nested is not None and nested is not result:
            val = parse_resume_score(nested)
            if val is not None:
                return val
        return None
    text = str(result).strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, (dict, int, float)):
            parsed = parse_resume_score(obj)
            if parsed is not None:
                return parsed
    except Exception:
        pass
    m_score = re.search(r"(?i)\bscore\b[^\d]*(\d{1,3})", text)
    if m_score:
        return max(0, min(100, int(m_score.group(1))))
    m_num = re.search(r"\b(100|[0-9]{1,2})\b", text)
    if m_num:
        return max(0, min(100, int(m_num.group(1))))
    return None


def _call_scoring_openai(prompt: str) -> str:
    from openai import OpenAI

    cfgs = load_linkedin_configs()
    sec = cfgs.get("secrets", {})
    base_url = str(sec.get("llm_api_url") or "https://openrouter.ai/api/v1").strip()
    model = str(sec.get("llm_model") or "openai/gpt-4o-mini").strip()
    timeout = float(sec.get("ai_request_timeout") or 60)

    attempts = max(1, ai_runtime.key_count())
    last_err: Exception | None = None
    for _ in range(attempts):
        key = ai_runtime.current_api_key()
        client = OpenAI(base_url=base_url, api_key=key, timeout=timeout)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            if ai_runtime.try_rotate_or_disable(e, source="Indeed Resume Score"):
                continue
            raise
        finally:
            try:
                client.close()
            except Exception:
                pass
    if last_err:
        raise last_err
    return ""


def apply_resume_score_gate(
    job_description: str,
    *,
    scorer_fn: Callable[[str], Any] | None = None,
) -> tuple[bool, int | None, str | None, str | None]:
    """
    Score JD vs Linkedin_Test user_information_all.
    Returns (continue_apply, score, skip_reason, skip_message).
    Passes by default if disabled, JD too short, AI disabled, error, or unparseable.
    """
    cfgs = load_linkedin_configs()
    sett = cfgs.get("settings", {})
    q = cfgs.get("questions", {})

    if not bool(sett.get("use_resume_score_gate", True)):
        return True, None, None, None

    min_jd = int(sett.get("min_jd_chars", 80))
    if not job_description or len(job_description.strip()) < min_jd:
        logger.info("Resume Score skipped — insufficient JD content (passing gate by default)")
        return True, None, None, None

    if not ai_runtime.is_ai_enabled() or (scorer_fn is None and not ai_runtime.current_api_key()):
        logger.info("Resume Score skipped — AI disabled or key unavailable (passing gate by default)")
        return True, None, None, None

    resume_text = str(q.get("user_information_all") or "").strip()
    if not resume_text:
        return True, None, None, None

    threshold = int(sett.get("resume_score_threshold", 80))
    prompt = _RESUME_SCORE_PROMPT.replace("<<<RESUME>>>", resume_text).replace(
        "<<<JD>>>", job_description
    )
    try:
        raw = scorer_fn(prompt) if scorer_fn else _call_scoring_openai(prompt)
    except Exception as e:
        logger.warning(f"Resume Score failed due to AI/system error (passing gate by default): {e}")
        return True, None, None, None

    score = parse_resume_score(raw)
    if score is None:
        logger.warning(f"Resume Score unparseable ({raw!r}) — passing gate by default")
        return True, None, None, None

    logger.info(f"JD passed eligibility — Resume Score: {score}/100 (threshold={threshold})")
    if score < threshold:
        return (
            False,
            score,
            "Low resume score",
            f"JD rejected — Resume Score: {score}/100 < threshold {threshold}",
        )
    return True, score, None, None


# ---------------------------------------------------------------------------
# RenderCv Resume Engine & AI_Outreach Integration
# ---------------------------------------------------------------------------

def get_default_resume_path() -> str:
    cfgs = load_linkedin_configs()
    path = str(cfgs.get("questions", {}).get("default_resume_path") or "").strip()
    if path and Path(path).is_file():
        return str(Path(path).resolve())
    local_pdfs = list((REPO_ROOT / "data" / "resumes").glob("*.pdf"))
    if local_pdfs:
        return str(local_pdfs[0].resolve())
    return path


def _resolve_rendercv_python(engine_root: Path) -> str:
    for candidate in (
        engine_root / "venv" / "Scripts" / "python.exe",
        engine_root / ".venv" / "Scripts" / "python.exe",
        engine_root / "venv" / "bin" / "python",
    ):
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def prepare_tailored_resume_pdf(
    job_title: str,
    company_name: str,
    location: str,
    job_description: str,
    *,
    cancel_check: Callable[[], bool] | None = None,
) -> str:
    """Generate tailored resume via RenderCv worker, falling back to default_resume_path."""
    fallback = get_default_resume_path()
    if not ai_runtime.is_resume_engine_enabled():
        return fallback

    cfgs = load_linkedin_configs()
    sett = cfgs.get("settings", {})
    min_jd = int(sett.get("min_jd_chars", 80))
    if not job_description or len(job_description.strip()) < min_jd:
        logger.info("Skipping Resume Engine (insufficient JD); using default resume.")
        return fallback

    raw_root = str(sett.get("resume_engine_root") or "../RenderCv")
    engine_root = Path(raw_root)
    if not engine_root.is_absolute():
        engine_root = (get_linkedin_bot_dir() / engine_root).resolve()
    if not (engine_root / "resume_engine").is_dir():
        logger.info(f"Resume Engine unavailable under {engine_root}; using default resume.")
        return fallback

    worker_script = WORKSPACE_ROOT / "connectors" / "resume_worker.py"
    if not worker_script.is_file():
        return fallback

    output_dir = (REPO_ROOT / "all resumes" / "generated").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    template_name = str(sett.get("resume_template_name") or "template2")
    timeout = float(sett.get("resume_generation_timeout", 900))
    job_text = (
        f"Job Title: {job_title}\n"
        f"Company: {company_name}\n"
        f"Location: {location}\n\n"
        f"{job_description.strip()}"
    )

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        result_json = Path(tf.name)

    try:
        py_exe = _resolve_rendercv_python(engine_root)
        cmd = [
            py_exe,
            "-m",
            "connectors.resume_worker",
            str(engine_root),
            str(output_dir),
            str(result_json),
            template_name,
        ]
        env = os.environ.copy()
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{WORKSPACE_ROOT}{os.pathsep}{existing_pp}".strip(os.pathsep)
        env["PYTHONUTF8"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        logger.info(
            f"Generating tailored resume via RenderCV for '{job_title}' at '{company_name}'..."
        )
        proc = subprocess.Popen(
            cmd,
            cwd=str(WORKSPACE_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        assert proc.stdin is not None
        proc.stdin.write(job_text)
        proc.stdin.close()

        def _stream_pipe(pipe: Any, level: str) -> None:
            try:
                while True:
                    raw_line = pipe.readline()
                    if not isinstance(raw_line, (str, bytes)) or not raw_line:
                        break
                    msg = (
                        raw_line.decode("utf-8", errors="replace")
                        if isinstance(raw_line, bytes)
                        else raw_line
                    ).strip()
                    if msg:
                        getattr(logger, level)(f"[RenderCV] {msg}")
            except Exception:
                pass
            finally:
                try:
                    pipe.close()
                except Exception:
                    pass

        stream_threads: list[threading.Thread] = []
        for pipe, level in ((getattr(proc, "stdout", None), "info"), (getattr(proc, "stderr", None), "warning")):
            if pipe is not None and callable(getattr(pipe, "readline", None)):
                t = threading.Thread(target=_stream_pipe, args=(pipe, level), daemon=True)
                t.start()
                stream_threads.append(t)

        deadline = time.monotonic() + timeout
        while proc.poll() is None:
            if cancel_check and cancel_check():
                proc.kill()
                proc.wait(timeout=5)
                raise InterruptedError("Resume generation cancelled by user")
            if time.monotonic() >= deadline:
                proc.kill()
                proc.wait(timeout=5)
                logger.warning("Resume Engine timed out; falling back to default resume.")
                return fallback
            time.sleep(0.25)

        for t in stream_threads:
            t.join(timeout=1.0)

        if result_json.is_file():
            data = json.loads(result_json.read_text(encoding="utf-8") or "{}")
            if not data.get("ok"):
                err = str(data.get("error") or "worker failed")
                ai_runtime.disable_resume_engine(err, source="Resume Engine")
                return fallback
            if data.get("success"):
                chosen = data.get("pdf_path") or data.get("docx_path")
                if chosen and Path(chosen).is_file():
                    logger.info(f"Tailored resume ready via RenderCv: {chosen}")
                    return str(Path(chosen).resolve())
        return fallback
    except InterruptedError:
        raise
    except Exception as e:
        ai_runtime.disable_resume_engine(e, source="Resume Engine")
        logger.warning(f"Resume Engine failed ({e}); using default resume.")
        return fallback
    finally:
        try:
            result_json.unlink(missing_ok=True)
        except OSError:
            pass


def maybe_send_jd_outreach(jd_text: str | None, resume_path: str | None) -> None:
    """Fire-and-forget AI_Outreach email when the JD contains an email address."""
    cfgs = load_linkedin_configs()
    sett = cfgs.get("settings", {})
    if not bool(sett.get("use_jd_email_outreach", True)):
        return
    text = (jd_text or "").strip()
    if not text or text == "Unknown":
        return

    outreach_root = Path(
        str(sett.get("ai_outreach_root") or (WORKSPACE_ROOT / "Python_Scrapper_Telegram" / "AI_Outreach"))
    )
    if not outreach_root.is_dir():
        return
    if str(outreach_root) not in sys.path:
        sys.path.insert(0, str(outreach_root))

    try:
        from ai_outreach.generator import emails_in_text

        emails = emails_in_text(text)
    except Exception as e:
        logger.debug(f"JD email outreach unavailable: {e}")
        return

    if not emails:
        return

    attach = str(Path(resume_path).resolve()) if resume_path and Path(resume_path).is_file() else get_default_resume_path()

    def _run():
        try:
            from ai_outreach import send_outreach

            send_outreach(jd=text, resume_path=attach)
        except Exception as exc:
            logger.warning(f"JD email outreach failed (Indeed apply continues): {exc}")

    logger.info(f"JD email outreach: found {len(emails)} email(s); firing send_outreach in background.")
    threading.Thread(target=_run, daemon=True, name="indeed-jd-outreach").start()


# ---------------------------------------------------------------------------
# Separate Indeed CSV History & Screenshots (LinkedIn CSV schema parity)
# ---------------------------------------------------------------------------

def _truncate_csv(data: Any, max_length: int = 131000) -> str:
    s = "" if data is None else str(data)
    if len(s) <= max_length:
        return s
    return s[: max_length - 15] + "...[TRUNCATED]"


def applied_csv_path() -> Path:
    return REPO_ROOT / "all excels" / "all_applied_applications_history.csv"


def failed_csv_path() -> Path:
    return REPO_ROOT / "all excels" / "all_failed_applications_history.csv"


def get_applied_job_ids(csv_path: Path | None = None) -> set[str]:
    target = csv_path or applied_csv_path()
    ids: set[str] = set()
    if not target.is_file():
        return ids
    try:
        with target.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                jid = (row.get("Job ID") or "").strip()
                if jid:
                    ids.add(jid)
    except Exception as e:
        logger.warning(f"Could not read applied CSV {target}: {e}")
    return ids


def extract_indeed_jk(url: str) -> str:
    m = re.search(r"[?&]jk=([A-Za-z0-9_-]+)", url or "")
    return m.group(1) if m else (url or "")


def record_applied_job(
    *,
    job_id: str,
    title: str,
    company: str,
    work_location: str,
    work_style: str,
    description: str,
    experience_required: Any,
    skills: Any,
    resume_path: str,
    date_posted: Any,
    job_link: str,
    application_link: str = "Indeed Easy Apply",
    questions_found: Any = None,
    csv_path: Path | None = None,
) -> None:
    target = csv_path or applied_csv_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    write_header = not target.exists() or target.stat().st_size == 0
    with target.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=APPLIED_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "Job ID": _truncate_csv(job_id),
                "Title": _truncate_csv(title),
                "Company": _truncate_csv(company),
                "Work Location": _truncate_csv(work_location),
                "Work Style": _truncate_csv(work_style),
                "About Job": _truncate_csv(description),
                "Experience required": _truncate_csv(experience_required),
                "Skills required": _truncate_csv(skills),
                "HR Name": "Unknown",
                "HR Link": "Unknown",
                "Resume": _truncate_csv(resume_path),
                "Re-posted": False,
                "Date Posted": _truncate_csv(date_posted or "Unknown"),
                "Date Applied": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Job Link": _truncate_csv(job_link),
                "External Job link": _truncate_csv(application_link),
                "Questions Found": _truncate_csv(questions_found),
                "Connect Request": "N/A",
            }
        )


def record_failed_job(
    *,
    job_id: str,
    job_link: str,
    resume_tried: str,
    date_listed: Any,
    reason: str,
    exception_text: Any = "",
    application_link: str = "Indeed Easy Apply",
    screenshot_name: str = "Not Available",
    csv_path: Path | None = None,
) -> None:
    target = csv_path or failed_csv_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    write_header = not target.exists() or target.stat().st_size == 0
    with target.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FAILED_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "Job ID": _truncate_csv(job_id),
                "Job Link": _truncate_csv(job_link),
                "Resume Tried": _truncate_csv(resume_tried),
                "Date listed": _truncate_csv(date_listed or "Unknown"),
                "Date Tried": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Assumed Reason": _truncate_csv(reason),
                "Stack Trace": _truncate_csv(exception_text),
                "External Job link": _truncate_csv(application_link),
                "Screenshot Name": _truncate_csv(screenshot_name),
            }
        )


def estimate_date_posted(posted_text: str | None) -> str:
    mins = parse_job_age_minutes(posted_text)
    if mins is None:
        return "Unknown"
    return (datetime.now() - timedelta(minutes=mins)).strftime("%Y-%m-%d %H:%M:%S")


async def capture_indeed_screenshot(page: Any, job_id: str, label: str) -> str:
    screenshots_dir = REPO_ROOT / "logs" / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H.%M.%S")
    safe_label = re.sub(r"[^\w\-. ]", "_", f"{job_id} - {label} - {stamp}")[:120]
    filename = f"{safe_label}.png"
    try:
        await page.screenshot(path=str(screenshots_dir / filename))
        return filename
    except Exception:
        return "Not Available"
