import time
from typing import Union

from playwright.sync_api import Page

from config.logger_config import logger
from src.dashboard.runtime import StopRequested
from src.job_manager.authenticator import BaseAuthenticator
from src.job_manager.indeed import indeed_base_url
from src.utils.bot_control import should_stop
from src.utils.browser_utils import find_element_safely, safe_click, safe_fill
from src.utils.utils import async_pause


class IndeedAuthenticator(BaseAuthenticator):
    """Class for Indeed login and session management"""

    # secure.indeed.com serves the login flow of every country domain
    INDEED_LOGIN_URL = "https://secure.indeed.com/account/login"
    INDEED_HOME_URL = f"{indeed_base_url()}/?lang=en"

    def __init__(self, page: Union[Page, any] = None):
        super().__init__(page)
        logger.info("Indeed authenticator initialized")

    def set_parameters(self, email: str, password: str = None) -> None:
        logger.info("Setting Indeed Authenticator parameters")
        super().set_parameters(email, password)

    @staticmethod
    async def is_security_challenge(page: any) -> bool:
        """Detect Indeed/Cloudflare human verification or CAPTCHA (never bypassed)."""
        try:
            raw_url = getattr(page, "url", None)
            if isinstance(raw_url, str):
                current_url = raw_url.lower()
                if "/challenge" in current_url or "captcha" in current_url or "__cf_chl" in current_url:
                    return True
            mod = getattr(type(page), "__module__", "") or ""
            if not mod.startswith(("playwright", "patchright")):
                return False
            if callable(getattr(page, "title", None)):
                title = await page.title()
                if isinstance(title, str) and (
                    "just a moment" in title.lower() or "security check" in title.lower()
                ):
                    return True
            for sel in ("[data-testid='captcha']", "#challenge-running"):
                loc = page.locator(sel)
                for i in range(await loc.count()):
                    if await loc.nth(i).is_visible():
                        return True
            # Ignore SmartApply's passive invisible reCAPTCHA/hCaptcha footer widgets; flag interactive challenges.
            for sel in ("iframe[src*='turnstile']", "iframe[src*='recaptcha']", "iframe[src*='hcaptcha']"):
                loc = page.locator(sel)
                for i in range(await loc.count()):
                    el = loc.nth(i)
                    if not await el.is_visible():
                        continue
                    box = await el.bounding_box()
                    if not box or box.get("width", 0) < 50 or box.get("height", 0) < 40:
                        continue
                    src = ((await el.get_attribute("src")) or "").lower()
                    if "recaptcha" in sel and "/bframe" not in src and (
                        "size=invisible" in src or ("/anchor" in src and box.get("height", 0) < 70)
                    ):
                        continue
                    if "hcaptcha" in sel and "invisible" in src and box.get("height", 0) < 120:
                        continue
                    return True
        except Exception:
            pass
        return False

    async def wait_for_manual_challenge_if_present(self, max_wait_seconds: int = 180) -> bool:
        """Pause and poll while user solves CAPTCHA/Cloudflare manually."""
        if not await self.is_security_challenge(self.page):
            return True
        logger.warning(
            "Indeed security verification / CAPTCHA detected — please complete it manually in the Chrome window."
        )
        deadline = time.monotonic() + max_wait_seconds
        while time.monotonic() < deadline:
            if should_stop():
                raise StopRequested("Stop requested while waiting on security verification")
            await async_pause(1.2, 2.5)
            if not await self.is_security_challenge(self.page):
                logger.info("Security verification cleared; continuing.")
                return True
        logger.error("Security verification timed out.")
        return False

    async def is_logged_in(self) -> bool:
        """Check if user is logged into Indeed"""
        try:
            if should_stop():
                raise StopRequested("Stop requested before Indeed login check")
            await self.page.goto(self.INDEED_HOME_URL, wait_until="domcontentloaded")
            logger.info("Checking if user is logged into Indeed...")

            if await self.is_security_challenge(self.page):
                await self.wait_for_manual_challenge_if_present()

            if should_stop():
                raise StopRequested("Stop requested during Indeed login check")

            current_url = self.page.url or ""
            if "login" in current_url or "signin" in current_url or "/auth" in current_url:
                logger.warning("Redirected to login page, user not authorized")
                return False

            mod = getattr(type(self.page), "__module__", "") or ""
            if mod.startswith(("playwright", "patchright")):
                try:
                    sign_in_cnt = await self.page.locator(
                        "[data-gnav-element-name='SignIn'], a:has-text('Sign in')"
                    ).count()
                    if sign_in_cnt > 0:
                        logger.warning("Sign in button visible on Indeed homepage — user is not logged in")
                        return False
                except Exception:
                    pass

            for selector in (
                "[data-gnav-element-name='AccountMenu']",
                "[data-gnav-element-name='Notifications']",
                "[data-gnav-element-name='Messages']",
                "button[aria-label*='Account']",
                "a[href*='/myjobs']",
            ):
                account_menu = await find_element_safely(self.page, selector, timeout=900)
                if account_menu:
                    logger.info("Account menu found, user is logged into Indeed")
                    return True

            if mod.startswith(("playwright", "patchright")):
                try:
                    welcome_cnt = await self.page.locator(":text-matches('Welcome,\\\\s+', 'i')").count()
                    if welcome_cnt > 0:
                        logger.info("Indeed personalized homepage detected ('Welcome, ...') — user is logged in")
                        return True
                except Exception:
                    pass

            logger.warning("Account menu not found, user is not logged in")
            return False

        except StopRequested:
            raise
        except Exception as e:
            logger.error(f"Error checking Indeed authorization status: {e}")
            return False

    async def handle_login(self) -> bool:
        """Navigate to Indeed login and authenticate"""
        logger.info("Navigating to Indeed login page...")
        try:
            if should_stop():
                raise StopRequested("Stop requested before Indeed login navigation")
            await self.page.goto(self.INDEED_LOGIN_URL)
            await async_pause(1, 2)
            if should_stop():
                raise StopRequested("Stop requested during Indeed login")
            if await self.is_security_challenge(self.page):
                await self.wait_for_manual_challenge_if_present()
            return await self.enter_credentials()
        except StopRequested:
            raise
        except Exception as e:
            logger.error(f"Error accessing Indeed login: {e}")
            return False

    async def enter_credentials(self) -> bool:
        """Enter email on Indeed login form and wait automatically for manual OTP/login completion"""
        logger.info("Entering email in Indeed login form...")
        try:
            if should_stop():
                raise StopRequested("Stop requested before entering Indeed credentials")
            mod = getattr(type(self.page), "__module__", "") or ""
            is_real = mod.startswith(("playwright", "patchright"))
            if is_real:
                try:
                    accept_btn = self.page.locator("#onetrust-accept-btn-handler")
                    if await accept_btn.count() > 0 and await accept_btn.first.is_visible():
                        await accept_btn.first.click()
                        await async_pause(0.5, 1)
                except Exception:
                    pass

            if not await safe_fill(self.page, "input[type='email']", self.email or ""):
                # Persistent session may already be signed in after challenge resolved
                if is_real and await self.check_login_success(max_attempts=5):
                    return True
                logger.error("Failed to fill email field")
                return False

            if should_stop():
                raise StopRequested("Stop requested after entering email")

            continue_selectors = [
                "button[data-tn-element='auth-page-email-submit-button']",
                "button[type='submit']",
                "#login-submit-button",
            ]
            for selector in continue_selectors:
                if await safe_click(self.page, selector, timeout=6000):
                    break

            logger.info(
                "Please complete Indeed login / OTP / Google / verification in Chrome. "
                "The bot will continue automatically once signed in (or press STOP BOT to abort)..."
            )
            return await self.check_login_success(max_attempts=120)

        except StopRequested:
            raise
        except Exception as e:
            logger.error(f"Error during Indeed credential entry: {e}")
            return False

    async def check_login_success(self, max_attempts: int = 30) -> bool:
        """Verify successful Indeed login"""
        try:
            for attempt in range(max_attempts):
                if should_stop():
                    raise StopRequested("Stop requested while waiting for Indeed login")

                current_url = self.page.url or ""
                if attempt == 0 or attempt % 5 == 0:
                    logger.debug(f"Login check attempt {attempt + 1}/{max_attempts}, current URL: {current_url}")

                if (
                    "/challenge" in current_url
                    or "captcha" in current_url
                    or "__cf_chl" in current_url
                ):
                    logger.warning(
                        "Indeed security challenge detected - waiting for manual resolution"
                    )
                    await async_pause(1.5, 3.0)
                    continue

                if "login" in current_url or "signin" in current_url or "/auth" in current_url:
                    error_locator = self.page.locator(".icl-Alert--error, .error-message")
                    if await error_locator.count() > 0:
                        error_text = await error_locator.first.text_content()
                        if error_text and error_text.strip():
                            logger.error(f"Indeed login error: {error_text.strip()}")
                            return False
                    await async_pause(1, 2)
                    continue

                success_patterns = [
                    "indeed.com/jobs",
                    "indeed.com/myjobs",
                    "indeed.com/?",
                    "indeed.com/account",
                    "indeed.com/settings/account",
                    "indeed.com/",
                ]
                if "secure.indeed.com" not in current_url and any(
                    pattern in current_url for pattern in success_patterns
                ):
                    sign_in_btn = await find_element_safely(
                        self.page, "[data-gnav-element-name='SignIn']", timeout=1500
                    )
                    if not sign_in_btn and not await self.is_security_challenge(self.page):
                        logger.info("Indeed login successful - reached authenticated page")
                        return True

                await async_pause(1, 2)

            logger.info("Indeed login timeout reached")
            return False

        except StopRequested:
            raise
        except Exception as e:
            logger.error(f"Error checking Indeed login success: {e}")
            return False
