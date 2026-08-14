"""Downloader middleware for UFCStats' JavaScript proof-of-work page."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Hashable
from urllib.parse import urlsplit

from scrapy import FormRequest, Request
from scrapy.http import Response
from scrapy.http.request import NO_CALLBACK
from twisted.internet.defer import Deferred
from twisted.python.failure import Failure


class UfcstatsChallengeError(RuntimeError):
    """Raised when UFCStats clearance cannot be obtained safely."""


@dataclass
class _Clearance:
    owner: Request
    waiters: list[tuple[Deferred, Request]] = field(default_factory=list)


class UfcstatsChallengeMiddleware:
    """Solve UFCStats' bounded SHA-256 challenge once per cookie jar."""

    _CONTROL_META = "_ufcstats_challenge_control"
    _RETRY_META = "_ufcstats_challenge_retries"
    _NONCE_RE = re.compile(r"\bnonce\s*=\s*['\"]([0-9a-fA-F]{1,128})['\"]")
    _DIFFICULTY_RE = re.compile(
        r"\btarget\s*=\s*new\s+Array\(\s*(\d+)\s*\+\s*1\s*\)"
        r"\.join\(\s*['\"]0['\"]\s*\)"
    )

    def __init__(
        self,
        max_difficulty: int = 5,
        max_work: int = 2_000_000,
        max_retries: int = 1,
        max_waiters: int = 64,
    ) -> None:
        self.max_difficulty = max_difficulty
        self.max_work = max_work
        self.max_retries = max_retries
        self.max_waiters = max_waiters
        self._clearances: dict[tuple[str, Hashable], _Clearance] = {}
        self._control_keys: dict[str, tuple[str, Hashable]] = {}
        self._next_control_id = 0

    @classmethod
    def from_crawler(cls, crawler):
        settings = crawler.settings
        return cls(
            max_difficulty=settings.getint(
                "UFCSTATS_CHALLENGE_MAX_DIFFICULTY", 5
            ),
            max_work=settings.getint("UFCSTATS_CHALLENGE_MAX_WORK", 2_000_000),
            max_retries=settings.getint("UFCSTATS_CHALLENGE_MAX_RETRIES", 1),
            max_waiters=settings.getint("UFCSTATS_CHALLENGE_MAX_WAITERS", 64),
        )

    def process_response(self, request, response, spider=None):
        control_id = request.meta.get(self._CONTROL_META)
        if control_id is not None:
            return self._finish_clearance(control_id, response)

        if not self._is_ufcstats(request.url):
            return response

        if self._has_interactive_captcha(response):
            raise UfcstatsChallengeError(
                f"UFCStats returned an interactive CAPTCHA for {request.url}"
            )

        if not self._is_known_challenge(response):
            return response

        retry_count = request.meta.get(self._RETRY_META, 0)
        if retry_count >= self.max_retries:
            raise UfcstatsChallengeError(
                f"UFCStats proof-of-work challenge persisted after {retry_count} "
                f"clearance attempt(s) for {request.url}"
            )

        key = self._clearance_key(request)
        active = self._clearances.get(key)
        if active is not None:
            if len(active.waiters) >= self.max_waiters:
                raise UfcstatsChallengeError(
                    f"UFCStats clearance queue exceeded {self.max_waiters} requests"
                )
            waiter = Deferred()
            waiter.addCallback(lambda _ignored: self._retry(request))
            active.waiters.append((waiter, request))
            return waiter

        nonce, difficulty = self._parse_challenge(response)
        answer = self._solve(nonce, difficulty)
        self._next_control_id += 1
        control_id = str(self._next_control_id)
        self._clearances[key] = _Clearance(owner=request)
        self._control_keys[control_id] = key

        meta = request.meta.copy()
        meta[self._CONTROL_META] = control_id
        return FormRequest(
            url=f"{urlsplit(request.url).scheme}://{urlsplit(request.url).netloc}/__c",
            method="POST",
            formdata={"nonce": nonce, "n": str(answer)},
            callback=NO_CALLBACK,
            dont_filter=True,
            priority=request.priority + 1000,
            meta=meta,
        )

    def process_exception(self, request, exception, spider=None):
        control_id = request.meta.get(self._CONTROL_META)
        if control_id is None:
            return None
        error = UfcstatsChallengeError(
            f"UFCStats clearance request failed: {exception}"
        )
        self._abort_clearance(control_id, error)
        raise error from exception

    @staticmethod
    def _is_ufcstats(url: str) -> bool:
        return (urlsplit(url).hostname or "").lower() in {
            "ufcstats.com",
            "www.ufcstats.com",
        }

    @staticmethod
    def _is_known_challenge(response: Response) -> bool:
        if response.status != 200:
            return False
        body = response.body.lower()
        return b"<title>loading" in body and b"checking your browser" in body

    @staticmethod
    def _has_interactive_captcha(response: Response) -> bool:
        body = response.body.lower()
        return any(
            marker in body
            for marker in (b"g-recaptcha", b"h-captcha", b"hcaptcha", b"cf-turnstile")
        )

    def _parse_challenge(self, response: Response) -> tuple[str, int]:
        try:
            script = response.text
        except AttributeError as exc:
            raise UfcstatsChallengeError(
                "UFCStats challenge was not an HTML text response"
            ) from exc

        if 'xhr.open(\'POST\',"/__c"' not in script and 'xhr.open("POST","/__c"' not in script:
            raise UfcstatsChallengeError(
                "UFCStats challenge endpoint format changed"
            )
        nonce_match = self._NONCE_RE.search(script)
        difficulty_match = self._DIFFICULTY_RE.search(script)
        if nonce_match is None or difficulty_match is None:
            raise UfcstatsChallengeError("UFCStats challenge format changed")

        difficulty = int(difficulty_match.group(1))
        if difficulty < 1 or difficulty > self.max_difficulty:
            raise UfcstatsChallengeError(
                f"UFCStats published unsupported proof difficulty {difficulty}; "
                f"maximum is {self.max_difficulty}"
            )
        return nonce_match.group(1), difficulty

    def _solve(self, nonce: str, difficulty: int) -> int:
        target = "0" * difficulty
        for answer in range(self.max_work):
            digest = hashlib.sha256(f"{nonce}:{answer}".encode()).hexdigest()
            if digest.startswith(target):
                return answer
        raise UfcstatsChallengeError(
            f"UFCStats proof exceeded the {self.max_work}-hash work limit"
        )

    @staticmethod
    def _clearance_key(request: Request) -> tuple[str, Hashable]:
        host = (urlsplit(request.url).hostname or "").lower()
        cookie_jar = request.meta.get("cookiejar")
        try:
            hash(cookie_jar)
        except TypeError as exc:
            raise UfcstatsChallengeError("Scrapy cookiejar key must be hashable") from exc
        return host, cookie_jar

    def _finish_clearance(self, control_id: str, response: Response) -> Request:
        key = self._control_keys.pop(control_id, None)
        clearance = self._clearances.pop(key, None) if key is not None else None
        if clearance is None:
            raise UfcstatsChallengeError("Unknown UFCStats clearance response")

        set_cookie = b"\n".join(response.headers.getlist("Set-Cookie"))
        if response.status != 204 or b"_fmc=" not in set_cookie:
            error = UfcstatsChallengeError(
                "UFCStats clearance was rejected or did not set the _fmc cookie "
                f"(HTTP {response.status})"
            )
            self._fail_waiters(clearance, error)
            raise error

        for waiter, _request in clearance.waiters:
            if not waiter.called:
                waiter.callback(None)
        return self._retry(clearance.owner)

    def _abort_clearance(self, control_id: str, error: Exception) -> None:
        key = self._control_keys.pop(control_id, None)
        clearance = self._clearances.pop(key, None) if key is not None else None
        if clearance is not None:
            self._fail_waiters(clearance, error)

    @staticmethod
    def _fail_waiters(clearance: _Clearance, error: Exception) -> None:
        for waiter, _request in clearance.waiters:
            if not waiter.called:
                waiter.errback(Failure(error))

    def _retry(self, request: Request) -> Request:
        retry_count = request.meta.get(self._RETRY_META, 0) + 1
        return request.replace(
            dont_filter=True,
            meta={**request.meta, self._RETRY_META: retry_count},
        )
