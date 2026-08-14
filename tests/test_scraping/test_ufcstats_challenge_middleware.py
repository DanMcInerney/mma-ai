import hashlib
import unittest
from urllib.parse import parse_qs

from scrapy import Request
from scrapy.downloadermiddlewares.cookies import CookiesMiddleware
from scrapy.http import HtmlResponse

from libs.scraping.ufcstats_challenge import (
    UfcstatsChallengeError,
    UfcstatsChallengeMiddleware,
)


CHALLENGE = b"""<!doctype html><html><head><title>Loading\xe2\x80\xa6</title></head>
<body><p>Checking your browser\xe2\x80\xa6</p><script>
function sha256(msg) { return msg; }
var nonce="0ff11e0123456789", target=new Array(2+1).join('0');
var n=0;
while(sha256(nonce+':'+n).slice(0,target.length)!==target){n++;}
var xhr=new XMLHttpRequest();
xhr.open('POST',"/__c",true);
xhr.send('nonce='+encodeURIComponent(nonce)+'&n='+n);
</script></body></html>"""


def response_for(request, body=CHALLENGE, status=200, headers=None):
    return HtmlResponse(
        request.url,
        request=request,
        body=body,
        status=status,
        headers=headers,
        encoding="utf-8",
    )


class UfcstatsChallengeMiddlewareTests(unittest.TestCase):
    def setUp(self):
        self.middleware = UfcstatsChallengeMiddleware(max_work=100_000)
        self.cookies = CookiesMiddleware()

    def test_serializes_clearance_and_retries_with_cookie(self):
        callback = lambda response: response
        first = Request(
            "http://ufcstats.com/statistics/events/completed?page=all",
            callback=callback,
        )
        second = Request(
            "http://ufcstats.com/statistics/fighters?char=a&page=all",
            callback=callback,
        )

        control = self.middleware.process_response(
            first, response_for(first), spider=None
        )
        waiting = self.middleware.process_response(
            second, response_for(second), spider=None
        )

        self.assertEqual(control.url, "http://ufcstats.com/__c")
        self.assertEqual(control.method, "POST")
        form = parse_qs(control.body.decode())
        self.assertEqual(form["nonce"], ["0ff11e0123456789"])
        answer = form["n"][0]
        self.assertTrue(
            hashlib.sha256(f"0ff11e0123456789:{answer}".encode())
            .hexdigest()
            .startswith("00")
        )
        self.assertFalse(waiting.called)

        cleared = response_for(
            control,
            body=b"",
            status=204,
            headers={"Set-Cookie": b"_fmc=offline-clearance; Path=/; HttpOnly"},
        )
        self.cookies.process_response(control, cleared)
        first_retry = self.middleware.process_response(control, cleared, spider=None)
        second_retry = waiting.result

        self.assertTrue(waiting.called)
        self.assertIs(first_retry.callback, callback)
        self.assertIs(second_retry.callback, callback)
        self.assertTrue(first_retry.dont_filter)
        self.assertTrue(second_retry.dont_filter)

        self.cookies.process_request(first_retry)
        self.cookies.process_request(second_retry)
        self.assertIn(b"_fmc=offline-clearance", first_retry.headers[b"Cookie"])
        self.assertIn(b"_fmc=offline-clearance", second_retry.headers[b"Cookie"])

    def test_persistent_challenge_fails_explicitly(self):
        request = Request(
            "http://ufcstats.com/statistics/events/completed?page=all",
            meta={"_ufcstats_challenge_retries": 1},
        )
        with self.assertRaisesRegex(UfcstatsChallengeError, "persisted"):
            self.middleware.process_response(request, response_for(request), spider=None)

    def test_rejects_difficulty_above_configured_maximum(self):
        middleware = UfcstatsChallengeMiddleware(max_difficulty=5)
        request = Request("http://ufcstats.com/statistics/events/completed?page=all")
        excessive = CHALLENGE.replace(b"new Array(2+1)", b"new Array(6+1)")

        with self.assertRaisesRegex(
            UfcstatsChallengeError, "unsupported proof difficulty 6"
        ):
            middleware.process_response(
                request, response_for(request, excessive), spider=None
            )

    def test_fails_when_proof_work_limit_is_exhausted(self):
        middleware = UfcstatsChallengeMiddleware(max_work=1)
        request = Request("http://ufcstats.com/statistics/events/completed?page=all")
        first_digest = hashlib.sha256(b"0ff11e0123456789:0").hexdigest()
        self.assertFalse(first_digest.startswith("00"))

        with self.assertRaisesRegex(UfcstatsChallengeError, "1-hash work limit"):
            middleware.process_response(request, response_for(request), spider=None)

    def test_rejects_requests_beyond_waiter_queue_bound(self):
        middleware = UfcstatsChallengeMiddleware(
            max_work=100_000,
            max_waiters=1,
        )
        requests = [
            Request(f"http://ufcstats.com/statistics/fighters?char={char}&page=all")
            for char in "abc"
        ]

        control = middleware.process_response(
            requests[0], response_for(requests[0]), spider=None
        )
        waiting = middleware.process_response(
            requests[1], response_for(requests[1]), spider=None
        )
        with self.assertRaisesRegex(UfcstatsChallengeError, "exceeded 1 requests"):
            middleware.process_response(
                requests[2], response_for(requests[2]), spider=None
            )

        self.assertEqual(control.url, "http://ufcstats.com/__c")
        self.assertFalse(waiting.called)

    def test_changed_format_fails_instead_of_guessing(self):
        request = Request("http://ufcstats.com/statistics/fighters?char=a&page=all")
        changed = CHALLENGE.replace(b"new Array(2+1)", b"difficulty(2)")
        with self.assertRaisesRegex(UfcstatsChallengeError, "format changed"):
            self.middleware.process_response(
                request, response_for(request, changed), spider=None
            )

    def test_interactive_captcha_fails_explicitly(self):
        request = Request("http://ufcstats.com/statistics/fighters?char=a&page=all")
        captcha = b"<html><div class='g-recaptcha'></div></html>"
        with self.assertRaisesRegex(UfcstatsChallengeError, "interactive CAPTCHA"):
            self.middleware.process_response(
                request, response_for(request, captcha), spider=None
            )


if __name__ == "__main__":
    unittest.main()
