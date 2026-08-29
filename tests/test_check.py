"""Preflight diagnostics tests.

Every one of these failures surfaces as a bare 403 or 400 in normal use, and
the four causes need four different fixes. Classifying them wrong sends someone
to regenerate a perfectly good key when the real problem was that the API was
never enabled - so the mapping is worth pinning down.
"""

import unittest

from ytengine.check import CheckResult, _classify, _message, check_key, render


class TestClassify(unittest.TestCase):
    def test_api_not_enabled(self):
        body = ('{"error":{"message":"YouTube Data API v3 has not been used in project '
                '123 before or it is disabled.","errors":[{"reason":"accessNotConfigured"}]}}')
        status, fix = _classify(403, body)
        self.assertEqual(status, "API not enabled")
        self.assertIn("Library", fix)

    def test_quota_exhausted_is_not_a_broken_key(self):
        status, fix = _classify(403, '{"error":{"message":"quotaExceeded"}}')
        self.assertEqual(status, "quota exhausted")
        self.assertIn("resets", fix)

    def test_invalid_key(self):
        status, _ = _classify(400, '{"error":{"message":"API key not valid."}}')
        self.assertEqual(status, "key invalid")

    def test_referrer_restriction_is_called_out(self):
        """The costliest failure: works in a browser, fails from every script,
        and the error never says so."""
        body = '{"error":{"message":"Requests from referer <empty> are blocked."}}'
        status, fix = _classify(403, body)
        self.assertEqual(status, "key restricted")
        self.assertIn("browser", fix)

    def test_unknown_403_still_gives_direction(self):
        status, fix = _classify(403, '{"error":{"message":"nope"}}')
        self.assertTrue(fix)


class TestMessage(unittest.TestCase):
    def test_extracts_the_human_sentence(self):
        body = '{"error":{"message":"API key not valid.","errors":[{"reason":"badRequest"}]}}'
        self.assertEqual(_message(body), "API key not valid. [badRequest]")

    def test_survives_non_json(self):
        self.assertEqual(_message("<html>502</html>"), "<html>502</html>")

    def test_survives_unexpected_shape(self):
        self.assertTrue(_message('{"error":"a string not an object"}'))


class TestCheck(unittest.TestCase):
    def test_missing_key_needs_no_network(self):
        r = check_key(None)
        self.assertFalse(r.ok)
        self.assertEqual(r.status, "no key")
        self.assertEqual(r.quota_spent, 0)

    def test_render_states_the_analytics_boundary(self):
        """A key gets public data only. CTR and retention need OAuth and only
        cover channels you own - worth saying before someone plans around it."""
        out = render(CheckResult(ok=True, status="working", detail="d",
                                 unlocked=["harvest  x"]))
        self.assertIn("retention", out)
        self.assertIn("OAuth", out)

    def test_render_shows_the_fix(self):
        self.assertIn("Enable", render(CheckResult(
            ok=False, status="API not enabled", detail="d", fix="Enable it")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
