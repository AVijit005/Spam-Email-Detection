from __future__ import annotations

import unittest
from unittest import mock

from fastapi import HTTPException


class TestRequireApiKey(unittest.TestCase):
    def setUp(self):
        self._settings_patch = mock.patch("app.core.auth.settings")
        self.mock_settings = self._settings_patch.start()

    def tearDown(self):
        self._settings_patch.stop()

    def _call(self, api_key=mock.sentinel.absent):
        from app.core.auth import require_api_key
        kwargs = {}
        if api_key is not mock.sentinel.absent:
            kwargs["api_key"] = api_key
        return require_api_key(**kwargs)

    def test_allows_when_no_key_configured(self):
        self.mock_settings.api_key = ""

        result = self._call(api_key=None)

        self.assertIsNone(result)

    def test_allows_when_no_key_configured_header_present_but_ignored(self):
        self.mock_settings.api_key = ""

        result = self._call(api_key="anything")

        self.assertIsNone(result)

    def test_rejects_missing_header_when_key_configured(self):
        self.mock_settings.api_key = "secret"

        with self.assertRaises(HTTPException) as ctx:
            self._call(api_key=None)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("Invalid or missing", ctx.exception.detail)

    def test_rejects_wrong_key_when_configured(self):
        self.mock_settings.api_key = "secret"

        with self.assertRaises(HTTPException) as ctx:
            self._call(api_key="wrong")
        self.assertEqual(ctx.exception.status_code, 401)

    def test_allows_correct_key(self):
        self.mock_settings.api_key = "secret"

        result = self._call(api_key="secret")

        self.assertIsNone(result)
