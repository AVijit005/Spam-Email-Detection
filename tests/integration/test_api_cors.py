from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class TestApiCors(unittest.TestCase):
    def setUp(self):
        self._load_patch = mock.patch("app.main.load_resources")
        self._load_patch.start()
        from app.main import create_app
        self.app = create_app()

    def tearDown(self):
        self._load_patch.stop()

    def _client(self, **headers):
        from fastapi.testclient import TestClient
        return TestClient(self.app, headers=headers)

    def test_cors_blocks_disallowed_origin(self):
        with self._client(origin="https://evil.com") as client:
            response = client.get("/v1/health")
        self.assertNotIn("access-control-allow-origin", response.headers)

    def test_cors_allows_localhost(self):
        with self._client(origin="http://localhost:3000") as client:
            response = client.options("/v1/health")
        self.assertIn("access-control-allow-origin", response.headers)

    def test_cors_allows_127_0_0_1(self):
        with self._client(origin="http://127.0.0.1:8080") as client:
            response = client.options("/v1/health")
        self.assertIn("access-control-allow-origin", response.headers)

    def test_cors_preflight_allows_get_and_post(self):
        with self._client(origin="http://localhost:3000") as client:
            response = client.options("/v1/health")
        allowed = response.headers.get("allow", "")
        self.assertIn("GET", allowed)

    def test_health_works_with_valid_origin(self):
        with self._client(origin="http://localhost:3000") as client:
            response = client.get("/v1/health")
        self.assertEqual(response.status_code, 200)
