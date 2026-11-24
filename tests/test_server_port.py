from __future__ import annotations

import sys
import unittest
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import app_server


class ExclusivePortTest(unittest.TestCase):
    def test_second_server_on_same_port_fails(self):
        first = app_server.ExclusiveHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
        try:
            port = first.server_address[1]
            with self.assertRaises(OSError):
                app_server.ExclusiveHTTPServer(("127.0.0.1", port), BaseHTTPRequestHandler)
        finally:
            first.server_close()


class AppHandlerSecurityAndErrorsTest(unittest.TestCase):
    def test_static_path_whitelist(self):
        self.assertTrue(app_server.is_allowed_static_path("/"))
        self.assertTrue(app_server.is_allowed_static_path("/index.html"))
        self.assertTrue(app_server.is_allowed_static_path("/icons/yr/01d.svg"))
        self.assertFalse(app_server.is_allowed_static_path("/mlflow.db"))
        self.assertFalse(app_server.is_allowed_static_path("/data/app_server.log"))
        self.assertFalse(app_server.is_allowed_static_path("/scripts/app_server.py"))
        self.assertFalse(app_server.is_allowed_static_path("/icons/../mlflow.db"))
        self.assertFalse(app_server.is_allowed_static_path("/icons/%2e%2e/mlflow.db"))

    def test_handle_forecast_api_returns_400_on_invalid_hours(self):
        handler = object.__new__(app_server.AppHandler)
        handler.send_api_error = mock.Mock()

        handler.handle_forecast_api("hours=abc")
        handler.send_api_error.assert_called_once()
        self.assertEqual(handler.send_api_error.call_args[0][0], HTTPStatus.BAD_REQUEST)

        handler.send_api_error.reset_mock()
        handler.handle_forecast_api("hours=0")
        self.assertEqual(handler.send_api_error.call_args[0][0], HTTPStatus.BAD_REQUEST)


if __name__ == "__main__":
    unittest.main()
