from __future__ import annotations

import sys
import unittest
from http.server import BaseHTTPRequestHandler
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
