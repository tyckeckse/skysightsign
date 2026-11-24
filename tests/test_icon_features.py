from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import icon_features


def _response(hourly: dict) -> mock.Mock:
    response = mock.Mock()
    response.json.return_value = {"hourly": hourly}
    response.raise_for_status.return_value = None
    return response


class AddIconFeaturesTest(unittest.TestCase):
    def test_aligns_on_time_and_fills_gaps_with_none(self):
        hourly = {"time": ["T1", "T2", "T3"]}
        icon = {"time": ["T2", "T3"], "cloud_cover": [40, 90], "cloud_cover_low": [10, 80],
                "cloud_cover_mid": [0, 0], "cloud_cover_high": [30, 10]}
        with mock.patch.object(icon_features.requests, "get", return_value=_response(icon)) as get:
            icon_features.add_icon_features("url", {"hourly": "x", "daily": "y", "latitude": "1"}, hourly)
        self.assertEqual(hourly["icon_cloud_cover"], [None, 40, 90])
        self.assertEqual(hourly["icon_cloud_cover_low"], [None, 10, 80])
        sent = get.call_args.kwargs["params"]
        self.assertEqual(sent["models"], "icon_seamless")
        self.assertNotIn("daily", sent)

    def test_failed_request_gives_none(self):
        hourly = {"time": ["T1"]}
        with mock.patch.object(icon_features.requests, "get", side_effect=requests.ConnectionError):
            icon_features.add_icon_features("url", {}, hourly)
        self.assertEqual(hourly["icon_cloud_cover"], [None])


if __name__ == "__main__":
    unittest.main()
