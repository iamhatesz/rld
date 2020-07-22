import unittest

from flask import Response

from rld.app import init
from rld.tests.resources.envs import collect_rollout, BoxObsDiscreteActionEnv


class TestApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rollout = collect_rollout(BoxObsDiscreteActionEnv())
        cls.client = init(rollout).test_client()

    def test_get_trajectories_returns_valid_response(self):
        response = self.client.get("/trajectories")
        self.assertIsInstance(response, Response)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/json")

    def test_get_trajectories_returns_valid_trajectories_indices(self):
        response = self.client.get("/trajectories")
        self.assertIsInstance(response.json, dict)
        self.assertListEqual(response.json["trajectories"], list(range(10)))
