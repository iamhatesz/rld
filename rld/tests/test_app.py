import unittest

from flask import Response

from rld.app import init
from rld.tests.resources.envs import collect_rollout, BoxObsDiscreteActionEnv


class TestApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rollout = collect_rollout(BoxObsDiscreteActionEnv())
        cls.client = init(cls.rollout).test_client()

    def test_accessing_invalid_endpoint_returns_error_message(self):
        response = self.client.get("/invalid")
        self.assertIsInstance(response, Response)
        self.assertEqual(response.status_code, 404)

    def test_get_trajectories_returns_valid_response(self):
        response = self.client.get("/trajectories")
        self.assertIsInstance(response, Response)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/json")

    def test_get_trajectories_returns_valid_trajectories_indices(self):
        response = self.client.get("/trajectories")
        self.assertIsInstance(response.json, dict)
        self.assertEqual(response.json["length"], len(self.rollout))
        self.assertIsInstance(response.json["trajectories"], list)
        self.assertGreater(len(response.json["trajectories"]), 0)

    def test_get_trajectory_returns_valid_response(self):
        response = self.client.get("/trajectory/0")
        self.assertIsInstance(response, Response)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/json")

    def test_get_trajectory_returns_valid_timesteps(self):
        response = self.client.get("/trajectory/0")
        self.assertIsInstance(response.json, dict)
        self.assertEqual(response.json["length"], len(self.rollout.trajectories[0]))
        self.assertIsInstance(response.json["timesteps"], list)
        self.assertGreater(len(response.json["timesteps"]), 0)
        self.assertIsInstance(response.json["timesteps"][0], dict)
        self.assertTrue("obs" in response.json["timesteps"][0])
        self.assertTrue("action" in response.json["timesteps"][0])
        self.assertTrue("reward" in response.json["timesteps"][0])
        self.assertTrue("done" in response.json["timesteps"][0])
        self.assertTrue("info" in response.json["timesteps"][0])
        self.assertTrue("attributations" in response.json["timesteps"][0])
