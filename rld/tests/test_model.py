import unittest

import gym
import numpy as np
import torch

from rld.model import pack_array, unpack_tensor


class TestPacking(unittest.TestCase):
    def test_pack_array_box(self):
        obs_space = gym.spaces.Box(-1.0, 1.0, shape=(10, 4))
        obs = obs_space.sample()
        packed = pack_array(obs, obs_space)
        self.assertEqual(packed.shape, (40,))

    def test_pack_array_dict(self):
        obs_space = gym.spaces.Dict(
            a=gym.spaces.Box(-1.0, 1.0, shape=(10, 4)),
            b=gym.spaces.Box(-1.0, 1.0, shape=(2, 5)),
        )
        obs = obs_space.sample()
        packed = pack_array(obs, obs_space)
        self.assertEqual(packed.shape, (50,))

    def test_unpack_tensor_box(self):
        obs_space = gym.spaces.Box(-1.0, 1.0, shape=(10, 4))
        obs = torch.rand((40,), dtype=torch.float32)
        unpacked = unpack_tensor(obs, obs_space)
        self.assertEqual(unpacked.shape, (10, 4))

    def test_unpack_tensor_box_with_batch_dim(self):
        obs_space = gym.spaces.Box(-1.0, 1.0, shape=(10, 4))
        batch_size = 5
        obs = torch.rand((batch_size, 40), dtype=torch.float32)
        unpacked = unpack_tensor(obs, obs_space)
        self.assertEqual(unpacked.shape, (batch_size, 10, 4))

    def test_pack_unpack_box_with_batch_dim(self):
        obs_space = gym.spaces.Box(-1.0, 1.0, shape=(10, 4))
        batch_size = 5

        origin_obs = [obs_space.sample() for _ in range(batch_size)]
        origin_obs_batch = np.stack(origin_obs, axis=0)

        packed = np.stack([pack_array(obs, obs_space) for obs in origin_obs], axis=0)
        packed_tensor = torch.tensor(packed)
        unpacked_obs_batch = unpack_tensor(packed_tensor, obs_space).cpu().numpy()

        np.testing.assert_almost_equal(origin_obs_batch, unpacked_obs_batch)

    def test_unpack_tensor_dict(self):
        obs_space = gym.spaces.Dict(
            a=gym.spaces.Box(-1.0, 1.0, shape=(10, 4)),
            b=gym.spaces.Box(-1.0, 1.0, shape=(2, 5)),
        )
        obs = torch.rand(50, dtype=torch.float32)
        unpacked = unpack_tensor(obs, obs_space)
        self.assertIsInstance(unpacked, dict)
        self.assertIn("a", unpacked)
        self.assertIn("b", unpacked)
        self.assertEqual(unpacked["a"].shape, (10, 4))
        self.assertEqual(unpacked["b"].shape, (2, 5))
