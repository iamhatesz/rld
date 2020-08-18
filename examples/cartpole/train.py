import ray
from ray.tune import run

if __name__ == "__main__":
    ray.init()

    run(
        "PPO",
        name="cartpole",
        local_dir="/tmp/ray-results",
        checkpoint_at_end=True,
        stop={"episode_reward_mean": 450.0},
        config={"env": "CartPole-v1", "framework": "torch",},
    )
