import ray
from ray.tune import run

if __name__ == "__main__":
    ray.init()

    run(
        "PPO",
        local_dir="/tmp/cartpole",
        checkpoint_at_end=True,
        stop={"episode_reward_mean": 200.0},
        config={"env": "CartPole-v1", "framework": "torch",},
    )
