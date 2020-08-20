import ray
from ray.tune import run

if __name__ == "__main__":
    ray.init()

    run(
        "PPO",
        name="atari-breakout",
        local_dir="/tmp/ray-results",
        checkpoint_at_end=True,
        stop={"training_iteration": 1},
        config={"env": "BreakoutNoFrameskip-v4", "framework": "torch",},
    )
