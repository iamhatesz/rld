import importlib
from pathlib import Path

import click

from rld.app import init
from rld.attributation import (
    AttributationTrajectoryIterator,
    attribute_trajectory,
    AttributationNormalizer,
)
from rld.config import Config
from rld.exception import InvalidConfigProvided
from rld.rollout import (
    RayFileRolloutReader,
    FileRolloutWriter,
    FileRolloutReader,
    Rollout,
)

_KNOWN_EXTRA_EXTENSIONS = [
    # These extensions are sometimes used by Python shelve module.
    ".dat",
    ".bak",
    ".dir",
]


@click.group()
def main():
    pass


@main.command()
@click.option("--out", default=str(Path.home() / "rollout.rld"))
@click.argument("rollout")
def convert(out: str, rollout: str):
    """
    This script converts rollout from Ray format (created using `rllib rollout`) into
    rld format.
    """
    rollout_path = Path(rollout)
    _verify_rollout_exists(rollout_path)
    out_path = Path(out)

    click.echo(f"Converting Ray rollout `{rollout}` to `{out}`...")

    ray_rollout_reader = RayFileRolloutReader(rollout_path)

    with FileRolloutWriter(out_path) as rollout_writer:
        for trajectory in ray_rollout_reader:
            rollout_writer.write(trajectory)

    click.echo("Done.")


@main.command()
@click.option("--out", default=str(Path.home() / "rollout.rld"))
@click.option("--rllib", default=False, is_flag=True)
@click.argument("config")
@click.argument("rollout")
def attribute(out: str, rllib: bool, config: str, rollout: str):
    """
    This script calculates attributations for the given `rollout`, using configuration
    (e.g. a model definition) stored in the `config` file.
    """
    config_path = Path(config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file `{config}` not found.")
    rollout_path = Path(rollout)
    _verify_rollout_exists(rollout_path)
    out_path = Path(out)

    config = _load_config(config_path)

    click.echo("Attributing rollout...")

    with FileRolloutWriter(out_path) as rollout_writer:
        if rllib:
            reader = RayFileRolloutReader(rollout_path)
        else:
            reader = FileRolloutReader(rollout_path)

        normalizer = AttributationNormalizer(
            obs_space=config.model.obs_space(),
            obs_image_channel_dim=config.normalize_obs_image_channel_dim,
            mode=config.normalize_sign,
            outlier_percentile=config.normalize_outlier_percentile,
        )

        for trajectory in reader:
            trajectory_it = AttributationTrajectoryIterator(
                trajectory,
                model=config.model,
                baseline=config.baseline,
                target=config.target,
            )
            attr_trajectory = attribute_trajectory(
                trajectory_it, model=config.model, normalizer=normalizer
            )
            rollout_writer.write(attr_trajectory)

    click.echo("Done.")


@main.command()
@click.option("--host", default="localhost")
@click.option("--port", default=5000)
@click.option("--viewer", default="none")
@click.option("--debug/--no-debug", default=False)
@click.option("--limit-steps", default=0)
@click.argument("rollout")
def start(
    host: str, port: int, viewer: str, debug: bool, limit_steps: int, rollout: str
):
    """
    This script runs web server serving application to visualize calculated
    attributations.
    """
    rollout_path = Path(rollout)
    _verify_rollout_exists(rollout_path)

    rollout_obj = FileRolloutReader(rollout_path)
    # TODO Refactor this
    app = init(
        Rollout([t for t in rollout_obj]),
        host=host,
        port=port,
        viewer=viewer,
        debug=debug,
        limit_steps=limit_steps,
    )
    app.run(host=host, port=port, debug=debug)


def _load_config(config_path: Path) -> Config:
    config_spec = importlib.util.spec_from_file_location("config", config_path)
    config_mod = importlib.util.module_from_spec(config_spec)
    config_spec.loader.exec_module(config_mod)
    if not hasattr(config_mod, "config"):
        raise InvalidConfigProvided(config_path)
    config = config_mod.config
    if not isinstance(config, Config):
        raise InvalidConfigProvided(config_path)
    return config


def _verify_rollout_exists(rollout_path: Path):
    if not rollout_path.exists():
        for extra_ext in _KNOWN_EXTRA_EXTENSIONS:
            rollout_path_with_extra_ext = rollout_path.with_suffix(
                rollout_path.suffix + extra_ext
            )
            if rollout_path_with_extra_ext.exists():
                break
        else:
            raise FileNotFoundError(f"Rollout file `{rollout_path}` not found.")


if __name__ == "__main__":
    main()
