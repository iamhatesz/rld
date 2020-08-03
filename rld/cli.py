import importlib
from pathlib import Path

import click

from rld.app import init
from rld.attributation import (
    AttributationTrajectoryIterator,
    attribute_trajectory,
    NormalizeAttributationProcessor,
    AttributationVisualizationSign,
)
from rld.config import Config
from rld.exception import InvalidConfigProvided
from rld.rollout import (
    RayRolloutReader,
    ToFileRolloutWriter,
    FromFileRolloutReader,
    Rollout,
)


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
    if not rollout_path.exists():
        raise FileNotFoundError(f"Rollout file `{rollout}` not found.")
    out_path = Path(out)

    click.echo(f"Converting Ray rollout `{rollout}` to `{out}`...")

    ray_rollout_reader = RayRolloutReader(rollout_path)

    with ToFileRolloutWriter(out_path) as rollout_writer:
        for trajectory in ray_rollout_reader:
            rollout_writer.write(trajectory)

    click.echo("Done.")


@main.command()
@click.option("--out", default=str(Path.home() / "rollout.rld"))
@click.option("--rllib", default=False, is_flag=True)
@click.option("--normalize", default=False, is_flag=True)
@click.option(
    "--sign",
    default="all",
    type=click.Choice(
        ["all", "positive", "negative", "absolute_value"], case_sensitive=False
    ),
)
@click.argument("config")
@click.argument("rollout")
def attribute(
    out: str, rllib: bool, normalize: bool, sign: str, config: str, rollout: str
):
    """
    This script calculates attributations for the given `rollout`, using configuration
    (e.g. a model definition) stored in the `config` file.
    """
    config_path = Path(config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file `{config}` not found.")
    rollout_path = Path(rollout)
    if not rollout_path.exists():
        raise FileNotFoundError(f"Rollout file `{rollout}` not found.")
    out_path = Path(out)

    config = load_config(config_path)

    click.echo("Attributing rollout...")

    with ToFileRolloutWriter(out_path) as rollout_writer:
        if rllib:
            reader = RayRolloutReader(rollout_path)
        else:
            reader = FromFileRolloutReader(rollout_path)

        if normalize:
            sign = AttributationVisualizationSign[sign.upper()]
            processor = NormalizeAttributationProcessor(
                config.model.obs_space(), sign=sign
            )
        else:
            processor = None

        for trajectory in reader:
            trajectory_it = AttributationTrajectoryIterator(
                trajectory,
                model=config.model,
                baseline=config.baseline,
                target=config.target,
            )
            attr_trajectory = attribute_trajectory(
                trajectory_it, model=config.model, processor=processor
            )
            rollout_writer.write(attr_trajectory)

    click.echo("Done.")


@main.command()
# @click.option("--debug", default=True)
@click.argument("rollout")
def start(rollout: str):
    """
    This script runs web server serving application to visualize calculated
    attributations.
    """
    rollout_path = Path(rollout)
    if not rollout_path.exists():
        raise FileNotFoundError(f"Rollout file `{rollout}` not found.")

    rollout_obj = FromFileRolloutReader(rollout_path)
    # TODO Refactor this
    app = init(Rollout([t for t in rollout_obj]))
    # TODO Make this dependent on the debug flag value
    app.run(debug=True)


def load_config(config_path: Path) -> Config:
    config_spec = importlib.util.spec_from_file_location("config", config_path)
    config_mod = importlib.util.module_from_spec(config_spec)
    config_spec.loader.exec_module(config_mod)
    if not hasattr(config_mod, "config"):
        raise InvalidConfigProvided(config_path)
    config = config_mod.config
    if not isinstance(config, Config):
        raise InvalidConfigProvided(config_path)
    return config


if __name__ == "__main__":
    main()
