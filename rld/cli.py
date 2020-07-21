from pathlib import Path

import click

from rld.app import init
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


if __name__ == "__main__":
    main()
