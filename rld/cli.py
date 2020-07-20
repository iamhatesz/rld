from pathlib import Path

import click

from rld.rollout import RayRolloutReader, ToFileRolloutWriter


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


if __name__ == "__main__":
    main()
