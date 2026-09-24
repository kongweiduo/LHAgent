"""Run the official grader with the same Docker platforms used for solving."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch


def _image_platform(image: str, platforms: dict[str, str]) -> str:
    try:
        return platforms[image]
    except KeyError as exc:
        raise ValueError(
            f"grader requested an image absent from the platform plan: {image}"
        ) from exc


@contextmanager
def docker_platforms(platforms: dict[str, str]):
    # The upstream harness uses the SDK, which ignores DOCKER_DEFAULT_PLATFORM.
    from docker.models.containers import ContainerCollection
    from docker.models.images import ImageCollection

    original_create = ContainerCollection.create
    original_pull = ImageCollection.pull

    def create(self, image, *args, **kwargs):
        kwargs["platform"] = _image_platform(image, platforms)
        return original_create(self, image, *args, **kwargs)

    def pull(self, repository, tag=None, *args, **kwargs):
        image = f"{repository}:{tag}" if tag else repository
        kwargs["platform"] = _image_platform(image, platforms)
        return original_pull(self, repository, tag, *args, **kwargs)

    with (
        patch.object(ContainerCollection, "create", create),
        patch.object(ImageCollection, "pull", pull),
    ):
        yield


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--platform-plan", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    platforms = json.loads(args.platform_plan.read_text())
    sys.argv = ["swebench.harness.run_evaluation", *remaining]
    with docker_platforms(platforms):
        runpy.run_module("swebench.harness.run_evaluation", run_name="__main__")


if __name__ == "__main__":
    main()
