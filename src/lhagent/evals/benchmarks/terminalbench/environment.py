"""Harbor Docker provider restricted to prebuilt images, with strict cleanup."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from harbor.environments.docker.docker import DockerEnvironment


class ResourceUnavailableError(RuntimeError):
    """The Docker host cannot provide the CPUs, memory or GPUs a task requests."""


# Docker/Compose messages emitted when a container's resource request cannot be satisfied.
RESOURCE_ERROR_PATTERNS = (
    "range of cpus",
    "cannot be greater than the number of cpus",
    "could not select device driver",
    "nvidia-container-cli",
    "unknown or invalid runtime name: nvidia",
    "insufficient memory",
    "minimum memory limit",
)


def is_resource_error(message: str) -> bool:
    message = message.lower()
    return any(pattern in message for pattern in RESOURCE_ERROR_PATTERNS)


async def docker(*args: str) -> str:
    process = await asyncio.create_subprocess_exec(
        "docker", *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        stdout, stderr = await process.communicate()
    except asyncio.CancelledError:
        process.kill()
        await process.wait()
        raise
    if process.returncode:
        raise RuntimeError(f"Docker {' '.join(args[:2])} failed: {stderr.decode(errors='replace')}")
    return stdout.decode()


async def image_platforms(image: str) -> set[str]:
    try:
        data = json.loads(await docker("manifest", "inspect", "--verbose", image))
    except RuntimeError:
        data = json.loads(await docker("image", "inspect", image))
    entries = data if isinstance(data, list) else data.get("manifests", [data])
    platforms = set()
    for entry in entries:
        info = entry.get("Descriptor", {}).get("platform", entry.get("platform", entry))
        os_name = info.get("os", info.get("Os"))
        arch = info.get("architecture", info.get("Architecture"))
        if os_name == "linux" and arch in {"amd64", "arm64"}:
            platforms.add(f"linux/{arch}")
    if not platforms:
        raise ValueError(f"no supported Linux platform found for image {image}")
    return platforms


def select_platform(platforms: set[str], bundles: set[str], native: str) -> str:
    for platform in (native, "linux/amd64", "linux/arm64"):
        if platform in platforms and platform in bundles:
            return platform
    raise ValueError(f"image platforms {sorted(platforms)} have no matching LHAgent bundle")


class PrebuiltDockerEnvironment(DockerEnvironment):
    def __init__(
        self,
        *,
        initial_images,
        cleanup_errors_path,
        task_images_path,
        platforms_path,
        bundle_platforms,
        native_platform,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.initial_images = set(initial_images)
        self.cleanup_errors_path = Path(cleanup_errors_path)
        self.task_images_path = Path(task_images_path)
        self.platforms_path = Path(platforms_path)
        self.bundle_platforms = set(bundle_platforms)
        self.native_platform = native_platform
        self.task_images = set()

    async def _ensure_egress_control_sidecar_image_built(self):
        raise RuntimeError(
            "This task requires a locally built Harbor network sidecar; "
            "the adapter permits prebuilt images only"
        )

    async def start(self, force_build=False):
        image = self.task_env_config.docker_image
        if not image or force_build:
            raise ValueError("an official prebuilt docker_image is required; local builds disabled")
        self.task_images.add(image)
        platform = select_platform(
            await image_platforms(image), self.bundle_platforms, self.native_platform
        )
        overlay = self.trial_paths.trial_dir / f"{self.session_id}.platform.json"
        overlay.write_text(json.dumps({"services": {"main": {"platform": platform}}}))
        self.extra_docker_compose_paths = [*self.extra_docker_compose_paths, overlay]
        plans = json.loads(self.platforms_path.read_text()) if self.platforms_path.exists() else {}
        plans[image] = platform
        self.platforms_path.write_text(json.dumps(plans, indent=2))
        await docker("pull", "--platform", platform, image)
        await super().start(force_build=False)

    async def _run_docker_compose_command(self, command, **kwargs):
        if command[0] == "build":
            raise RuntimeError("local environment builds are disabled")
        if command[0] == "up":
            result = await super()._run_docker_compose_command(["config", "--format", "json"])
            compose = json.loads(result.stdout)
            for service in compose["services"].values():
                if service.get("build") or not service.get("image"):
                    raise ValueError("every task service must use a prebuilt image")
                self.task_images.add(service["image"])
            command = ["up", "--no-build", *command[1:]]
            try:
                return await super()._run_docker_compose_command(command, **kwargs)
            except RuntimeError as exc:
                if is_resource_error(str(exc)):
                    raise ResourceUnavailableError(
                        f"Docker host cannot satisfy task resources: {exc}"
                    ) from exc
                raise
        return await super()._run_docker_compose_command(command, **kwargs)

    async def stop(self, delete=True):
        try:
            # Avoid Harbor's best-effort cleanup and --rmi local, which can remove cached images.
            try:
                await self.prepare_logs_for_host()
            finally:
                await self._run_docker_compose_command(["down", "--volumes", "--remove-orphans"])
        except Exception as exc:
            self.cleanup_errors_path.parent.mkdir(parents=True, exist_ok=True)
            errors = (
                json.loads(self.cleanup_errors_path.read_text())
                if self.cleanup_errors_path.exists()
                else []
            )
            errors.append(str(exc))
            self.cleanup_errors_path.write_text(json.dumps(errors, indent=2))
            raise
        finally:
            images = (
                set(json.loads(self.task_images_path.read_text()))
                if self.task_images_path.exists()
                else set()
            )
            self.task_images_path.write_text(json.dumps(sorted(images | self.task_images)))
            self._cleanup_mounts_compose_file()
            self._cleanup_resources_compose_file()
            self._cleanup_env_compose_file()
            self._cleanup_egress_control_services_compose_file()


async def cleanup_images(attempt_dir: Path, initial_images: set[str]):
    """Delete task images only after the official report and logs have been saved."""
    path = attempt_dir / "images.json"
    if not path.exists():
        return
    try:
        for image in json.loads(path.read_text()):
            try:
                image_id = (await docker("image", "inspect", "--format", "{{.Id}}", image)).strip()
            except RuntimeError as exc:
                # A failed pull may leave no image, or another reference was already removed.
                if "No such image" in str(exc) or "No such object" in str(exc):
                    continue
                raise
            if image_id and image_id not in initial_images:
                await docker("image", "rm", image)
    except Exception as exc:
        error_path = attempt_dir / "cleanup-errors.json"
        errors = json.loads(error_path.read_text()) if error_path.exists() else []
        error_path.write_text(json.dumps([*errors, str(exc)], indent=2))
