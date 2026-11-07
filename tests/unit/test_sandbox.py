import asyncio
import os
import shutil
import subprocess
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from vendorops.models import CodeArtifact
from vendorops.sandbox import SandboxExecutor

DOCKER_LABEL = "vendorops.sandbox"


@pytest.fixture(scope="module", autouse=True)
def require_docker() -> None:
    docker = shutil.which("docker")
    available = docker is not None
    if available:
        try:
            result = subprocess.run(
                [docker, "info"],
                check=False,
                capture_output=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            available = False
        else:
            available = result.returncode == 0
    if available:
        return
    if os.environ.get("REQUIRE_DOCKER") == "1":
        pytest.fail("Docker is required but unavailable", pytrace=False)
    pytest.skip("Docker is unavailable")


@pytest.fixture
def executor() -> SandboxExecutor:
    return SandboxExecutor(timeout_seconds=3.0, max_output_bytes=4096)


@pytest.fixture(autouse=True)
def assert_containers_are_removed() -> Iterator[None]:
    yield
    _assert_no_sandbox_containers()


def _assert_no_sandbox_containers() -> None:
    result = subprocess.run(
        [
            "docker",
            "ps",
            "-aq",
            "--filter",
            f"label={DOCKER_LABEL}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert not result.stdout.strip()


@pytest.mark.asyncio
async def test_sandbox_executes_python_and_removes_container(executor: SandboxExecutor) -> None:
    artifact = await executor.execute("print('hello from sandbox')")

    assert isinstance(artifact, CodeArtifact)
    assert artifact.exit_code == 0
    assert "hello from sandbox" in artifact.stdout
    assert artifact.timed_out is False
    _assert_no_sandbox_containers()


@pytest.mark.asyncio
async def test_sandbox_has_no_network(executor: SandboxExecutor) -> None:
    artifact = await executor.execute(
        "import socket\n"
        "sock = socket.socket()\n"
        "sock.settimeout(1)\n"
        "try:\n"
        "    sock.connect(('1.1.1.1', 53))\n"
        "except OSError:\n"
        "    print('network blocked')\n"
        "else:\n"
        "    raise SystemExit('network unexpectedly available')\n"
        "finally:\n"
        "    sock.close()\n"
    )

    assert artifact.exit_code == 0
    assert "network blocked" in artifact.stdout


@pytest.mark.asyncio
async def test_infinite_loop_is_killed_at_wall_clock_timeout() -> None:
    artifact = await SandboxExecutor(timeout_seconds=2.0).execute("while True: pass")

    assert artifact.timed_out is True
    assert artifact.exit_code is None or artifact.exit_code != 0


@pytest.mark.asyncio
async def test_fork_bomb_is_contained_by_pids_limit(executor: SandboxExecutor) -> None:
    code = (
        "import os, signal\n"
        "children = []\n"
        "for _ in range(128):\n"
        "    try:\n"
        "        pid = os.fork()\n"
        "    except OSError:\n"
        "        print('pids limit reached')\n"
        "        break\n"
        "    if pid == 0:\n"
        "        import time\n"
        "        time.sleep(2)\n"
        "        os._exit(0)\n"
        "    children.append(pid)\n"
        "for pid in children:\n"
        "    try:\n"
        "        os.kill(pid, signal.SIGKILL)\n"
        "    except ProcessLookupError:\n"
        "        pass\n"
        "for pid in children:\n"
        "    try:\n"
        "        os.waitpid(pid, 0)\n"
        "    except ChildProcessError:\n"
        "        pass\n"
    )
    artifact = await executor.execute(code)

    assert "pids limit reached" in artifact.stdout
    assert not artifact.timed_out


@pytest.mark.asyncio
async def test_memory_hog_is_oom_killed(executor: SandboxExecutor) -> None:
    artifact = await executor.execute(
        "chunks = []\n"
        "while True:\n"
        "    chunks.append(bytearray(8 * 1024 * 1024))\n"
    )

    assert artifact.exit_code != 0
    assert not artifact.timed_out


@pytest.mark.asyncio
async def test_root_filesystem_is_read_only(executor: SandboxExecutor) -> None:
    artifact = await executor.execute(
        "try:\n"
        "    open('/root/sandbox-write-test', 'w').write('no')\n"
        "except OSError:\n"
        "    print('read only')\n"
        "else:\n"
        "    raise SystemExit('root filesystem is writable')\n"
        "open('/tmp/sandbox-write-test', 'w').write('allowed')\n"
    )

    assert artifact.exit_code == 0
    assert "read only" in artifact.stdout


@pytest.mark.asyncio
async def test_host_filesystem_is_not_mounted(
    executor: SandboxExecutor,
    tmp_path: Path,
) -> None:
    secret = f"host-only-{uuid.uuid4().hex}"
    host_file = tmp_path / "host-secret.txt"
    host_file.write_text(secret, encoding="utf-8")
    artifact = await executor.execute(
        "from pathlib import Path\n"
        f"path = Path({str(host_file)!r})\n"
        "print('host file absent' if not path.exists() else path.read_text())\n"
    )

    assert artifact.exit_code == 0
    assert "host file absent" in artifact.stdout
    assert secret not in artifact.stdout


@pytest.mark.asyncio
async def test_stdout_and_stderr_are_truncated_at_output_cap() -> None:
    artifact = await SandboxExecutor(
        timeout_seconds=3.0,
        max_output_bytes=1024,
    ).execute(
        "print('x' * 10000)\n"
        "import sys\n"
        "print('y' * 10000, file=sys.stderr)\n"
    )

    assert len(artifact.stdout.encode("utf-8")) <= 1024
    assert len(artifact.stderr.encode("utf-8")) <= 1024
    assert artifact.stdout_truncated
    assert artifact.stderr_truncated


@pytest.mark.asyncio
async def test_twenty_concurrent_executions_all_clean_up() -> None:
    executor = SandboxExecutor(timeout_seconds=30.0, max_output_bytes=4096)
    artifacts = await asyncio.gather(
        *(executor.execute(f"print('task-{index}')") for index in range(20))
    )

    assert len(artifacts) == 20
    assert all(artifact.exit_code == 0 for artifact in artifacts)
    assert {artifact.stdout.strip() for artifact in artifacts} == {
        f"task-{index}" for index in range(20)
    }


@pytest.mark.asyncio
async def test_container_is_removed_when_execution_is_cancelled() -> None:
    executor = SandboxExecutor(timeout_seconds=30.0)
    execution = asyncio.create_task(executor.execute("while True: pass"))
    await asyncio.sleep(0.5)
    execution.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(execution, timeout=15.0)
