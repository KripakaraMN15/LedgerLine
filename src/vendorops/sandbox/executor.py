from __future__ import annotations

import asyncio
import math
import re
import uuid
from asyncio.subprocess import Process
from typing import Final

from vendorops.models import CodeArtifact

_CONTAINER_LABEL: Final = "vendorops.sandbox=true"
_EXECUTION_SCRIPT: Final = (
    "import sys; exec(compile(sys.stdin.read(), '<sandbox>', 'exec'))"
)


class SandboxExecutionError(RuntimeError):
    pass


class SandboxExecutor:
    def __init__(
        self,
        *,
        image: str = "python:3.11-slim",
        timeout_seconds: float = 10.0,
        max_output_bytes: int = 65_536,
        max_code_bytes: int = 1_000_000,
        memory_limit: str = "128m",
        cpu_limit: float = 0.5,
        pids_limit: int = 32,
        docker_command: str = "docker",
    ) -> None:
        if timeout_seconds <= 0 or max_output_bytes < 1 or max_code_bytes < 1:
            raise ValueError("Timeout and output/code limits must be positive")
        if not math.isfinite(cpu_limit) or cpu_limit <= 0 or pids_limit < 1:
            raise ValueError("CPU and process limits must be positive")
        if (
            re.fullmatch(
                r"[1-9][0-9]*(?:[bBkKmMgG]|[kKmMgG][iI][bB])?",
                memory_limit,
            )
            is None
        ):
            raise ValueError("Memory limit must be a positive Docker memory value")
        if not image or not docker_command:
            raise ValueError("Docker image and command must not be empty")
        self._image = image
        self._timeout_seconds = timeout_seconds
        self._max_output_bytes = max_output_bytes
        self._max_code_bytes = max_code_bytes
        self._memory_limit = memory_limit
        self._cpu_limit = cpu_limit
        self._pids_limit = pids_limit
        self._docker_command = docker_command

    async def execute(
        self,
        source: str,
        *,
        timeout_seconds: float | None = None,
    ) -> CodeArtifact:
        source_bytes = source.encode("utf-8")
        if len(source_bytes) > self._max_code_bytes:
            raise ValueError("Sandbox source exceeds the configured code-size limit")
        timeout = self._timeout_seconds if timeout_seconds is None else timeout_seconds
        if timeout <= 0:
            raise ValueError("Execution timeout must be positive")

        container_name = f"vendorops-sandbox-{uuid.uuid4().hex}"
        process: Process | None = None
        stdout_task: asyncio.Task[tuple[bytes, bool]] | None = None
        stderr_task: asyncio.Task[tuple[bytes, bool]] | None = None
        stdin_task: asyncio.Task[None] | None = None
        timed_out = False
        try:
            process = await asyncio.create_subprocess_exec(
                self._docker_command,
                "run",
                "--name",
                container_name,
                "--label",
                _CONTAINER_LABEL,
                "--network",
                "none",
                "--user",
                "65534:65534",
                "--read-only",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,nodev,size=16m,mode=1777",
                "--cpus",
                str(self._cpu_limit),
                "--memory",
                self._memory_limit,
                "--memory-swap",
                self._memory_limit,
                "--pids-limit",
                str(self._pids_limit),
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges:true",
                "--workdir",
                "/tmp",
                self._image,
                "python",
                "-I",
                "-S",
                "-B",
                "-c",
                _EXECUTION_SCRIPT,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            if process.stdin is None or process.stdout is None or process.stderr is None:
                raise SandboxExecutionError("Docker did not provide the required I/O streams")

            stdin_task = asyncio.create_task(self._write_source(process, source_bytes))
            stdout_task = asyncio.create_task(
                self._capture(process.stdout, self._max_output_bytes)
            )
            stderr_task = asyncio.create_task(
                self._capture(process.stderr, self._max_output_bytes)
            )
            try:
                await asyncio.wait_for(process.wait(), timeout=timeout)
            except TimeoutError:
                timed_out = True
                await self._docker_action("kill", container_name, ignore_failure=True)
                await self._wait_for_process(process)

            await stdin_task
            stdout, stdout_truncated = await stdout_task
            stderr, stderr_truncated = await stderr_task
            exit_code = process.returncode
            if exit_code == 125:
                raise SandboxExecutionError("Docker failed to start the sandbox container")
            return CodeArtifact(
                source=source,
                stdout=stdout.decode("utf-8", errors="ignore"),
                stderr=stderr.decode("utf-8", errors="ignore"),
                exit_code=exit_code,
                timed_out=timed_out,
                stdout_truncated=stdout_truncated,
                stderr_truncated=stderr_truncated,
            )
        except asyncio.CancelledError:
            raise
        except FileNotFoundError as error:
            raise SandboxExecutionError("Docker executable is unavailable") from error
        finally:
            try:
                if process is not None:
                    cleanup = asyncio.create_task(self._cleanup(container_name, process))
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        await cleanup
                        raise
            finally:
                stream_tasks = [
                    task
                    for task in (stdin_task, stdout_task, stderr_task)
                    if task is not None
                ]
                for task in stream_tasks:
                    if not task.done():
                        task.cancel()
                if stream_tasks:
                    await asyncio.gather(*stream_tasks, return_exceptions=True)

    async def _cleanup(self, container_name: str, process: Process) -> None:
        try:
            if process.returncode is None:
                try:
                    await self._docker_action("kill", container_name, ignore_failure=True)
                finally:
                    await self._wait_for_process(process)
        finally:
            await self._docker_action("rm", "-f", container_name, ignore_failure=False)

    async def _wait_for_process(self, process: Process) -> None:
        try:
            await asyncio.wait_for(process.wait(), timeout=5.0)
        except TimeoutError:
            process.kill()
            await process.wait()

    async def _docker_action(
        self,
        *arguments: str,
        ignore_failure: bool,
    ) -> None:
        process: Process | None = None
        communication: asyncio.Task[tuple[bytes, bytes]] | None = None
        try:
            process = await asyncio.create_subprocess_exec(
                self._docker_command,
                *arguments,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            communication = asyncio.create_task(process.communicate())
            _, stderr = await asyncio.wait_for(
                asyncio.shield(communication),
                timeout=10.0,
            )
        except FileNotFoundError as error:
            raise SandboxExecutionError("Docker executable is unavailable") from error
        except TimeoutError as error:
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()
                if communication is not None:
                    await communication
            raise SandboxExecutionError("Docker operation timed out") from error

        if process.returncode != 0 and not ignore_failure:
            error_text = stderr.decode("utf-8", errors="replace").strip()
            if "no such container" not in error_text.lower():
                raise SandboxExecutionError("Docker could not remove the sandbox container")

    @staticmethod
    async def _write_source(
        process: Process,
        source: bytes,
    ) -> None:
        if process.stdin is None:
            return
        try:
            process.stdin.write(source)
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            process.stdin.close()
            try:
                await process.stdin.wait_closed()
            except (BrokenPipeError, ConnectionResetError):
                pass

    @staticmethod
    async def _capture(
        stream: asyncio.StreamReader,
        limit: int,
    ) -> tuple[bytes, bool]:
        captured = bytearray()
        truncated = False
        while chunk := await stream.read(8192):
            remaining = limit - len(captured)
            if remaining > 0:
                captured.extend(chunk[:remaining])
            if len(chunk) > remaining:
                truncated = True
        return bytes(captured), truncated
