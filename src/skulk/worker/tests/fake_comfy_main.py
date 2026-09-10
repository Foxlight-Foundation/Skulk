# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportExplicitAny=false
"""A stand-in ComfyUI server for the comfy runner tests.

Speaks the subset of ComfyUI's HTTP and WebSocket protocol the runner uses
(``/system_stats``, ``/ws``, ``/prompt``, ``/history/{id}``, ``/api/jobs``,
``/interrupt``) and executes a submitted graph by walking its nodes in a
fixed order, emitting the same events a real server would, and writing the
files the save nodes name. Behaviour is steered by environment variables:
``FAKE_COMFY_STEP_SECONDS`` (sampling pace), ``FAKE_COMFY_FAIL_NODE`` (fail
when that node executes). Copied into a temporary checkout as ``main.py``
and started by the runner exactly like the real thing; it imports nothing
from Skulk.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from aiohttp import WSMsgType, web
from PIL import Image

NODE_ORDER = (
    "unet",
    "clip",
    "video_vae",
    "audio_vae",
    "condition",
    "sampler",
    "decode_video",
    "decode_audio",
    "create_video",
    "save_video",
    "thumbnail_frame",
    "save_thumbnail",
)


class FakeComfy:
    def __init__(self, input_dir: Path, output_dir: Path) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.sockets: dict[str, web.WebSocketResponse] = {}
        self.history: dict[str, dict[str, Any]] = {}
        self.running: str | None = None
        self.interrupt = False
        self.queue: asyncio.Queue[tuple[str, dict[str, Any], str]] = asyncio.Queue()
        self.step_seconds = float(os.environ.get("FAKE_COMFY_STEP_SECONDS", "0.01"))
        self.fail_node = os.environ.get("FAKE_COMFY_FAIL_NODE") or None

    async def send(self, event: str, data: dict[str, Any], client_id: str) -> None:
        socket = self.sockets.get(client_id)
        if socket is not None and not socket.closed:
            await socket.send_str(json.dumps({"type": event, "data": data}))

    async def system_stats(self, request: web.Request) -> web.Response:
        return web.json_response({"system": {"os": "fake"}, "devices": []})

    async def websocket(self, request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        client_id = request.rel_url.query.get("clientId") or uuid.uuid4().hex
        self.sockets[client_id] = socket
        await socket.send_str(json.dumps({"type": "status", "data": {"status": {"exec_info": {"queue_remaining": 0}}, "sid": client_id}}))
        async for message in socket:
            if message.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                break
        self.sockets.pop(client_id, None)
        return socket

    async def prompt(self, request: web.Request) -> web.Response:
        body = await request.json()
        graph: dict[str, dict[str, Any]] = body["prompt"]
        prompt_id = body.get("prompt_id") or str(uuid.uuid4())
        node_errors: dict[str, Any] = {}
        for node_id, node in graph.items():
            if "class_type" not in node:
                return web.json_response({"error": {"type": "invalid_prompt", "message": f"{node_id} has no class_type"}, "node_errors": {}}, status=400)
            inputs = node.get("inputs", {})
            for key in ("image", "file", "audio"):
                name = inputs.get(key)
                loads_media = node["class_type"] in ("LoadImage", "LoadVideo", "LoadAudio")
                if loads_media and isinstance(name, str) and not (self.input_dir / name).is_file():
                    node_errors[node_id] = {"errors": [{"type": "value_not_in_list", "message": f"{key} not in list: {name}"}], "class_type": node["class_type"]}
        if node_errors:
            return web.json_response({"error": {"type": "prompt_outputs_failed_validation", "message": "Prompt outputs failed validation"}, "node_errors": node_errors}, status=400)
        (self.output_dir / f"{prompt_id}.prompt.json").parent.mkdir(parents=True, exist_ok=True)
        (self.output_dir / f"{prompt_id}.prompt.json").write_text(json.dumps(graph))
        await self.queue.put((prompt_id, graph, body.get("client_id", "")))
        return web.json_response({"prompt_id": prompt_id, "number": 0, "node_errors": {}})

    async def executor(self) -> None:
        while True:
            prompt_id, graph, client_id = await self.queue.get()
            self.running = prompt_id
            self.interrupt = False
            try:
                await self.execute(prompt_id, graph, client_id)
            finally:
                self.running = None

    def _save(self, prefix: str, extension: str, payload: bytes) -> dict[str, str]:
        subfolder, _, stem = prefix.rpartition("/")
        directory = self.output_dir / subfolder
        directory.mkdir(parents=True, exist_ok=True)
        filename = f"{stem}_00001_.{extension}"
        (directory / filename).write_bytes(payload)
        return {"filename": filename, "subfolder": subfolder, "type": "output"}

    async def execute(self, prompt_id: str, graph: dict[str, dict[str, Any]], client_id: str) -> None:
        outputs: dict[str, Any] = {}
        messages: list[Any] = [["execution_start", {"prompt_id": prompt_id, "timestamp": int(time.time() * 1000)}]]
        status: dict[str, Any]
        steps = int(graph.get("scheduler", {}).get("inputs", {}).get("steps", 1))
        node_id = "unet"
        try:
            for node_id in NODE_ORDER:
                if node_id not in graph:
                    continue
                await self.send("executing", {"node": node_id, "display_node": node_id, "prompt_id": prompt_id}, client_id)
                if self.fail_node == node_id:
                    raise RuntimeError(f"fake failure at {node_id}")
                if node_id == "sampler":
                    for step in range(1, steps + 1):
                        await asyncio.sleep(self.step_seconds)
                        if self.interrupt:
                            raise InterruptedError
                        await self.send(
                            "progress_state",
                            {"prompt_id": prompt_id, "nodes": {"sampler": {"value": step, "max": steps, "state": "running", "node_id": "sampler", "prompt_id": prompt_id}}},
                            client_id,
                        )
                elif node_id == "save_video":
                    prefix = graph[node_id]["inputs"]["filename_prefix"]
                    saved = self._save(prefix, "mp4", b"\x00\x00\x00\x18ftypisom" + b"\x00" * 64)
                    outputs[node_id] = {"images": [saved], "animated": [True]}
                elif node_id == "save_thumbnail":
                    prefix = graph[node_id]["inputs"]["filename_prefix"]
                    import io

                    buffer = io.BytesIO()
                    Image.new("RGB", (8, 6), (200, 40, 40)).save(buffer, format="PNG")
                    saved = self._save(prefix, "png", buffer.getvalue())
                    outputs[node_id] = {"images": [saved]}
                await self.send("executed", {"node": node_id, "display_node": node_id, "output": outputs.get(node_id), "prompt_id": prompt_id}, client_id)
        except InterruptedError:
            data = {"prompt_id": prompt_id, "node_id": "sampler", "node_type": "SamplerCustomAdvanced", "executed": []}
            messages.append(["execution_interrupted", data])
            status = {"status_str": "error", "completed": False, "messages": messages}
            self.history[prompt_id] = {"prompt": [0, prompt_id, graph, {}, []], "outputs": outputs, "status": status}
            await self.send("execution_interrupted", data, client_id)
        except Exception as error:  # noqa: BLE001 - mirrors ComfyUI's handler
            data = {"prompt_id": prompt_id, "node_id": node_id, "node_type": graph[node_id]["class_type"], "exception_message": str(error), "exception_type": type(error).__name__, "traceback": [], "current_inputs": {}, "current_outputs": {}}
            messages.append(["execution_error", data])
            status = {"status_str": "error", "completed": False, "messages": messages}
            self.history[prompt_id] = {"prompt": [0, prompt_id, graph, {}, []], "outputs": outputs, "status": status}
            await self.send("execution_error", data, client_id)
        else:
            messages.append(["execution_success", {"prompt_id": prompt_id, "timestamp": int(time.time() * 1000)}])
            status = {"status_str": "success", "completed": True, "messages": messages}
            self.history[prompt_id] = {"prompt": [0, prompt_id, graph, {}, []], "outputs": outputs, "status": status}
            await self.send("execution_success", {"prompt_id": prompt_id}, client_id)
        await self.send("executing", {"node": None, "prompt_id": prompt_id}, client_id)

    async def history_entry(self, request: web.Request) -> web.Response:
        prompt_id = request.match_info["prompt_id"]
        entry = self.history.get(prompt_id)
        return web.json_response({prompt_id: entry} if entry else {})

    def _job_status(self, prompt_id: str) -> str | None:
        if prompt_id == self.running:
            return "in_progress"
        entry = self.history.get(prompt_id)
        if entry is None:
            return None
        if entry["status"]["completed"]:
            return "completed"
        names = {message[0] for message in entry["status"]["messages"]}
        return "cancelled" if "execution_interrupted" in names else "failed"

    async def job(self, request: web.Request) -> web.Response:
        prompt_id = request.match_info["job_id"]
        status = self._job_status(prompt_id)
        if status is None:
            return web.json_response({"error": "Job not found"}, status=404)
        error = None
        entry = self.history.get(prompt_id)
        if entry is not None and status == "failed":
            for name, data in entry["status"]["messages"]:
                if name == "execution_error":
                    error = data
        return web.json_response({"id": prompt_id, "status": status, "execution_error": error})

    async def cancel(self, request: web.Request) -> web.Response:
        prompt_id = request.match_info["job_id"]
        if prompt_id == self.running:
            self.interrupt = True
            return web.json_response({"cancelled": True})
        return web.json_response({"cancelled": False})

    async def interrupt_all(self, request: web.Request) -> web.Response:
        self.interrupt = True
        return web.Response(status=200)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--input-directory", required=True)
    parser.add_argument("--output-directory", required=True)
    args, _ = parser.parse_known_args()
    fake = FakeComfy(Path(args.input_directory), Path(args.output_directory))
    app = web.Application()
    app.router.add_get("/system_stats", fake.system_stats)
    app.router.add_get("/ws", fake.websocket)
    app.router.add_post("/prompt", fake.prompt)
    app.router.add_get("/history/{prompt_id}", fake.history_entry)
    app.router.add_get("/api/jobs/{job_id}", fake.job)
    app.router.add_post("/api/jobs/{job_id}/cancel", fake.cancel)
    app.router.add_post("/interrupt", fake.interrupt_all)

    async def start_executor(app: web.Application) -> None:
        app["executor"] = asyncio.create_task(fake.executor())

    app.on_startup.append(start_executor)
    print(f"fake comfy listening on {args.port}", file=sys.stderr, flush=True)
    web.run_app(app, host="127.0.0.1", port=args.port, print=None)


if __name__ == "__main__":
    main()
