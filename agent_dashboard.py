"""Local dashboard for running and observing the Stage 5 agent flow.

This server intentionally exposes only fixed project commands:
start services, stop services, run test_client.py, and read log tails.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
HOST = "127.0.0.1"
PORT = int(os.getenv("AGENT_DASHBOARD_PORT", "8088"))

SERVICES = {
    "registry": 10000,
    "customer_agent": 10100,
    "law_agent": 10101,
    "tax_agent": 10102,
    "compliance_agent": 10103,
}

MODULES = {
    "registry": "registry",
    "customer_agent": "customer_agent",
    "law_agent": "law_agent",
    "tax_agent": "tax_agent",
    "compliance_agent": "compliance_agent",
}

START_ORDER = [
    "registry",
    "tax_agent",
    "compliance_agent",
    "law_agent",
    "customer_agent",
]

PYTHON_EXE = ROOT / ".venv" / "Scripts" / "python.exe"
PYTHON_CMD = str(PYTHON_EXE) if PYTHON_EXE.exists() else "python"
OPTIMIZATION_ENABLED = True

LOG_FILES = [
    "customer_agent.err.log",
    "law_agent.err.log",
    "tax_agent.err.log",
    "compliance_agent.err.log",
    "registry.err.log",
    "customer_agent.out.log",
    "law_agent.out.log",
    "tax_agent.out.log",
    "compliance_agent.out.log",
    "registry.out.log",
]


class RunState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running = False
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.exit_code: int | None = None
        self.output = ""
        self.question = ""

    def snapshot(self) -> dict:
        with self.lock:
            elapsed = None
            if self.started_at:
                end = self.finished_at or time.time()
                elapsed = round(end - self.started_at, 3)
            return {
                "running": self.running,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "elapsed_seconds": elapsed,
                "exit_code": self.exit_code,
                "output": self.output,
                "question": self.question,
            }


RUN_STATE = RunState()


def run_command(command: list[str], env: dict[str, str] | None = None) -> tuple[int, str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    proc = subprocess.Popen(
        command,
        cwd=ROOT,
        env=merged_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output, _ = proc.communicate()
    return proc.returncode, output


def run_test_client(question: str, optimized: bool) -> None:
    with RUN_STATE.lock:
        if RUN_STATE.running:
            return
        RUN_STATE.running = True
        RUN_STATE.started_at = time.time()
        RUN_STATE.finished_at = None
        RUN_STATE.exit_code = None
        RUN_STATE.output = ""
        RUN_STATE.question = question

    code, output = run_command(
        [PYTHON_CMD, "test_client.py"],
        env={
            "LEGAL_TEST_QUESTION": question,
            "CUSTOMER_DIRECT_DELEGATION": "true" if optimized else "false",
        },
    )

    with RUN_STATE.lock:
        RUN_STATE.running = False
        RUN_STATE.finished_at = time.time()
        RUN_STATE.exit_code = code
        RUN_STATE.output = output


def tail_file(path: Path, max_chars: int = 12000) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    return data[-max_chars:].decode("utf-8", errors="replace")


def all_log_text() -> str:
    chunks = []
    for name in LOG_FILES:
        text = tail_file(LOG_DIR / name, max_chars=8000)
        if text:
            chunks.append(f"===== {name} =====\n{text}")
    return "\n\n".join(chunks)


def extract_trace_ids(text: str) -> list[str]:
    ids = re.findall(r"trace[=:]\s*([0-9a-fA-F-]{32,36})", text)
    seen: list[str] = []
    for trace_id in ids:
        if trace_id not in seen:
            seen.append(trace_id)
    return seen[-8:]


def port_status() -> dict[str, dict[str, object]]:
    status: dict[str, dict[str, object]] = {}
    try:
        code, output = run_command(["netstat", "-ano"])
    except Exception:
        code, output = 1, ""

    for service, port in SERVICES.items():
        pattern = re.compile(rf":{port}\s+.*LISTENING\s+(\d+)", re.IGNORECASE)
        match = pattern.search(output)
        status[service] = {
            "port": port,
            "running": bool(match),
            "pid": match.group(1) if match else None,
        }
    status["_netstat_exit_code"] = {"port": 0, "running": code == 0, "pid": None}
    return status


def stop_services_by_port() -> str:
    status = port_status()
    stopped = []
    for service, info in status.items():
        if service.startswith("_"):
            continue
        pid = info.get("pid")
        if pid:
            run_command(["taskkill", "/PID", str(pid), "/F"])
            stopped.append(f"{service}({pid})")
    return "Stopped: " + ", ".join(stopped) if stopped else "No running services found."


def stop_service(service: str) -> str:
    if service not in SERVICES:
        return f"Unknown service: {service}"

    info = port_status().get(service, {})
    pid = info.get("pid")
    if not pid:
        return f"{service} is already stopped."
    run_command(["taskkill", "/PID", str(pid), "/F"])
    return f"Stopped {service}({pid})."


def start_service(service: str) -> str:
    if service not in SERVICES:
        return f"Unknown service: {service}"

    if port_status().get(service, {}).get("running"):
        return f"{service} is already online."

    LOG_DIR.mkdir(exist_ok=True)
    stdout_path = LOG_DIR / f"{service}.out.log"
    stderr_path = LOG_DIR / f"{service}.err.log"
    stdout = stdout_path.open("a", encoding="utf-8")
    stderr = stderr_path.open("a", encoding="utf-8")
    env = os.environ.copy()
    if service == "customer_agent":
        env["CUSTOMER_DIRECT_DELEGATION"] = "true" if OPTIMIZATION_ENABLED else "false"

    proc = subprocess.Popen(
        [PYTHON_CMD, "-m", MODULES[service]],
        cwd=ROOT,
        env=env,
        stdout=stdout,
        stderr=stderr,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    stdout.close()
    stderr.close()
    with (LOG_DIR / "services.pids").open("a", encoding="utf-8") as handle:
        handle.write(f"{service},{proc.pid},{SERVICES[service]}\n")
    return f"Started {service}({proc.pid})."


def restart_customer_for_optimization(enabled: bool) -> str:
    global OPTIMIZATION_ENABLED
    OPTIMIZATION_ENABLED = enabled
    stopped = stop_service("customer_agent")
    time.sleep(1)
    started = start_service("customer_agent")
    return f"Optimization set to {enabled}. {stopped} {started}"


def start_services() -> str:
    LOG_DIR.mkdir(exist_ok=True)
    pid_file = LOG_DIR / "services.pids"
    pid_file.write_text("", encoding="utf-8")

    started = []

    for service in START_ORDER:
        started.append(start_service(service))
        time.sleep(2 if service == "registry" else 1)

    return "\n".join(started)


def json_response(handler: BaseHTTPRequestHandler, payload: object, status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            html = (ROOT / "agent_dashboard.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return

        if parsed.path == "/api/status":
            log_text = all_log_text()
            payload = {
                "run": RUN_STATE.snapshot(),
                "services": port_status(),
                "trace_ids": extract_trace_ids(log_text),
                "optimization_enabled": OPTIMIZATION_ENABLED,
            }
            json_response(self, payload)
            return

        if parsed.path == "/api/logs":
            qs = parse_qs(parsed.query)
            selected = qs.get("file", ["all"])[0]
            if selected == "all":
                text = all_log_text()
            elif selected in LOG_FILES:
                text = tail_file(LOG_DIR / selected, max_chars=20000)
            else:
                json_response(self, {"error": "Unknown log file"}, 400)
                return
            json_response(self, {"file": selected, "text": text, "trace_ids": extract_trace_ids(text)})
            return

        json_response(self, {"error": "Not found"}, 404)

    def do_POST(self) -> None:
        if self.path == "/api/run":
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(raw or "{}")
            question = payload.get("question") or (
                "If a company breaks a contract and avoids taxes, "
                "what are the legal and regulatory consequences?"
            )
            optimized = bool(payload.get("optimized", OPTIMIZATION_ENABLED))

            with RUN_STATE.lock:
                already_running = RUN_STATE.running
            if already_running:
                json_response(self, {"error": "A test run is already in progress."}, 409)
                return

            thread = threading.Thread(target=run_test_client, args=(question, optimized), daemon=True)
            thread.start()
            json_response(self, {"ok": True, "message": "Started test_client.py"})
            return

        if self.path == "/api/start-services":
            message = start_services()
            json_response(self, {"ok": True, "message": message})
            return

        if self.path == "/api/stop-services":
            message = stop_services_by_port()
            json_response(self, {"ok": True, "message": message})
            return

        if self.path == "/api/service":
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(raw or "{}")
            service = payload.get("service", "")
            action = payload.get("action", "")
            if action == "start":
                message = start_service(service)
            elif action == "stop":
                message = stop_service(service)
            elif action == "restart":
                message = stop_service(service) + " " + start_service(service)
            else:
                json_response(self, {"error": "action must be start, stop, or restart"}, 400)
                return
            json_response(self, {"ok": True, "message": message})
            return

        if self.path == "/api/optimization":
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(raw or "{}")
            enabled = bool(payload.get("enabled", True))
            message = restart_customer_for_optimization(enabled)
            json_response(self, {"ok": True, "message": message, "enabled": enabled})
            return

        json_response(self, {"error": "Not found"}, 404)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), DashboardHandler)
    try:
        print(f"Agent dashboard: http://{HOST}:{PORT}", flush=True)
        print("Press Ctrl+C to stop the dashboard server.", flush=True)
    except OSError:
        pass
    server.serve_forever()


if __name__ == "__main__":
    main()
