#!/usr/bin/env python3
"""Opt-in TITAN RTX integration trial. Does not run on installation/test discovery."""
import argparse
import json
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Explicitly allow downloads and GPU inference')
    parser.add_argument('--apptainer', default='apptainer')
    parser.add_argument('--device', default='0')
    parser.add_argument('--port', type=int, default=18089)
    parser.add_argument('--timeout', type=int, default=1800, help='Total trial timeout including container conversion and model download')
    parser.add_argument('--output', type=Path, default=Path('.trial'))
    args = parser.parse_args()
    if not args.run:
        print('Not run. Add --run on the trial machine to download weights and test GPU inference.')
        print('Profile: tiny-qwen; GGUF Q4_K_M; 2048 tokens; one slot; no speculative decoding.')
        return 0
    if not 1024 <= args.port <= 65535 or args.timeout <= 0:
        parser.error('Use an unprivileged port and a positive timeout')
    # Do not accidentally test a pre-existing service.
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', args.port))
    gpu = subprocess.check_output(['nvidia-smi', '-i', args.device,
            '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], text=True).strip()
    if 'TITAN RTX' not in gpu:
        raise RuntimeError(f'This trial targets TITAN RTX; detected {gpu}. Adapt the trial for other hardware.')
    destination = args.output.resolve() / str(time.time_ns())
    destination.mkdir(parents=True)
    launcher_log = destination / 'launcher.log'
    prefix = [sys.executable, '-m', 'llm_deploy']
    cmd = prefix + ['start', '--profile', 'tiny-qwen', '--apptainer', args.apptainer,
                    '--device', args.device, '--port', str(args.port), '--timeout', '600']
    process = None
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    started = time.monotonic()
    report = {'status': 'failed', 'gpu': gpu, 'command': cmd, 'speculative': False}
    try:
        with launcher_log.open('w') as stream:
            process = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = started + args.timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f'Launcher exited ({process.returncode}); see {launcher_log}')
                text = launcher_log.read_text()
                if 'Ready: ' in text:
                    break
                time.sleep(1)
            else:
                raise TimeoutError(f'Trial exceeded {args.timeout}s; see {launcher_log}')
            log_line = next(line for line in text.splitlines() if line.startswith('Log: '))
            engine_log = Path(log_line.removeprefix('Log: '))
            engine_text = engine_log.read_text(errors='replace')
            if not re.search(r'offloaded\s+[1-9]\d*/\d+\s+layers to GPU', engine_text):
                raise RuntimeError(f'No GPU layer-offload evidence in {engine_log}; CPU fallback is not a passing GPU trial')
            result = subprocess.check_output(prefix + ['smoke', '--profile', 'tiny-qwen',
                        '--port', str(args.port)], text=True, timeout=150)
            report.update(status='passed', elapsed_seconds=time.monotonic() - started,
                          response=json.loads(result), engine_log=str(engine_log))
            print(f'PASS: GPU offload, model identity, health, and chat completion. Results: {destination}')
            return 0
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if process is not None and process.poll() is None:
            # Launcher owns its engine process group and forwards shutdown to it.
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=25)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        (destination / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f'Trial log: {launcher_log}')


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as exc:
        print(f'Trial failed: {exc}', file=sys.stderr)
        raise SystemExit(1)
