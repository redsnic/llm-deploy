"""Local foreground serving and single-allocation Slurm execution."""
import argparse
import base64
import hashlib
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import tempfile
import urllib.error
import urllib.request

import yaml
from jsonschema import Draft202012Validator

PACKAGE = Path(__file__).parent
DEFAULTS = {
    'llama-cpp': dict(context=2048, parallel=1, gpu_layers=99, batch=128,
                      ubatch=128, flash_attention=False, cache_type='f16'),
    'vllm': dict(context=4096, parallel=1, dtype='auto', tensor_parallel=1,
                 gpu_memory_utilization=0.85, language_model_only=False),
}


# Immutable linux/amd64 OCI manifests from the original working deployments.
IMAGES = {
    'llama-cpp': 'docker://ghcr.io/ggml-org/llama.cpp@sha256:3e847b7b8d616411d4dd8f3d1a79caf02cc7a66b6b92c8212156ea73835bf7f3',
    'vllm': 'docker://docker.io/vllm/vllm-openai@sha256:2286e8533ca8b6bc777594bae30524f1426ba46ca21797524e06df6a94b06635',
}


def load_profile(value):
    path = Path(value).expanduser()
    if not path.is_file():
        path = PACKAGE / 'profiles' / (value + '.yaml')
    if not path.is_file():
        raise ValueError(f'Unknown profile: {value}')
    data = yaml.safe_load(path.read_text())
    schema = json.loads((PACKAGE / 'profile.schema.json').read_text())
    Draft202012Validator(schema).validate(data)
    defaults = DEFAULTS[data['engine']]
    runtime = data.get('runtime', {})
    unknown = set(runtime) - set(defaults)
    if unknown:
        raise ValueError(f"Runtime options unsupported by {data['engine']}: {sorted(unknown)}")
    data['runtime'] = dict(defaults, **runtime)
    data.setdefault('container', {'image': IMAGES[data['engine']]})
    if data['engine'] == 'llama-cpp' and data['runtime']['ubatch'] > data['runtime']['batch']:
        raise ValueError('ubatch must not exceed batch')
    if data['engine'] == 'vllm' and set(data['model']) - {'repo', 'revision'}:
        raise ValueError('vLLM expects a HF snapshot, not GGUF file metadata')
    return data


def artifact_dir(cache, model):
    return cache / 'models' / model['repo'].replace('/', '--') / model['revision']


def download(model, cache):
    # Imported only on execution; planning and schema validation never download.
    from huggingface_hub import hf_hub_download, snapshot_download
    destination = artifact_dir(cache, model)
    if 'file' in model:
        path = Path(hf_hub_download(repo_id=model['repo'], revision=model['revision'],
                                   filename=model['file'], local_dir=str(destination)))
        if path.stat().st_size != model['bytes']:
            raise ValueError(f'Wrong file size: {path}')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != model['sha256'].lower():
            raise ValueError(f'Checksum mismatch: {path}; remove the corrupt cached file before retrying')
    else:
        snapshot_download(repo_id=model['repo'], revision=model['revision'], local_dir=str(destination))
        # Read the checkpoint index rather than assuming Qwen's shard count.
        index = destination / 'model.safetensors.index.json'
        if index.is_file():
            names = set(json.loads(index.read_text())['weight_map'].values())
            if not names:
                raise ValueError('Empty checkpoint weight index')
            for name in names:
                weight = (destination / name).resolve()
                if not weight.is_relative_to(destination.resolve()) or not weight.is_file() or weight.stat().st_size == 0:
                    raise ValueError(f'Missing/invalid checkpoint shard: {name}')
        elif not (destination / 'model.safetensors').is_file():
            raise ValueError('Expected a safetensors checkpoint or weight index')
        if not (destination / 'config.json').is_file():
            raise ValueError('Missing model config.json')


def engine_command(profile, cache, host, port):
    model, runtime = profile['model'], profile['runtime']
    path = artifact_dir(cache, model)
    if profile['engine'] == 'llama-cpp':
        return ['/app/llama-server', '--model', str(path / model['file']),
                '--alias', profile['name'], '--host', host, '--port', str(port),
                '--ctx-size', str(runtime['context']), '--parallel', str(runtime['parallel']),
                '--n-gpu-layers', str(runtime['gpu_layers']), '--batch-size', str(runtime['batch']),
                '--ubatch-size', str(runtime['ubatch']), '--flash-attn', 'on' if runtime['flash_attention'] else 'off',
                '--cache-type-k', runtime['cache_type'], '--cache-type-v', runtime['cache_type']]
    args = ['vllm', 'serve', str(path), '--served-model-name', profile['name'],
            '--host', host, '--port', str(port), '--dtype', runtime['dtype'],
            '--load-format', 'safetensors', '--max-model-len', str(runtime['context']),
            '--max-num-seqs', str(runtime['parallel']), '--tensor-parallel-size', str(runtime['tensor_parallel']),
            '--gpu-memory-utilization', str(runtime['gpu_memory_utilization'])]
    if runtime['language_model_only']:
        args += ['--language-model-only']
    parsers = profile.get('parsers', {})
    if 'reasoning' in parsers:
        args += ['--reasoning-parser', parsers['reasoning']]
    if 'tool_calls' in parsers:
        args += ['--enable-auto-tool-choice', '--tool-call-parser', parsers['tool_calls']]
    if 'speculative' in profile:
        spec = profile['speculative']
        args += ['--speculative-config', json.dumps(dict(method=spec['method'],
                 model=str(artifact_dir(cache, spec)), num_speculative_tokens=spec['num_speculative_tokens']))]
    return args


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def image_path(profile, cache):
    digest = profile['container']['image'].rsplit('sha256:', 1)[1]
    return cache / 'containers' / (digest + '.sif')


def prepare_image(profile, cache, apptainer):
    image = image_path(profile, cache)
    image.parent.mkdir(parents=True, exist_ok=True)
    checksum = image.with_suffix('.sif.sha256')
    # Separate launches sharing the image store must not race during conversion.
    with image.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if image.is_file() and checksum.is_file() and file_sha256(image) == checksum.read_text().strip():
            print(f'Reusing verified container: {image}', flush=True)
            return
        env = dict(os.environ)
        env['APPTAINER_CACHEDIR'] = str(cache / 'apptainer-cache')
        if os.environ.get('SLURM_TMPDIR'):
            env.setdefault('APPTAINER_TMPDIR', os.environ['SLURM_TMPDIR'])
        Path(env['APPTAINER_CACHEDIR']).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.pull-', dir=image.parent) as staging:
            temporary = Path(staging) / 'image.sif'
            print(f'Preparing container: {profile["container"]["image"]}', flush=True)
            subprocess.run([apptainer, 'pull', str(temporary), profile['container']['image']],
                           env=env, check=True)
            digest = file_sha256(temporary)
            temporary.replace(image)
            temporary_checksum = Path(staging) / 'checksum'
            temporary_checksum.write_text(digest + '\n')
            temporary_checksum.replace(checksum)


def compile_dir(profile, cache):
    fingerprint = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()[:16]
    return cache / 'compile' / fingerprint


def command(profile, cache, apptainer, host, port, run_dir=None):
    run_dir = run_dir or cache / 'runs' / (profile['name'] + '-preview')
    # Apptainer bind specifications use colon/comma delimiters; reject ambiguous paths.
    for path in (cache, run_dir):
        if any(character in str(path) for character in ':,\n'):
            raise ValueError('Cache and run paths cannot contain colon, comma, or newline')
    args = [apptainer or 'apptainer', 'exec', '--cleanenv']
    if profile['engine'] == 'vllm' or profile['runtime']['gpu_layers'] > 0:
        args += ['--nv']
    args += ['--home', str(run_dir) + ':/home/llm', '--pwd', '/runtime',
             '--bind', str(cache / 'models') + ':/models:ro',
             '--bind', str(compile_dir(profile, cache)) + ':/compile',
             '--bind', str(run_dir) + ':/runtime',
             '--env', 'HF_HUB_OFFLINE=1', '--env', 'TRANSFORMERS_OFFLINE=1',
             '--env', 'HF_HOME=/compile/huggingface', '--env', 'VLLM_CACHE_ROOT=/compile/vllm',
             '--env', 'XDG_CACHE_HOME=/compile/xdg',
             str(image_path(profile, cache))]
    # Engine paths refer to the read-only /models mount, never host cache paths.
    return args + engine_command(profile, Path('/'), host, port)


def request(base, route, body=None, timeout=5):
    data = None if body is None else json.dumps(body).encode()
    headers = {'Content-Type': 'application/json'}
    req = urllib.request.Request(base.rstrip('/') + route, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw) if raw else {}


def smoke(base, name, profile=None):
    models = request(base, '/v1/models')['data']
    if not any(item['id'] == name for item in models):
        raise ValueError(f'Server does not advertise model {name}')
    payload = dict(model=name, messages=[dict(role='user', content='Say hello in one short sentence.')],
                   max_tokens=64, temperature=0)
    if profile and profile.get('parsers', {}).get('reasoning') == 'qwen3':
        payload['chat_template_kwargs'] = {'enable_thinking': False}
    result = request(base, '/v1/chat/completions', payload, timeout=120)
    if not result.get('choices') or not result['choices'][0]['message'].get('content'):
        raise ValueError('Chat smoke test returned no text')
    return result


def wait_ready(process, base, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'Server exited with code {process.returncode}')
        try:
            request(base, '/health', timeout=2)
            return
        except (OSError, ValueError, urllib.error.URLError):
            time.sleep(1)
    raise TimeoutError(f'Server did not become healthy within {timeout} seconds')


def start_local(profile, args):
    cache = args.cache_dir.expanduser().resolve()
    if not shutil.which(args.apptainer):
        raise ValueError(f'Apptainer not found: {args.apptainer}. Install Apptainer or load its cluster module.')
    command(profile, cache, args.apptainer, args.host, args.port)  # Validate paths before any downloads.
    # Refuse occupied ports rather than accidentally checking another server.
    with socket.socket() as probe:
        probe.bind((args.host, args.port))
    prepare_image(profile, cache, args.apptainer)
    download(profile['model'], cache)
    if 'speculative' in profile:
        download(profile['speculative'], cache)
    run_dir = cache / 'runs' / (profile['name'] + '-' + str(time.time_ns()))
    run_dir.mkdir(parents=True)
    compile_dir(profile, cache).mkdir(parents=True, exist_ok=True)
    cmd = command(profile, cache, args.apptainer, args.host, args.port, run_dir)
    logfile = run_dir / 'server.log'
    (run_dir / 'profile.json').write_text(json.dumps(profile, indent=2) + '\n')
    (run_dir / 'command.json').write_text(json.dumps(cmd, indent=2) + '\n')
    env = dict(os.environ)
    visible = args.device if args.device is not None else env.get('CUDA_VISIBLE_DEVICES')
    if visible is not None:
        env['CUDA_VISIBLE_DEVICES'] = visible
        env['APPTAINERENV_CUDA_VISIBLE_DEVICES'] = visible
    else:
        env.pop('APPTAINERENV_CUDA_VISIBLE_DEVICES', None)
    base = f"http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}"
    print(f'Log: {logfile}', flush=True)
    process = None
    previous = signal.getsignal(signal.SIGTERM)
    def terminate(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    try:
        with logfile.open('w') as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                       env=env, start_new_session=True)
            wait_ready(process, base, args.timeout)
            result = smoke(base, profile['name'], profile)
            (run_dir / 'smoke.json').write_text(json.dumps(result, indent=2) + '\n')
            print(f'Ready: {base}/v1 ; model={profile["name"]} ; speculative={"speculative" in profile}', flush=True)
            return process.wait()
    finally:
        # Kill only the process group we created, including engine workers.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=15)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        signal.signal(signal.SIGTERM, previous)


def slurm_script(profile, args):
    # Profile embedded in the job: queued runs cannot silently change with YAML edits.
    payload = base64.b64encode(json.dumps(profile).encode()).decode()
    python = os.path.abspath(sys.executable)  # Preserve virtualenv path, not its symlink target.
    argv = [python, '-m', 'llm_deploy', 'start', '--executor', 'local',
            '--cache-dir', str(args.cache_dir.expanduser().resolve()), '--host', args.host,
            '--port', str(args.port), '--timeout', str(args.timeout)]
    argv += ['--apptainer', args.apptainer]
    module_line = ''
    if args.apptainer_module:
        module_line = 'module load ' + shlex.quote(args.apptainer_module) + '\n'
    # Slurm owns CUDA_VISIBLE_DEVICES; no local device override here.
    script = ('#!/bin/bash -l\nset -euo pipefail\n'
              + module_line +
              'profile_file=$(mktemp)\ntrap \'rm -f "$profile_file"\' EXIT\n'
              f"printf %s {shlex.quote(payload)} | base64 --decode > \"$profile_file\"\n"
              + shlex.join(argv) + ' --profile "$profile_file" &\nchild=$!\n'
              'trap \'kill -TERM "$child" 2>/dev/null || true; wait "$child" || true; exit 143\' TERM INT\n'
              'wait "$child"\n')
    return script


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['profiles', 'validate', 'plan', 'start', 'health', 'smoke'])
    p.add_argument('--profile', default='tiny-qwen')
    p.add_argument('--executor', choices=['local', 'slurm'], default='local')
    p.add_argument('--cache-dir', type=Path, default=Path(os.environ.get('XDG_CACHE_HOME', '~/.cache')) / 'llm-deploy')
    p.add_argument('--apptainer', default='apptainer', help='Apptainer executable on the execution machine')
    p.add_argument('--apptainer-module', help='Optional module name loaded inside the Slurm job')
    p.add_argument('--host', default='127.0.0.1', help='IPv4 bind address; loopback by default')
    p.add_argument('--port', type=int, default=18081)
    p.add_argument('--timeout', type=int, default=600)
    p.add_argument('--device', help='Local CUDA_VISIBLE_DEVICES override; omitted by default')
    p.add_argument('--cpu', action='store_true', help='llama.cpp: explicitly disable GPU offload')
    p.add_argument('--dry-run', action='store_true', help='Print plan without downloads, writes or processes')
    p.add_argument('--partition')
    p.add_argument('--account')
    p.add_argument('--node')
    p.add_argument('--gres', default='gpu:1')
    p.add_argument('--cpus', type=int, default=4)
    p.add_argument('--mem', default='8G')
    p.add_argument('--time', default='02:00:00')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.action == 'profiles':
            print('\n'.join(p.stem for p in sorted((PACKAGE / 'profiles').glob('*.yaml'))))
            return 0
        profile = load_profile(args.profile)
        if not 1024 <= args.port <= 65535 or args.timeout <= 0 or args.cpus <= 0:
            raise ValueError('Port must be 1024..65535; timeout and CPUs must be positive')
        if args.cpu:
            if profile['engine'] != 'llama-cpp':
                raise ValueError('--cpu is only supported for llama.cpp')
            profile['runtime']['gpu_layers'] = 0
        if (args.executor == 'slurm' or os.environ.get('SLURM_JOB_ID')) and args.device is not None:
            raise ValueError('Slurm controls GPU visibility; omit --device')
        if args.apptainer_module and args.executor != 'slurm':
            raise ValueError('--apptainer-module requires Slurm; load modules in your shell for local execution')
        if args.action == 'validate':
            print(json.dumps(profile, indent=2))
            return 0
        if args.action == 'plan' or args.dry_run:
            print(json.dumps(dict(executor=args.executor, runtime='apptainer', profile=profile,
                  image_pull=[args.apptainer, 'pull', str(image_path(profile, args.cache_dir.expanduser().resolve())), profile['container']['image']],
                  command=command(profile, args.cache_dir.expanduser().resolve(), args.apptainer, args.host, args.port)), indent=2))
            if args.executor == 'slurm':
                print(slurm_script(profile, args))
            return 0
        if args.action in ('health', 'smoke'):
            base = f'http://{args.host}:{args.port}'
            result = request(base, '/health') if args.action == 'health' else smoke(base, profile['name'], profile)
            print(json.dumps(result, indent=2))
            return 0
        if args.executor == 'local':
            return start_local(profile, args)
        if os.environ.get('SLURM_JOB_ID'):
            raise ValueError('Already in a Slurm job; use --executor local inside the allocation')
        cmd = ['sbatch', '--parsable', '--export=ALL', '--job-name=' + profile['name'],
               '--gres=' + args.gres, '--cpus-per-task=' + str(args.cpus),
               '--mem=' + args.mem, '--time=' + args.time, '--signal=B:TERM@60']
        for field in ('partition', 'account', 'node'):
            if getattr(args, field):
                flag = 'nodelist' if field == 'node' else field
                cmd += ['--' + flag + '=' + getattr(args, field)]
        subprocess.run(cmd, input=slurm_script(profile, args), text=True, check=True)
        return 0
    except KeyboardInterrupt:
        print('Stopped.', file=sys.stderr)
        return 130
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
