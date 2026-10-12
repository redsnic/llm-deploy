"""Offline unit tests. No model downloads, engines, GPU, or Slurm calls."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from llm_deploy.cli import command, load_profile, main, parser, slurm_script, prepare_image, image_path


class Profiles(unittest.TestCase):
    def test_bundled_profiles(self):
        for name in ('tiny-qwen', 'qwen-h200', 'qwen-dflash'):
            self.assertEqual(load_profile(name)['name'], name)

    def test_target_only_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.yaml'
            path.write_text('schema_version: 1\nname: other\nengine: vllm\nmodel:\n  repo: owner/model\n  revision: ' + 'a' * 40 + '\n')
            profile = load_profile(str(path))
            args = command(profile, Path(tmp), None, '127.0.0.1', 18081)
            self.assertNotIn('--speculative-config', args)
            self.assertNotIn('--reasoning-parser', args)
            self.assertNotIn('--tool-call-parser', args)
            self.assertEqual(profile['runtime']['parallel'], 1)
            self.assertEqual(profile['runtime']['context'], 4096)
            self.assertIn('auto', args)

    def test_tiny_has_no_draft_and_conservative_settings(self):
        profile = load_profile('tiny-qwen')
        args = command(profile, Path('/tmp/cache'), None, '127.0.0.1', 18081)
        self.assertNotIn('speculative', profile)
        self.assertEqual(args[args.index('--flash-attn') + 1], 'off')
        self.assertEqual(args[args.index('--ctx-size') + 1], '2048')
        self.assertEqual(args[args.index('--parallel') + 1], '1')
        self.assertEqual(profile['model']['bytes'], 491400032)

    def test_speculation_is_explicit(self):
        args = command(load_profile('qwen-dflash'), Path('/tmp/cache'), None, '127.0.0.1', 18081)
        spec = json.loads(args[args.index('--speculative-config') + 1])
        self.assertEqual(spec['num_speculative_tokens'], 7)
        self.assertEqual(spec['method'], 'dflash')

    def test_apptainer_mounts_and_container_paths(self):
        profile = load_profile('tiny-qwen')
        args = command(profile, Path('/tmp/cache'), None, '127.0.0.1', 18081)
        self.assertEqual(args[:3], ['apptainer', 'exec', '--cleanenv'])
        self.assertIn('--nv', args)
        self.assertIn('/tmp/cache/models:/models:ro', args)
        self.assertIn('/app/llama-server', args)
        model = args[args.index('--model') + 1]
        self.assertTrue(model.startswith('/models/'))
        self.assertNotIn('/tmp/cache', model)
        profile['runtime']['gpu_layers'] = 0
        self.assertNotIn('--nv', command(profile, Path('/tmp/cache'), None, '127.0.0.1', 18081))

    def test_local_and_slurm_share_pinned_container(self):
        profile = load_profile('tiny-qwen')
        args = parser().parse_args(['start', '--executor', 'slurm', '--apptainer-module', 'apptainer/1.5.2'])
        script = slurm_script(profile, args)
        self.assertIn('module load apptainer/1.5.2', script)
        self.assertIn('--apptainer apptainer', script)
        self.assertIn('--executor local', script)
        self.assertIn('@sha256:', profile['container']['image'])

    def test_image_cache_reuse_and_corruption(self):
        import hashlib
        profile = load_profile('tiny-qwen')
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            image = image_path(profile, cache)
            image.parent.mkdir(parents=True)
            image.write_bytes(b'cached image')
            image.with_suffix('.sif.sha256').write_text(hashlib.sha256(b'cached image').hexdigest())
            with patch('llm_deploy.cli.subprocess.run') as run:
                prepare_image(profile, cache, 'apptainer')
                run.assert_not_called()
            image.write_bytes(b'corrupted')
            def fake_pull(argv, **kwargs):
                Path(argv[2]).write_bytes(b'rebuilt image')
            with patch('llm_deploy.cli.subprocess.run', side_effect=fake_pull) as run:
                prepare_image(profile, cache, 'apptainer')
                run.assert_called_once()
                self.assertEqual(run.call_args.args[0][-1], profile['container']['image'])
            self.assertEqual(image.read_bytes(), b'rebuilt image')

    def test_unknown_settings_fail(self):
        for change in ('unknown', 'runtime', 'speculative', 'revision'):
            with tempfile.TemporaryDirectory() as tmp:
                profile = load_profile('tiny-qwen')
                if change == 'unknown':
                    profile['typo'] = True
                elif change == 'runtime':
                    profile['runtime']['dtype'] = 'bfloat16'
                elif change == 'speculative':
                    profile['speculative'] = load_profile('qwen-dflash')['speculative']
                else:
                    profile['model']['revision'] = 'main'
                path = Path(tmp) / 'bad.yaml'
                path.write_text(json.dumps(profile))
                with self.assertRaises(Exception):
                    load_profile(str(path))

    def test_plan_is_offline(self):
        with patch('llm_deploy.cli.download', side_effect=AssertionError('download')), \
             patch('llm_deploy.cli.subprocess.Popen', side_effect=AssertionError('launch')), \
             patch('llm_deploy.cli.subprocess.run', side_effect=AssertionError('submit')), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(main(['start', '--dry-run']), 0)
            self.assertEqual(main(['plan', '--executor', 'slurm']), 0)

    def test_slurm_submits_once_and_runs_local_inside_job(self):
        with patch('llm_deploy.cli.subprocess.run') as run, \
             patch.dict('os.environ', {}, clear=True):
            self.assertEqual(main(['start', '--executor', 'slurm']), 0)
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0][0], 'sbatch')
        script = run.call_args.kwargs['input']
        self.assertIn('--executor local', script)
        self.assertNotIn('sbatch', script)
        self.assertNotIn('CUDA_VISIBLE_DEVICES', script)
        self.assertIn('base64 --decode', script)


if __name__ == '__main__':
    unittest.main()
