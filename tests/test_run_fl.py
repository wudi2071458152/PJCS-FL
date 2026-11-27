"""Portable binding checks and small CPU parity fixtures; no benchmark training."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('public_run_fl', ROOT / 'run_fl.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class PortableRunnerTests(unittest.TestCase):
    def workspace(self):
        base = ROOT / 'archive/test_runs'
        base.mkdir(parents=True, exist_ok=True)
        path = base / ('portable_fl_' + uuid.uuid4().hex)
        path.mkdir()
        def cleanup():
            if not path.resolve().is_relative_to(base.resolve()):
                raise ValueError('Test cleanup escaped workspace')
            shutil.rmtree(path)
        self.addCleanup(cleanup)
        return path

    def test_partition_coverage_and_output_isolation(self):
        rows = [np.array([i, i + 32]) for i in range(32)]
        self.assertEqual(len(runner.partition_indices({'client_indices': rows}, 64)), 32)
        rows[-1] = rows[0]
        with self.assertRaisesRegex(ValueError, 'cover the real'):
            runner.partition_indices({'client_indices': rows}, 64)
        root = self.workspace()
        for output in (root / 'inputs/child', root):
            with self.assertRaisesRegex(ValueError, 'overlaps'):
                runner.isolated_output(output, root / 'inputs')
        runner.isolated_output(root / 'outputs', root / 'inputs')

    def test_default_preflight_does_not_train_or_create_output(self):
        root = self.workspace()
        args = ['--dataset', 'cifar10', '--alpha', '0p05', '--route', 'fedavg',
                '--data-root', str(root / 'data'), '--output', str(root / 'out')]
        with patch.object(runner, 'prepare', return_value=(None, {}, {'ready': True})), patch.object(runner, 'train') as train, patch('builtins.print'):
            runner.main(args)
        train.assert_not_called()
        self.assertFalse((root / 'out').exists())

    def test_hash_rejected_before_deserializing_and_foreign_output_preserved(self):
        root = self.workspace()
        release = root / 'release'
        (release / 'assets').mkdir(parents=True)
        (release / 'configs').mkdir()
        (release / 'run_fl.py').write_bytes(b'# fixture')
        (release / 'configs/paper_lr001.json').write_text('{}', encoding='utf-8')
        (release / 'assets/source_manifest.json').write_text('{"files": []}', encoding='utf-8')
        inventory = {'support_files': [dict(dataset='cifar10', alpha='0p05',
            role='inputs/manifest.json', asset_path='wrong.pt', sha256='0' * 64)]}
        (release / 'assets/proxy_manifest.json').write_text(json.dumps(inventory), encoding='utf-8')
        artifacts = root / 'inputs'
        artifacts.mkdir()
        (artifacts / 'wrong.pt').write_bytes(b'not a trusted tensor file')
        args = SimpleNamespace(dataset='cifar10', alpha='0p05', route='fedavg', seed=2025,
            rounds=1500, artifacts_root=artifacts, data_root=root / 'data', output=root / 'output')
        with patch.object(runner, 'ROOT', release), patch.object(runner.c, 'load') as load:
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                runner.prepare(args)
        load.assert_not_called()
        (root / 'foreign.txt').write_bytes(b'keep')
        with self.assertRaisesRegex(ValueError, 'nonempty output'):
            runner.train(SimpleNamespace(output=root), None)
        self.assertEqual((root / 'foreign.txt').read_bytes(), b'keep')

    def test_initialization_uses_saved_theta0_and_refuses_wrong_new_seed(self):
        root = self.workspace()
        base = torch.nn.Linear(2, 2)
        saved = copy.deepcopy(base.state_dict())
        for tensor in saved.values():
            tensor.add_(1.)
        expected = runner.multi.state_fingerprint(saved)
        model_path = root / 'initial.pt'
        torch.save(saved, model_path)
        manifest = {'datasets': {'cifar10': {'2025': expected, '2027': 'f' * 64}}}
        fake_root = root / 'release'
        (fake_root / 'assets').mkdir(parents=True)
        (fake_root / 'assets/initialization_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        with patch.object(runner, 'ROOT', fake_root), patch.object(runner.c, 'model', side_effect=lambda seed: copy.deepcopy(base)):
            actual, sha = runner.initial_model('cifar10', '0p05', 2025, root, model_path, lambda *args: None)
            self.assertEqual(sha, expected)
            for name, value in actual.state_dict().items():
                self.assertTrue(torch.equal(value, saved[name]))
            with self.assertRaisesRegex(RuntimeError, 'Initial model differs'):
                runner.initial_model('cifar10', '0p05', 2027, root, model_path, lambda *args: None)

    def test_ten_routes_preserve_direct_kernel_selection_and_updates(self):
        net = torch.nn.Linear(2, 2)
        real = [(torch.full((i + 2, 2), float(i)), torch.arange(i + 2) % 2) for i in range(32)]
        syn = [(x + .25, y, torch.ones(2)) for x, y in real]
        weights = np.arange(2, 34, dtype=float); weights /= weights.sum()
        def update(net, x, y, seed, *args):
            value = float(x[0, 0]) + 1.
            return torch.tensor([value, value * value / 10., float(seed % 7) + .1, -value / 3., .2, .3])
        def query(net, x, y, gradient=False):
            return update(net, x, y, 0) if gradient else float(x[0, 0])
        with patch.object(runner.c, 'DEVICE', torch.device('cpu')):
            for route in runner.CONFIG['routes']:
                with self.subTest(route=route):
                    left = runner.worker.initial_state(net, route, weights)
                    right = copy.deepcopy(left)
                    delta, row, _ = runner.worker.one_round(net, real, syn, weights, route, 2025, 1, left, update_fn=update, query_fn=query)
                    if route in runner.legacy.algorithms.ROUTES:
                        expected, direct, _ = runner.legacy.algorithms.one_round(net, real, weights, route, 2025, 1, right, update, query)
                    elif route == 'hcs_static_gram_equal':
                        expected, direct, _ = runner.legacy.one_round(net, real, syn, weights, route, 2025, 1, right, update_fn=update)
                    else:
                        expected, direct = runner.multi.one_round(net, real, syn, weights, route, 2025, 1, history=right, update_fn=update)
                    torch.testing.assert_close(delta, expected, atol=0, rtol=0)
                    self.assertEqual(row['selected'], direct['selected'])
                    self.assertEqual(row.get('pi'), direct.get('pi'))
                    self.assertEqual(row['communication'], runner.legacy.costs.communication(route, row, 6))

    def test_retained_checkpoint_loop_resumes_same_small_fixture(self):
        root = self.workspace()
        torch.manual_seed(19)
        initial = torch.nn.Linear(2, 2)
        real = [(torch.full((i + 2, 2), float(i)), torch.arange(i + 2) % 2) for i in range(32)]
        weights = np.arange(2, 34, dtype=float); weights /= weights.sum()
        test = (torch.tensor([[0., 0.], [1., 1.]]), torch.tensor([0, 1]))
        protocol = dict(suite='pjcs_portable_lr001', route='fedavg', seed=2025, alpha='0p05',
            rounds=2, weights=weights.tolist(), model_parameters=6, files_sha256={}, files_paths={},
            communication=runner.legacy.costs.policy(initial, None),
            initial_model_state_sha256=runner.legacy.cv.tensor_fingerprint(initial.state_dict()))
        protocol.update(runner.worker.route_definition('fedavg'))
        def update(net, x, y, seed, *args):
            return torch.full((6,), float(seed % 7) / 1000.)
        original_round = runner.worker.one_round
        def one_round(*args, **kwargs):
            return original_round(*args, **dict(kwargs, update_fn=update))
        def run(folder, net, deadline):
            args = SimpleNamespace(output=folder, route='fedavg', seed=2025, alpha='0p05', rounds=2, deadline=deadline)
            prepared = net, real, None, weights, test, copy.deepcopy(protocol)
            return runner.train(args, prepared)
        resumed, fresh = copy.deepcopy(initial), copy.deepcopy(initial)
        with patch.object(runner.worker, 'one_round', side_effect=one_round), patch.object(runner.c, 'DEVICE', torch.device('cpu')):
            self.assertFalse(run(root / 'resumed', resumed, 0.))
            self.assertTrue((root / 'resumed/resume.pt').is_file())
            self.assertTrue(run(root / 'resumed', resumed, None))
            self.assertTrue(run(root / 'fresh', fresh, None))
        for name, value in fresh.state_dict().items():
            torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)
        for folder in ('fresh', 'resumed'):
            runner.verify_output(root / folder, expected_rounds=2)
        summary = runner.c.read(root / 'resumed/summary.json')
        self.assertEqual(summary['rounds'], 2)
        self.assertIn('mean_last300_test_accuracy', summary)


if __name__ == '__main__':
    unittest.main()
