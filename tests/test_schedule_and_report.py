import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / (name + '.py'))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


class ScheduleAndReportTests(unittest.TestCase):
    def scheduler_fixture(self):
        base = ROOT / 'archive/test_runs'
        base.mkdir(parents=True, exist_ok=True)
        folder = base / ('schedule_' + uuid.uuid4().hex)
        folder.mkdir()
        self.addCleanup(lambda: shutil.rmtree(folder) if folder.resolve().is_relative_to(base.resolve()) else None)
        return SimpleNamespace(output_root=folder / 'output', artifacts_root=folder / 'artifacts',
                               cifar_data_root=folder / 'cifar', fmnist_data_root=folder / 'fmnist',
                               workers=2, gpus=['0', '1'])

    def test_matrix_and_first_seed_barrier(self):
        config = json.loads((ROOT / 'configs/paper_lr001.json').read_text(encoding='utf-8'))
        worker = module('schedule_fl')
        jobs = worker.plan(config)
        self.assertEqual(len(jobs), 500)
        self.assertEqual({j['seed'] for j in jobs[:100]}, {2025})
        self.assertEqual({j['phase'] for j in jobs[100:]}, {'remaining_seeds'})
        self.assertEqual(len({(j['dataset'], j['alpha'], j['method'], j['seed']) for j in jobs}), 500)
        with self.assertRaises(ValueError):
            worker.plan(config, seeds=[2025, 2025])
        with self.assertRaises(ValueError):
            worker.plan(config, methods=['scaffold'])

    def test_dispatch_environment_and_failed_status_without_training(self):
        worker = module('schedule_fl')
        config = json.loads((ROOT / 'configs/paper_lr001.json').read_text(encoding='utf-8'))
        jobs = worker.plan(config, seeds=[2025, 2027], methods=['fedavg'])
        args = self.scheduler_fixture()
        calls = []

        class CompletedProcess:
            pid = 123
            def __init__(self, command, **kwargs):
                if '--route' not in command or '--method' in command:
                    raise AssertionError('Scheduler must use the portable worker CLI')
                calls.append((command, kwargs['env']))
            def poll(self): return 0
            def wait(self): return 0
            def terminate(self): raise AssertionError('Completed process must not be terminated')

        with patch.object(worker.subprocess, 'Popen', CompletedProcess):
            worker.run(args, jobs)
        self.assertEqual(len(calls), 20)
        self.assertTrue(all(command[command.index('--seed') + 1] == '2025' for command, _ in calls[:10]))
        self.assertEqual({env['CUDA_VISIBLE_DEVICES'] for _, env in calls}, {'0', '1'})
        self.assertTrue(all(env['OMP_NUM_THREADS'] == '2' for _, env in calls))
        self.assertEqual(json.loads((args.output_root / 'scheduler_status.json').read_text())['state'], 'complete')

        args = self.scheduler_fixture()
        class FailedProcess(CompletedProcess):
            def poll(self): return 1
        with patch.object(worker.subprocess, 'Popen', FailedProcess), self.assertRaises(RuntimeError):
            worker.run(args, jobs)
        self.assertEqual(json.loads((args.output_root / 'scheduler_status.json').read_text())['state'], 'failed')
        self.assertFalse((args.output_root / 'first_seed_complete.json').exists())

    def test_recorded_table_uses_complete_five_seed_matrix(self):
        value = json.loads((ROOT / 'results/lr001_tail300.json').read_text(encoding='utf-8'))
        report = module('report_results')
        text = report.render(value)
        self.assertEqual(text.count(' ★**'), 10)
        value['records'][0]['mean_percent'] += 0.1
        with self.assertRaisesRegex(ValueError, 'Recorded mean differs'):
            report.render(value)


if __name__ == '__main__':
    unittest.main()
