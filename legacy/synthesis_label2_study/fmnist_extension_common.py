"""Independent alpha=0.2/0.3 inputs using the frozen FMNIST numerical rules."""
import copy
import fmnist_common as _original
from fmnist_extension_adapter import functions

functions(_original, globals())
SUITE = 'fmnist_extension_20261017'
INPUTS = ROOT / 'inputs' / SUITE
ALPHAS = ('0p2', '0p3')
_source_protocol = protocol
_source_preflight = preflight
_source_prepare_inputs = prepare_inputs
context.__wrapped__.__defaults__ = (INPUTS, True)


def validate_config(path=None):
    expected = copy.deepcopy(_original.validate_config())
    expected.update(suite=SUITE, alphas=list(ALPHAS), first_seed_routes=36,
                    remaining_seed_routes=144, total_fl_routes=180, physical_jobs=252)
    value = c.read(Path(path) if path is not None else INPUTS / 'config.json')
    if value != expected:
        raise RuntimeError('Frozen FMNIST extension configuration differs from the confirmed matrix')
    return value


def protocol(inputs_root=INPUTS):
    value = _source_protocol(inputs_root)
    for name in ('run_fmnist_fl.py', 'fmnist_extension_adapter.py', 'fmnist_extension_common.py',
                 'run_fmnist_extension_synthesis.py', 'run_fmnist_extension_fl.py'):
        rel = 'synthesis_label2_study/' + name
        value['code_sha256'][rel] = c.digest(ROOT / rel)
    return json.loads(json.dumps(value))


def preflight(data_root, inputs_root=INPUTS):
    value = _source_preflight(data_root, inputs_root)
    value.update(reference_jobs=4, calibration_jobs=2, synthesis_jobs=64,
                 vector_jobs=2, fl_jobs=180)
    return value


def prepare_inputs(data_root, inputs_root=INPUTS, download=False):
    return _source_prepare_inputs(data_root, inputs_root, download)
