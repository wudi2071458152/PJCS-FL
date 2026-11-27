"""The nine unchanged proxy FL methods, bound to continued 32000-step images."""
from functools import lru_cache
from types import FunctionType
import run_fmnist_fl as _frozen
from fmnist_extension_adapter import functions

functions(_frozen, globals())
import fmnist_continue32000_common as common

ALPHAS, SEEDS, SUITE = common.ALPHAS, common.FL_SEEDS, common.SUITE
ITERATION = 32000


def _image_binding(function, completion=False):
    """Replace only the frozen adapter's image milestone, never numerical kernels."""
    raw = getattr(function, '__wrapped__', function)
    constants = raw.__code__.co_consts
    if sum(type(v) is int and v == 8000 for v in constants) != 1:
        raise RuntimeError('Frozen FL image-binding layout changed: ' + raw.__name__)
    if completion and constants.count('COMPLETED_8000.json') != 1:
        raise RuntimeError('Frozen FL completion-binding layout changed')
    constants = tuple(ITERATION if type(v) is int and v == 8000 else
                      'COMPLETED_32000.json' if completion and v == 'COMPLETED_8000.json' else v
                      for v in constants)
    bound = FunctionType(raw.__code__.replace(co_consts=constants), globals(), raw.__name__,
                         raw.__defaults__, raw.__closure__)
    bound.__kwdefaults__ = raw.__kwdefaults__
    return lru_cache(**function.cache_parameters())(bound) if hasattr(function, 'cache_parameters') else bound


_prepared_data = _image_binding(_prepared_data, completion=True)
_prepare = _image_binding(prepare)
_verify_protocol = _image_binding(verify_protocol)


def prepare(a):
    if a.route not in PROXY:
        raise ValueError('Only proxy-dependent methods are rerun at 32000 steps')
    return _prepare(a)


def verify_protocol(folder):
    value = _verify_protocol(folder)
    if value['route'] not in PROXY:
        raise RuntimeError('A nonproxy result must be reused, not created in the 32000 suite')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs-root', 'source-root', 'data-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--alpha', choices=ALPHAS, required=True)
    parser.add_argument('--seed', choices=SEEDS, type=int, required=True)
    parser.add_argument('--route', choices=PROXY, required=True)
    parser.add_argument('--rounds', type=int, default=1500)
    parser.add_argument('--deadline', type=float)
    parser.add_argument('--run', action='store_true'); a = parser.parse_args()
    if a.rounds != 1500: parser.error('Formal FMNIST FL requires 1500 rounds')
    if a.run and not torch.cuda.is_available(): raise RuntimeError('Formal FL requires CUDA')
    prepared = prepare(a)
    if not a.run:
        print(json.dumps(dict(state='preflight_ok', formal_training_started=False, protocol=prepared[-1])))
        return
    def stop(*_): a.deadline = 0.
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    import schedule_dirichlet_screen as queue
    with queue.lock(a.output): raise SystemExit(0 if train(a, prepared) else 75)


if __name__ == '__main__': main()
