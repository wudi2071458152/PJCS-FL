"""Keep frozen FL numerics and decorated evaluation; fix project provenance."""
import inspect
import sys
from pathlib import Path
from unittest.mock import patch
import run_dirichlet_proxy_fl as p
import run_dirichlet_proxy_fl_gamma24 as gammafl

PREPARE = p.prepare
EVALUATE = p.evaluate
POLICY = 'delegate unchanged no_grad evaluation; fingerprint unwrapped project source'


def evaluate(*args, **kwargs):
    return EVALUATE(*args, **kwargs)


def prepare(a):
    with patch.object(p, 'evaluate', evaluate):
        result = PREPARE(a)
    protocol = result[-1]
    source = Path(inspect.unwrap(EVALUATE).__code__.co_filename).resolve()
    for file in (source, Path(__file__).resolve()):
        protocol['files_sha256']['code/' + file.relative_to(p.c.ROOT).as_posix()] = p.c.digest(file)
    protocol['evaluation_fingerprint_policy'] = POLICY
    return result


def main():
    new_images = '--gamma24' in sys.argv
    if new_images:
        sys.argv.remove('--gamma24')
    with patch.object(p, 'evaluate', evaluate), patch.object(p, 'prepare', prepare), \
         patch.object(gammafl, 'PREPARE', prepare):
        (gammafl.main if new_images else p.main)()


if __name__ == '__main__':
    main()
