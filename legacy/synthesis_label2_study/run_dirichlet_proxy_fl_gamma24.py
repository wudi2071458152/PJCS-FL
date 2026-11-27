"""Keep FL equations unchanged; require the new branch's complete gamma24 binding."""
from pathlib import Path
from unittest.mock import patch
import dirichlet_gamma24 as g
import dirichlet_continue32000_worker as worker
import run_dirichlet_proxy_fl as p

PREPARE = p.prepare


def prepare(a):
    prepared = PREPARE(a)
    protocol = prepared[-1]
    if protocol['proxy_loss'] != 'low_dm_direction' or protocol['iteration'] != 32000:
        raise RuntimeError('Gamma24 FL entry only accepts new-loss 32000 images')
    source_protocol = p.c.read(Path(a.proxy_root)/'protocol.json')
    if source_protocol['loss_settings'].get('gamma_policy') != g.POLICY:
        raise RuntimeError('New proxy branch is not bound to gamma24 calibration')
    protocol['gamma_policy'] = g.POLICY
    for file in (Path(__file__),Path(g.__file__)):
        protocol['files_sha256']['code/'+file.relative_to(p.c.ROOT).as_posix()] = p.c.digest(file)
    return prepared


def main():
    with patch.object(worker,'gamma_manifest',g.gamma_manifest), patch.object(p,'prepare',prepare):
        p.main()


if __name__=='__main__': main()
