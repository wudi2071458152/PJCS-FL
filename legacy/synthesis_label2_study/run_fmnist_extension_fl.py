"""Eighteen unchanged FMNIST methods for alpha=0.2/0.3 and five FL seeds."""
import run_fmnist_fl as _original
from fmnist_extension_adapter import functions

functions(_original, globals())
import fmnist_extension_common as common
ALPHAS, SUITE = common.ALPHAS, common.SUITE

if __name__ == '__main__':
    main()
