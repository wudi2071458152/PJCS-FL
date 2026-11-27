"""Frozen FMNIST construction and diagnostics for the two additional alphas."""
import run_fmnist_synthesis as _original
from fmnist_extension_adapter import functions

functions(_original, globals())
import fmnist_extension_common as common

if __name__ == '__main__':
    main()
