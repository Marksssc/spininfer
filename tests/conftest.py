import pytest
import numpy as np
import numba
numba.set_num_threads(min(8, numba.config.NUMBA_NUM_THREADS))
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

try:
    import jax
    jax.config.update("jax_enable_x64", True)
except ImportError:
    pass

from spininfer.models.ising import IsingModel

ALL_BACKENDS = ["numpy", "numba", "cupy", "jax"]


def _available(backend: str) -> bool:
    """Get the installed libraries"""
    try:
        IsingModel(n_sites=2, backend=backend)
        return True
    except ValueError:
        return False


def pytest_addoption(parser):
    parser.addoption("--backend", action="append", choices=ALL_BACKENDS,
                     help="run backend-dependent tests only for this backend (repeatable)")


def pytest_generate_tests(metafunc):
    if "backend" not in metafunc.fixturenames:
        return
    requested = metafunc.config.getoption("backend")
    backends = requested or [b for b in ALL_BACKENDS if _available(b)]
    params = [
        pytest.param(b, marks=pytest.mark.skipif(not _available(b), reason=f"{b} backend not installed"))
        for b in backends
    ]
    metafunc.parametrize("backend", params)

@pytest.fixture
def to_numpy():
    return lambda x: x.get() if hasattr(x, "get") else np.asarray(x)
