import importlib

def build_backend_dict(required: dict, optional: dict) -> dict:
    """
    required: {"numpy": ising_numpy.simulate, "numba": ising_numba.simulate}
              — these must exist, no fallback.
    optional: {"cupy": ("stats.mcmc.ising_cupy", "simulate"),
               "jax":  ("stats.mcmc.ising_jax", "simulate")}
              — (module path, function name) to try importing; skip if missing.
    """
    backends = dict(required)
    for name, (module_path, attr) in optional.items():
        try:
            module = importlib.import_module(module_path)
            backends[name] = getattr(module, attr)
        except ImportError:
            pass
    return backends