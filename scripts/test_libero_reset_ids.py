"""Exercise LiberoEnv._get_ordered_reset_state_ids (eval pool branch) in isolation, from a given libero_env.py."""
import ast, sys, types
import numpy as np

def load_method(path):
    src = open(path).read(); tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_get_ordered_reset_state_ids":
            code = ast.get_source_segment(src, node)
            import textwrap
            ns = {"np": np}; exec(textwrap.dedent(code), ns); return ns["_get_ordered_reset_state_ids"]
    raise SystemExit("method not found")

def run(path, n_requests=20000, seed=0):
    f = load_method(path)
    rng = np.random.default_rng(seed)
    me = types.SimpleNamespace(specific_reset_id=None, is_eval=True, _eval_reset_pool=np.arange(50), start_idx=0)
    n_neg = n_total = 0; visits = np.zeros(50, int)
    for _ in range(n_requests):
        k = int(rng.choice([1, 1, 1, 2, 2]))          # envs finishing together (2 parallel envs)
        r = f(me, k); n_total += k; n_neg += int((r < 0).sum())
        for v in r[r >= 0]: visits[v] += 1
    return n_neg, n_total, visits.min(), visits.max()

for p in sys.argv[1:]:
    neg, tot, vmin, vmax = run(p)
    print(f"{p}: {neg} of {tot} requested reset ids were -1 ({100*neg/tot:.3f}%) | visits per init state min/max = {vmin}/{vmax}")
