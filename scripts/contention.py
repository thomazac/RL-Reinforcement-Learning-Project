"""Per-process env speed when N copies of the Phase 1a env run at once: random actions, no policy, no
RLGym-Learn. Compared with scripts/throughput.py at the same process count, it separates CPU contention (cores,
memory bandwidth) from the cost of RLGym-Learn's lockstep exchange with the main process.

    python scripts/contention.py 1 8 12 16 20
"""

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = "1"

import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

WARMUP_STEPS, STEPS = 500, 4000


def run(seed, start, results):
    import numpy as np

    from playstyle.env import make_env

    env = make_env(coordination=False, seed=seed)
    rng = np.random.default_rng(seed)
    env.reset()

    def steps(n):
        for _ in range(n):
            _, _, term, trunc = env.step({a: np.array([rng.integers(90)]) for a in env.agents})
            if any(term.values()) or any(trunc.values()):
                env.reset()

    steps(WARMUP_STEPS)
    start.wait()  # all copies time the same stretch
    t = time.perf_counter()
    steps(STEPS)
    results.put(4 * STEPS / (time.perf_counter() - t))


def main():
    ctx = mp.get_context("spawn")
    for n in map(int, sys.argv[1:] or ["1", "8"]):
        start, results = ctx.Barrier(n), ctx.Queue()
        procs = [ctx.Process(target=run, args=(i, start, results)) for i in range(n)]
        for p in procs:
            p.start()
        rates = [results.get() for _ in procs]
        for p in procs:
            p.join()
        print(f"{n:2d} copies: {sum(rates) / n:6,.0f} agent-steps/s per copy (slowest {min(rates):,.0f}), "
              f"{sum(rates):7,.0f} in total", flush=True)


if __name__ == "__main__":
    main()
