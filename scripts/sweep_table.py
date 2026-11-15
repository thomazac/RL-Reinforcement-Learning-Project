"""Table of scripts/throughput.py results appended to a JSON-lines file.

    python scripts/sweep_table.py runs/sweep.jsonl
"""

import json
import sys

for line in open(sys.argv[1] if len(sys.argv) > 1 else "runs/sweep.jsonl"):
    r = json.loads(line)
    exits = [c for c in r["env_process_exit_codes"] if c is not None]
    print(f"width {r['width']}  {r['procs']:2d} procs  actor after {r['inference_procs']:2d}  "
          f"collect {r['collect_sps']:7,.0f} ({r['collect_sps_per_proc']:5,.0f}/proc)  update {r['update_s']:.2f} s  "
          f"overall {r['overall_sps']:7,.0f}  wait {r['wait_share']:.0%}  inference {r['inference_share']:.0%}  "
          f"jump {r['monitors_worst']['Largest Observation Jump']:.3f}  exited {exits}")
