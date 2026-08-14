"""ncu-view: Nsight-Compute-style kernel profiles as a web report.

Profile a CUDA kernel (on Modal or anywhere ncu runs), then read the report
like Nsight Compute's GUI: a Summary page with prioritized recommendations
and a per-kernel Details page with the Speed Of Light, Occupancy, Warp State,
Compute/Memory Workload and Instruction sections.

Inputs: results/ncu_counters.json probe files (see ncu_profile_all.py), raw
ncu CSV dumps (`ncu --page raw --csv`), or .ncu-rep report files (via
NVIDIA's own ncu_report module).

    ncu-view report  path/to/report.ncu-rep --outdir out/
    ncu-view summary path/to/raw.csv
    ncu-view serve   path/to/report.ncu-rep --port 8000
    ncu-view profile kernels/cute_dsl/B200 --outdir out/
"""

__version__ = "0.2.0"

from .cli import main

__all__ = ["main"]
