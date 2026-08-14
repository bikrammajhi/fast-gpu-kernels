"""GPU hardware catalog and auto-detection.

Every derived number in the report (TFLOPS, DRAM %, theoretical occupancy)
is computed against the *detected* device's peak specs, so reports are
correct on any GPU ncu can profile. Detection is a two-step cascade:

1. Device attributes measured by ncu itself (device__attribute_* in the
   .ncu-rep) always win — they are the hardware's own numbers.
2. Otherwise the device name (from the report or --gpu) is looked up in
   :data:`DEVICES`; an unknown name falls back to :data:`FALLBACK` with a
   warning flag so the report stays honest instead of silently wrong.

Peak TFLOPS are the vendor's dense FP16 (tensor core) numbers — the units
the pipeline's TFLOPS row is expressed in. `dram_peak` is the memory
bandwidth spec (bytes/s). All other fields are per-SM limits needed by the
theoretical-occupancy rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# dense FP16 tensor TFLOPS, memory bandwidth bytes/s, per-SM limits
DEVICES: dict[str, dict] = {
    # ---- datacenter Hopper ----
    "H100 SXM": dict(tensor_peak=989.0, dram_peak=3.35e12, max_warps_per_sm=64,
                     smsp_per_sm=4, max_smem_per_sm=228.0 * 2 ** 20, nvcc_arch="sm_90a"),
    "H100 PCIe": dict(tensor_peak=989.0, dram_peak=2.0e12, max_warps_per_sm=64,
                      smsp_per_sm=4, max_smem_per_sm=228.0 * 2 ** 20, nvcc_arch="sm_90a"),
    "H200": dict(tensor_peak=989.0, dram_peak=4.8e12, max_warps_per_sm=64,
                 smsp_per_sm=4, max_smem_per_sm=228.0 * 2 ** 20, nvcc_arch="sm_90a"),
    # ---- datacenter Blackwell ----
    "B200": dict(tensor_peak=2250.0, dram_peak=8.0e12, max_warps_per_sm=64,
                 smsp_per_sm=4, max_smem_per_sm=227.0 * 2 ** 20, nvcc_arch="sm_100a"),
    "GB200": dict(tensor_peak=2250.0, dram_peak=8.0e12, max_warps_per_sm=64,
                  smsp_per_sm=4, max_smem_per_sm=227.0 * 2 ** 20, nvcc_arch="sm_100a"),
    "B100": dict(tensor_peak=1800.0, dram_peak=8.0e12, max_warps_per_sm=64,
                 smsp_per_sm=4, max_smem_per_sm=227.0 * 2 ** 20, nvcc_arch="sm_100a"),
    "RTX PRO 6000 Blackwell": dict(tensor_peak=2500.0, dram_peak=1.79e12, max_warps_per_sm=64,
                                   smsp_per_sm=4, max_smem_per_sm=227.0 * 2 ** 20, nvcc_arch="sm_120a"),
    # ---- datacenter Ampere ----
    "A100": dict(tensor_peak=312.0, dram_peak=1.555e12, max_warps_per_sm=64,
                 smsp_per_sm=4, max_smem_per_sm=164.0 * 2 ** 20, nvcc_arch="sm_80"),
    "A100 SXM4 40 GB": dict(tensor_peak=312.0, dram_peak=1.555e12, max_warps_per_sm=64,
                            smsp_per_sm=4, max_smem_per_sm=164.0 * 2 ** 20, nvcc_arch="sm_80"),
    "A100 SXM4 80 GB": dict(tensor_peak=312.0, dram_peak=2.039e12, max_warps_per_sm=64,
                            smsp_per_sm=4, max_smem_per_sm=164.0 * 2 ** 20, nvcc_arch="sm_80"),
    "A40": dict(tensor_peak=149.7, dram_peak=696.0e9, max_warps_per_sm=64,
                smsp_per_sm=4, max_smem_per_sm=164.0 * 2 ** 20, nvcc_arch="sm_86"),
    "A10": dict(tensor_peak=125.0, dram_peak=600.0e9, max_warps_per_sm=64,
                smsp_per_sm=4, max_smem_per_sm=164.0 * 2 ** 20, nvcc_arch="sm_86"),
    "A30": dict(tensor_peak=165.0, dram_peak=933.0e9, max_warps_per_sm=64,
                smsp_per_sm=4, max_smem_per_sm=164.0 * 2 ** 20, nvcc_arch="sm_80"),
    # ---- Ada Lovelace ----
    "RTX 4090": dict(tensor_peak=330.3, dram_peak=1.008e12, max_warps_per_sm=64,
                     smsp_per_sm=4, max_smem_per_sm=100.0 * 2 ** 20, nvcc_arch="sm_89"),
    "RTX 4080": dict(tensor_peak=290.0, dram_peak=716.8e9, max_warps_per_sm=64,
                     smsp_per_sm=4, max_smem_per_sm=100.0 * 2 ** 20, nvcc_arch="sm_89"),
    "RTX 4070": dict(tensor_peak=234.0, dram_peak=504.2e9, max_warps_per_sm=64,
                     smsp_per_sm=4, max_smem_per_sm=100.0 * 2 ** 20, nvcc_arch="sm_89"),
    "L40": dict(tensor_peak=181.0, dram_peak=864.0e9, max_warps_per_sm=64,
                smsp_per_sm=4, max_smem_per_sm=100.0 * 2 ** 20, nvcc_arch="sm_89"),
    "L4": dict(tensor_peak=121.0, dram_peak=300.0e9, max_warps_per_sm=64,
               smsp_per_sm=4, max_smem_per_sm=100.0 * 2 ** 20, nvcc_arch="sm_89"),
    # ---- Volta / Turing ----
    "V100": dict(tensor_peak=112.0, dram_peak=900.0e9, max_warps_per_sm=64,
                 smsp_per_sm=4, max_smem_per_sm=96.0 * 2 ** 20, nvcc_arch="sm_70"),
    "T4": dict(tensor_peak=65.0, dram_peak=320.0e9, max_warps_per_sm=48,
               smsp_per_sm=4, max_smem_per_sm=64.0 * 2 ** 20, nvcc_arch="sm_75"),
    "RTX 2080": dict(tensor_peak=54.0, dram_peak=448.0e9, max_warps_per_sm=48,
                     smsp_per_sm=4, max_smem_per_sm=64.0 * 2 ** 20, nvcc_arch="sm_75"),
}

# Conservative generic defaults for an unknown device (honest, flagged).
FALLBACK = dict(tensor_peak=1000.0, dram_peak=2.0e12, max_warps_per_sm=64,
                smsp_per_sm=4, max_smem_per_sm=164.0 * 2 ** 20, nvcc_arch="sm_90a")

_ATTR_PEAKS = {
    # measured device attribute (in .ncu-rep / raw CSV) -> config key
    "device__attribute_max_warps_per_multiprocessor": "max_warps_per_sm",
    "device__attribute_max_shared_memory_per_multiprocessor": "max_smem_per_sm",
}


_VENDOR_WORDS = ("nvidia", "geforce", "quadro", "tesla")


def _norm(s: str) -> str:
    n = "".join(c for c in s.lower() if c.isalnum())
    for v in _VENDOR_WORDS:
        n = n.replace(v, "")
    return n


def _match_name(name: str) -> str | None:
    """Catalog match on a display name like 'NVIDIA B200' or 'NVIDIA H100'.

    Exact match first; then prefix match (display names usually omit the
    memory/variant suffix, e.g. 'H100' -> 'H100 SXM'); ties prefer SXM.
    """
    if not name:
        return None
    n = _norm(name)
    if not n:
        return None
    keys = {k: _norm(k) for k in DEVICES}
    exact = [k for k, kn in keys.items() if kn == n]
    if exact:
        return exact[0]
    cands = [k for k, kn in keys.items() if kn.startswith(n)]
    if cands:
        sxm = [k for k in cands if "sxm" in k.lower()]
        return (sxm or cands)[0]
    return None


@dataclass
class Device:
    """The hardware a profile ran on, plus the spec used for derived rows."""

    name: str | None = None
    detected: bool = False          # True: name came from the profile itself
    matched: bool = False           # True: catalog entry found
    spec: dict = field(default_factory=dict)
    note: str = ""


def detect_device(name: str | None = None,
                  attributes: dict | None = None,
                  override: dict | None = None) -> Device:
    """Resolve the spec to use for derived rows.

    `override` (user --config / --gpu) wins; then measured device
    attributes; then the catalog by name; then the fallback.
    """
    attrs = attributes or {}
    spec: dict = {}
    if override:
        spec.update({k: v for k, v in override.items() if v is not None})
    matched = False
    note = ""
    if not spec.get("tensor_peak"):
        key = _match_name(name)
        if key:
            spec.update(DEVICES[key])
            matched = True
        elif name:
            spec.update(FALLBACK)
            note = (f"device '{name}' not in catalog; "
                    f"using generic peaks — override with --gpu/--config")
        else:
            note = "device not recorded in input; using default peak specs"
    for attr, key in _ATTR_PEAKS.items():
        v = attrs.get(attr)
        if v and v > 0:
            spec[key] = float(v)
    per_sched = attrs.get("device__attribute_max_warps_per_scheduler")
    max_w = spec.get("max_warps_per_sm")
    if per_sched and max_w:
        spec["smsp_per_sm"] = max(1, int(max_w) // int(per_sched))
    if "M" not in spec:
        spec["M"] = None
    return Device(name=name or "unknown", detected=name is not None,
                  matched=matched, spec=spec, note=note)


def cfg_for_device(device: Device, user_cfg: dict | None = None) -> dict:
    """The config dict sections consume, resolved for the detected device."""
    from .sections import CONFIG_DEFAULTS

    cfg = {**CONFIG_DEFAULTS,
           **{k: v for k, v in (device.spec or {}).items() if v is not None}}
    if user_cfg:
        cfg.update({k: v for k, v in user_cfg.items() if v is not None})
    return cfg
