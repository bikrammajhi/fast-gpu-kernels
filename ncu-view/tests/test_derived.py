"""Derived metrics (tagged ours): ncu-view calculations over NVIDIA rows."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ncu_view.report import build  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
GOLDEN_REP = ROOT / "kernels" / "cute_dsl" / "B200" / "results" / "golden" / "matmul_v1.ncu-rep"
RAW_CSV = ROOT / "kernels" / "cute_dsl" / "B200" / "results" / "golden" / "matmul_v1.raw.csv"
BLOG_JSON = ROOT / "kernels" / "cute_dsl" / "B200" / "results" / "ncu_counters.json"


def _derived(path):
    r = build(str(path))
    return {d["name"]: d for d in r["kernels"][0]["derived"]}


def test_golden_derived_values():
    d = _derived(GOLDEN_REP)
    assert d["CTAs launched"]["value"] == 2048
    assert d["Total threads"]["value"] == 262144
    assert abs(d["Instructions per thread"]["value"] - 84.8) < 0.5
    assert abs(d["Kernel-wide IPC"]["value"] - 4.95) < 0.05
    assert abs(d["Duration per CTA"]["value"] - 1.18) < 0.05
    assert abs(d["Occupancy utilization"]["value"] - 100.2) < 1.0
    assert abs(d["Warp-slot stall share"]["value"] - 1.16) < 0.2
    assert abs(d["Implied SM clock"]["value"] - 1.86) < 0.05


def test_derived_marked_ours_with_formula():
    r = build(str(GOLDEN_REP))
    for d in r["kernels"][0]["derived"]:
        assert d["src"] == "ours"
        assert d["formula"] and d["sources"]
        assert d.get("desc"), f"{d['name']} missing one-line definition"
        assert d.get("group"), f"{d['name']} missing display group"


def test_derived_grouped_and_ordered():
    r = build(str(GOLDEN_REP))
    names = [d["name"] for d in r["kernels"][0]["derived"]]
    order = {"FMA FLOPS (SASS)": 0, "DRAM bytes moved": 1, "Arithmetic intensity": 2,
             "CTAs launched": 3, "Duration per CTA": 4, "PM sampler buffer": 5}
    pos = {n: i for i, n in enumerate(names) if n in order}
    assert list(pos) == sorted(pos, key=order.get), \
        "tiles must be grouped COMPUTE → MEMORY → ROOFLINE → OCCUPANCY → TIMING → PM SAMPLING"


def test_derived_zero_tensor_tile_on_fma_only_rep():
    rep = ROOT / "kernels" / "cuda" / "gpu" / "matmul_v1-ncu-report" / "matmul_v1.cu-1786854886.ncu-rep"
    if not rep.exists():
        return
    d = _derived(rep)
    tile = d["Tensor FLOPS (NVIDIA ops-path)"]
    assert tile["value"] == 0
    assert "No tensor-core work" in tile["note"]


def test_derived_raw_csv_metric_tiles_only():
    r = build(str(RAW_CSV))
    for k in r["kernels"]:
        names = [d["name"] for d in k["derived"]]
        assert "DRAM bytes moved" in names
        assert "CTAs launched" not in names, \
            "section-row-derived tiles must not appear without sections"
        for d in k["derived"]:
            assert d["src"] == "ours" and d["formula"] and d["sources"]


def test_golden_roofline():
    r = build(str(GOLDEN_REP))
    rl = r["kernels"][0]["roofline"]
    a = rl["achieved"]
    assert a["flops"] == 1099511722496
    assert abs(a["ai"] - 501.6) < 0.5
    assert abs(a["flops_s"] / 1e12 - 453.6) < 1.0
    assert len(a["fma"]) == 3
    assert sum(f["flops"] for f in a["fma"]) == 94720
    assert a["tensor"][0]["name"] == "src_fp16_dst_fp32"
    assert a["tensor"][0]["flops"] == 1099511627776
    assert abs(a["tensor"][0]["pct"] - 10.23) < 0.05
    env = rl["envelope"]
    assert abs(env["peak_dram_bw"] / 1e12 - 7.67) < 0.1
    assert abs(env["peak_fma_flops"] / 1e12 - 138.7) < 1.0
    assert env["peak_compute_source"] == "tensor"
    assert abs(env["peak_compute_flops"] / 1e15 - 4.43) < 0.05
    assert abs(env["ridge"] - 578.0) < 1.0
    assert len(rl["levels"]) == 3
    assert rl["levels"][0]["level"] == "L1 (SM↔L2)"
    assert rl["levels"][2]["level"] == "DRAM"
    names = [d["name"] for d in r["kernels"][0]["derived"]]
    assert "Tensor FLOPS (NVIDIA ops-path)" in names
    assert "Tensor instructions (utcmma (tcgen05))" in names
    for d in r["kernels"][0]["derived"]:
        if d["name"] in ("Warp-sampling period", "Warp samples collected"):
            assert d["value"]


def test_golden_memory_model():
    r = build(str(GOLDEN_REP))
    mem = r["kernels"][0]["memory"]
    units = {u["name"]: u for u in mem["units"]}
    assert units["Kernel"]["kind"] == "logical"
    assert units["L1/TEX Cache"]["kind"] == "physical"
    assert units["Global"]["active"] is True
    assert units["Local"]["active"] is False
    links = {(ln["from"], ln["to"], ln["kind"]): ln for ln in mem["links"]}
    gk = links[("Kernel", "Global", "inst")]
    assert gk["value"] == 2097152 and gk["unit"] == "inst"
    g1 = links[("Global", "L1/TEX Cache", "req")]
    assert g1["value"] == 2097152
    assert abs(g1["pct"] - 0.3197) < 0.001
    l12 = links[("L1/TEX Cache", "L2 Cache", "sectors")]
    assert l12["value"] == 1189590
    assert abs(l12["pct"] - 16.307) < 0.01
    d = links[("L2 Cache", "Device Memory", "bytes")]
    assert d["value"] == 2192105216
    assert abs(d["pct"] - 11.063) < 0.01
    assert links[("L2 Cache", "Peer Memory", "sectors")]["value"] == 0
    for ln in mem["links"]:
        assert ln["src"] == "ours" and ln["formula"] and ln["sources"]

    t = mem["tables"]
    assert set(t) == {"shared", "l1", "l2", "texops", "evict", "dram"}
    sh = {row["label"]: row["cells"] for row in t["shared"]["rows"]}
    assert sh["Loads"]["Instructions"]["value"] == 12288
    assert sh["Stores"]["Bank Conflicts"]["value"] == 143
    assert sh["Atomics"]["Requests"]["value"] == 8192
    assert sh["Total"]["Wavefronts"]["value"] == 43151
    l1 = {row["label"]: row["cells"] for row in t["l1"]["rows"]}
    assert l1["Global Stores"]["Sectors"]["value"] == 67108864
    assert abs(l1["Global Stores"]["Hit Rate"]["value"] - 98.2274) < 0.001
    assert l1["Global Stores"]["Sector Misses to L2"]["value"] == 1189590
    assert abs(l1["Total"]["Hit Rate"]["value"] - 98.23) < 0.05
    l2 = {row["label"]: row["cells"] for row in t["l2"]["rows"]}
    assert l2["L1/TEX Total"]["Sectors"]["value"] == 284965052
    assert l2["L2 Fabric Total"]["Sectors"]["value"] == 89159305
    assert l2["ECC Total"]["Sectors"]["value"] == 123666
    assert l2["GPU Total"]["Sectors"]["value"] == 373777023
    assert abs(l2["GPU Total"]["Hit Rate"]["value"] - 59.087) < 0.01
    ev = {row["label"]: row["cells"] for row in t["evict"]["rows"]}
    assert ev["Normal"]["Sectors"]["value"] == 373529415
    dm = {row["label"]: row["cells"] for row in t["dram"]["rows"]}
    assert dm["Reads"]["Sectors"]["value"] == 64291024
    assert dm["Total"]["Bytes"]["value"] == 2192105216
    assert abs(dm["Total"]["% Peak"]["value"] - 11.788) < 0.01


def test_counters_memory_model_honest():
    """Counters-only input: only the tables the raw counters support."""
    r = build(str(BLOG_JSON))
    mem = r["kernels"][0]["memory"]
    assert [u["name"] for u in mem["units"]] == ["Kernel", "Device Memory"]
    assert set(mem["tables"]) == {"dram"}
    for ln in mem["links"]:
        assert ln["from"] == "L2 Cache" and ln["to"] == "Device Memory"


def _metric_ref(path):
    r = build(str(path))
    return {f["sid"]: f for f in r["kernels"][0]["metric_ref"]}


def test_golden_metric_ref_families():
    fams = _metric_ref(GOLDEN_REP)
    assert list(fams) == ["launch", "occupancy", "device", "pcsamp",
                          "pcsamp-not-issued", "warpidsamp", "warpidsamp-not-issued",
                          "source", "evict"]
    for f in fams.values():
        assert f["title"] and f["guide"] and f["intro"]
        assert 0 <= f["present"] <= f["total"] == len(f["rows"])
        for r in f["rows"]:
            assert r["name"] and r["desc"]
            assert r["present"] == (("value" in r) or ("str" in r))
            assert not (r["present"] and "value" in r and "str" in r)


def test_golden_metric_ref_values():
    fams = _metric_ref(GOLDEN_REP)
    la = {r["name"]: r for r in fams["launch"]["rows"]}
    assert fams["launch"]["present"] == 55 and fams["launch"]["total"] == 58
    assert la["launch__grid_size"]["value"] == 2048
    assert la["launch__thread_count"]["value"] == 262144
    assert la["launch__waves_per_multiprocessor"]["value"] == 13.837837837837839
    assert not la["launch__execution_model"]["present"]
    assert la["launch__cluster_scheduling_policy"]["str"] == "PolicySpread"

    occ = {r["name"]: r for r in fams["occupancy"]["rows"]}
    assert fams["occupancy"]["present"] == 3
    assert occ["sm__maximum_warps_per_active_cycle_pct"]["value"] == 6.25
    assert occ["sm__maximum_warps_avg_per_active_cycle"]["value"] == 4.0

    dev = {r["name"]: r for r in fams["device"]["rows"]}
    assert dev["device__attribute_display_name"]["str"] == "NVIDIA B200"
    assert "device__attribute_l2s_count" not in dev, \
        "undocumented attributes are not listed (profile exports 164, guide documents 27)"

    pc = {r["name"]: r for r in fams["pcsamp"]["rows"]}
    assert fams["pcsamp"]["present"] == 17
    assert pc["smsp__pcsamp_warps_issue_stalled_long_scoreboard"]["value"] == 143856
    assert pc["smsp__pcsamp_warps_issue_stalled_barrier"]["value"] == 417
    assert pc["smsp__pcsamp_warps_issue_stalled_imc_miss"]["present"] is False  # not on B200
    assert fams["pcsamp-not-issued"]["present"] == 17

    assert fams["warpidsamp"]["present"] == 0
    assert fams["warpidsamp"]["note"]
    assert fams["warpidsamp-not-issued"]["present"] == 0

    src = {r["name"]: r for r in fams["source"]["rows"]}
    assert src["inst_executed"]["value"] == 35339164
    assert src["thread_inst_executed"]["value"] == 1129835392
    assert src["branch_inst_executed"]["present"] is False
    assert src["memory_l2_theoretical_sectors_global"]["value"] == 67108864

    ev = {r["name"]: r for r in fams["evict"]["rows"]}
    assert fams["evict"]["present"] == 7
    assert ev["smsp__sass_inst_executed_memdesc_explicit_evict_type"]["value"] == 0.0


def test_raw_csv_metric_ref_honest():
    """Raw-CSV input: families still emit, absent metrics stay honest."""
    fams = _metric_ref(RAW_CSV)
    assert list(fams) == ["launch", "occupancy", "device", "pcsamp",
                          "pcsamp-not-issued", "warpidsamp", "warpidsamp-not-issued",
                          "source", "evict"]
    for f in fams.values():
        for r in f["rows"]:
            assert r["desc"] and r["present"] == ("value" in r or "str" in r)
