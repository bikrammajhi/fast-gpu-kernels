"""Metric reference families — NVIDIA Profiling Guide section 2.4.

Metric names and descriptions are NVIDIA's own, transcribed from the Nsight
Compute Profiling Guide (https://docs.nvidia.com/nsight-compute/
ProfilingGuide/index.html). Values are read from the profile's own metrics
at build time; a metric the export does not carry renders '—'. Nothing here
is invented and no hardware specifications are assumed.
"""

FAMILIES = [
    {
        "sid": "launch",
        "title": "Launch Metrics",
        "guide": "2.4.2",
        "intro": "Launch configuration metrics. These metrics are collected at launch time and are "
                 "available with kernel replay, application replay, range replay, application range "
                 "replay, and command list replay.",
        "metrics": [
            ("launch__barrier_count", "Maximum number of barriers per grid"),
            ("launch__block_dim_x", "Maximum x-dimension of a block"),
            ("launch__block_dim_y", "Maximum y-dimension of a block"),
            ("launch__block_dim_z", "Maximum z-dimension of a block"),
            ("launch__block_size", "Number of threads per block"),
            ("launch__cluster_dim_x", "Maximum x-dimension of a cluster"),
            ("launch__cluster_dim_y", "Maximum y-dimension of a cluster"),
            ("launch__cluster_dim_z", "Maximum z-dimension of a cluster"),
            ("launch__cluster_max_active", "Maximum number of active clusters"),
            ("launch__cluster_max_potential_size", "Maximum number of blocks in a cluster"),
            ("launch__cluster_size", "Number of blocks per cluster"),
            ("launch__cluster_scheduling_policy", "Cluster scheduling policy"),
            ("launch__context_id", "Context ID"),
            ("launch__device_id", "Device ID"),
            ("launch__execution_model", "Kernel execution model"),
            ("launch__function_pcs", "Function PCs"),
            ("launch__grid_dim_x", "Maximum x-dimension of a grid"),
            ("launch__grid_dim_y", "Maximum y-dimension of a grid"),
            ("launch__grid_dim_z", "Maximum z-dimension of a grid"),
            ("launch__grid_size", "Maximum total number of blocks in a grid"),
            ("launch__kernel_name", "Kernel name"),
            ("launch__occupancy_cluster_gpu_pct", "Percentage of GPU clusters at which kernel can run with maximum cluster size"),
            ("launch__occupancy_cluster_pct", "Percentage of GPU clusters at which kernel can run with preferred cluster size"),
            ("launch__occupancy_limit_barriers", "Number of blocks per SM based on barrier limit"),
            ("launch__occupancy_limit_blocks", "Number of blocks per SM based on block limit"),
            ("launch__occupancy_limit_registers", "Number of blocks per SM based on register limit"),
            ("launch__occupancy_limit_shared_mem", "Number of blocks per SM based on shared memory limit"),
            ("launch__occupancy_limit_warps", "Number of blocks per SM based on warp limit"),
            ("launch__occupancy_per_barrier_count", "Instance value: Number of blocks per SM based on barrier limit"),
            ("launch__occupancy_per_block_size", "Instance value: Number of blocks per SM based on block size"),
            ("launch__occupancy_per_cluster_size", "Instance value: Number of blocks per SM based on cluster size"),
            ("launch__occupancy_per_register_count", "Instance value: Number of blocks per SM based on register limit"),
            ("launch__occupancy_per_shared_mem_size", "Instance value: Number of blocks per SM based on shared memory limit"),
            ("launch__persisting_l2_cache_size", "Size of persisting L2 cache"),
            ("launch__preferred_cluster_dim_x", "Preferred x-dimension of a cluster"),
            ("launch__preferred_cluster_dim_y", "Preferred y-dimension of a cluster"),
            ("launch__preferred_cluster_dim_z", "Preferred z-dimension of a cluster"),
            ("launch__preferred_cluster_size", "Preferred number of blocks in a cluster"),
            ("launch__registers_per_thread", "Number of registers per thread"),
            ("launch__registers_per_thread_allocated", "Number of registers per thread allocated"),
            ("launch__shared_mem_config_size", "Size of shared memory per block based on shared memory configuration"),
            ("launch__shared_mem_per_block", "Size of shared memory per block"),
            ("launch__shared_mem_per_block_allocated", "Size of shared memory per block allocated"),
            ("launch__shared_mem_per_block_driver", "Size of shared memory per block driver"),
            ("launch__shared_mem_per_block_dynamic", "Size of dynamic shared memory per block"),
            ("launch__shared_mem_per_block_static", "Size of static shared memory per block"),
            ("launch__sm_count", "Number of streaming multiprocessors (SMs)"),
            ("launch__stack_size", "Stack size"),
            ("launch__stream_id", "Stream ID"),
            ("launch__sub_launch_name", "Sub kernel name"),
            ("launch__thread_count", "Maximum total number of threads in a grid"),
            ("launch__tpc_count", "Number of TPCs (GPCs)"),
            ("launch__uses_cdp", "Uses CUDA Dynamic Parallelism"),
            ("launch__uses_green_context", "Uses green context"),
            ("launch__uses_mps", "Uses Multi-Process Service"),
            ("launch__uses_nvlink_centric_scheduling", "Uses NVLINK centric scheduling"),
            ("launch__uses_vgpu", "Uses virtual GPU"),
            ("launch__waves_per_multiprocessor", "Waves per multiprocessor"),
        ],
    },
    {
        "sid": "occupancy",
        "title": "Occupancy Metrics",
        "guide": "2.4.3",
        "intro": "Occupancy metrics provide information about the theoretical maximum occupancy of "
                 "the GPU. These metrics are collected at launch time and are available with kernel "
                 "replay, application replay, range replay, application range replay, and command "
                 "list replay.",
        "metrics": [
            ("sm__maximum_warps_avg_per_active_cycle", "The theoretical maximum number of warps resident on an SM"),
            ("sm__maximum_warps_per_active_cycle_pct", "Percentage of the theoretical maximum warps resident on an SM"),
            ("smsp__maximum_warps_avg_per_active_cycle", "The theoretical maximum number of warps resident on an SMSP"),
        ],
    },
    {
        "sid": "device",
        "title": "Device Attributes",
        "guide": "2.4.6",
        "intro": "Device attributes provide information about the GPU on which the application was "
                 "profiled. They are not associated with any particular kernel, and are collected at "
                 "the start of profiling. Available with kernel replay, application replay, range "
                 "replay, application range replay, and command list replay.",
        "note": "This profile exports more device__attribute_* metrics than the guide documents; "
                "the documented subset is listed below.",
        "metrics": [
            ("device__attribute_architecture", "Compute architecture"),
            ("device__attribute_can_flush_l2", "Device can flush L2 cache"),
            ("device__attribute_chip_name", "Chip name"),
            ("device__attribute_clock_rate", "Device clock rate"),
            ("device__attribute_compute_capability_major", "Compute capability major version"),
            ("device__attribute_compute_capability_minor", "Compute capability minor version"),
            ("device__attribute_device_index", "Device index"),
            ("device__attribute_display_name", "Device display name"),
            ("device__attribute_ecc_enabled", "ECC support enabled"),
            ("device__attribute_global_memory_bus_width", "Global memory bus width"),
            ("device__attribute_implementation", "Device implementation"),
            ("device__attribute_l2_cache_size", "L2 cache size"),
            ("device__attribute_max_blocks_per_multiprocessor", "Maximum number of blocks per SM"),
            ("device__attribute_max_clock_rate", "Maximum device clock rate"),
            ("device__attribute_max_ipc_per_multiprocessor", "Maximum IPC per SM"),
            ("device__attribute_max_warps_per_multiprocessor", "Maximum number of warps per SM"),
            ("device__attribute_memory_clock_rate", "Memory clock rate"),
            ("device__attribute_multiprocessor_count", "Number of SMs"),
            ("device__attribute_pci_bus_id", "PCI bus ID"),
            ("device__attribute_pci_device_id", "PCI device ID"),
            ("device__attribute_pci_domain_id", "PCI domain ID"),
            ("device__attribute_sysmem_l2_partition_size", "Size of L2 partition of sysmem"),
            ("device__attribute_total_memory", "Total memory size"),
            ("device__attribute_tpc_count", "Number of TPCs (GPCs)"),
            ("device__attribute_ufm_enabled", "Device UFM mode"),
            ("device__attribute_unified_addressing", "Device supports unified addressing"),
            ("device__attribute_unified_memory_support", "Device supports unified memory"),
        ],
    },
    {
        "sid": "pcsamp",
        "title": "Warp Stall Reasons",
        "guide": "2.4.7",
        "intro": "Warp stall reasons describe the reasons why a warp was not selected to issue an "
                 "instruction. Warp stall reasons are collected via sampling and require a sampling "
                 "interval to be specified. Available with kernel replay, application replay, range "
                 "replay, application range replay, and command list replay.",
        "metrics": [
            ("smsp__pcsamp_warps_issue_stalled_barrier", "Warp was stalled waiting at a barrier."),
            ("smsp__pcsamp_warps_issue_stalled_branch_resolving", "Warp was stalled waiting for a branch to be resolved."),
            ("smsp__pcsamp_warps_issue_stalled_dispatch_stall", "Warp was stalled waiting for a free warp slot to be available for warp launch."),
            ("smsp__pcsamp_warps_issue_stalled_drain", "Warp was stalled waiting for all dependent warps to complete."),
            ("smsp__pcsamp_warps_issue_stalled_imc_miss", "Warp was stalled waiting for an instruction memory cache (IMC) miss to resolve."),
            ("smsp__pcsamp_warps_issue_stalled_lg_throttle", "Warp was stalled waiting for the local/global (LG) throttle to clear."),
            ("smsp__pcsamp_warps_issue_stalled_long_scoreboard", "Warp was stalled waiting for a memory operation (e.g., load, store, atomic) to complete."),
            ("smsp__pcsamp_warps_issue_stalled_math_pipe_throttle", "Warp was stalled waiting for a math pipe to become available."),
            ("smsp__pcsamp_warps_issue_stalled_membar", "Warp was stalled waiting for a memory barrier to clear."),
            ("smsp__pcsamp_warps_issue_stalled_mio_throttle", "Warp was stalled waiting for a memory input/output (MIO) throttle to clear."),
            ("smsp__pcsamp_warps_issue_stalled_misc", "Warp was stalled for a reason not covered by other stall reasons."),
            ("smsp__pcsamp_warps_issue_stalled_no_instructions", "Warp was stalled because there were no instructions to issue."),
            ("smsp__pcsamp_warps_issue_stalled_not_selected", "Warp was eligible to issue, but was not selected."),
            ("smsp__pcsamp_warps_issue_stalled_selected", "Warp was selected to issue an instruction, but it was not eligible to issue."),
            ("smsp__pcsamp_warps_issue_stalled_short_scoreboard", "Warp was stalled waiting for a memory operation (e.g., load, store, atomic) to complete (short scoreboard)."),
            ("smsp__pcsamp_warps_issue_stalled_sleeping", "Warp was stalled waiting for a sleep instruction to complete."),
            ("smsp__pcsamp_warps_issue_stalled_tex_throttle", "Warp was stalled waiting for the texture (tex) throttle to clear."),
            ("smsp__pcsamp_warps_issue_stalled_wait", "Warp was stalled waiting for a fixed-latency execution dependency to clear."),
        ],
    },
    {
        "sid": "pcsamp-not-issued",
        "title": "Warp Stall Reasons (Not Issued)",
        "guide": "2.4.8",
        "intro": "Warp stall reasons (not issued) describe the reasons why a warp was not issued an "
                 "instruction. Warp stall reasons are collected via sampling and require a sampling "
                 "interval to be specified. Available with kernel replay, application replay, range "
                 "replay, application range replay, and command list replay.",
        "metrics": [
            ("smsp__pcsamp_warps_issue_stalled_barrier_not_issued", "Warp was not issued because it was stalled waiting at a barrier."),
            ("smsp__pcsamp_warps_issue_stalled_branch_resolving_not_issued", "Warp was not issued because it was stalled waiting for a branch to be resolved."),
            ("smsp__pcsamp_warps_issue_stalled_dispatch_stall_not_issued", "Warp was not issued because it was stalled waiting for a free warp slot to be available for warp launch."),
            ("smsp__pcsamp_warps_issue_stalled_drain_not_issued", "Warp was not issued because it was stalled waiting for all dependent warps to complete."),
            ("smsp__pcsamp_warps_issue_stalled_imc_miss_not_issued", "Warp was not issued because it was stalled waiting for an instruction memory cache (IMC) miss to resolve."),
            ("smsp__pcsamp_warps_issue_stalled_lg_throttle_not_issued", "Warp was not issued because it was stalled waiting for the local/global (LG) throttle to clear."),
            ("smsp__pcsamp_warps_issue_stalled_long_scoreboard_not_issued", "Warp was not issued because it was stalled waiting for a memory operation (e.g., load, store, atomic) to complete."),
            ("smsp__pcsamp_warps_issue_stalled_math_pipe_throttle_not_issued", "Warp was not issued because it was stalled waiting for a math pipe to become available."),
            ("smsp__pcsamp_warps_issue_stalled_membar_not_issued", "Warp was not issued because it was stalled waiting for a memory barrier to clear."),
            ("smsp__pcsamp_warps_issue_stalled_mio_throttle_not_issued", "Warp was not issued because it was stalled waiting for a memory input/output (MIO) throttle to clear."),
            ("smsp__pcsamp_warps_issue_stalled_misc_not_issued", "Warp was not issued because it was stalled for a reason not covered by other stall reasons."),
            ("smsp__pcsamp_warps_issue_stalled_no_instructions_not_issued", "Warp was not issued because there were no instructions to issue."),
            ("smsp__pcsamp_warps_issue_stalled_not_selected_not_issued", "Warp was eligible to issue, but was not selected."),
            ("smsp__pcsamp_warps_issue_stalled_selected_not_issued", "Warp was selected to issue an instruction, but it was not eligible to issue."),
            ("smsp__pcsamp_warps_issue_stalled_short_scoreboard_not_issued", "Warp was not issued because it was stalled waiting for a memory operation (e.g., load, store, atomic) to complete (short scoreboard)."),
            ("smsp__pcsamp_warps_issue_stalled_sleeping_not_issued", "Warp was not issued because it was stalled waiting for a sleep instruction to complete."),
            ("smsp__pcsamp_warps_issue_stalled_tex_throttle_not_issued", "Warp was not issued because it was stalled waiting for the texture (tex) throttle to clear."),
            ("smsp__pcsamp_warps_issue_stalled_wait_not_issued", "Warp was not issued because it was stalled waiting for a fixed-latency execution dependency to clear."),
        ],
    },
    {
        "sid": "warpidsamp",
        "title": "Warp Stalls per Warp ID",
        "guide": "2.4.9",
        "intro": "Warp stall reasons per warp ID describe the reasons why a warp was not selected to "
                 "issue an instruction, associated with the warp ID. These metrics are collected via "
                 "warp ID sampling and require a sampling interval to be specified. Available with "
                 "kernel replay, application replay, range replay, application range replay, and "
                 "command list replay.",
        "note0": "These per-warp-ID metrics are not available in this profile's ncu "
                 "(13.x): the warpidsamp metric family is absent from the metric "
                 "database on every tested architecture (A100, H100, B200). The "
                 "same stall reasons ARE collected via PC sampling — see the "
                 "Warp Stall Reasons card (guide 2.4.7).",
        "metrics": [
            ("smsp__warpidsamp_warps_issue_stalled_barrier", "Warp was stalled waiting at a barrier."),
            ("smsp__warpidsamp_warps_issue_stalled_branch_resolving", "Warp was stalled waiting for a branch to be resolved."),
            ("smsp__warpidsamp_warps_issue_stalled_dispatch_stall", "Warp was stalled waiting for a free warp slot to be available for warp launch."),
            ("smsp__warpidsamp_warps_issue_stalled_drain", "Warp was stalled waiting for all dependent warps to complete."),
            ("smsp__warpidsamp_warps_issue_stalled_imc_miss", "Warp was stalled waiting for an instruction memory cache (IMC) miss to resolve."),
            ("smsp__warpidsamp_warps_issue_stalled_lg_throttle", "Warp was stalled waiting for the local/global (LG) throttle to clear."),
            ("smsp__warpidsamp_warps_issue_stalled_long_scoreboard", "Warp was stalled waiting for a memory operation (e.g., load, store, atomic) to complete."),
            ("smsp__warpidsamp_warps_issue_stalled_math_pipe_throttle", "Warp was stalled waiting for a math pipe to become available."),
            ("smsp__warpidsamp_warps_issue_stalled_membar", "Warp was stalled waiting for a memory barrier to clear."),
            ("smsp__warpidsamp_warps_issue_stalled_mio_throttle", "Warp was stalled waiting for a memory input/output (MIO) throttle to clear."),
            ("smsp__warpidsamp_warps_issue_stalled_misc", "Warp was stalled for a reason not covered by other stall reasons."),
            ("smsp__warpidsamp_warps_issue_stalled_no_instructions", "Warp was stalled because there were no instructions to issue."),
            ("smsp__warpidsamp_warps_issue_stalled_not_selected", "Warp was eligible to issue, but was not selected."),
            ("smsp__warpidsamp_warps_issue_stalled_selected", "Warp was selected to issue an instruction, but it was not eligible to issue."),
            ("smsp__warpidsamp_warps_issue_stalled_short_scoreboard", "Warp was stalled waiting for a memory operation (e.g., load, store, atomic) to complete (short scoreboard)."),
            ("smsp__warpidsamp_warps_issue_stalled_sleeping", "Warp was stalled waiting for a sleep instruction to complete."),
            ("smsp__warpidsamp_warps_issue_stalled_tex_throttle", "Warp was stalled waiting for the texture (tex) throttle to clear."),
            ("smsp__warpidsamp_warps_issue_stalled_wait", "Warp was stalled waiting for a fixed-latency execution dependency to clear."),
        ],
    },
    {
        "sid": "warpidsamp-not-issued",
        "title": "Warp Stalls per Warp ID (Not Issued)",
        "guide": "2.4.10",
        "intro": "Warp stall reasons per warp ID (not issued) describe the reasons why a warp was not "
                 "issued an instruction, associated with the warp ID. These metrics are collected via "
                 "warp ID sampling and require a sampling interval to be specified. Available with "
                 "kernel replay, application replay, range replay, application range replay, and "
                 "command list replay.",
        "note0": "These per-warp-ID metrics are not available in this profile's ncu "
                 "(13.x): the warpidsamp metric family is absent from the metric "
                 "database on every tested architecture (A100, H100, B200). The "
                 "same stall reasons ARE collected via PC sampling — see the "
                 "Warp Stall Reasons (Not Issued) card (guide 2.4.8).",
        "metrics": [
            ("smsp__warpidsamp_warps_issue_stalled_barrier_not_issued", "Warp was not issued because it was stalled waiting at a barrier."),
            ("smsp__warpidsamp_warps_issue_stalled_branch_resolving_not_issued", "Warp was not issued because it was stalled waiting for a branch to be resolved."),
            ("smsp__warpidsamp_warps_issue_stalled_dispatch_stall_not_issued", "Warp was not issued because it was stalled waiting for a free warp slot to be available for warp launch."),
            ("smsp__warpidsamp_warps_issue_stalled_drain_not_issued", "Warp was not issued because it was stalled waiting for all dependent warps to complete."),
            ("smsp__warpidsamp_warps_issue_stalled_imc_miss_not_issued", "Warp was not issued because it was stalled waiting for an instruction memory cache (IMC) miss to resolve."),
            ("smsp__warpidsamp_warps_issue_stalled_lg_throttle_not_issued", "Warp was not issued because it was stalled waiting for the local/global (LG) throttle to clear."),
            ("smsp__warpidsamp_warps_issue_stalled_long_scoreboard_not_issued", "Warp was not issued because it was stalled waiting for a memory operation (e.g., load, store, atomic) to complete."),
            ("smsp__warpidsamp_warps_issue_stalled_math_pipe_throttle_not_issued", "Warp was not issued because it was stalled waiting for a math pipe to become available."),
            ("smsp__warpidsamp_warps_issue_stalled_membar_not_issued", "Warp was not issued because it was stalled waiting for a memory barrier to clear."),
            ("smsp__warpidsamp_warps_issue_stalled_mio_throttle_not_issued", "Warp was not issued because it was stalled waiting for a memory input/output (MIO) throttle to clear."),
            ("smsp__warpidsamp_warps_issue_stalled_misc_not_issued", "Warp was not issued because it was stalled for a reason not covered by other stall reasons."),
            ("smsp__warpidsamp_warps_issue_stalled_no_instructions_not_issued", "Warp was not issued because there were no instructions to issue."),
            ("smsp__warpidsamp_warps_issue_stalled_not_selected_not_issued", "Warp was eligible to issue, but was not selected."),
            ("smsp__warpidsamp_warps_issue_stalled_selected_not_issued", "Warp was selected to issue an instruction, but it was not eligible to issue."),
            ("smsp__warpidsamp_warps_issue_stalled_short_scoreboard_not_issued", "Warp was not issued because it was stalled waiting for a memory operation (e.g., load, store, atomic) to complete (short scoreboard)."),
            ("smsp__warpidsamp_warps_issue_stalled_sleeping_not_issued", "Warp was not issued because it was stalled waiting for a sleep instruction to complete."),
            ("smsp__warpidsamp_warps_issue_stalled_tex_throttle_not_issued", "Warp was not issued because it was stalled waiting for the texture (tex) throttle to clear."),
            ("smsp__warpidsamp_warps_issue_stalled_wait_not_issued", "Warp was not issued because it was stalled waiting for a fixed-latency execution dependency to clear."),
        ],
    },
    {
        "sid": "source",
        "title": "Source Metrics",
        "guide": "2.4.11",
        "intro": "Source metrics are calculated by Nsight Compute's instruction source profiling. "
                 "They are available with kernel replay, application replay, range replay, "
                 "application range replay, and command list replay, and are not available when "
                 "graph profiling is enabled.",
        "metrics": [
            ("branch_inst_executed", "Total number of executed branch instructions."),
            ("inst_executed", "Total number of executed instructions."),
            ("thread_inst_executed", "Total number of executed thread instructions."),
            ("derived__avg_thread_executed", "Average number of thread instructions executed per thread."),
            ("derived__avg_thread_executed_true", "True average number of thread instructions executed per thread."),
            ("derived__avg_thread_unexecuted_true", "True average number of thread instructions not executed per thread."),
            ("derived__memory_l1_conflicts_shared_nway", "Conflicts (per bank, 1-based) count the number of shared memory instructions that access the same bank. The value indicates the maximum level of conflicts among banks."),
            ("derived__memory_l1_wavefronts_shared_excessive", "Total number of excessive shared memory wavefronts."),
            ("derived__memory_l2_theoretical_sectors_global_excessive", "Theoretical number of excessive sectors accessed by all global memory operations."),
            ("derived__pct_occupancy_per_barrier_count", "Percentage of the theoretical maximum occupancy based on barrier count."),
            ("derived__pct_occupancy_per_block_size", "Percentage of the theoretical maximum occupancy based on block size."),
            ("derived__pct_occupancy_per_register_count", "Percentage of the theoretical maximum occupancy based on register count."),
            ("derived__pct_occupancy_per_shared_mem_size", "Percentage of the theoretical maximum occupancy based on shared memory size."),
            ("memory_l1_wavefronts_shared", "Total number of shared memory wavefronts."),
            ("memory_l1_wavefronts_shared_load", "Total number of shared memory load wavefronts."),
            ("memory_l1_wavefronts_shared_store", "Total number of shared memory store wavefronts."),
            ("memory_l2_theoretical_sectors_global", "Theoretical number of sectors accessed by all global memory operations. The theoretical sector count can be less than the actual sector count when memory operations access data with low utilization efficiency (e.g., when accessing 8B of a 32B sector)."),
            ("memory_l2_theoretical_sectors_global_load", "Theoretical number of sectors accessed by all global memory load operations."),
            ("memory_l2_theoretical_sectors_global_store", "Theoretical number of sectors accessed by all global memory store operations."),
            ("memory_l2_theoretical_sectors_local", "Theoretical number of sectors accessed by all local memory operations."),
            ("memory_l2_theoretical_sectors_local_load", "Theoretical number of sectors accessed by all local memory load operations."),
            ("memory_l2_theoretical_sectors_local_store", "Theoretical number of sectors accessed by all local memory store operations."),
            ("memory_l2_theoretical_sectors_global_atom", "Theoretical number of sectors accessed by all global memory atom operations."),
            ("memory_l2_theoretical_sectors_global_red", "Theoretical number of sectors accessed by all global memory reduction operations."),
            ("memory_l2_theoretical_sectors_global_atom_response", "Theoretical number of sectors accessed by all global memory atom operations returning data to the SM."),
            ("memory_l2_theoretical_sectors_global_red_response", "Theoretical number of sectors accessed by all global memory reduction operations returning data to the SM."),
        ],
    },
    {
        "sid": "evict",
        "title": "L2 Cache Eviction Metrics",
        "guide": "2.4.12",
        "intro": "L2 cache eviction metrics describe the executed memory descriptor instructions with "
                 "explicit eviction policies. These metrics are available with kernel replay, "
                 "application replay, range replay, application range replay, and command list replay.",
        "note": "The profile exports this family under the target architecture's counter names "
                "(hitprop evict normal demote and missprop policies) in the guide's wording pattern.",
        "metrics": [
            ("smsp__sass_inst_executed_memdesc_explicit_evict_type", "Executed explicit evict type memory descriptor instructions."),
            ("smsp__sass_inst_executed_memdesc_explicit_hitprop_evict_normal", "Executed explicit hitprop evict normal memory descriptor instructions."),
            ("smsp__sass_inst_executed_memdesc_explicit_hitprop_evict_normal_demote", "Executed explicit hitprop evict normal demote memory descriptor instructions."),
            ("smsp__sass_inst_executed_memdesc_explicit_hitprop_evict_first", "Executed explicit hitprop evict first memory descriptor instructions."),
            ("smsp__sass_inst_executed_memdesc_explicit_hitprop_evict_last", "Executed explicit hitprop evict last memory descriptor instructions."),
            ("smsp__sass_inst_executed_memdesc_explicit_missprop_evict_first", "Executed explicit missprop evict first memory descriptor instructions."),
            ("smsp__sass_inst_executed_memdesc_explicit_missprop_evict_normal", "Executed explicit missprop evict normal memory descriptor instructions."),
        ],
    },
]


def metric_ref(kp) -> list[dict]:
    """Per-kernel metric reference: guide families with profile values.

    Each family carries the NVIDIA name + description verbatim plus the
    profile's own value (numbers and strings). A metric the export does not
    carry has present=False and renders '—' — nothing is invented.
    """
    m, s = kp.metrics, kp.str_metrics
    fams = []
    for fam in FAMILIES:
        rows, present = [], 0
        for name, desc in fam["metrics"]:
            if name in s:
                rows.append({"name": name, "desc": desc, "present": True, "str": s[name]})
                present += 1
            elif name in m:
                rows.append({"name": name, "desc": desc, "present": True, "value": m[name]})
                present += 1
            else:
                rows.append({"name": name, "desc": desc, "present": False})
        note = fam.get("note0", "") if present == 0 else fam.get("note", "")
        fams.append(
            {
                "sid": fam["sid"],
                "title": fam["title"],
                "guide": fam["guide"],
                "intro": fam["intro"],
                "note": note,
                "present": present,
                "total": len(rows),
                "rows": rows,
            }
        )
    return fams
