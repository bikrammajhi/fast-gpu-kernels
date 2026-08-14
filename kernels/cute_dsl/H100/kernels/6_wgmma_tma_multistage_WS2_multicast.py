import os
os.environ.setdefault("CUTE_DSL_ARCH", "sm_90a")

import torch
import math
from typing import Tuple
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
from cutlass.cute.runtime import from_dlpack
import cutlass.pipeline as pipeline
from cutlass.pipeline import PipelineTmaAsync, CooperativeGroup, Agent, make_pipeline_state, PipelineUserType
import cutlass.utils.hopper_helpers as sm90_utils
import cutlass.utils as utils

@dataclass
class GemmConfig:
    """Configuration for a single-CTA Hopper GEMM with warp-specialized TMA + WGMMA."""
    cta_tiler: Tuple[int, int, int] = (128, 256, 64)
    num_stages: int = 4
    cluster_shape_mn: Tuple[int, int] = (2, 2)

    def __post_init__(self):
        self.BM, self.BN, self.BK = self.cta_tiler
        self.atom_layout_mnk = (2, 1, 1) if self.BM > 64 and self.BN > 64 else (1, 1, 1)
        self.mma_warp_groups = math.prod(self.atom_layout_mnk)
        self.threads_per_warp_group = 128
        self.num_mma_warps = self.mma_warp_groups * 4
        self.mma_warp_ids = tuple(range(self.num_mma_warps))
        self.tma_warp_id = self.num_mma_warps
        self.threads_per_cta = (self.num_mma_warps + 1) * 32
        
        self.num_mcast_ctas_a = self.cluster_shape_mn[1]
        self.num_mcast_ctas_b = self.cluster_shape_mn[0]
        self.mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1


class HopperGemmWarpSpecialized:
    def __init__(self, config: GemmConfig = None):
        self.config = config or GemmConfig()
        assert self.config.num_stages > 1, "Only multistage supported."
        assert self.config.cluster_shape_mn in ((1, 1), (2, 1), (2, 2)), "Only cluster_shape_mn (1, 1), (2, 1), and (2, 2) are supported."

    def run(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> float:
        K = A.shape[1]
        if K % self.config.BK != 0:
            raise ValueError(f"K ({K}) must be a multiple of BK ({self.config.BK})")

        a_cute = from_dlpack(A, assumed_align=16)
        b_cute = from_dlpack(B, assumed_align=16)
        c_cute = from_dlpack(C, assumed_align=16)

        compiled = cute.compile(self, a_cute, b_cute, c_cute)
        compiled(a_cute, b_cute, c_cute)
        return 0.0

    @cute.jit
    def __call__(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
        cfg = self.config

        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.acc_dtype = cutlass.Float32
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        self.a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout=self.a_layout, mma_tiler_mnk=cfg.cta_tiler, a_dtype=self.a_dtype, num_stages=cfg.num_stages
        )
        self.b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            b_layout=self.b_layout, mma_tiler_mnk=cfg.cta_tiler, b_dtype=self.b_dtype, num_stages=cfg.num_stages
        )

        num_mcast_ctas_a = cfg.cluster_shape_mn[1]
        num_mcast_ctas_b = cfg.cluster_shape_mn[0]

        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            a, self.a_smem_layout_staged, (cfg.BM, cfg.BK), num_mcast_ctas_a
        )
        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            b, self.b_smem_layout_staged, (cfg.BN, cfg.BK), num_mcast_ctas_b
        )

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, cfg.num_stages * 2]
            sA: cute.struct.Align[cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged)], 1024]
            sB: cute.struct.Align[cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged)], 1024]

        self.shared_storage = SharedStorage

        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype, self.b_dtype, self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(), self.acc_dtype, cfg.atom_layout_mnk, tiler_mn=(64, cfg.BN)
        )

        grid_m = cute.ceil_div(c.shape[0], cfg.BM * cfg.cluster_shape_mn[0]) * cfg.cluster_shape_mn[0]
        grid_n = cute.ceil_div(c.shape[1], cfg.BN * cfg.cluster_shape_mn[1]) * cfg.cluster_shape_mn[1]
        grid_dim = (grid_m, grid_n, 1)
        block_dim = (cfg.threads_per_cta, 1, 1)

        self.kernel(
            tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b,
            self.tiled_mma, c, self.a_smem_layout_staged, self.b_smem_layout_staged,
        ).launch(grid=grid_dim, block=block_dim, cluster=(*cfg.cluster_shape_mn, 1))

    @cute.jit
    def _make_tma_atoms_and_tensors(self, tensor, smem_layout_staged, smem_tile: tuple[int, int], mcast_dim: int):
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        op = (
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
        )
        return cute.nvgpu.cpasync.make_tiled_tma_atom(op, tensor, smem_layout, smem_tile, num_multicast=mcast_dim)

    @cute.kernel
    def kernel(
        self, tma_atom_a: cute.CopyAtom, mA_tma_tensor: cute.Tensor,
        tma_atom_b: cute.CopyAtom, mB_tma_tensor: cute.Tensor,
        tiled_mma: cute.TiledMma, mC: cute.Tensor,
        a_smem_layout_staged: cute.ComposedLayout, b_smem_layout_staged: cute.ComposedLayout,
    ):
        cfg = self.config

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        
        tidx, _, _ = cute.arch.thread_idx()
        cidx, cidy, _ = cute.arch.cluster_idx()
        cdimx, cdimy, _ = cute.arch.cluster_dim()
        cluster_id = cidx + cdimx * cidy

        # CTA Swizzle to promote L2 data reuse
        group_size_m = 8
        s_shape = ((group_size_m, cdimx // group_size_m), cdimy)
        s_stride = ((1, cdimy * group_size_m), group_size_m)
        s_layout = cute.make_layout(s_shape, stride=s_stride)
        num_reg_cids = cute.size(s_shape)
        
        cid_m, cid_n = s_layout.get_flat_coord(cluster_id % num_reg_cids)

        if cluster_id >= num_reg_cids:
            tail_size_m = cdimx % group_size_m
            tail_layout = cute.make_layout((tail_size_m, cdimy), stride=(1, tail_size_m))
            tail_cid = cluster_id - num_reg_cids
            tail_cid_m, tail_cid_n = tail_layout.get_flat_coord(tail_cid)
            cid_m = cute.size(s_shape, mode=[0]) + tail_cid_m
            cid_n = tail_cid_n

        bidx_in_cluster = cute.arch.block_in_cluster_idx()
        pid_m = cid_m * cfg.cluster_shape_mn[0] + bidx_in_cluster[0]
        pid_n = cid_n * cfg.cluster_shape_mn[1] + bidx_in_cluster[1]

        tile_coord_mnk = (pid_m, pid_n, None)

        is_tma_warp = warp_idx == cfg.tma_warp_id
        is_mma_warp = warp_idx <= cfg.mma_warp_ids[-1]

        if is_tma_warp:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        mainloop_mbar_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

        sA = storage.sA.get_tensor(layout=a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(layout=b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)

        gA = cute.local_tile(mA_tma_tensor, cfg.cta_tiler, tile_coord_mnk, (1, None, 1))
        gB = cute.local_tile(mB_tma_tensor, cfg.cta_tiler, tile_coord_mnk, (None, 1, 1))
        gC = cute.local_tile(mC, cfg.cta_tiler, tile_coord_mnk, (1, 1, None))

        cta_layout_mnk = cute.make_layout((cfg.cluster_shape_mn[0], cfg.cluster_shape_mn[1], 1))
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        a_mcast_mask = cute.make_layout_image_mask(cta_layout_mnk, cluster_coord_mnk, mode=1)
        b_mcast_mask = cute.make_layout_image_mask(cta_layout_mnk, cluster_coord_mnk, mode=0)

        is_a_mcast = cfg.cluster_shape_mn[1] > 1
        is_b_mcast = cfg.cluster_shape_mn[0] > 1
        
        a_mcast_mask = a_mcast_mask if is_a_mcast else 0
        b_mcast_mask = b_mcast_mask if is_b_mcast else 0

        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        sA_for_tma = cute.group_modes(sA, 0, 2)
        gA_for_tma = cute.group_modes(gA, 0, 2)
        sA_tma, gA_tma = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a, a_cta_crd, a_cta_layout, sA_for_tma, gA_for_tma
        )

        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        sB_for_tma = cute.group_modes(sB, 0, 2)
        gB_for_tma = cute.group_modes(gB, 0, 2)
        sB_tma, gB_tma = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b, b_cta_crd, b_cta_layout, sB_for_tma, gB_for_tma
        )

        tma_transaction_bytes = cute.size_in_bytes(self.a_dtype, a_smem_layout) + cute.size_in_bytes(self.b_dtype, b_smem_layout)

        mainloop_pipeline_producer_group = CooperativeGroup(Agent.Thread)
        consumer_arrive_cnt = cfg.mcast_size * cfg.num_mma_warps
        mainloop_pipeline_consumer_group = CooperativeGroup(Agent.Thread, consumer_arrive_cnt)

        cta_layout_vmnk = cute.make_layout((1, cfg.cluster_shape_mn[0], cfg.cluster_shape_mn[1], 1))
        mainloop_pipeline = PipelineTmaAsync.create(
            barrier_storage=mainloop_mbar_ptr,
            num_stages=cfg.num_stages,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_transaction_bytes,
            cta_layout_vmnk=cta_layout_vmnk,
        )

        is_clustered = cfg.cluster_shape_mn[0] > 1 or cfg.cluster_shape_mn[1] > 1
        if is_clustered:
            cute.arch.cluster_arrive_relaxed()

        producer_state = make_pipeline_state(PipelineUserType.Producer, cfg.num_stages)
        consumer_state = make_pipeline_state(PipelineUserType.Consumer, cfg.num_stages)

        k_tile_cnt = mA_tma_tensor.shape[1] // cfg.BK
        num_k_blocks = cfg.BK // tiled_mma.shape_mnk[2]

        if is_clustered:
            cute.arch.cluster_wait()
        else:
            cute.arch.sync_threads()

        # =========================================================================
        # FIX: MMA Partitioning and Fragment Allocation (MUST BE BEFORE CONTROL FLOW)
        # =========================================================================
        warp_group_idx_raw = tidx // cfg.threads_per_warp_group
        warp_group_idx = cute.arch.make_warp_uniform(cutlass.min(warp_group_idx_raw, cfg.mma_warp_groups - 1))
        warp_group_thread_layout = cute.make_layout(cfg.mma_warp_groups, stride=cfg.threads_per_warp_group)
        thr_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx))

        tCgC = thr_mma.partition_C(gC)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        
        # These MUST be defined here so the compiler sees them before the `if is_mma_warp:` block
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        accumulators = cute.make_rmem_tensor(tCgC.shape, self.acc_dtype)

        # =========================================================================
        # Strict Warp-Specialized Loops
        # =========================================================================
        
        # 1. TMA Thread: Single contiguous producer loop
        if is_tma_warp:
            for k_tile in range(k_tile_cnt):
                mainloop_pipeline.producer_acquire(producer_state)
                
                tAgA_k = gA_tma[(None, producer_state.count)]
                tAsA_pipe = sA_tma[(None, producer_state.index)]
                tBgB_k = gB_tma[(None, producer_state.count)]
                tBsB_pipe = sB_tma[(None, producer_state.index)]

                cute.copy(tma_atom_a, tAgA_k, tAsA_pipe, tma_bar_ptr=mainloop_pipeline.producer_get_barrier(producer_state), mcast_mask=a_mcast_mask)
                cute.copy(tma_atom_b, tBgB_k, tBsB_pipe, tma_bar_ptr=mainloop_pipeline.producer_get_barrier(producer_state), mcast_mask=b_mcast_mask)
                
                mainloop_pipeline.producer_commit(producer_state)
                producer_state.advance()

        # 2. MMA Warps: Single contiguous consumer loop
        if is_mma_warp:
            tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
            
            for k_tile in range(k_tile_cnt):
                mainloop_pipeline.consumer_wait(consumer_state)
                
                cute.nvgpu.warpgroup.fence()
                for k_block_idx in range(num_k_blocks, unroll_full=True):
                    k_block_coord = (None, None, k_block_idx, consumer_state.index)
                    tCrA_1phase = tCrA[k_block_coord]
                    tCrB_1phase = tCrB[k_block_coord]
                    
                    cute.gemm(tiled_mma, accumulators, tCrA_1phase, tCrB_1phase, accumulators)
                    tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                
                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(0)
                
                mainloop_pipeline.consumer_release(consumer_state)
                consumer_state.advance()
                
                if k_tile < k_tile_cnt - 1:
                    mainloop_pipeline.consumer_try_wait(consumer_state)

        cute.nvgpu.warpgroup.wait_group(0)
        
        if is_clustered:
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
        else:
            cute.arch.sync_threads()

        # 4. EPILOGUE
        thr_mma_store = tiled_mma.get_slice(tidx)
        tCgC_store = thr_mma_store.partition_C(gC)
        atom_universal = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
        tCrC_out = cute.make_fragment_like(accumulators, dtype=self.c_dtype)

        for reg_idx in range(cute.size(tCrC_out)):
            tCrC_out[reg_idx] = self.c_dtype(accumulators[reg_idx])
        cute.copy(atom=atom_universal, src=tCrC_out, dst=tCgC_store)


def main():
    M, N, K = 4096*2, 4096*2, 4096*2
    
    warmup_iterations = 5
    iterations = 20
    use_cold_l2 = True

    A = torch.randn((M, K), device="cuda", dtype=torch.float16)
    B = torch.randn((N, K), device="cuda", dtype=torch.float16)
    C = torch.empty((M, N), device="cuda", dtype=torch.float16)

    config = GemmConfig(cluster_shape_mn=(2, 1))    # (2,1) performs best
    gemm = HopperGemmWarpSpecialized(config)

    a_cute = from_dlpack(A, assumed_align=16)
    b_cute = from_dlpack(B, assumed_align=16)
    c_cute = from_dlpack(C, assumed_align=16)
    
    compiled = cute.compile(gemm, a_cute, b_cute, c_cute)
    compiled(a_cute, b_cute, c_cute)
    
    ref = torch.matmul(A, B.T)
    assert torch.allclose(C, ref, atol=5e1, rtol=5e-1), "CORRECTNESS FAILED"
    print("CORRECTNESS PASS")

    def generate_tensors():
        a_w = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b_w = torch.randn((N, K), device="cuda", dtype=torch.float16)
        c_w = torch.empty((M, N), device="cuda", dtype=torch.float16)
        return testing.JitArguments(
            from_dlpack(a_w, assumed_align=16),
            from_dlpack(b_w, assumed_align=16),
            from_dlpack(c_w, assumed_align=16)
        )

    workspace_count = 1
    if use_cold_l2:
        one_workspace_bytes = (A.numel() * A.element_size() + B.numel() * B.element_size() + C.numel() * C.element_size())
        workspace_count = testing.get_workspace_count(one_workspace_bytes, warmup_iterations, iterations)

    exec_time = testing.benchmark(
        compiled,
        workspace_generator=generate_tensors,
        workspace_count=workspace_count,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )

    tflops = (2 * M * N * K) / (exec_time * 1e6)
    print(f"DURATION: {exec_time:>5.4f} µs\nTFLOPS: {tflops:>5.4f}")


if __name__ == "__main__":
    main()