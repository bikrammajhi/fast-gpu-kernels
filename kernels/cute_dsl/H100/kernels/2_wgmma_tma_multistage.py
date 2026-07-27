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
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils.hopper_helpers as sm90_utils
import cutlass.utils as utils


@dataclass
class GemmConfig:
    """Configuration for a single-CTA Hopper GEMM using TMA + WGMMA."""
    cta_tiler: Tuple[int, int, int] = (128, 128, 128)
    num_stages: int = 3  # multistage pipeline depth

    def __post_init__(self):
        self.BM, self.BN, self.BK = self.cta_tiler
        self.atom_layout_mnk = (2, 1, 1) if self.BM > 64 and self.BN > 64 else (1, 1, 1)
        self.mma_warp_groups = math.prod(self.atom_layout_mnk)
        self.threads_per_cta = self.mma_warp_groups * 128


class HopperGemm:
    """Single-CTA Hopper GEMM with multi-stage TMA+WGMMA pipeline."""

    def __init__(self, config: GemmConfig = None):
        self.config = config or GemmConfig()
        assert self.config.num_stages > 1, "Only multistage supported."

    def run(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> float:
        """Compile, launch, and benchmark the kernel."""
        K = A.shape[1]
        if K % self.config.BK != 0:
            raise ValueError(f"K ({K}) must be a multiple of BK ({self.config.BK})")

        a_cute = from_dlpack(A, assumed_align=16)
        b_cute = from_dlpack(B, assumed_align=16)
        c_cute = from_dlpack(C, assumed_align=16)

        compiled = cute.compile(self, a_cute, b_cute, c_cute)
        compiled(a_cute, b_cute, c_cute)

        M, N = A.shape[0], B.shape[0]
        time_us = testing.benchmark(
            compiled,
            kernel_arguments=testing.JitArguments(a_cute, b_cute, c_cute),
        )
        return time_us

    @cute.jit
    def __call__(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
        cfg = self.config

        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.acc_dtype = cutlass.Float32
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)

        # SMEM layouts for A/B (multistage)
        self.a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout=self.a_layout,
            mma_tiler_mnk=cfg.cta_tiler,
            a_dtype=self.a_dtype,
            num_stages=cfg.num_stages,
        )
        self.b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            b_layout=self.b_layout,
            mma_tiler_mnk=cfg.cta_tiler,
            b_dtype=self.b_dtype,
            num_stages=cfg.num_stages,
        )

        # TMA load atoms and tiled tensors
        tma_atom_a, tma_tensor_a = self._make_tma_load_atoms_and_tensors(
            a, self.a_smem_layout_staged, (cfg.BM, cfg.BK),
        )
        tma_atom_b, tma_tensor_b = self._make_tma_load_atoms_and_tensors(
            b, self.b_smem_layout_staged, (cfg.BN, cfg.BK),
        )

        # WGMMA descriptor
        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            cfg.atom_layout_mnk,
            tiler_mn=(64, cfg.BN),
        )

        @cute.struct
        class SharedStorage:
            # Pipeline barriers: 2x num_stages for ping-pong
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, cfg.num_stages * 2
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged)],
                1024,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged)],
                1024,
            ]

        self.shared_storage = SharedStorage

        grid_dim = (*cute.ceil_div(c.shape, (cfg.BM, cfg.BN)), 1)
        block_dim = (cfg.threads_per_cta, 1, 1)

        self.kernel(
            tma_atom_a, tma_tensor_a,
            tma_atom_b, tma_tensor_b,
            self.tiled_mma,
            c,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
        ).launch(grid=grid_dim, block=block_dim)

    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_tma_tensor: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_tma_tensor: cute.Tensor,
        tiled_mma: cute.TiledMma,
        mC: cute.Tensor,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
    ):
        cfg = self.config

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)

        bidx, bidy, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        tile_coord_mnk = (bidx, bidy, None)

        gA = cute.local_tile(mA_tma_tensor, cfg.cta_tiler, tile_coord_mnk, (1, None, 1))
        gB = cute.local_tile(mB_tma_tensor, cfg.cta_tiler, tile_coord_mnk, (None, 1, 1))
        gC = cute.local_tile(mC,            cfg.cta_tiler, tile_coord_mnk, (1, 1, None))

        # SMEM allocation
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        mainloop_mbar_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)

        # TMA load partitions
        a_cta_layout = cute.make_layout(cute.slice_(cute.make_layout((1, 1, 1)), (0, None, 0)).shape)
        sA_tma, gA_tma = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a, 0, a_cta_layout, cute.group_modes(sA, 0, 2), cute.group_modes(gA, 0, 2),
        )

        b_cta_layout = cute.make_layout(cute.slice_(cute.make_layout((1, 1, 1)), (None, 0, 0)).shape)
        sB_tma, gB_tma = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b, 0, b_cta_layout, cute.group_modes(sB, 0, 2), cute.group_modes(gB, 0, 2),
        )

        tma_transaction_bytes = (
            cute.size_in_bytes(self.a_dtype, a_smem_layout)
            + cute.size_in_bytes(self.b_dtype, b_smem_layout)
        )

        # Pipeline participants
        num_warps = cfg.threads_per_cta // 32
        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_warps
        )

        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=mainloop_mbar_ptr,
            num_stages=cfg.num_stages,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_transaction_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            defer_sync=True,
        )

        # Barrier init: arrive from all CTAs, then wait
        pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)

        # MMA thread partitioning
        warp_group_idx = cute.arch.make_warp_uniform(tidx // 128)
        warp_group_thread_layout = cute.make_layout(cfg.mma_warp_groups, stride=128)
        thr_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx))

        tCgC = thr_mma.partition_C(gC)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)

        accumulators = cute.make_rmem_tensor(tCgC.shape, self.acc_dtype)
        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)

        pipeline_init_wait(cluster_shape_mn=(1, 1))

        # Prefetch: fill pipeline before main loop
        k_tile_cnt = cute.size(gA, mode=[2])
        prefetch_k_tile_cnt = cutlass.max(cutlass.min(cfg.num_stages, k_tile_cnt), 0)

        mainloop_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, cfg.num_stages
        )
        if warp_idx == 0:
            for prefetch_idx in cutlass.range(prefetch_k_tile_cnt, unroll=1):
                mainloop_pipeline.producer_acquire(mainloop_producer_state)

                tAgA_k = gA_tma[(None, mainloop_producer_state.count)]
                tAsA_pipe = sA_tma[(None, mainloop_producer_state.index)]
                tBgB_k = gB_tma[(None, mainloop_producer_state.count)]
                tBsB_pipe = sB_tma[(None, mainloop_producer_state.index)]

                cute.copy(
                    tma_atom_a, tAgA_k, tAsA_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_producer_state),
                )
                cute.copy(
                    tma_atom_b, tBgB_k, tBsB_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_producer_state),
                )
                mainloop_pipeline.producer_commit(mainloop_producer_state)
                mainloop_producer_state.advance()

        # Prologue: warm up MMA pipeline
        k_pipe_mmas = 1
        mainloop_consumer_read_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, cfg.num_stages
        )
        mainloop_consumer_release_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, cfg.num_stages
        )

        peek_ab_full_status = cutlass.Boolean(1)
        if mainloop_consumer_read_state.count < k_tile_cnt:
            peek_ab_full_status = mainloop_pipeline.consumer_try_wait(mainloop_consumer_read_state)

        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
        num_k_blocks = cute.size(tCrA, mode=[2])
        for k_tile in cutlass.range_constexpr(k_pipe_mmas):
            mainloop_pipeline.consumer_wait(mainloop_consumer_read_state, peek_ab_full_status)

            cute.nvgpu.warpgroup.fence()
            for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                k_block_coord = (None, None, k_block_idx, mainloop_consumer_read_state.index)
                cute.gemm(
                    tiled_mma, accumulators,
                    tCrA[k_block_coord], tCrB[k_block_coord], accumulators,
                )
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

            cute.nvgpu.warpgroup.commit_group()
            mainloop_consumer_read_state.advance()
            peek_ab_full_status = cutlass.Boolean(1)
            if mainloop_consumer_read_state.count < k_tile_cnt:
                peek_ab_full_status = mainloop_pipeline.consumer_try_wait(mainloop_consumer_read_state)

        # Main loop: overlap WGMMA with TMA loads
        for k_tile in cutlass.range(k_pipe_mmas, k_tile_cnt, 1, unroll=1):
            mainloop_pipeline.consumer_wait(mainloop_consumer_read_state, peek_ab_full_status)

            cute.nvgpu.warpgroup.fence()
            for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                k_block_coord = (None, None, k_block_idx, mainloop_consumer_read_state.index)
                cute.gemm(
                    tiled_mma, accumulators,
                    tCrA[k_block_coord], tCrB[k_block_coord], accumulators,
                )

            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(k_pipe_mmas)

            mainloop_pipeline.consumer_release(mainloop_consumer_release_state)
            mainloop_consumer_read_state.advance()
            mainloop_consumer_release_state.advance()

            peek_ab_full_status = cutlass.Boolean(1)
            if mainloop_consumer_read_state.count < k_tile_cnt:
                peek_ab_full_status = mainloop_pipeline.consumer_try_wait(mainloop_consumer_read_state)

            # TMA load next stage
            if warp_idx == 0 and mainloop_producer_state.count < k_tile_cnt:
                mainloop_pipeline.producer_acquire(mainloop_producer_state)

                tAgA_k = gA_tma[(None, mainloop_producer_state.count)]
                tAsA_pipe = sA_tma[(None, mainloop_producer_state.index)]
                tBgB_k = gB_tma[(None, mainloop_producer_state.count)]
                tBsB_pipe = sB_tma[(None, mainloop_producer_state.index)]

                cute.copy(
                    tma_atom_a, tAgA_k, tAsA_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_producer_state),
                )
                cute.copy(
                    tma_atom_b, tBgB_k, tBsB_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_producer_state),
                )
                mainloop_pipeline.producer_commit(mainloop_producer_state)
                mainloop_producer_state.advance()

        cute.nvgpu.warpgroup.wait_group(0)
        cute.arch.sync_threads()

        self._store_tile(accumulators, tiled_mma, tidx, mC, tile_coord_mnk)

    @cute.jit
    def _store_tile(self, accumulators, tiled_mma, tidx, c_local, tile_coord_mnk):
        """Store accumulators to GMEM via rmem -> smem -> TMA."""
        gC = cute.local_tile(c_local, self.config.cta_tiler, tile_coord_mnk, (1, 1, None))
        thr_mma_store = tiled_mma.get_slice(tidx)
        tCgC_store = thr_mma_store.partition_C(gC)

        atom_universal = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            c_local.element_type,
        )

        tCrC_out = cute.make_fragment_like(accumulators, dtype=cutlass.Float16)
        for reg_idx in range(cute.size(tCrC_out)):
            tCrC_out[reg_idx] = cutlass.Float16(accumulators[reg_idx])

        cute.copy(atom=atom_universal, src=tCrC_out, dst=tCgC_store)

    @cute.jit
    def _make_tma_load_atoms_and_tensors(self, tensor, smem_layout_staged, smem_tile: tuple[int, int]):
        """Build TMA atom and tiled tensor for GMEM -> SMEM loads."""
        op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        return cute.nvgpu.cpasync.make_tiled_tma_atom(op, tensor, smem_layout, smem_tile)


def main():
    M, N, K = 4096*2, 4096*2, 4096*2

    A = torch.randn((M, K), device="cuda", dtype=torch.float16)
    B = torch.randn((N, K), device="cuda", dtype=torch.float16)
    C = torch.empty((M, N), device="cuda", dtype=torch.float16)

    config = GemmConfig(cta_tiler=(128, 128, 128))
    gemm = HopperGemm(config)

    time_us = gemm.run(A, B, C)

    ref = torch.matmul(A, B.T)
    assert torch.allclose(C, ref, atol=5e1, rtol=5e-1), "CORRECTNESS FAILED"
    print("CORRECTNESS PASS")
    tflops = (2 * M * N * K) / (time_us * 1e6)
    print(f"DURATION: {time_us:>5.4f} µs\nTFLOPS: {tflops:>5.4f}")


if __name__ == "__main__":
    main()
