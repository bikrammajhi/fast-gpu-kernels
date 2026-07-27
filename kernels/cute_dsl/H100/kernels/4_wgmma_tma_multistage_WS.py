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
    cta_tiler: Tuple[int, int, int] = (128, 128, 128)
    num_stages: int = 3

    def __post_init__(self):
        self.BM, self.BN, self.BK = self.cta_tiler
        self.atom_layout_mnk = (2, 1, 1) if self.BM > 64 and self.BN > 64 else (1, 1, 1)
        self.mma_warp_groups = math.prod(self.atom_layout_mnk)
        self.threads_per_warp_group = 128
        self.num_mma_warp_groups = self.mma_warp_groups
        self.num_mma_warps = self.num_mma_warp_groups * 4
        self.mma_warp_ids = tuple(range(self.num_mma_warps))
        self.tma_warp_id = self.num_mma_warps
        self.threads_per_cta = (self.num_mma_warps + 1) * 32


class HopperGemmWarpSpecialized:
    """Single-CTA Hopper GEMM with warp-specialized TMA producer / MMA consumer pipeline."""

    def __init__(self, config: GemmConfig = None):
        self.config = config or GemmConfig()
        assert self.config.num_stages > 1, "Only multistage supported."

    def run(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> float:
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
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        self.a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout=self.a_layout, mma_tiler_mnk=cfg.cta_tiler, a_dtype=self.a_dtype, num_stages=cfg.num_stages
        )
        self.b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            b_layout=self.b_layout, mma_tiler_mnk=cfg.cta_tiler, b_dtype=self.b_dtype, num_stages=cfg.num_stages
        )

        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(a, self.a_smem_layout_staged, (cfg.BM, cfg.BK))
        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(b, self.b_smem_layout_staged, (cfg.BN, cfg.BK))

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

        grid_dim = (*cute.ceil_div(c.shape, (cfg.BM, cfg.BN)), 1)
        block_dim = (cfg.threads_per_cta, 1, 1)

        self.kernel(
            tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b,
            self.tiled_mma, c, self.a_smem_layout_staged, self.b_smem_layout_staged,
        ).launch(grid=grid_dim, block=block_dim)

    @cute.jit
    def _make_tma_atoms_and_tensors(self, tensor, smem_layout_staged, smem_tile: tuple[int, int]):
        op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        return cute.nvgpu.cpasync.make_tiled_tma_atom(op, tensor, smem_layout, smem_tile)

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
        bidx, bidy, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

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

        tile_coord_mnk = (bidx, bidy, None)
        gA = cute.local_tile(mA_tma_tensor, cfg.cta_tiler, tile_coord_mnk, (1, None, 1))
        gB = cute.local_tile(mB_tma_tensor, cfg.cta_tiler, tile_coord_mnk, (None, 1, 1))
        gC = cute.local_tile(mC, cfg.cta_tiler, tile_coord_mnk, (1, 1, None))

        sA_tma, gA_tma = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a, 0, cute.make_layout(1), cute.group_modes(sA, 0, 2), cute.group_modes(gA, 0, 2)
        )
        sB_tma, gB_tma = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b, 0, cute.make_layout(1), cute.group_modes(sB, 0, 2), cute.group_modes(gB, 0, 2)
        )

        tma_transaction_bytes = cute.size_in_bytes(self.a_dtype, a_smem_layout) + cute.size_in_bytes(self.b_dtype, b_smem_layout)

        mainloop_pipeline = PipelineTmaAsync.create(
            num_stages=cfg.num_stages,
            producer_group=CooperativeGroup(Agent.Thread, 1),
            consumer_group=CooperativeGroup(Agent.Thread, cfg.num_mma_warps),
            barrier_storage=mainloop_mbar_ptr,
            tx_count=tma_transaction_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1))
        )

        producer_state = make_pipeline_state(PipelineUserType.Producer, cfg.num_stages)
        consumer_state = make_pipeline_state(PipelineUserType.Consumer, cfg.num_stages)

        k_tile_cnt = mA_tma_tensor.shape[1] // cfg.BK
        num_k_blocks = cfg.BK // tiled_mma.shape_mnk[2]

        if is_tma_warp:
            for kidx in range(k_tile_cnt):
                mainloop_pipeline.producer_acquire(producer_state)
                cute.copy(tma_atom_a, gA_tma[None, producer_state.count], sA_tma[None, producer_state.index], tma_bar_ptr=mainloop_pipeline.producer_get_barrier(producer_state))
                cute.copy(tma_atom_b, gB_tma[None, producer_state.count], sB_tma[None, producer_state.index], tma_bar_ptr=mainloop_pipeline.producer_get_barrier(producer_state))
                mainloop_pipeline.producer_commit(producer_state)
                producer_state.advance()

        # MMA partitioning for ALL threads (so accumulators are available for epilogue)
        warp_group_idx_raw = tidx // cfg.threads_per_warp_group
        warp_group_idx = cute.arch.make_warp_uniform(
            cutlass.min(warp_group_idx_raw, cfg.num_mma_warp_groups - 1)
        )
        warp_group_thread_layout = cute.make_layout(cfg.mma_warp_groups, stride=cfg.threads_per_warp_group)
        thr_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx))

        tCgC = thr_mma.partition_C(gC)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)

        accumulators = cute.make_rmem_tensor(tCgC.shape, self.acc_dtype)

        if is_mma_warp:
            tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)

            for kidx in range(k_tile_cnt):
                mainloop_pipeline.consumer_wait(consumer_state)
                cute.nvgpu.warpgroup.fence()

                for k_block_idx in range(num_k_blocks, unroll_full=True):
                    cute.gemm(
                        tiled_mma, accumulators,
                        tCrA[None, None, k_block_idx, consumer_state.index],
                        tCrB[None, None, k_block_idx, consumer_state.index],
                        accumulators,
                    )
                    tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(0)
                mainloop_pipeline.consumer_release(consumer_state)
                consumer_state.advance()

            cute.nvgpu.warpgroup.wait_group(0)

            # =========================================================================
            # 4. EPILOGUE: Direct register -> global memory copy
            # =========================================================================
            thr_mma_store = tiled_mma.get_slice(tidx)
            tCgC_store = thr_mma_store.partition_C(gC)

            atom_universal = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
            tCrC_out = cute.make_fragment_like(accumulators, dtype=self.c_dtype)

            for reg_idx in range(cute.size(tCrC_out)):
                tCrC_out[reg_idx] = self.c_dtype(accumulators[reg_idx])

            cute.copy(atom=atom_universal, src=tCrC_out, dst=tCgC_store)


def main():
    M, N, K = 4096*2, 4096*2, 4096*2

    A = torch.randn((M, K), device="cuda", dtype=torch.float16)
    B = torch.randn((N, K), device="cuda", dtype=torch.float16)
    C = torch.empty((M, N), device="cuda", dtype=torch.float16)

    config = GemmConfig()
    gemm = HopperGemmWarpSpecialized(config)

    time_us = gemm.run(A, B, C)

    ref = torch.matmul(A, B.T)
    assert torch.allclose(C, ref, atol=5e1, rtol=5e-1), "CORRECTNESS FAILED"
    print("CORRECTNESS PASS")
    tflops = (2 * M * N * K) / (time_us * 1e6)
    print(f"DURATION: {time_us:>5.4f} µs\nTFLOPS: {tflops:>5.4f}")


if __name__ == "__main__":
    main()