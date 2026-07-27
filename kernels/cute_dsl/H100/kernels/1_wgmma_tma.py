import torch
import math
from typing import Tuple
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.hopper_helpers as sm90_utils
import cutlass.utils as utils
from cutlass.testing import benchmark, JitArguments


@dataclass
class GemmConfig:
    """
    Configuration for a single‑CTA Hopper GEMM using TMA + WGMMA.
    
    SMEM budget (single stage): A (BM×BK×2B) + B (BN×BK×2B) = 64 KB for (128,128,128).
    sA is reused for sC, keeping total SMEM at 64 KB.
    """
    cta_tiler: Tuple[int, int, int] = (128, 128, 128)
    num_stages: int = 1  # only single‑stage is supported in this version

    def __post_init__(self):
        self.BM, self.BN, self.BK = self.cta_tiler

        # MMA atom layout: 2 warp groups when BM > 64 and BN > 64
        # For large tile size, using two warp groups is preferred because using only one warp
        # group may result in register spill
        self.atom_layout_mnk = (2, 1, 1) if self.BM > 64 and self.BN > 64 else (1, 1, 1)
        self.mma_warp_groups = math.prod(self.atom_layout_mnk)
        self.threads_per_cta = self.mma_warp_groups * 128   # 128 threads per warp group


class HopperGemm:
    """
    Single‑CTA Hopper GEMM reference kernel.
    
    Algorithm (no warp specialisation, no pipelining):
      TMA load -> mbarrier wait -> WGMMA -> epilogue (register → smem → TMA store)
    
    Shared memory is reused: after WGMMA, sA is repurposed as the output buffer sC.
    """

    def __init__(self, config: GemmConfig = None):
        self.config = config or GemmConfig()
        assert self.config.num_stages == 1, "Only single‑stage supported in this reference."

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> float:
        """
        Compile, launch, and benchmark the kernel.
        Returns execution time in microseconds.
        """
        # Validate K alignment
        K = A.shape[1]
        if K % self.config.BK != 0:
            raise ValueError(f"K ({K}) must be a multiple of BK ({self.config.BK})")

        a_cute = from_dlpack(A, assumed_align=16)
        b_cute = from_dlpack(B, assumed_align=16)
        c_cute = from_dlpack(C, assumed_align=16)

        compiled = cute.compile(self, a_cute, b_cute, c_cute)
        compiled(a_cute, b_cute, c_cute)

        M, N = A.shape[0], B.shape[0]          # B is N×K for C = A @ B.T
        time_us = benchmark(compiled, kernel_arguments=JitArguments(a_cute, b_cute, c_cute))
        return time_us

    # ------------------------------------------------------------------
    # Kernel entry point (JIT compiled)
    # ------------------------------------------------------------------
    @cute.jit
    def __call__(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
        cfg = self.config

        # Tensor metadata
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.acc_dtype = cutlass.Float32
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)

        # SMEM layouts for A and B (single stage)
        self.a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout=self.a_layout,
            mma_tiler_mnk=cfg.cta_tiler,
            a_dtype=self.a_dtype,
            num_stages=1,
        )
        self.b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            b_layout=self.b_layout,
            mma_tiler_mnk=cfg.cta_tiler,
            b_dtype=self.b_dtype,
            num_stages=1,
        )

        # Build TMA atoms and tiled tensors for GMEM ↔ SMEM copies
        tma_atom_a, tma_tensor_a = self._make_tma_load_atoms_and_tensors(
            a, self.a_smem_layout_staged, (cfg.BM, cfg.BK),
        )
        tma_atom_b, tma_tensor_b = self._make_tma_load_atoms_and_tensors(
            b, self.b_smem_layout_staged, (cfg.BN, cfg.BK),
        )

        # Tiled MMA descriptor (WGMMA)
        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            cfg.atom_layout_mnk,
            tiler_mn=(64, cfg.BN),
        )

        # Shared memory structure – sA will later be reused as sC
        @cute.struct
        class SharedStorage:
            mbarrier_array_ptr: cute.struct.MemRange[cutlass.Int64, 2]   # 2 mbarrier slots
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

    # ------------------------------------------------------------------
    # The device kernel
    # ------------------------------------------------------------------
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
        # ---------- thread / block ids ----------
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        # Prefetch TMA desc
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)

        bidx, bidy, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        tile_coord_mnk = (bidx, bidy, None)

        gA = cute.local_tile(mA_tma_tensor, cfg.cta_tiler, tile_coord_mnk, (1, None, 1))
        gB = cute.local_tile(mB_tma_tensor, cfg.cta_tiler, tile_coord_mnk, (None, 1, 1))
        gC = cute.local_tile(mC,            cfg.cta_tiler, tile_coord_mnk, (1, 1, None))

        # ---------- SMEM allocation ----------
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)

        # TMA load partitions
        sA_tma, gA_tma = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a, 0, cute.make_layout(1),
            cute.group_modes(sA, 0, 2), cute.group_modes(gA, 0, 2),
        )
        sB_tma, gB_tma = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b, 0, cute.make_layout(1),
            cute.group_modes(sB, 0, 2), cute.group_modes(gB, 0, 2),
        )

        tma_transaction_bytes = (
            cute.size_in_bytes(self.a_dtype, a_smem_layout) +
            cute.size_in_bytes(self.b_dtype, b_smem_layout)
        )
        mbar_ptr = storage.mbarrier_array_ptr.data_ptr()

        # ---------- MMA thread partitioning ----------
        warp_group_idx = cute.arch.make_warp_uniform(tidx // 128)
        warp_group_thread_layout = cute.make_layout(cfg.mma_warp_groups, stride=128)
        thr_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx))

        tCgC = thr_mma.partition_C(gC)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)

        # ---------- Accumulators ----------
        accumulators = cute.make_rmem_tensor(tCgC.shape, self.acc_dtype)
        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)   # accumulators = 0

        # mbarrier initialisation: single arrival (we use an explicit arrive+tx later)
        if warp_idx == 0 and tidx == 0:
            cute.arch.mbarrier_init(mbar_ptr, cnt=1)
            cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        phase = 0
        K_tiles = mA_tma_tensor.shape[1] // cfg.BK
        c_local = mC

        # ---------- Main loop ----------
        for k_idx in range(K_tiles):
            self._load_tile(warp_idx, tidx, tma_atom_a, gA_tma, sA_tma,
                            tma_atom_b, gB_tma, sB_tma, mbar_ptr,
                            tma_transaction_bytes, k_idx)
            cute.arch.mbarrier_wait(mbar_ptr, phase)
            phase ^= 1
            cute.nvgpu.warpgroup.fence()
            BK = cfg.BK
            for k_block in range(BK // tiled_mma.shape_mnk[2]):
                k_coord = (None, None, k_block, 0)
                cute.gemm(tiled_mma, accumulators, tCrA[k_coord], tCrB[k_coord], accumulators)
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)

        # ---------- Epilogue ----------
        self._store_tile(accumulators, tiled_mma, tidx, c_local, tile_coord_mnk)

    # ------------------------------------------------------------------
    # Helper stages
    # ------------------------------------------------------------------
    @cute.jit
    def _load_tile(self, warp_idx, tidx, tma_a, gA, sA, tma_b, gB, sB,
                   mbar_ptr, tma_bytes, k_idx):
        """Issue TMA loads for the current K‑tile. Only warp 0 participates."""
        if warp_idx == 0:
            if tidx == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, tma_bytes)
            cute.copy(tma_a, gA[None, k_idx], sA[None, 0], tma_bar_ptr=mbar_ptr)
            cute.copy(tma_b, gB[None, k_idx], sB[None, 0], tma_bar_ptr=mbar_ptr)
                
    @cute.jit
    def _compute_tile(self, accumulators, tiled_mma, tCrA, tCrB):
        """WGMMA on the current K‑block."""
        BK = self.config.BK
        for k_block in range(BK // tiled_mma.shape_mnk[2]):  # inner K steps
            k_coord = (None, None, k_block, 0)
            cute.gemm(tiled_mma, accumulators, tCrA[k_coord], tCrB[k_coord], accumulators)
            tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
            
    @cute.jit
    def _store_tile(self, accumulators, tiled_mma, tidx, c_local, tile_coord_mnk):
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

        cute.copy(
            atom=atom_universal,
            src=tCrC_out,
            dst=tCgC_store,
        )
            
    # ------------------------------------------------------------------
    # TMA helpers
    # ------------------------------------------------------------------
    @cute.jit
    def _make_tma_load_atoms_and_tensors(self, tensor, smem_layout_staged, smem_tile: tuple[int, int]):
        op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        return cute.nvgpu.cpasync.make_tiled_tma_atom(op, tensor, smem_layout, smem_tile)

# ------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------
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