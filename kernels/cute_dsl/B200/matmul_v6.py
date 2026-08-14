import argparse
from typing import Tuple

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack

io_dtype = cutlass.Float16
acc_dtype = cutlass.Float32
use_2cta_instrs = True
cluster_shape_mnk = (2, 1, 1) if use_2cta_instrs else (1, 1, 1)
mma_inst_shape_mnk = (256, 256, 16)
mma_tiler_mnk = (256, 256, 64)
threads_in_epilogue = 128

ab_stages = 6
epi_stages = 2
acc_stages = 1


@cute.struct
class SharedStorage:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ab_stages * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, acc_stages * 2]
    tmem_dealloc_mbar: cutlass.Int64
    tmem_holding_buffer: cutlass.Int32


@cute.kernel()
def kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB_nkl: cute.Tensor,
    tma_atom_c: cute.CopyAtom,
    mC_mnl: cute.Tensor,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
    c_smem_layout_kind: cutlass.Constexpr,
    epi_smem_layout_staged: cute.ComposedLayout,
    epi_tile: cute.Tile,
    cta_layout_vmnk: cute.Layout,
):
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)

    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    cta_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(cta_rank_in_cluster)

    mma_coord_vmnk = (
        bidx % cute.size(cta_layout_vmnk, mode=[0]),
        bidx // cute.size(cta_layout_vmnk, mode=[0]),
        bidy,
        None,
    )
    mma_coord_mnk = mma_coord_vmnk[1:]
    is_leader_cta = mma_coord_vmnk[0] == 0

    epilogue_warp_ids = (0, 1, 2, 3)
    mma_warp_id = 4
    tma_warp_id = 5

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sA = smem.allocate_tensor(
        element_type=io_dtype,
        layout=a_smem_layout.outer,
        byte_alignment=128,
        swizzle=a_smem_layout.inner,
    )
    sB = smem.allocate_tensor(
        element_type=io_dtype,
        layout=b_smem_layout.outer,
        byte_alignment=128,
        swizzle=b_smem_layout.inner,
    )
    sC = smem.allocate_tensor(
        element_type=io_dtype,
        layout=epi_smem_layout_staged.outer,
        byte_alignment=128,
        swizzle=epi_smem_layout_staged.inner,
    )

    if warp_idx == tma_warp_id:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)
        cpasync.prefetch_descriptor(tma_atom_c)

    num_mcast_participants = (
        cute.size(cta_layout_vmnk, mode=[1]) + cute.size(cta_layout_vmnk, mode=[2]) - 1
    )

    tma_mcast_mask_a = cute.nvgpu.cpasync.create_tma_multicast_mask(
        cta_layout_vmnk, cta_in_cluster_coord_vmnk, mcast_mode=2
    )
    tma_mcast_mask_b = cute.nvgpu.cpasync.create_tma_multicast_mask(
        cta_layout_vmnk, cta_in_cluster_coord_vmnk, mcast_mode=1
    )

    gA = cute.local_tile(mA_mkl, mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1))
    gB = cute.local_tile(mB_nkl, mma_tiler_mnk, mma_coord_mnk, proj=(None, 1, 1))
    gC = cute.local_tile(mC_mnl, mma_tiler_mnk, mma_coord_mnk, proj=(1, 1, None))

    thr_mma = tiled_mma.get_slice(mma_coord_vmnk[0])
    tCgA = thr_mma.partition_A(gA)
    tCgB = thr_mma.partition_B(gB)
    tCgC = thr_mma.partition_C(gC)

    tCrA = tiled_mma.make_fragment_A(sA)
    tCrB = tiled_mma.make_fragment_B(sB)
    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

    epilogue_sync_barrier = pipeline.NamedBarrier(
        barrier_id=1, num_threads=threads_in_epilogue
    )
    tmem_alloc_barrier = pipeline.NamedBarrier(
        barrier_id=2,
        num_threads=32 * len((mma_warp_id, *epilogue_warp_ids)),
    )
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buffer,
        barrier_for_retrieve=tmem_alloc_barrier,
        allocator_warp_id=epilogue_warp_ids[0],
        is_two_cta=True,
        two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar,
    )

    tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
        tma_atom_a,
        cta_in_cluster_coord_vmnk[2],
        cute.make_layout(cute.size(cta_layout_vmnk, mode=[2])),
        cute.group_modes(sA, 0, 3),
        cute.group_modes(tCgA, 0, 3),
    )
    tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
        tma_atom_b,
        cta_in_cluster_coord_vmnk[1],
        cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])),
        cute.group_modes(sB, 0, 3),
        cute.group_modes(tCgB, 0, 3),
    )

    tCgC_epi = cute.flat_divide(tCgC[((None, None), 0, 0)], epi_tile)
    tCsC, tCgC_tma = cute.nvgpu.cpasync.tma_partition(
        tma_atom_c, 0, cute.make_layout(1),
        cute.group_modes(sC, 0, 2), cute.group_modes(tCgC_epi, 0, 2),
    )

    num_tma_copy_bytes = (
        cute.size_in_bytes(io_dtype, cute.select(a_smem_layout, mode=[0, 1, 2]))
        + cute.size_in_bytes(io_dtype, cute.select(b_smem_layout, mode=[0, 1, 2]))
    ) * cute.size(cta_layout_vmnk, mode=[0])

    mainloop_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
        pipeline.Agent.Thread, size=num_mcast_participants
    )
    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        barrier_storage=storage.ab_mbar_ptr.data_ptr(),
        num_stages=ab_stages,
        producer_group=mainloop_pipeline_producer_group,
        consumer_group=mainloop_pipeline_consumer_group,
        tx_count=num_tma_copy_bytes,
        cta_layout_vmnk=cta_layout_vmnk,
    ).make_participants()

    acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    acc_pipeline_consumer_group = pipeline.CooperativeGroup(
        pipeline.Agent.Thread,
        size=cute.size(cta_layout_vmnk, mode=[0]) * len(epilogue_warp_ids),
    )
    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
        num_stages=acc_stages,
        producer_group=acc_pipeline_producer_group,
        consumer_group=acc_pipeline_consumer_group,
        cta_layout_vmnk=cta_layout_vmnk,
    ).make_participants()

    num_k_tiles = cute.size(gA, mode=[2])

    if warp_idx == tma_warp_id:
        for k_tile_idx in range(num_k_tiles):
            handle = ab_producer.acquire_and_advance()
            cute.copy(tma_atom_a, tAgA[(None, k_tile_idx)], tAsA[(None, handle.index)],
                      tma_bar_ptr=handle.barrier, mcast_mask=tma_mcast_mask_a)
            cute.copy(tma_atom_b, tBgB[(None, k_tile_idx)], tBsB[(None, handle.index)],
                      tma_bar_ptr=handle.barrier, mcast_mask=tma_mcast_mask_b)
        ab_producer.tail()

    elif warp_idx == mma_warp_id:
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(acc_dtype)
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

        if is_leader_cta:
            acc_empty = acc_producer.acquire_and_advance()
            for k_tile_idx in range(num_k_tiles):
                handle = ab_consumer.wait_and_advance()
                tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile_idx != 0)
                tile_crd = (None, None, None, handle.index)
                cute.gemm(tiled_mma, tCtAcc, tCrA[tile_crd], tCrB[tile_crd], tCtAcc)
                handle.release()
            acc_empty.commit()

    elif warp_idx < mma_warp_id:
        num_tmem_cols = 512
        tmem.allocate(num_tmem_cols)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(acc_dtype)
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

        epilogue_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=epi_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, size=128),
        )

        copy_atom_t2r = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition.x32), cutlass.Float32,
        )
        acc_consumer.wait_and_advance()
        tCtAcc_epi = cute.flat_divide(tCtAcc[((None, None), 0, 0)], epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_epi[(None, None, 0, 0)])
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tCtAcc_epi)
        tTR_gC = thr_copy_t2r.partition_D(tCgC_epi)
        tTR_rAcc = cute.make_rmem_tensor(tTR_gC[(None, None, None, 0, 0)].shape, cutlass.Float32)
        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

        copy_atom_r2s = cutlass.utils.blackwell_helpers.get_smem_store_op(
            c_smem_layout_kind, cutlass.Float32, cutlass.Float32, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        tRS_rAcc = tiled_copy_r2s.retile(tTR_rAcc)
        tRS_rC = cute.make_rmem_tensor(tRS_rAcc.shape, io_dtype)
        tCgC_grouped = cute.group_modes(tCgC_tma, 1, cute.rank(tCgC_tma))
        subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])

        for subtile_idx in cutlass.range(subtile_cnt):
            cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, subtile_idx)], tTR_rAcc)
            c_buffer = subtile_idx % epi_stages
            tRS_rC.store(tRS_rAcc.load().to(io_dtype))
            cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])
            cute.arch.fence_view_async_shared()
            epilogue_sync_barrier.arrive_and_wait()
            if warp_idx == epilogue_warp_ids[0]:
                cute.copy(tma_atom_c, tCsC[(None, c_buffer)], tCgC_grouped[(None, subtile_idx)])
                epilogue_pipeline.producer_commit()
                epilogue_pipeline.producer_acquire()
            epilogue_sync_barrier.arrive_and_wait()

        epilogue_pipeline.producer_tail()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)


@cute.jit
def host_function(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    op = tcgen05.MmaF16BF16Op(
        io_dtype, acc_dtype, mma_inst_shape_mnk,
        tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
    )
    tiled_mma = cute.make_tiled_mma(op)

    a_smem_layout = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, a.element_type, ab_stages)
    b_smem_layout = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, b.element_type, ab_stages)
    c_smem_layout_kind = utils.LayoutEnum.from_tensor(c)

    cta_layout_mnk = cute.make_layout(cluster_shape_mnk)
    cta_layout_vmnk = cute.tiled_divide(cta_layout_mnk, (tiled_mma.thr_id,))

    op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp(
        tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE
    )
    a_smem_layout_slice = cute.slice_(a_smem_layout, (None, None, None, 0))
    a_tma_atom, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
        op, a, a_smem_layout_slice, mma_tiler_mnk, tiled_mma, cta_layout_vmnk.shape,
    )
    b_smem_layout_slice = cute.slice_(b_smem_layout, (None, None, None, 0))
    b_tma_atom, b_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
        op, b, b_smem_layout_slice, mma_tiler_mnk, tiled_mma, cta_layout_vmnk.shape,
    )

    cta_tile_shape_mnk = (
        mma_tiler_mnk[0] // cute.size(tiled_mma.thr_id),
        mma_tiler_mnk[1],
        mma_tiler_mnk[2],
    )
    epi_tile = utils.compute_epilogue_tile_shape(
        cta_tile_shape_mnk, use_2cta_instrs, c_smem_layout_kind, io_dtype,
    )
    epi_smem_layout_staged = cutlass.utils.blackwell_helpers.make_smem_layout_epi(
        io_dtype, c_smem_layout_kind, epi_tile, epi_stages,
    )
    epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
    c_tma_atom, c_tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(), c, epi_smem_layout, epi_tile,
    )

    grid_shape = cute.round_up(
        (
            cute.ceil_div(c.layout.shape[0], mma_tiler_mnk[0] // (2 if use_2cta_instrs else 1)),
            cute.ceil_div(c.layout.shape[1], mma_tiler_mnk[1]),
            1,
        ),
        cluster_shape_mnk,
    )

    kernel(
        tiled_mma, a_tma_atom, a_tma_tensor, b_tma_atom, b_tma_tensor,
        c_tma_atom, c_tma_tensor, a_smem_layout, b_smem_layout,
        c_smem_layout_kind, epi_smem_layout_staged, epi_tile, cta_layout_vmnk,
    ).launch(grid=grid_shape, block=[192, 1, 1], cluster=cluster_shape_mnk)


def run_dense_gemm(mnk: Tuple[int, int, int], tolerance: float):
    global torch, cutlass_torch
    import torch
    import cutlass.torch as cutlass_torch

    m, n, k = mnk
    torch.manual_seed(1111)

    def make_tensors(mn, k, dtype):
        shape = (mn, k)
        return torch.empty(*shape, dtype=torch.int32).random_(-2, 2).to(device="cuda", dtype=dtype)

    a = make_tensors(m, k, cutlass_torch.dtype(io_dtype))
    b = make_tensors(n, k, cutlass_torch.dtype(io_dtype))
    c = make_tensors(m, n, cutlass_torch.dtype(io_dtype))
    a_memref = from_dlpack(a).mark_layout_dynamic()
    b_memref = from_dlpack(b).mark_layout_dynamic()
    c_memref = from_dlpack(c).mark_layout_dynamic()

    compiled_kernel = cute.compile(host_function, a_memref, b_memref, c_memref)
    avg_time_us = cutlass.testing.benchmark(
        compiled_kernel,
        kernel_arguments=cutlass.testing.JitArguments(a_memref, b_memref, c_memref),
        warmup_iterations=1,
        iterations=2,
    )

    total_float_ops = m * n * k * 2
    achieved_tflops = total_float_ops / (avg_time_us * 1000000)

    print(f"Problem Size")
    print(f"------------")
    print(f"  M x N x K        : {m} x {n} x {k}")
    print(f"  A ({a.shape})  [{a.dtype}]")
    print(f"  B ({b.shape})  [{b.dtype}]")
    print(f"  C ({c.shape})  [{c.dtype}]")
    print(f"  IO dtype        : {io_dtype}")
    print(f"  Accum dtype     : {acc_dtype}")
    print()
    print(f"Performance")
    print(f"----------")
    print(f"  Kernel time     : {avg_time_us:.4f} us")
    print(f"  Throughput      : {achieved_tflops:.2f} TFLOPs")
    print(f"BENCH_RESULT m={m} n={n} k={k} tflops={achieved_tflops:.2f} time_us={avg_time_us:.4f}")
    print()

    ref = (torch.einsum("mk,nk->mn", a.to(torch.float32), b.to(torch.float32))).cpu()
    torch.testing.assert_close(
        c.cpu(), ref.to(cutlass_torch.dtype(io_dtype)), atol=tolerance, rtol=1e-05
    )


if __name__ == "__main__":
    def parse_comma_separated_ints(s: str) -> list[int]:
        try:
            return [int(x.strip()) for x in s.split(",")]
        except ValueError:
            raise argparse.ArgumentTypeError("Invalid format. Expected comma-separated integers.")

    from cuda.bindings import driver as cu_driver
    cu_driver.cuInit(0)
    err, device_count = cu_driver.cuDeviceGetCount()
    if err != cu_driver.CUresult.CUDA_SUCCESS or device_count < 1:
        raise RuntimeError("A GPU is required to run this example")

    parser = argparse.ArgumentParser(description="Blackwell fp16 GEMM v8 - Warp Specialized")
    parser.add_argument("--mnk", type=parse_comma_separated_ints, default=(8192, 8192, 8192),
                        help="MNK dimensions (comma-separated)")
    parser.add_argument("--tolerance", type=float, default=1e-01, help="Tolerance for validation")
    args = parser.parse_args()
    if len(args.mnk) != 3:
        parser.error("--mnk must contain exactly 3 values")

    run_dense_gemm(tuple(args.mnk), args.tolerance)
    print("PASS")