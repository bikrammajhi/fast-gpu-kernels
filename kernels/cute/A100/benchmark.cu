#define BENCHMARK_SUITE
#include "matmul_v1.cu"

int main()
{
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);
  printf("Device: %s (SM%d, %d SMs)\n", props.name, props.major * 10 + props.minor, props.multiProcessorCount);
  printf("%8s %8s %8s   %8s %8s   %8s %8s   %5s\n",
         "M", "N", "K", "CuTe_TF", "CuTe_ms", "cubl_TF", "cubl_ms", "Rel%");

  struct Problem { int m, n, k; };
  Problem problems[] = {
    {128, 128, 128},
    {256, 256, 256},
    {512, 512, 512},
    {1024, 1024, 1024},
    {2048, 2048, 2048},
    {4096, 4096, 4096},
    {5120, 5120, 4096},
    {8192, 8192, 8192},
  };

  cublasHandle_t handle;
  cublasCreate(&handle);
  __nv_bfloat16 alpha_bf16 = __float2bfloat16(1.0f);
  __nv_bfloat16 beta_bf16  = __float2bfloat16(0.0f);
  GPU_Clock timer;
  srand(42);

  for (int p = 0; p < sizeof(problems)/sizeof(problems[0]); ++p) {
    int m = problems[p].m, n = problems[p].n, k = problems[p].k;
    double gflops = (2.0 * m * n * k) * 1e-9;

    int timing_iterations = (m < 512) ? 1000 : (m < 2048) ? 200 : 100;

    thrust::host_vector<bf16> h_A(m * k);
    thrust::host_vector<bf16> h_B(n * k);
    thrust::host_vector<bf16> h_C(m * n);
    for (int j = 0; j < m * k; ++j) h_A[j] = static_cast<bf16>(2.0f * (rand() / double(RAND_MAX)) - 1.0f);
    for (int j = 0; j < n * k; ++j) h_B[j] = static_cast<bf16>(2.0f * (rand() / double(RAND_MAX)) - 1.0f);
    for (int j = 0; j < m * n; ++j) h_C[j] = static_cast<bf16>(-1.0f);

    thrust::device_vector<bf16> d_A = h_A;
    thrust::device_vector<bf16> d_B = h_B;
    thrust::device_vector<bf16> d_C = h_C;

    int ldA = k, ldB = k, ldC = n;

    // --- CuTe kernel ---
    gemm(m, n, k, d_A.data().get(), ldA, d_B.data().get(), ldB, d_C.data().get(), ldC);
    CUTE_CHECK_LAST();
    timer.start();
    for (int i = 0; i < timing_iterations; ++i)
      gemm(m, n, k, d_A.data().get(), ldA, d_B.data().get(), ldB, d_C.data().get(), ldC);
    double cute_time = timer.seconds() / timing_iterations;
    CUTE_CHECK_LAST();

    // --- cuBLAS ---
    auto* raw_A = reinterpret_cast<const __nv_bfloat16*>(d_A.data().get());
    auto* raw_B = reinterpret_cast<const __nv_bfloat16*>(d_B.data().get());
    auto* raw_C = reinterpret_cast<__nv_bfloat16*>(d_C.data().get());

    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
        &alpha_bf16, raw_B, CUDA_R_16BF, n,
                      raw_A, CUDA_R_16BF, k,
        &beta_bf16,  raw_C, CUDA_R_16BF, n,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.start();
    for (int i = 0; i < timing_iterations; ++i)
      cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
          &alpha_bf16, raw_B, CUDA_R_16BF, n,
                        raw_A, CUDA_R_16BF, k,
          &beta_bf16,  raw_C, CUDA_R_16BF, n,
          CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    double cublas_time = timer.seconds() / timing_iterations;
    CUDA_CHECK(cudaDeviceSynchronize());

    double cute_tflops = gflops / cute_time / 1000.0;
    double cublas_tflops = gflops / cublas_time / 1000.0;
    printf("%8d %8d %8d   %8.1f %8.4f   %8.1f %8.4f   %5.1f%%\n",
           m, n, k, cute_tflops, cute_time * 1000, cublas_tflops, cublas_time * 1000,
           cute_tflops / cublas_tflops * 100.0);
  }

  cublasDestroy(handle);
  return 0;
}
