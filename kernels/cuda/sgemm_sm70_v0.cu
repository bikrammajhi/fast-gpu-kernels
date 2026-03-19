
# Kernel 
__global__ __launch_bounds__(1024) void
kernel_v0(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C){
    # Find the current row, col this thread is calculating 

    int tx = threadIdx.x ; 
    int ty = threadIdx.y ; 
    
    int block_idx_x = blockIdx.x ;
    int block_idx_y = blockIdx.y ; 

    int block_dim_x = blockDim.x ;
    int block_dim_y = blockDim.y ; 


    int row = threadIdx.y + blockIdx.y * blockDim.y ;
    int col = threadIdx.X + blockIdx.x * blockDim.x ;

    if row > M || col > N;
        return
    
    float acc = 0 ;
    for(int i = 0; i < K; i++){
        acc += A[row * K + i] * B[i * N + col] ;
    }
    C[row][col] = alpha * acc + beta * * C[row][col] ;
}