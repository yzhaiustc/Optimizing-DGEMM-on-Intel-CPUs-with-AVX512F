#include "immintrin.h"
#define A(i,j) A[(i)+(j)*LDA]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]
#define M_BLOCKING 192
#define N_BLOCKING 2048
#define K_BLOCKING 384
void scale_c_k8(double *C,int M, int N, int LDC, double scalar){
    int i,j;
    for (i=0;i<M;i++){
        for (j=0;j<N;j++){
            C(i,j)*=scalar;
        }
    }
}

void mydgemm_cpu_opt_k8(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    int i,j,k;
    if (beta != 1.0) scale_c_k8(C,M,N,LDC,beta);
    for (i=0;i<M;i++){
        for (j=0;j<N;j++){
            double tmp=C(i,j);
            for (k=0;k<K;k++){
                tmp += alpha*A(i,k)*B(k,j);
            }
            C(i,j) = tmp;
        }
    }
}

#define KERNEL_K1_8x4_avx2_intrinsics\
    a0 = _mm256_mul_pd(valpha, _mm256_loadu_pd(&A(i,k)));\
    a1 = _mm256_mul_pd(valpha, _mm256_loadu_pd(&A(i+4,k)));\
    b0 = _mm256_broadcast_sd(&B(k,j));\
    b1 = _mm256_broadcast_sd(&B(k,j+1));\
    b2 = _mm256_broadcast_sd(&B(k,j+2));\
    b3 = _mm256_broadcast_sd(&B(k,j+3));\
    c00 = _mm256_fmadd_pd(a0,b0,c00);\
    c01 = _mm256_fmadd_pd(a1,b0,c01);\
    c10 = _mm256_fmadd_pd(a0,b1,c10);\
    c11 = _mm256_fmadd_pd(a1,b1,c11);\
    c20 = _mm256_fmadd_pd(a0,b2,c20);\
    c21 = _mm256_fmadd_pd(a1,b2,c21);\
    c30 = _mm256_fmadd_pd(a0,b3,c30);\
    c31 = _mm256_fmadd_pd(a1,b3,c31);\
    k++;

void macro_kernel_gemm_k8(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double *C, int LDC){
    int i,j,k;
    int M8=M&-8,N4=N&-4,K4=K&-4;
    __m256d a0,a1,b0,b1,b2,b3;
    __m256d valpha = _mm256_set1_pd(alpha);
    for (i=0;i<M8;i+=8){
        for (j=0;j<N4;j+=4){
            __m256d c00 = _mm256_setzero_pd();
            __m256d c01 = _mm256_setzero_pd();
            __m256d c10 = _mm256_setzero_pd();
            __m256d c11 = _mm256_setzero_pd();
            __m256d c20 = _mm256_setzero_pd();
            __m256d c21 = _mm256_setzero_pd();
            __m256d c30 = _mm256_setzero_pd();
            __m256d c31 = _mm256_setzero_pd();
            // unroll the loop by four times
            for (k=0;k<K4;){
                KERNEL_K1_8x4_avx2_intrinsics
                KERNEL_K1_8x4_avx2_intrinsics
                KERNEL_K1_8x4_avx2_intrinsics
                KERNEL_K1_8x4_avx2_intrinsics
            }
            // deal with the edge case for K
            for (k=K4;k<K;){
                KERNEL_K1_8x4_avx2_intrinsics
            }
            _mm256_storeu_pd(&C(i,j), _mm256_add_pd(c00,_mm256_loadu_pd(&C(i,j))));
            _mm256_storeu_pd(&C(i+4,j), _mm256_add_pd(c01,_mm256_loadu_pd(&C(i+4,j))));
            _mm256_storeu_pd(&C(i,j+1), _mm256_add_pd(c10,_mm256_loadu_pd(&C(i,j+1))));
            _mm256_storeu_pd(&C(i+4,j+1), _mm256_add_pd(c11,_mm256_loadu_pd(&C(i+4,j+1))));
            _mm256_storeu_pd(&C(i,j+2), _mm256_add_pd(c20,_mm256_loadu_pd(&C(i,j+2))));
            _mm256_storeu_pd(&C(i+4,j+2), _mm256_add_pd(c21,_mm256_loadu_pd(&C(i+4,j+2))));
            _mm256_storeu_pd(&C(i,j+3), _mm256_add_pd(c30,_mm256_loadu_pd(&C(i,j+3))));
            _mm256_storeu_pd(&C(i+4,j+3), _mm256_add_pd(c31,_mm256_loadu_pd(&C(i+4,j+3))));
        }
    }
    if (M8==M&&N4==N) return;
    // boundary conditions
    if (M8!=M) mydgemm_cpu_opt_k8(M-M8,N,K,alpha,A+M8,LDA,B,LDB,1.0,&C(M8,0),LDC);
    if (N4!=N) mydgemm_cpu_opt_k8(M8,N-N4,K,alpha,A,LDA,&B(0,N4),LDB,1.0,&C(0,N4),LDC);
}

void mydgemm_cpu_v8(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    
    if (beta != 1.0) scale_c_k8(C,M,N,LDC,beta);
    
    int m_count, n_count, k_count;
    int m_inc, n_inc, k_inc;
    for (n_count=0;n_count<N;n_count+=n_inc){
        n_inc = (N-n_count>N_BLOCKING)?N_BLOCKING:N-n_count;
        for (k_count=0;k_count<K;k_count+=k_inc){
            k_inc = (K-k_count>K_BLOCKING)?K_BLOCKING:K-k_count;
            for (m_count=0;m_count<M;m_count+=m_inc){
                m_inc = (M-m_count>M_BLOCKING)?M_BLOCKING:N-m_count;
                //macro kernel: to compute C += A_tilt * B_tilt
                macro_kernel_gemm_k8(m_inc,n_inc,k_inc,alpha,&A(m_count,k_count), LDA, &B(k_count,n_count), LDB, &C(m_count, n_count), LDC);
            }
        }
    }

}
