#include "immintrin.h"
#define A(i,j) A[(i)+(j)*LDA]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]
#define M_BLOCKING 192
#define N_BLOCKING 2048
#define K_BLOCKING 384
void scale_c_k9(double *C,int M, int N, int LDC, double scalar){
    int i,j;
    for (i=0;i<M;i++){
        for (j=0;j<N;j++){
            C(i,j)*=scalar;
        }
    }
}

void packing_a_k9(double *src, double *dst, int leading_dim, int dim_first, int dim_second){
    //dim_first: M, dim_second: K
    double *tosrc,*todst;
    todst=dst;
    int count_first,count_second,count_sub=dim_first;
    for (count_first=0;count_sub>7;count_first+=8,count_sub-=8){
        tosrc=src+count_first;
        for(count_second=0;count_second<dim_second;count_second++){
            _mm512_store_pd(todst,_mm512_loadu_pd(tosrc));
            tosrc+=leading_dim;
            todst+=8;
        }
    }
    for (;count_sub>3;count_first+=4,count_sub-=4){
        tosrc=src+count_first;
        for(count_second=0;count_second<dim_second;count_second++){
            _mm256_store_pd(todst,_mm256_loadu_pd(tosrc));
            tosrc+=leading_dim;
            todst+=4;
        }
    }
}

void packing_b_k9(double *src,double *dst,int leading_dim,int dim_first,int dim_second){
    //dim_first:K,dim_second:N
    double *tosrc1,*tosrc2,*tosrc3,*tosrc4,*todst;
    todst=dst;
    int count_first,count_second;
    for (count_second=0;count_second<dim_second;count_second+=4){
        tosrc1=src+count_second*leading_dim;tosrc2=tosrc1+leading_dim;
        tosrc3=tosrc2+leading_dim;tosrc4=tosrc3+leading_dim;
        for (count_first=0;count_first<dim_first;count_first++){
            *todst=*tosrc1;tosrc1++;todst++;
            *todst=*tosrc2;tosrc2++;todst++;
            *todst=*tosrc3;tosrc3++;todst++;
            *todst=*tosrc4;tosrc4++;todst++;
        }
    }
}

void mydgemm_cpu_opt_k9(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    int i,j,k;
    if (beta != 1.0) scale_c_k9(C,M,N,LDC,beta);
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

#define KERNEL_K1_8x4_avx2_intrinsics_packing\
    a0 = _mm256_mul_pd(valpha, _mm256_load_pd(ptr_packing_a));\
    a1 = _mm256_mul_pd(valpha, _mm256_load_pd(ptr_packing_a+4));\
    b0 = _mm256_broadcast_sd(ptr_packing_b);\
    b1 = _mm256_broadcast_sd(ptr_packing_b+1);\
    b2 = _mm256_broadcast_sd(ptr_packing_b+2);\
    b3 = _mm256_broadcast_sd(ptr_packing_b+3);\
    c00 = _mm256_fmadd_pd(a0,b0,c00);\
    c01 = _mm256_fmadd_pd(a1,b0,c01);\
    c10 = _mm256_fmadd_pd(a0,b1,c10);\
    c11 = _mm256_fmadd_pd(a1,b1,c11);\
    c20 = _mm256_fmadd_pd(a0,b2,c20);\
    c21 = _mm256_fmadd_pd(a1,b2,c21);\
    c30 = _mm256_fmadd_pd(a0,b3,c30);\
    c31 = _mm256_fmadd_pd(a1,b3,c31);\
    ptr_packing_a+=8;ptr_packing_b+=4;k++;
#define macro_kernel_8xkx4_packing\
    c00 = _mm256_setzero_pd();\
    c01 = _mm256_setzero_pd();\
    c10 = _mm256_setzero_pd();\
    c11 = _mm256_setzero_pd();\
    c20 = _mm256_setzero_pd();\
    c21 = _mm256_setzero_pd();\
    c30 = _mm256_setzero_pd();\
    c31 = _mm256_setzero_pd();\
    for (k=0;k<K4;){\
        KERNEL_K1_8x4_avx2_intrinsics_packing\
        KERNEL_K1_8x4_avx2_intrinsics_packing\
        KERNEL_K1_8x4_avx2_intrinsics_packing\
        KERNEL_K1_8x4_avx2_intrinsics_packing\
    }\
    for (k=K4;k<K;){\
        KERNEL_K1_8x4_avx2_intrinsics_packing\
    }\
    _mm256_storeu_pd(&C(i,j), _mm256_add_pd(c00,_mm256_loadu_pd(&C(i,j))));\
    _mm256_storeu_pd(&C(i+4,j), _mm256_add_pd(c01,_mm256_loadu_pd(&C(i+4,j))));\
    _mm256_storeu_pd(&C(i,j+1), _mm256_add_pd(c10,_mm256_loadu_pd(&C(i,j+1))));\
    _mm256_storeu_pd(&C(i+4,j+1), _mm256_add_pd(c11,_mm256_loadu_pd(&C(i+4,j+1))));\
    _mm256_storeu_pd(&C(i,j+2), _mm256_add_pd(c20,_mm256_loadu_pd(&C(i,j+2))));\
    _mm256_storeu_pd(&C(i+4,j+2), _mm256_add_pd(c21,_mm256_loadu_pd(&C(i+4,j+2))));\
    _mm256_storeu_pd(&C(i,j+3), _mm256_add_pd(c30,_mm256_loadu_pd(&C(i,j+3))));\
    _mm256_storeu_pd(&C(i+4,j+3), _mm256_add_pd(c31,_mm256_loadu_pd(&C(i+4,j+3))));
#define KERNEL_K1_4x4_avx2_intrinsics_packing\
    a0 = _mm256_mul_pd(valpha, _mm256_load_pd(ptr_packing_a));\
    b0 = _mm256_broadcast_sd(ptr_packing_b);\
    b1 = _mm256_broadcast_sd(ptr_packing_b+1);\
    b2 = _mm256_broadcast_sd(ptr_packing_b+2);\
    b3 = _mm256_broadcast_sd(ptr_packing_b+3);\
    c00 = _mm256_fmadd_pd(a0,b0,c00);\
    c10 = _mm256_fmadd_pd(a0,b1,c10);\
    c20 = _mm256_fmadd_pd(a0,b2,c20);\
    c30 = _mm256_fmadd_pd(a0,b3,c30);\
    ptr_packing_a+=4;ptr_packing_b+=4;k++;
#define macro_kernel_4xkx4_packing\
    c00 = _mm256_setzero_pd();\
    c10 = _mm256_setzero_pd();\
    c20 = _mm256_setzero_pd();\
    c30 = _mm256_setzero_pd();\
    for (k=0;k<K4;){\
        KERNEL_K1_4x4_avx2_intrinsics_packing\
        KERNEL_K1_4x4_avx2_intrinsics_packing\
        KERNEL_K1_4x4_avx2_intrinsics_packing\
        KERNEL_K1_4x4_avx2_intrinsics_packing\
    }\
    for (k=K4;k<K;){\
        KERNEL_K1_4x4_avx2_intrinsics_packing\
    }\
    _mm256_storeu_pd(&C(i,j), _mm256_add_pd(c00,_mm256_loadu_pd(&C(i,j))));\
    _mm256_storeu_pd(&C(i,j+1), _mm256_add_pd(c10,_mm256_loadu_pd(&C(i,j+1))));\
    _mm256_storeu_pd(&C(i,j+2), _mm256_add_pd(c20,_mm256_loadu_pd(&C(i,j+2))));\
    _mm256_storeu_pd(&C(i,j+3), _mm256_add_pd(c30,_mm256_loadu_pd(&C(i,j+3))));
void macro_kernel_gemm_k9(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double *C, int LDC){
    int i,j,k;
    int M8=M&-8,N4=N&-4,K4=K&-4;
    double *ptr_packing_a = A;
    double *ptr_packing_b = B;
    __m256d valpha = _mm256_set1_pd(alpha);//broadcast alpha to a 256-bit vector
    __m256d a,a0,a1,b0,b1,b2,b3;
    __m256d c00,c01,c10,c11,c20,c21,c30,c31;
    __m256d c0,c1,c2,c3;
    for (i=0;i<M8;i+=8){
        for (j=0;j<N4;j+=4){
            ptr_packing_a=A+i*K;ptr_packing_b=B+j*K;
            macro_kernel_8xkx4_packing
        }
    }
    for (i=M8;i<M;i+=4){
        for (j=0;j<N4;j+=4){
            ptr_packing_a=A+i*K;ptr_packing_b=B+j*K;
            macro_kernel_4xkx4_packing
        }
    }
}

void mydgemm_cpu_v9(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    
    if (beta != 1.0) scale_c_k9(C,M,N,LDC,beta);
    double *b_buffer = (double *)aligned_alloc(4096,K_BLOCKING*N_BLOCKING*sizeof(double));
    double *a_buffer = (double *)aligned_alloc(4096,K_BLOCKING*M_BLOCKING*sizeof(double));
    int m_count, n_count, k_count;
    int m_inc, n_inc, k_inc;
    for (n_count=0;n_count<N;n_count+=n_inc){
        n_inc = (N-n_count>N_BLOCKING)?N_BLOCKING:N-n_count;
        for (k_count=0;k_count<K;k_count+=k_inc){
            k_inc = (K-k_count>K_BLOCKING)?K_BLOCKING:K-k_count;
            packing_b_k9(B+k_count+n_count*LDB,b_buffer,LDB,k_inc,n_inc);
            for (m_count=0;m_count<M;m_count+=m_inc){
                m_inc = (M-m_count>M_BLOCKING)?M_BLOCKING:N-m_count;
                packing_a_k9(A+m_count+k_count*LDA,a_buffer,LDA,m_inc,k_inc);
                //macro kernel: to compute C += A_tilt * B_tilt
                macro_kernel_gemm_k9(m_inc,n_inc,k_inc,alpha,a_buffer, LDA, b_buffer, LDB, &C(m_count, n_count), LDC);
            }
        }
    }
    free(a_buffer);free(b_buffer);
}
