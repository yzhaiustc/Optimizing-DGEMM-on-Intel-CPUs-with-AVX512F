#include "immintrin.h"
#define A(i,j) A[(i)+(j)*LDA]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]

void scale_c(double *C,int M, int N, int LDC, double scalar){
    int i,j;
    for (i=0;i<M;i++){
        for (j=0;j<N;j++){
            C(i,j)*=scalar;
        }
    }
}

void dgemm_ijk(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    int i,j,k;
    if (beta != 1.0) scale_c(C,M,N,LDC,beta);
    for (i=0;i<M;i++){
        for (j=0;j<N;j++){
            for (k=0;k<K;k++){
                C(i,j) += alpha*A(i,k)*B(k,j);
            }
        }
    }
}

void dgemm_ijk_opt(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    int i,j,k;
    if (beta != 1.0) scale_c(C,M,N,LDC,beta);
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

void dgemm_reg_blocking_2x2(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    int i,j,k;
    if (beta != 1.0) scale_c(C,M,N,LDC,beta);
    int M2=M&-2,N2=N&-2;
    for (i=0;i<M2;i+=2){
        for (j=0;j<N2;j+=2){
            double c00=C(i,j);
            double c01=C(i,j+1);
            double c10=C(i+1,j);
            double c11=C(i+1,j+1);
            for (k=0;k<K;k++){
                double a0 = alpha*A(i,k);
                double a1 = alpha*A(i+1,k);
                double b0 = B(k,j);
                double b1 = B(k,j+1);
                c00 += a0*b0;
                c01 += a0*b1;
                c10 += a1*b0;
                c11 += a1*b1;
            }
            C(i,j) = c00;
            C(i,j+1) = c01;
            C(i+1,j) = c10;
            C(i+1,j+1) = c11;
        }
    }
    if (M2==M&&N2==N) return;
    // boundary conditions
    if (M2!=M) dgemm_ijk_opt(M-M2,N,K,alpha,A+M2,LDA,B,LDB,1.0,&C(M2,0),LDC);
    if (N2!=N) dgemm_ijk_opt(M2,N-N2,K,alpha,A,LDA,&B(0,N2),LDB,1.0,&C(0,N2),LDC);
}

void dgemm_reg_blocking_4x4(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    int i,j,k;
    if (beta != 1.0) scale_c(C,M,N,LDC,beta);
    int M4=M&-4,N4=N&-4;
    for (i=0;i<M4;i+=4){
        for (j=0;j<N4;j+=4){
            double c00=C(i,j);
            double c01=C(i,j+1);
            double c02=C(i,j+2);
            double c03=C(i,j+3);
            double c10=C(i+1,j);
            double c11=C(i+1,j+1);
            double c12=C(i+1,j+2);
            double c13=C(i+1,j+3);
            double c20=C(i+2,j);
            double c21=C(i+2,j+1);
            double c22=C(i+2,j+2);
            double c23=C(i+2,j+3);
            double c30=C(i+3,j);
            double c31=C(i+3,j+1);
            double c32=C(i+3,j+2);
            double c33=C(i+3,j+3);
            for (k=0;k<K;k++){
                double a0 = alpha*A(i,k);
                double a1 = alpha*A(i+1,k);
                double a2 = alpha*A(i+2,k);
                double a3 = alpha*A(i+3,k);
                double b0 = B(k,j);
                double b1 = B(k,j+1);
                double b2 = B(k,j+2);
                double b3 = B(k,j+3);
                c00 += a0*b0;
                c01 += a0*b1;
                c02 += a0*b2;
                c03 += a0*b3;
                c10 += a1*b0;
                c11 += a1*b1;
                c12 += a1*b2;
                c13 += a1*b3;
                c20 += a2*b0;
                c21 += a2*b1;
                c22 += a2*b2;
                c23 += a2*b3;
                c30 += a3*b0;
                c31 += a3*b1;
                c32 += a3*b2;
                c33 += a3*b3;
            }
            C(i,j) = c00;
            C(i,j+1) = c01;
            C(i,j+2) = c02;
            C(i,j+3) = c03;
            C(i+1,j) = c10;
            C(i+1,j+1) = c11;
            C(i+1,j+2) = c12;
            C(i+1,j+3) = c13;
            C(i+2,j) = c20;
            C(i+2,j+1) = c21;
            C(i+2,j+2) = c22;
            C(i+2,j+3) = c23;
            C(i+3,j) = c30;
            C(i+3,j+1) = c31;
            C(i+3,j+2) = c32;
            C(i+3,j+3) = c33;
        }
    }
    if (M4==M&&N4==N) return;
    // boundary conditions
    if (M4!=M) dgemm_ijk_opt(M-M4,N,K,alpha,A+M4,LDA,B,LDB,1.0,&C(M4,0),LDC);
    if (N4!=N) dgemm_ijk_opt(M4,N-N4,K,alpha,A,LDA,&B(0,N4),LDB,1.0,&C(0,N4),LDC);
}


void dgemm_reg_blocking_4x4_avx2(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    int i,j,k;
    if (beta != 1.0) scale_c(C,M,N,LDC,beta);
    int M4=M&-4,N4=N&-4;
    __m256d valpha = _mm256_set1_pd(alpha);//broadcast alpha to a 256-bit vector
    for (i=0;i<M4;i+=4){
        for (j=0;j<N4;j+=4){
            __m256d c0 = _mm256_setzero_pd();
            __m256d c1 = _mm256_setzero_pd();
            __m256d c2 = _mm256_setzero_pd();
            __m256d c3 = _mm256_setzero_pd();
            for (k=0;k<K;k++){
                __m256d a = _mm256_mul_pd(valpha, _mm256_loadu_pd(&A(i,k)));
                __m256d b0 = _mm256_broadcast_sd(&B(k,j));
                __m256d b1 = _mm256_broadcast_sd(&B(k,j+1));
                __m256d b2 = _mm256_broadcast_sd(&B(k,j+2));
                __m256d b3 = _mm256_broadcast_sd(&B(k,j+3));
                c0 = _mm256_fmadd_pd(a,b0,c0);
                c1 = _mm256_fmadd_pd(a,b1,c1);
                c2 = _mm256_fmadd_pd(a,b2,c2);
                c3 = _mm256_fmadd_pd(a,b3,c3);
            }
            _mm256_storeu_pd(&C(i,j), _mm256_add_pd(c0,_mm256_loadu_pd(&C(i,j))));
            _mm256_storeu_pd(&C(i,j+1), _mm256_add_pd(c1,_mm256_loadu_pd(&C(i,j+1))));
            _mm256_storeu_pd(&C(i,j+2), _mm256_add_pd(c2,_mm256_loadu_pd(&C(i,j+2))));
            _mm256_storeu_pd(&C(i,j+3), _mm256_add_pd(c3,_mm256_loadu_pd(&C(i,j+3))));
        }
    }
    if (M4==M&&N4==N) return;
    // boundary conditions
    if (M4!=M) dgemm_ijk_opt(M-M4,N,K,alpha,A+M4,LDA,B,LDB,1.0,&C(M4,0),LDC);
    if (N4!=N) dgemm_ijk_opt(M4,N-N4,K,alpha,A,LDA,&B(0,N4),LDB,1.0,&C(0,N4),LDC);
}

#define KERNEL_K1_4x4_avx2_intrinsics\
    a = _mm256_mul_pd(valpha, _mm256_loadu_pd(&A(i,k)));\
    b0 = _mm256_broadcast_sd(&B(k,j));\
    b1 = _mm256_broadcast_sd(&B(k,j+1));\
    b2 = _mm256_broadcast_sd(&B(k,j+2));\
    b3 = _mm256_broadcast_sd(&B(k,j+3));\
    c0 = _mm256_fmadd_pd(a,b0,c0);\
    c1 = _mm256_fmadd_pd(a,b1,c1);\
    c2 = _mm256_fmadd_pd(a,b2,c2);\
    c3 = _mm256_fmadd_pd(a,b3,c3);\
    k++;

#define KERNEL_K1_4x1_avx2_intrinsics\
    a = _mm256_mul_pd(valpha, _mm256_loadu_pd(&A(i,k)));\
    b0 = _mm256_broadcast_sd(&B(k,j));\
    c0 = _mm256_fmadd_pd(a,b0,c0);\
    k++;

void dgemm_reg_blocking_4x4_avx2_template_unrollx4(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    int i,j,k;
    if (beta != 1.0) scale_c(C,M,N,LDC,beta);
    int M4=M&-4,N4=N&-4,K4=K&-4;
    __m256d valpha = _mm256_set1_pd(alpha);//broadcast alpha to a 256-bit vector
    __m256d a,b0,b1,b2,b3;
    for (i=0;i<M4;i+=4){
        for (j=0;j<N4;j+=4){
            __m256d c0 = _mm256_setzero_pd();
            __m256d c1 = _mm256_setzero_pd();
            __m256d c2 = _mm256_setzero_pd();
            __m256d c3 = _mm256_setzero_pd();
            // unroll the loop by four times
            for (k=0;k<K4;){
                KERNEL_K1_4x4_avx2_intrinsics
                KERNEL_K1_4x4_avx2_intrinsics
                KERNEL_K1_4x4_avx2_intrinsics
                KERNEL_K1_4x4_avx2_intrinsics
            }
            // deal with the edge case for K
            for (k=K4;k<K;){
                KERNEL_K1_4x4_avx2_intrinsics
            }
            _mm256_storeu_pd(&C(i,j), _mm256_add_pd(c0,_mm256_loadu_pd(&C(i,j))));
            _mm256_storeu_pd(&C(i,j+1), _mm256_add_pd(c1,_mm256_loadu_pd(&C(i,j+1))));
            _mm256_storeu_pd(&C(i,j+2), _mm256_add_pd(c2,_mm256_loadu_pd(&C(i,j+2))));
            _mm256_storeu_pd(&C(i,j+3), _mm256_add_pd(c3,_mm256_loadu_pd(&C(i,j+3))));
        }
    }
    if (M4==M&&N4==N) return;
    // boundary conditions
    if (M4!=M) dgemm_ijk_opt(M-M4,N,K,alpha,A+M4,LDA,B,LDB,1.0,&C(M4,0),LDC);
    if (N4!=N) dgemm_ijk_opt(M4,N-N4,K,alpha,A,LDA,&B(0,N4),LDB,1.0,&C(0,N4),LDC);
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

void dgemm_reg_blocking_8x4_avx2_template_unrollx4(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    int i,j,k;
    if (beta != 1.0) scale_c(C,M,N,LDC,beta);
    int M8=M&-8,N4=N&-4,K4=K&-4;
    __m256d valpha = _mm256_set1_pd(alpha);//broadcast alpha to a 256-bit vector
    __m256d a0,a1,b0,b1,b2,b3;
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
    if (M8!=M) dgemm_ijk_opt(M-M8,N,K,alpha,A+M8,LDA,B,LDB,1.0,&C(M8,0),LDC);
    if (N4!=N) dgemm_ijk_opt(M8,N-N4,K,alpha,A,LDA,&B(0,N4),LDB,1.0,&C(0,N4),LDC);
}



#define macro_kernel_4xkx4\
    c0 = _mm256_setzero_pd();\
    c1 = _mm256_setzero_pd();\
    c2 = _mm256_setzero_pd();\
    c3 = _mm256_setzero_pd();\
    for (k=k_start;k<K4;){\
        KERNEL_K1_4x4_avx2_intrinsics\
        KERNEL_K1_4x4_avx2_intrinsics\
        KERNEL_K1_4x4_avx2_intrinsics\
        KERNEL_K1_4x4_avx2_intrinsics\
    }\
    for (k=K4;k<k_end;){\
        KERNEL_K1_4x4_avx2_intrinsics\
    }\
    _mm256_storeu_pd(&C(i,j), _mm256_add_pd(c0,_mm256_loadu_pd(&C(i,j))));\
    _mm256_storeu_pd(&C(i,j+1), _mm256_add_pd(c1,_mm256_loadu_pd(&C(i,j+1))));\
    _mm256_storeu_pd(&C(i,j+2), _mm256_add_pd(c2,_mm256_loadu_pd(&C(i,j+2))));\
    _mm256_storeu_pd(&C(i,j+3), _mm256_add_pd(c3,_mm256_loadu_pd(&C(i,j+3))));


#define macro_kernel_4xkx1\
    c0 = _mm256_setzero_pd();\
    for (k=k_start;k<k_end;){\
        KERNEL_K1_4x1_avx2_intrinsics\
    }\
    _mm256_storeu_pd(&C(i,j), _mm256_add_pd(c0,_mm256_loadu_pd(&C(i,j))));


#define macro_kernel_8xkx4\
    c00 = _mm256_setzero_pd();\
    c01 = _mm256_setzero_pd();\
    c10 = _mm256_setzero_pd();\
    c11 = _mm256_setzero_pd();\
    c20 = _mm256_setzero_pd();\
    c21 = _mm256_setzero_pd();\
    c30 = _mm256_setzero_pd();\
    c31 = _mm256_setzero_pd();\
    for (k=k_start;k<K4;){\
        KERNEL_K1_8x4_avx2_intrinsics\
        KERNEL_K1_8x4_avx2_intrinsics\
        KERNEL_K1_8x4_avx2_intrinsics\
        KERNEL_K1_8x4_avx2_intrinsics\
    }\
    for (k=K4;k<k_end;){\
        KERNEL_K1_8x4_avx2_intrinsics\
    }\
    _mm256_storeu_pd(&C(i,j), _mm256_add_pd(c00,_mm256_loadu_pd(&C(i,j))));\
    _mm256_storeu_pd(&C(i+4,j), _mm256_add_pd(c01,_mm256_loadu_pd(&C(i+4,j))));\
    _mm256_storeu_pd(&C(i,j+1), _mm256_add_pd(c10,_mm256_loadu_pd(&C(i,j+1))));\
    _mm256_storeu_pd(&C(i+4,j+1), _mm256_add_pd(c11,_mm256_loadu_pd(&C(i+4,j+1))));\
    _mm256_storeu_pd(&C(i,j+2), _mm256_add_pd(c20,_mm256_loadu_pd(&C(i,j+2))));\
    _mm256_storeu_pd(&C(i+4,j+2), _mm256_add_pd(c21,_mm256_loadu_pd(&C(i+4,j+2))));\
    _mm256_storeu_pd(&C(i,j+3), _mm256_add_pd(c30,_mm256_loadu_pd(&C(i,j+3))));\
    _mm256_storeu_pd(&C(i+4,j+3), _mm256_add_pd(c31,_mm256_loadu_pd(&C(i+4,j+3))));


#define macro_kernel_1xkx4\
    sc0=sc1=sc2=sc3=0.;\
    for (k=k_start;k<k_end;k++){\
        sa=alpha*A(i,k);\
        sb0=B(k,j);sb1=B(k,j+1);sb2=B(k,j+2);sb3=B(k,j+3);\
        sc0+=sa*sb0;sc1+=sa*sb1;sc2+=sa*sb2;sc3+=sa*sb3;\
    }\
    C(i,j)+=sc0;C(i,j+1)+=sc1;C(i,j+2)+=sc2;C(i,j+3)+=sc3;

#define macro_kernel_1xkx1\
    sc0=0.;\
    for (k=k_start;k<k_end;k++){\
        sa=alpha*A(i,k);\
        sb0=B(k,j);\
        sc0+=sa*sb0;\
    }\
    C(i,j)+=sc0;


#define M_BLOCKING 192
#define N_BLOCKING 96
#define K_BLOCKING 384

void dgemm_cache_blocking_reg_blocking_4x4_avx2_template_unrollx4(\
    int M, \
    int N, \
    int K, \
    double alpha, \
    double *A, \
    int LDA, \
    double *B, \
    int LDB, \
    double beta, \
    double *C, \
    int LDC)\
{
    int i,j,k;
    if (beta != 1.0) scale_c(C,M,N,LDC,beta);
    int M4,N4,K4;
    __m256d valpha = _mm256_set1_pd(alpha);//broadcast alpha to a 256-bit vector
    __m256d a,b0,b1,b2,b3;
    __m256d c0,c1,c2,c3;
    double sc0,sc1,sc2,sc3,sa,sb0,sb1,sb2,sb3;
    int M_MAIN = M/M_BLOCKING*M_BLOCKING,M_EDGE=M-M_MAIN;
    int N_MAIN = N/N_BLOCKING*N_BLOCKING,N_EDGE=N-N_MAIN;
    int K_MAIN = K/K_BLOCKING*K_BLOCKING,K_EDGE=K-K_MAIN;
    int m_count,n_count,k_count;
    int m_inc,n_inc,k_inc,k_start,k_end;
    for (k_count=0;k_count<K;k_count+=k_inc){
        //printf("k_count=%d\n",k_count);
        k_inc=(K-k_count>K_BLOCKING)?K_BLOCKING:K-k_count;
        for (n_count=0;n_count<N_MAIN;n_count+=N_BLOCKING){
            for (m_count=0;m_count<M_MAIN;m_count+=M_BLOCKING){
                int inner_m_count,inner_n_count;
                for (inner_n_count=0;inner_n_count<N_BLOCKING;inner_n_count+=4){
                    for (inner_m_count=0;inner_m_count<M_BLOCKING;inner_m_count+=4){
                        i=m_count+inner_m_count;j=n_count+inner_n_count;
                        k_start=k_count;k_end=k_start+k_inc;K4=(k_start+k_inc)&-4;
                        macro_kernel_4xkx4
                        //printf("m_count=%d\n",m_count);
                    }
                }
            }
            for (m_count=M_MAIN;m_count<M;m_count++){
                int inner_n_count;
                for (inner_n_count=0;inner_n_count<N_BLOCKING;inner_n_count+=4){
                    i=m_count;j=n_count+inner_n_count;
                    k_start=k_count;k_end=k_start+k_inc;
                    macro_kernel_1xkx4
                }
            }
        }
        for (n_count=N_MAIN;n_count<N;n_count++){
            for (m_count=0;m_count<M_MAIN;m_count+=M_BLOCKING){
                int inner_m_count;
                for (inner_m_count=0;inner_m_count<M_BLOCKING;inner_m_count+=4){
                    i=m_count+inner_m_count;j=n_count;
                    k_start=k_count;k_end=k_start+k_inc;K4=(k_start+k_inc)&-4;
                    macro_kernel_4xkx1
                }
            }
            for (m_count=M_MAIN;m_count<M;m_count++){
                i=m_count;j=n_count;
                k_start=k_count;k_end=k_start+k_inc;
                macro_kernel_1xkx1
            }
        }
    }
}


void dgemm_cache_blocking_reg_blocking_8x4_avx2_template_unrollx4(\
    int M, \
    int N, \
    int K, \
    double alpha, \
    double *A, \
    int LDA, \
    double *B, \
    int LDB, \
    double beta, \
    double *C, \
    int LDC)\
{
    int i,j,k;
    if (beta != 1.0) scale_c(C,M,N,LDC,beta);
    int M4,N4,K4;
    __m256d valpha = _mm256_set1_pd(alpha);//broadcast alpha to a 256-bit vector
    __m256d a,a0,a1,b0,b1,b2,b3;
    __m256d c00,c01,c10,c11,c20,c21,c30,c31;
    __m256d c0,c1,c2,c3;
    double sc0,sc1,sc2,sc3,sa,sb0,sb1,sb2,sb3;
    int M_MAIN = M/M_BLOCKING*M_BLOCKING,M_EDGE=M-M_MAIN;
    int N_MAIN = N/N_BLOCKING*N_BLOCKING,N_EDGE=N-N_MAIN;
    int K_MAIN = K/K_BLOCKING*K_BLOCKING,K_EDGE=K-K_MAIN;
    int m_count,n_count,k_count;
    int m_inc,n_inc,k_inc,k_start,k_end;
    for (k_count=0;k_count<K;k_count+=k_inc){
        //printf("k_count=%d\n",k_count);
        k_inc=(K-k_count>K_BLOCKING)?K_BLOCKING:K-k_count;
        for (n_count=0;n_count<N_MAIN;n_count+=N_BLOCKING){
            for (m_count=0;m_count<M_MAIN;m_count+=M_BLOCKING){
                int inner_m_count,inner_n_count;
                for (inner_n_count=0;inner_n_count<N_BLOCKING;inner_n_count+=4){
                    for (inner_m_count=0;inner_m_count<M_BLOCKING;inner_m_count+=8){
                        i=m_count+inner_m_count;j=n_count+inner_n_count;
                        k_start=k_count;k_end=k_start+k_inc;K4=(k_start+k_inc)&-4;
                        macro_kernel_8xkx4
                        //printf("m_count=%d\n",m_count);
                    }
                }
            }
            for (m_count=M_MAIN;m_count<M;m_count++){
                int inner_n_count;
                for (inner_n_count=0;inner_n_count<N_BLOCKING;inner_n_count+=4){
                    i=m_count;j=n_count+inner_n_count;
                    k_start=k_count;k_end=k_start+k_inc;
                    macro_kernel_1xkx4
                }
            }
        }
        for (n_count=N_MAIN;n_count<N;n_count++){
            for (m_count=0;m_count<M_MAIN;m_count+=M_BLOCKING){
                int inner_m_count;
                for (inner_m_count=0;inner_m_count<M_BLOCKING;inner_m_count+=4){
                    i=m_count+inner_m_count;j=n_count;
                    k_start=k_count;k_end=k_start+k_inc;K4=(k_start+k_inc)&-4;
                    macro_kernel_4xkx1
                }
            }
            for (m_count=M_MAIN;m_count<M;m_count++){
                i=m_count;j=n_count;
                k_start=k_count;k_end=k_start+k_inc;
                macro_kernel_1xkx1
            }
        }
    }
}

void packing_a(double *src, double *dst, int leading_dim, int dim_first, int dim_second){
    //dim_first: M, dim_second: K
    double *tosrc,*todst;
    todst=dst;
    int count_first,count_second;
    for (count_first=0;count_first<dim_first;count_first+=8){
        tosrc=src+count_first;
        for(count_second=0;count_second<dim_second;count_second++){
            _mm512_store_pd(todst,_mm512_loadu_pd(tosrc));
            tosrc+=leading_dim;
            todst+=8;
        }
    }
}

void packing_b(double *src,double *dst,int leading_dim,int dim_first,int dim_second){
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
    for (k=k_start;k<K4;){\
        KERNEL_K1_8x4_avx2_intrinsics_packing\
        KERNEL_K1_8x4_avx2_intrinsics_packing\
        KERNEL_K1_8x4_avx2_intrinsics_packing\
        KERNEL_K1_8x4_avx2_intrinsics_packing\
    }\
    for (k=K4;k<k_end;){\
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

void dgemm_packing_cache_blocking_reg_blocking_8x4_avx2_template_unrollx4(\
    int M, \
    int N, \
    int K, \
    double alpha, \
    double *A, \
    int LDA, \
    double *B, \
    int LDB, \
    double beta, \
    double *C, \
    int LDC)\
{
    int i,j,k;
    if (beta != 1.0) scale_c(C,M,N,LDC,beta);
    int M4,N4,K4;
    __m256d valpha = _mm256_set1_pd(alpha);//broadcast alpha to a 256-bit vector
    __m256d a,a0,a1,b0,b1,b2,b3;
    __m256d c00,c01,c10,c11,c20,c21,c30,c31;
    __m256d c0,c1,c2,c3;
    double *ptr_packing_a,*ptr_packing_b;
    double sc0,sc1,sc2,sc3,sa,sb0,sb1,sb2,sb3;
    double *b_buffer = (double *)aligned_alloc(4096,K_BLOCKING*N_BLOCKING*sizeof(double));
    double *a_buffer = (double *)aligned_alloc(4096,K_BLOCKING*M_BLOCKING*sizeof(double));
    int M_MAIN = M/M_BLOCKING*M_BLOCKING,M_EDGE=M-M_MAIN;
    int N_MAIN = N/N_BLOCKING*N_BLOCKING,N_EDGE=N-N_MAIN;
    int K_MAIN = K/K_BLOCKING*K_BLOCKING,K_EDGE=K-K_MAIN;
    int m_count,n_count,k_count;
    int m_inc,n_inc,k_inc,k_start,k_end;
    for (k_count=0;k_count<K;k_count+=k_inc){
        //printf("k_count=%d\n",k_count);
        k_inc=(K-k_count>K_BLOCKING)?K_BLOCKING:K-k_count;
        for (n_count=0;n_count<N_MAIN;n_count+=N_BLOCKING){
            packing_b(B+k_count+n_count*LDB,b_buffer,LDB,k_inc,N_BLOCKING);
            for (m_count=0;m_count<M_MAIN;m_count+=M_BLOCKING){
                packing_a(A+m_count+k_count*LDA,a_buffer,LDA,M_BLOCKING,k_inc);
                int inner_m_count,inner_n_count;
                for (inner_n_count=0;inner_n_count<N_BLOCKING;inner_n_count+=4){
                    for (inner_m_count=0;inner_m_count<M_BLOCKING;inner_m_count+=8){
                        i=m_count+inner_m_count;j=n_count+inner_n_count;
                        k_start=k_count;k_end=k_start+k_inc;K4=(k_start+k_inc)&-4;
                        ptr_packing_a=a_buffer+inner_m_count*k_inc;ptr_packing_b=b_buffer+inner_n_count*k_inc;
                        macro_kernel_8xkx4_packing
                        //printf("m_count=%d\n",m_count);
                    }
                }
            }
            for (m_count=M_MAIN;m_count<M;m_count++){
                int inner_n_count;
                for (inner_n_count=0;inner_n_count<N_BLOCKING;inner_n_count+=4){
                    i=m_count;j=n_count+inner_n_count;
                    k_start=k_count;k_end=k_start+k_inc;
                    macro_kernel_1xkx4
                }
            }
        }
        for (n_count=N_MAIN;n_count<N;n_count++){
            for (m_count=0;m_count<M_MAIN;m_count+=M_BLOCKING){
                int inner_m_count;
                for (inner_m_count=0;inner_m_count<M_BLOCKING;inner_m_count+=4){
                    i=m_count+inner_m_count;j=n_count;
                    k_start=k_count;k_end=k_start+k_inc;K4=(k_start+k_inc)&-4;
                    macro_kernel_4xkx1
                }
            }
            for (m_count=M_MAIN;m_count<M;m_count++){
                i=m_count;j=n_count;
                k_start=k_count;k_end=k_start+k_inc;
                macro_kernel_1xkx1
            }
        }
    }
}

#define KERNEL_K1_24x8_avx512_intrinsics_packing\
    a0 = _mm512_mul_pd(valpha, _mm512_load_pd(ptr_packing_a));\
    a1 = _mm512_mul_pd(valpha, _mm512_load_pd(ptr_packing_a+8));\
    a2 = _mm512_mul_pd(valpha, _mm512_load_pd(ptr_packing_a+16));\
    b0 = _mm512_set1_pd(*ptr_packing_b);\
    b1 = _mm512_set1_pd(*(ptr_packing_b+1));\
    c00 = _mm512_fmadd_pd(a0,b0,c00);\
    c01 = _mm512_fmadd_pd(a1,b0,c01);\
    c02 = _mm512_fmadd_pd(a2,b0,c02);\
    c10 = _mm512_fmadd_pd(a0,b1,c10);\
    c11 = _mm512_fmadd_pd(a1,b1,c11);\
    c12 = _mm512_fmadd_pd(a2,b1,c12);\
    b0 = _mm512_set1_pd(*(ptr_packing_b+2));\
    b1 = _mm512_set1_pd(*(ptr_packing_b+3));\
    c20 = _mm512_fmadd_pd(a0,b0,c20);\
    c21 = _mm512_fmadd_pd(a1,b0,c21);\
    c22 = _mm512_fmadd_pd(a2,b0,c22);\
    c30 = _mm512_fmadd_pd(a0,b1,c30);\
    c31 = _mm512_fmadd_pd(a1,b1,c31);\
    c32 = _mm512_fmadd_pd(a2,b1,c32);\
    b0 = _mm512_set1_pd(*(ptr_packing_b+4));\
    b1 = _mm512_set1_pd(*(ptr_packing_b+5));\
    c40 = _mm512_fmadd_pd(a0,b0,c40);\
    c41 = _mm512_fmadd_pd(a1,b0,c41);\
    c42 = _mm512_fmadd_pd(a2,b0,c42);\
    c50 = _mm512_fmadd_pd(a0,b1,c50);\
    c51 = _mm512_fmadd_pd(a1,b1,c51);\
    c52 = _mm512_fmadd_pd(a2,b1,c52);\
    b0 = _mm512_set1_pd(*(ptr_packing_b+6));\
    b1 = _mm512_set1_pd(*(ptr_packing_b+7));\
    c60 = _mm512_fmadd_pd(a0,b0,c60);\
    c61 = _mm512_fmadd_pd(a1,b0,c61);\
    c62 = _mm512_fmadd_pd(a2,b0,c62);\
    c70 = _mm512_fmadd_pd(a0,b1,c70);\
    c71 = _mm512_fmadd_pd(a1,b1,c71);\
    c72 = _mm512_fmadd_pd(a2,b1,c72);\
    ptr_packing_a+=24;ptr_packing_b+=8;k++;

void packing_a_24x8(double *src, double *dst, int leading_dim, int dim_first, int dim_second){
    //dim_first: M, dim_second: K
    double *tosrc,*todst;
    todst=dst;
    int count_first,count_second;
    for (count_first=0;count_first<dim_first;count_first+=24){
        tosrc=src+count_first;
        for(count_second=0;count_second<dim_second;count_second++){
            _mm512_store_pd(todst,_mm512_loadu_pd(tosrc));
            _mm512_store_pd(todst+8,_mm512_loadu_pd(tosrc+8));
            _mm512_store_pd(todst+16,_mm512_loadu_pd(tosrc+16));
            tosrc+=leading_dim;
            todst+=24;
        }
    }
}

void packing_b_24x8_version1(double *src,double *dst,int leading_dim,int dim_first,int dim_second){
    //dim_first:K,dim_second:N
    double *tosrc1,*tosrc2,*tosrc3,*tosrc4,*tosrc5,*tosrc6,*tosrc7,*tosrc8,*todst;
    todst=dst;
    int count_first,count_second;
    for (count_second=0;count_second<dim_second;count_second+=8){
        tosrc1=src+count_second*leading_dim;tosrc2=tosrc1+leading_dim;
        tosrc3=tosrc2+leading_dim;tosrc4=tosrc3+leading_dim;
        tosrc5=tosrc4+leading_dim;tosrc6=tosrc5+leading_dim;
        tosrc7=tosrc6+leading_dim;tosrc8=tosrc7+leading_dim;
        for (count_first=0;count_first<dim_first;count_first++){
            *todst=*tosrc1;tosrc1++;todst++;
            *todst=*tosrc2;tosrc2++;todst++;
            *todst=*tosrc3;tosrc3++;todst++;
            *todst=*tosrc4;tosrc4++;todst++;
            *todst=*tosrc5;tosrc5++;todst++;
            *todst=*tosrc6;tosrc6++;todst++;
            *todst=*tosrc7;tosrc7++;todst++;
            *todst=*tosrc8;tosrc8++;todst++;
        }
    }
}

#define KERNEL_K1_8x1_avx512_intrinsics\
    a = _mm512_mul_pd(valpha, _mm512_loadu_pd(&A(i,k)));\
    b0 = _mm512_set1_pd(B(k,j));\
    c0 = _mm512_fmadd_pd(a,b0,c0);\
    k++;

#define macro_kernel_8xkx1_avx512\
    c0 = _mm512_setzero_pd();\
    for (k=k_start;k<k_end;){\
        KERNEL_K1_8x1_avx512_intrinsics\
    }\
    _mm512_storeu_pd(&C(i,j), _mm512_add_pd(c0,_mm512_loadu_pd(&C(i,j))));


#define macro_kernel_24xkx8_packing_avx512_v1\
    c00 = _mm512_setzero_pd();\
    c01 = _mm512_setzero_pd();\
    c02 = _mm512_setzero_pd();\
    c10 = _mm512_setzero_pd();\
    c11 = _mm512_setzero_pd();\
    c12 = _mm512_setzero_pd();\
    c20 = _mm512_setzero_pd();\
    c21 = _mm512_setzero_pd();\
    c22 = _mm512_setzero_pd();\
    c30 = _mm512_setzero_pd();\
    c31 = _mm512_setzero_pd();\
    c32 = _mm512_setzero_pd();\
    c40 = _mm512_setzero_pd();\
    c41 = _mm512_setzero_pd();\
    c42 = _mm512_setzero_pd();\
    c50 = _mm512_setzero_pd();\
    c51 = _mm512_setzero_pd();\
    c52 = _mm512_setzero_pd();\
    c60 = _mm512_setzero_pd();\
    c61 = _mm512_setzero_pd();\
    c62 = _mm512_setzero_pd();\
    c70 = _mm512_setzero_pd();\
    c71 = _mm512_setzero_pd();\
    c72 = _mm512_setzero_pd();\
    for (k=k_start;k<K4;){\
        KERNEL_K1_24x8_avx512_intrinsics_packing\
        KERNEL_K1_24x8_avx512_intrinsics_packing\
        KERNEL_K1_24x8_avx512_intrinsics_packing\
        KERNEL_K1_24x8_avx512_intrinsics_packing\
    }\
    for (k=K4;k<k_end;){\
        KERNEL_K1_24x8_avx512_intrinsics_packing\
    }\
    _mm512_storeu_pd(&C(i,j), _mm512_add_pd(c00,_mm512_loadu_pd(&C(i,j))));\
    _mm512_storeu_pd(&C(i+8,j), _mm512_add_pd(c01,_mm512_loadu_pd(&C(i+8,j))));\
    _mm512_storeu_pd(&C(i+16,j), _mm512_add_pd(c02,_mm512_loadu_pd(&C(i+16,j))));\
    _mm512_storeu_pd(&C(i,j+1), _mm512_add_pd(c10,_mm512_loadu_pd(&C(i,j+1))));\
    _mm512_storeu_pd(&C(i+8,j+1), _mm512_add_pd(c11,_mm512_loadu_pd(&C(i+8,j+1))));\
    _mm512_storeu_pd(&C(i+16,j+1), _mm512_add_pd(c12,_mm512_loadu_pd(&C(i+16,j+1))));\
    _mm512_storeu_pd(&C(i,j+2), _mm512_add_pd(c20,_mm512_loadu_pd(&C(i,j+2))));\
    _mm512_storeu_pd(&C(i+8,j+2), _mm512_add_pd(c21,_mm512_loadu_pd(&C(i+8,j+2))));\
    _mm512_storeu_pd(&C(i+16,j+2), _mm512_add_pd(c22,_mm512_loadu_pd(&C(i+16,j+2))));\
    _mm512_storeu_pd(&C(i,j+3), _mm512_add_pd(c30,_mm512_loadu_pd(&C(i,j+3))));\
    _mm512_storeu_pd(&C(i+8,j+3), _mm512_add_pd(c31,_mm512_loadu_pd(&C(i+8,j+3))));\
    _mm512_storeu_pd(&C(i+16,j+3), _mm512_add_pd(c32,_mm512_loadu_pd(&C(i+16,j+3))));\
    _mm512_storeu_pd(&C(i,j+4), _mm512_add_pd(c40,_mm512_loadu_pd(&C(i,j+4))));\
    _mm512_storeu_pd(&C(i+8,j+4), _mm512_add_pd(c41,_mm512_loadu_pd(&C(i+8,j+4))));\
    _mm512_storeu_pd(&C(i+16,j+4), _mm512_add_pd(c42,_mm512_loadu_pd(&C(i+16,j+4))));\
    _mm512_storeu_pd(&C(i,j+5), _mm512_add_pd(c50,_mm512_loadu_pd(&C(i,j+5))));\
    _mm512_storeu_pd(&C(i+8,j+5), _mm512_add_pd(c51,_mm512_loadu_pd(&C(i+8,j+5))));\
    _mm512_storeu_pd(&C(i+16,j+5), _mm512_add_pd(c52,_mm512_loadu_pd(&C(i+16,j+5))));\
    _mm512_storeu_pd(&C(i,j+6), _mm512_add_pd(c60,_mm512_loadu_pd(&C(i,j+6))));\
    _mm512_storeu_pd(&C(i+8,j+6), _mm512_add_pd(c61,_mm512_loadu_pd(&C(i+8,j+6))));\
    _mm512_storeu_pd(&C(i+16,j+6), _mm512_add_pd(c62,_mm512_loadu_pd(&C(i+16,j+6))));\
    _mm512_storeu_pd(&C(i,j+7), _mm512_add_pd(c70,_mm512_loadu_pd(&C(i,j+7))));\
    _mm512_storeu_pd(&C(i+8,j+7), _mm512_add_pd(c71,_mm512_loadu_pd(&C(i+8,j+7))));\
    _mm512_storeu_pd(&C(i+16,j+7), _mm512_add_pd(c72,_mm512_loadu_pd(&C(i+16,j+7))));


void dgemm_packing_cache_blocking_reg_blocking_24x8_avx512_template_unrollx4(\
    int M, \
    int N, \
    int K, \
    double alpha, \
    double *A, \
    int LDA, \
    double *B, \
    int LDB, \
    double beta, \
    double *C, \
    int LDC)\
{
    int i,j,k;
    if (beta != 1.0) scale_c(C,M,N,LDC,beta);
    int M4,N4,K4;
    __m512d valpha = _mm512_set1_pd(alpha);//broadcast alpha to a 256-bit vector
    __m512d a,a0,a1,a2,b0,b1,b2,b3;
    __m512d c00,c01,c02,c10,c11,c12,c20,c21,c22,c30,c31,c32,c40,c41,c42,c50,c51,c52,c60,c61,c62,c70,c71,c72;
    __m512d c0,c1,c2,c3;
    double *ptr_packing_a,*ptr_packing_b;
    double sc0,sc1,sc2,sc3,sa,sb0,sb1,sb2,sb3;
    double *b_buffer = (double *)aligned_alloc(4096,K_BLOCKING*N_BLOCKING*sizeof(double));
    double *a_buffer = (double *)aligned_alloc(4096,K_BLOCKING*M_BLOCKING*sizeof(double));
    int M_MAIN = M/M_BLOCKING*M_BLOCKING,M_EDGE=M-M_MAIN;
    int N_MAIN = N/N_BLOCKING*N_BLOCKING,N_EDGE=N-N_MAIN;
    int K_MAIN = K/K_BLOCKING*K_BLOCKING,K_EDGE=K-K_MAIN;
    int m_count,n_count,k_count;
    int m_inc,n_inc,k_inc,k_start,k_end;
    for (k_count=0;k_count<K;k_count+=k_inc){
        //printf("k_count=%d\n",k_count);
        k_inc=(K-k_count>K_BLOCKING)?K_BLOCKING:K-k_count;
        for (n_count=0;n_count<N_MAIN;n_count+=N_BLOCKING){
            packing_b_24x8_version1(B+k_count+n_count*LDB,b_buffer,LDB,k_inc,N_BLOCKING);
            for (m_count=0;m_count<M_MAIN;m_count+=M_BLOCKING){
                packing_a_24x8(A+m_count+k_count*LDA,a_buffer,LDA,M_BLOCKING,k_inc);
                int inner_m_count,inner_n_count;
                for (inner_n_count=0;inner_n_count<N_BLOCKING;inner_n_count+=8){
                    for (inner_m_count=0;inner_m_count<M_BLOCKING;inner_m_count+=24){
                        i=m_count+inner_m_count;j=n_count+inner_n_count;
                        k_start=k_count;k_end=k_start+k_inc;K4=(k_start+k_inc)&-4;
                        ptr_packing_a=a_buffer+inner_m_count*k_inc;ptr_packing_b=b_buffer+inner_n_count*k_inc;
                        macro_kernel_24xkx8_packing_avx512_v1
                        //printf("m_count=%d\n",m_count);
                    }
                }
            }
            for (m_count=M_MAIN;m_count<M;m_count++){
                int inner_n_count;
                for (inner_n_count=0;inner_n_count<N_BLOCKING;inner_n_count+=4){
                    i=m_count;j=n_count+inner_n_count;
                    k_start=k_count;k_end=k_start+k_inc;
                    macro_kernel_1xkx4
                }
            }
        }
        for (n_count=N_MAIN;n_count<N;n_count++){
            for (m_count=0;m_count<M_MAIN;m_count+=M_BLOCKING){
                int inner_m_count;
                for (inner_m_count=0;inner_m_count<M_BLOCKING;inner_m_count+=8){
                    i=m_count+inner_m_count;j=n_count;
                    k_start=k_count;k_end=k_start+k_inc;K4=(k_start+k_inc)&-4;
                    macro_kernel_8xkx1_avx512
                }
            }
            for (m_count=M_MAIN;m_count<M;m_count++){
                i=m_count;j=n_count;
                k_start=k_count;k_end=k_start+k_inc;
                macro_kernel_1xkx1
            }
        }
    }
}



void dgemm_packing_cache_blocking_reg_blocking_24x8_avx512_template_unrollx4_v2(\
    int M, \
    int N, \
    int K, \
    double alpha, \
    double *A, \
    int LDA, \
    double *B, \
    int LDB, \
    double beta, \
    double *C, \
    int LDC)\
{
    int i,j,k;
    if (beta != 1.0) scale_c(C,M,N,LDC,beta);
    int M4,N8=N&-8,K4;
    __m512d valpha = _mm512_set1_pd(alpha);//broadcast alpha to a 256-bit vector
    __m512d a,a0,a1,a2,b0,b1,b2,b3;
    __m512d c00,c01,c02,c10,c11,c12,c20,c21,c22,c30,c31,c32,c40,c41,c42,c50,c51,c52,c60,c61,c62,c70,c71,c72;
    __m512d c0,c1,c2,c3;
    double *ptr_packing_a,*ptr_packing_b;
    double sc0,sc1,sc2,sc3,sa,sb0,sb1,sb2,sb3;
    double *b_buffer = (double *)aligned_alloc(4096,K_BLOCKING*N_BLOCKING*sizeof(double));
    double *a_buffer = (double *)aligned_alloc(4096,K_BLOCKING*M_BLOCKING*sizeof(double));
    int M_MAIN = M/M_BLOCKING*M_BLOCKING,M_EDGE=M-M_MAIN;
    int N_MAIN = N/N_BLOCKING*N_BLOCKING,N_EDGE=N-N_MAIN;
    int K_MAIN = K/K_BLOCKING*K_BLOCKING,K_EDGE=K-K_MAIN;
    int m_count,n_count,k_count;
    int m_inc,n_inc,k_inc,k_start,k_end;
    for (k_count=0;k_count<K;k_count+=k_inc){
        //printf("k_count=%d\n",k_count);
        k_inc=(K-k_count>K_BLOCKING)?K_BLOCKING:K-k_count;
        for (m_count=0;m_count<M_MAIN;m_count+=M_BLOCKING){
            packing_a_24x8(A+m_count+k_count*LDA,a_buffer,LDA,M_BLOCKING,k_inc);
            for (n_count=0;n_count<N_MAIN;n_count+=N_BLOCKING){
                packing_b_24x8_version1(B+k_count+n_count*LDB,b_buffer,LDB,k_inc,N_BLOCKING);
                int inner_m_count,inner_n_count;
                for (inner_n_count=0;inner_n_count<N_BLOCKING;inner_n_count+=8){
                    for (inner_m_count=0;inner_m_count<M_BLOCKING;inner_m_count+=24){
                        i=m_count+inner_m_count;j=n_count+inner_n_count;
                        k_start=k_count;k_end=k_start+k_inc;K4=(k_start+k_inc)&-4;
                        ptr_packing_a=a_buffer+inner_m_count*k_inc;ptr_packing_b=b_buffer+inner_n_count*k_inc;
                        macro_kernel_24xkx8_packing_avx512_v1
                        //printf("m_count=%d\n",m_count);
                    }
                }
            }
            if (N8-N_MAIN>0){
                packing_b_24x8_version1(B+k_count+N_MAIN*LDB,b_buffer,LDB,k_inc,N8-N_MAIN);
                for (n_count=N_MAIN;n_count<N8;n_count+=8){
                    int inner_m_count;
                    for (inner_m_count=0;inner_m_count<M_BLOCKING;inner_m_count+=24){
                        i=m_count+inner_m_count;j=n_count;
                        k_start=k_count;k_end=k_start+k_inc;K4=(k_start+k_inc)&-4;
                        ptr_packing_a=a_buffer+inner_m_count*k_inc;ptr_packing_b=b_buffer+(n_count-N_MAIN)*k_inc;
                        macro_kernel_24xkx8_packing_avx512_v1
                    }
                }
            }
            if (N-N8>0){
                for (n_count=N8;n_count<N;n_count++){
                    int inner_m_count;
                    for (inner_m_count=0;inner_m_count<M_BLOCKING;inner_m_count+=8){
                        i=m_count+inner_m_count;j=n_count;
                        k_start=k_count;k_end=k_start+k_inc;K4=(k_start+k_inc)&-4;
                        macro_kernel_8xkx1_avx512
                    }
                }
            }

        }

        for (m_count=M_MAIN;m_count<M;m_count++){
            for (n_count=0;n_count<N_MAIN;n_count+=N_BLOCKING){
                int inner_n_count;
                for (inner_n_count=0;inner_n_count<N_BLOCKING;inner_n_count+=4){
                    i=m_count;j=n_count+inner_n_count;
                    k_start=k_count;k_end=k_start+k_inc;
                    macro_kernel_1xkx4
                }
            }
            for (n_count=N_MAIN;n_count<N;n_count++){
                i=m_count;j=n_count;
                k_start=k_count;k_end=k_start+k_inc;
                macro_kernel_1xkx1
            }
        }
    }
}

#define OUT_M_BLOCKING 1152
#define OUT_N_BLOCKING 9216


void packing_a_24x8_edge(double *src, double *dst, int leading_dim, int dim_first, int dim_second){
    //dim_first: M, dim_second: K
    double *tosrc,*todst;
    todst=dst;
    int count_first,count_second,count_sub=dim_first;
    for (count_first=0;count_sub>23;count_first+=24,count_sub-=24){
        tosrc=src+count_first;
        for(count_second=0;count_second<dim_second;count_second++){
            _mm512_store_pd(todst,_mm512_loadu_pd(tosrc));
            _mm512_store_pd(todst+8,_mm512_loadu_pd(tosrc+8));
            _mm512_store_pd(todst+16,_mm512_loadu_pd(tosrc+16));
            tosrc+=leading_dim;
            todst+=24;
        }
    }
    // edge case
    for (;count_sub>7;count_first+=8,count_sub-=8){
        tosrc=src+count_first;
        for(count_second=0;count_second<dim_second;count_second++){
            _mm512_store_pd(todst,_mm512_loadu_pd(tosrc));
            tosrc+=leading_dim;
            todst+=8;
        }
    }
    for (;count_sub>1;count_first+=2,count_sub-=2){
        tosrc=src+count_first;
        for(count_second=0;count_second<dim_second;count_second++){
            _mm_store_pd(todst,_mm_loadu_pd(tosrc));
            tosrc+=leading_dim;
            todst+=2;
        }
    }
    for (;count_sub>0;count_first+=1,count_sub-=1){
        tosrc=src+count_first;
        for(count_second=0;count_second<dim_second;count_second++){
            *todst=*tosrc;
            tosrc+=leading_dim;
            todst++;
        }
    }
}

void packing_b_24x8_edge_version1(double *src,double *dst,int leading_dim,int dim_first,int dim_second){
    //dim_first:K,dim_second:N
    double *tosrc1,*tosrc2,*tosrc3,*tosrc4,*tosrc5,*tosrc6,*tosrc7,*tosrc8,*todst;
    todst=dst;
    int count_first,count_second,count_sub=dim_second;
    for (count_second=0;count_sub>7;count_second+=8,count_sub-=8){
        tosrc1=src+count_second*leading_dim;tosrc2=tosrc1+leading_dim;
        tosrc3=tosrc2+leading_dim;tosrc4=tosrc3+leading_dim;
        tosrc5=tosrc4+leading_dim;tosrc6=tosrc5+leading_dim;
        tosrc7=tosrc6+leading_dim;tosrc8=tosrc7+leading_dim;
        for (count_first=0;count_first<dim_first;count_first++){
            *todst=*tosrc1;tosrc1++;todst++;
            *todst=*tosrc2;tosrc2++;todst++;
            *todst=*tosrc3;tosrc3++;todst++;
            *todst=*tosrc4;tosrc4++;todst++;
            *todst=*tosrc5;tosrc5++;todst++;
            *todst=*tosrc6;tosrc6++;todst++;
            *todst=*tosrc7;tosrc7++;todst++;
            *todst=*tosrc8;tosrc8++;todst++;
        }
    }
    for (;count_sub>3;count_second+=4,count_sub-=4){
        tosrc1=src+count_second*leading_dim;tosrc2=tosrc1+leading_dim;
        tosrc3=tosrc2+leading_dim;tosrc4=tosrc3+leading_dim;
        for (count_first=0;count_first<dim_first;count_first++){
            *todst=*tosrc1;tosrc1++;todst++;
            *todst=*tosrc2;tosrc2++;todst++;
            *todst=*tosrc3;tosrc3++;todst++;
            *todst=*tosrc4;tosrc4++;todst++;
        }
    }
    for (;count_sub>1;count_second+=2,count_sub-=2){
        tosrc1=src+count_second*leading_dim;tosrc2=tosrc1+leading_dim;
        for (count_first=0;count_first<dim_first;count_first++){
            *todst=*tosrc1;tosrc1++;todst++;
            *todst=*tosrc2;tosrc2++;todst++;
        }
    }
    for (;count_sub>0;count_second++,count_sub-=1){
        tosrc1=src+count_second*leading_dim;
        for (count_first=0;count_first<dim_first;count_first++){
            *todst=*tosrc1;tosrc1++;todst++;
        }
    }
}



void kernel_n_8(double *a_buffer,double *b_buffer,double *c_ptr,int m,int K,int LDC,double alpha){
    int m_count,m_count_sub;
    int i,j,k;
    double *C=c_ptr;
    __m512d valpha = _mm512_set1_pd(alpha);//broadcast alpha to a 256-bit vector
    __m512d a,a0,a1,a2,b0,b1,b2,b3;
    __m512d c00,c01,c02,c10,c11,c12,c20,c21,c22,c30,c31,c32,c40,c41,c42,c50,c51,c52,c60,c61,c62,c70,c71,c72;
    __m512d c0,c1,c2,c3;
    double *ptr_packing_a,*ptr_packing_b;
    int k_start,k_end,K4;
    K4=K&-4;k_end=K;k_start=0;
    // printf("*****\n");
    // print_matrix(C,m,8);
    // printf("*****\n");
    for (m_count_sub=m,m_count=0;m_count_sub>23;m_count_sub-=24,m_count+=24){
        //call the micro kernel: m24n8;
        i=m_count;j=0;ptr_packing_a=a_buffer+m_count*K;ptr_packing_b=b_buffer;
        macro_kernel_24xkx8_packing_avx512_v1
    }
    for (;m_count_sub>7;m_count_sub-=8,m_count+=8){
        //call the micro kernel: m8n8;
    }
    for (;m_count_sub>1;m_count_sub-=2,m_count+=2){
        //call the micro kernel: m2n8;
    }
    for (;m_count_sub>0;m_count_sub-=1,m_count+=1){
        //call the micro kernel: m1n8;
    }
}

void kernel_n_4(double *a_buffer,double *b_buffer,double *C,int m,int K,int LDC){
    int m_count,m_count_sub;
    for (m_count_sub=m,m_count_sub=0;m_count_sub>23;m_count_sub-=24,m_count+=24){
        //call the micro kernel: m24n4;
    }
    for (;m_count_sub>7;m_count_sub-=8,m_count+=8){
        //call the micro kernel: m8n4;
    }
    for (;m_count_sub>1;m_count_sub-=2,m_count+=2){
        //call the micro kernel: m2n4;
    }
    for (;m_count_sub>0;m_count_sub-=1,m_count+=1){
        //call the micro kernel: m1n4;
    }
}

void kernel_n2(double *a_buffer,double *b_buffer,double *C,int m,int K,int LDC){
    int m_count,m_count_sub;
    for (m_count_sub=m,m_count_sub=0;m_count_sub>23;m_count_sub-=24,m_count+=24){
        //call the micro kernel: m24n2;
    }
    for (;m_count_sub>7;m_count_sub-=8,m_count+=8){
        //call the micro kernel: m8n2;
    }
    for (;m_count_sub>1;m_count_sub-=2,m_count+=2){
        //call the micro kernel: m2n2;
    }
    for (;m_count_sub>0;m_count_sub-=1,m_count+=1){
        //call the micro kernel: m1n2;
    }
}

void kernel_n1(double *a_buffer,double *b_buffer,double *C,double m,double k,int LDC){
    int m_count,m_count_sub;
    for (m_count_sub=m,m_count_sub=0;m_count_sub>23;m_count_sub-=24,m_count+=24){
        //call the micro kernel: m24n1;
    }
    for (;m_count_sub>7;m_count_sub-=8,m_count+=8){
        //call the micro kernel: m8n1;
    }
    for (;m_count_sub>1;m_count_sub-=2,m_count+=2){
        //call the micro kernel: m2n1;
    }
    for (;m_count_sub>0;m_count_sub-=1,m_count+=1){
        //call the micro kernel: m1n1;
    }
}

void macro_kernel(double *a_buffer,double *b_buffer,int m,int n,int k,double *C, int LDC,int k_inc,double alpha){
    int m_count,n_count,m_count_sub,n_count_sub;
    // printf("m= %d, n=%d, k = %d\n",m,n,k);
    for (n_count_sub=n,n_count=0;n_count_sub>7;n_count_sub-=8,n_count+=8){
        //call the m layer with n=8;
        kernel_n_8(a_buffer,b_buffer+n_count*k_inc,C+n_count*LDC,m,k_inc,LDC,alpha);
    }
    for (;n_count_sub>3;n_count_sub-=4,n_count+=4){
        //call the m layer with n=4
    }
    for (;n_count_sub>1;n_count_sub-=2,n_count+=2){
        //call the m layer with n=2
    }
    for (;n_count_sub>0;n_count_sub-=1,n_count+=1){
        //call the m layer with n=1
    }
}

/*
 * fuse edge cases into packing routine
 * 
 **/
void dgemm_packing_cache_blocking_reg_blocking_24x8_avx512_template_unrollx4_v3(\
    int M, \
    int N, \
    int K, \
    double alpha, \
    double *A, \
    int LDA, \
    double *B, \
    int LDB, \
    double beta, \
    double *C, \
    int LDC)\
{
    int i,j,k;
    if (beta != 1.0) scale_c(C,M,N,LDC,beta);
    if (alpha == 0.||K==0) return;
    int M4,N8=N&-8,K4;
    
    double sc0,sc1,sc2,sc3,sa,sb0,sb1,sb2,sb3;
    double *b_buffer = (double *)aligned_alloc(4096,K_BLOCKING*OUT_N_BLOCKING*sizeof(double));
    double *a_buffer = (double *)aligned_alloc(4096,K_BLOCKING*OUT_M_BLOCKING*sizeof(double));
    int second_m_count,second_n_count,second_m_inc,second_n_inc;
    int m_count,n_count,k_count;
    int m_inc,n_inc,k_inc;
    for (k_count=0;k_count<K;k_count+=k_inc){
        k_inc=(K-k_count>K_BLOCKING)?K_BLOCKING:K-k_count;
        for (n_count=0;n_count<N;n_count+=n_inc){
            n_inc=(N-n_count>OUT_N_BLOCKING)?OUT_N_BLOCKING:N-n_count;
            packing_b_24x8_edge_version1(B+k_count+n_count*LDB,b_buffer,LDB,k_inc,n_inc);
            //print_matrix(b_buffer,k_inc,n_inc);
            for (m_count=0;m_count<M;m_count+=m_inc){
                m_inc=(M-m_count>OUT_M_BLOCKING)?OUT_M_BLOCKING:M-m_count;
                packing_a_24x8_edge(A+m_count+k_count*LDA,a_buffer,LDA,m_inc,k_inc);
                //print_matrix(a_buffer,m_inc,k_inc);
                for (second_m_count=m_count;second_m_count<m_count+m_inc;second_m_count+=second_m_inc){
                    second_m_inc=(m_count+m_inc-second_m_count>M_BLOCKING)?M_BLOCKING:m_count+m_inc-second_m_count;
                    for (second_n_count=n_count;second_n_count<n_count+n_inc;second_n_count+=second_n_inc){
                        second_n_inc=(n_count+n_inc-second_n_count>N_BLOCKING)?N_BLOCKING:n_count+n_inc-second_n_count;
                        macro_kernel(a_buffer+(second_m_count-m_count)*k_inc,b_buffer+(second_n_count-n_count)*k_inc,second_m_inc,second_n_inc,k_inc,&C(second_m_count,second_n_count),LDC,k_inc,alpha);
                        //printf("m=%d,m_inc=%d,n=%d,n_inc=%d\n",second_m_count,second_m_inc,second_n_count,second_n_inc);
                    }
                }
            }
        }
    }
    free(a_buffer);free(b_buffer);
}