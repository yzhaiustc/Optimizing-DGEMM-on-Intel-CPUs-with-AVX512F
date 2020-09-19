#include "immintrin.h"
#define A(i,j) A[(i)+(j)*LDA]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]

void scale_c_k8(double *C,int M, int N, int LDC, double scalar){
    int i,j;
    for (i=0;i<M;i++){
        for (j=0;j<N;j++){
            C(i,j)*=scalar;
        }
    }
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


#define macro_packing_kernel_1xkx8\
    sc0=sc1=sc2=sc3=sc4=sc5=sc6=sc7=0.;\
    for (k=k_start;k<k_end;k++){\
        sa=alpha*(*ptr_packing_a);\
        sb0=*(ptr_packing_b);sb1=*(ptr_packing_b+1);sb2=*(ptr_packing_b+2);sb3=*(ptr_packing_b+3);\
        sb4=*(ptr_packing_b+4);sb5=*(ptr_packing_b+5);sb6=*(ptr_packing_b+6);sb7=*(ptr_packing_b+7);\
        sc0+=sa*sb0;sc1+=sa*sb1;sc2+=sa*sb2;sc3+=sa*sb3;\
        sc4+=sa*sb4;sc5+=sa*sb5;sc6+=sa*sb6;sc7+=sa*sb7;\
        ptr_packing_a++;ptr_packing_b+=8;\
    }\
    C(i,j)+=sc0;C(i,j+1)+=sc1;C(i,j+2)+=sc2;C(i,j+3)+=sc3;\
    C(i,j+4)+=sc4;C(i,j+5)+=sc5;C(i,j+6)+=sc6;C(i,j+7)+=sc7;

#define macro_packing_kernel_1xkx8_v2\
    sc0=sc1=sc2=sc3=sc4=sc5=sc6=sc7=0.;\
    for (k=k_start;k<k_end;k++){\
        sa=alpha*(*ptr_packing_a);\
        sb0=*(ptr_packing_b0);sb1=*(ptr_packing_b0+1);sb2=*(ptr_packing_b1);sb3=*(ptr_packing_b1+1);\
        sb4=*(ptr_packing_b2);sb5=*(ptr_packing_b2+1);sb6=*(ptr_packing_b3);sb7=*(ptr_packing_b3+1);\
        sc0+=sa*sb0;sc1+=sa*sb1;sc2+=sa*sb2;sc3+=sa*sb3;\
        sc4+=sa*sb4;sc5+=sa*sb5;sc6+=sa*sb6;sc7+=sa*sb7;\
        ptr_packing_a++;ptr_packing_b0+=2;\
        ptr_packing_b1+=2;ptr_packing_b2+=2;ptr_packing_b3+=2;\
    }\
    C(i,j)+=sc0;C(i,j+1)+=sc1;C(i,j+2)+=sc2;C(i,j+3)+=sc3;\
    C(i,j+4)+=sc4;C(i,j+5)+=sc5;C(i,j+6)+=sc6;C(i,j+7)+=sc7;

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

void mydgemm_cpu_v8(\
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
    if (beta != 1.0) scale_c_k8(C,M,N,LDC,beta);
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