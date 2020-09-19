#include "immintrin.h"
#define A(i,j) A[(i)+(j)*LDA]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]
#define M_BLOCKING 192
#define N_BLOCKING 96
#define K_BLOCKING 384

void scale_c_k10(double *C,int M, int N, int LDC, double scalar){
    int i,j;
    for (i=0;i<M;i++){
        for (j=0;j<N;j++){
            C(i,j)*=scalar;
        }
    }
}

void packing_b_24x8_version1_k10(double *src,double *dst,int leading_dim,int dim_first,int dim_second){
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

void packing_a_24x8_k10(double *src, double *dst, int leading_dim, int dim_first, int dim_second){
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

#define macro_kernel_1xkx4\
    sc0=sc1=sc2=sc3=0.;\
    for (k=k_start;k<k_end;k++){\
        sa=alpha*A(i,k);\
        sb0=B(k,j);sb1=B(k,j+1);sb2=B(k,j+2);sb3=B(k,j+3);\
        sc0+=sa*sb0;sc1+=sa*sb1;sc2+=sa*sb2;sc3+=sa*sb3;\
    }\
    C(i,j)+=sc0;C(i,j+1)+=sc1;C(i,j+2)+=sc2;C(i,j+3)+=sc3;



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

#define macro_kernel_1xkx1\
    sc0=0.;\
    for (k=k_start;k<k_end;k++){\
        sa=alpha*A(i,k);\
        sb0=B(k,j);\
        sc0+=sa*sb0;\
    }\
    C(i,j)+=sc0;

void mydgemm_cpu_v10(\
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
    if (beta != 1.0) scale_c_k10(C,M,N,LDC,beta);
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
            packing_b_24x8_version1_k10(B+k_count+n_count*LDB,b_buffer,LDB,k_inc,N_BLOCKING);
            for (m_count=0;m_count<M_MAIN;m_count+=M_BLOCKING){
                packing_a_24x8_k10(A+m_count+k_count*LDA,a_buffer,LDA,M_BLOCKING,k_inc);
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