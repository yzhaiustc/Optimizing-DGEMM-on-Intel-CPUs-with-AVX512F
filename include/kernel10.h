#include "immintrin.h"
#define A(i,j) A[(i)+(j)*LDA]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]
#define M_BLOCKING 192
#define N_BLOCKING 2048
#define K_BLOCKING 384
void scale_c_k10(double *C,int M, int N, int LDC, double scalar){
    int i,j;
    for (i=0;i<M;i++){
        for (j=0;j<N;j++){
            C(i,j)*=scalar;
        }
    }
}

void packing_a_k10(double *src, double *dst, int leading_dim, int dim_first, int dim_second){
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

void packing_b_k10(double *src,double *dst,int leading_dim,int dim_first,int dim_second){
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

#define KERNEL_K1_8x8_avx512_intrinsics_packing\
    a0 = _mm512_mul_pd(valpha, _mm512_load_pd(ptr_packing_a));\
    b0 = _mm512_set1_pd(*ptr_packing_b);\
    b1 = _mm512_set1_pd(*(ptr_packing_b+1));\
    c00 = _mm512_fmadd_pd(a0,b0,c00);\
    c10 = _mm512_fmadd_pd(a0,b1,c10);\
    b0 = _mm512_set1_pd(*(ptr_packing_b+2));\
    b1 = _mm512_set1_pd(*(ptr_packing_b+3));\
    c20 = _mm512_fmadd_pd(a0,b0,c20);\
    c30 = _mm512_fmadd_pd(a0,b1,c30);\
    b0 = _mm512_set1_pd(*(ptr_packing_b+4));\
    b1 = _mm512_set1_pd(*(ptr_packing_b+5));\
    c40 = _mm512_fmadd_pd(a0,b0,c40);\
    c50 = _mm512_fmadd_pd(a0,b1,c50);\
    b0 = _mm512_set1_pd(*(ptr_packing_b+6));\
    b1 = _mm512_set1_pd(*(ptr_packing_b+7));\
    c60 = _mm512_fmadd_pd(a0,b0,c60);\
    c70 = _mm512_fmadd_pd(a0,b1,c70);\
    ptr_packing_a+=8;ptr_packing_b+=8;k++;

#define macro_kernel_8xkx8_packing_avx512_v1\
    c00 = _mm512_setzero_pd();\
    c10 = _mm512_setzero_pd();\
    c20 = _mm512_setzero_pd();\
    c30 = _mm512_setzero_pd();\
    c40 = _mm512_setzero_pd();\
    c50 = _mm512_setzero_pd();\
    c60 = _mm512_setzero_pd();\
    c70 = _mm512_setzero_pd();\
    for (k=k_start;k<K4;){\
        KERNEL_K1_8x8_avx512_intrinsics_packing\
        KERNEL_K1_8x8_avx512_intrinsics_packing\
        KERNEL_K1_8x8_avx512_intrinsics_packing\
        KERNEL_K1_8x8_avx512_intrinsics_packing\
    }\
    for (k=K4;k<k_end;){\
        KERNEL_K1_8x8_avx512_intrinsics_packing\
    }\
    _mm512_storeu_pd(&C(i,j), _mm512_add_pd(c00,_mm512_loadu_pd(&C(i,j))));\
    _mm512_storeu_pd(&C(i,j+1), _mm512_add_pd(c10,_mm512_loadu_pd(&C(i,j+1))));\
    _mm512_storeu_pd(&C(i,j+2), _mm512_add_pd(c20,_mm512_loadu_pd(&C(i,j+2))));\
    _mm512_storeu_pd(&C(i,j+3), _mm512_add_pd(c30,_mm512_loadu_pd(&C(i,j+3))));\
    _mm512_storeu_pd(&C(i,j+4), _mm512_add_pd(c40,_mm512_loadu_pd(&C(i,j+4))));\
    _mm512_storeu_pd(&C(i,j+5), _mm512_add_pd(c50,_mm512_loadu_pd(&C(i,j+5))));\
    _mm512_storeu_pd(&C(i,j+6), _mm512_add_pd(c60,_mm512_loadu_pd(&C(i,j+6))));\
    _mm512_storeu_pd(&C(i,j+7), _mm512_add_pd(c70,_mm512_loadu_pd(&C(i,j+7))));

#define KERNEL_K1_2x8_avx512_intrinsics_packing\
    da0 = _mm_mul_pd(dvalpha, _mm_load_pd(ptr_packing_a));\
    db0 = _mm_set1_pd(*ptr_packing_b);\
    db1 = _mm_set1_pd(*(ptr_packing_b+1));\
    dc00 = _mm_fmadd_pd(da0,db0,dc00);\
    dc10 = _mm_fmadd_pd(da0,db1,dc10);\
    db0 = _mm_set1_pd(*(ptr_packing_b+2));\
    db1 = _mm_set1_pd(*(ptr_packing_b+3));\
    dc20 = _mm_fmadd_pd(da0,db0,dc20);\
    dc30 = _mm_fmadd_pd(da0,db1,dc30);\
    db0 = _mm_set1_pd(*(ptr_packing_b+4));\
    db1 = _mm_set1_pd(*(ptr_packing_b+5));\
    dc40 = _mm_fmadd_pd(da0,db0,dc40);\
    dc50 = _mm_fmadd_pd(da0,db1,dc50);\
    db0 = _mm_set1_pd(*(ptr_packing_b+6));\
    db1 = _mm_set1_pd(*(ptr_packing_b+7));\
    dc60 = _mm_fmadd_pd(da0,db0,dc60);\
    dc70 = _mm_fmadd_pd(da0,db1,dc70);\
    ptr_packing_a+=2;ptr_packing_b+=8;k++;

#define macro_kernel_2xkx8_packing_avx512_v1\
    dc00 = _mm_setzero_pd();\
    dc10 = _mm_setzero_pd();\
    dc20 = _mm_setzero_pd();\
    dc30 = _mm_setzero_pd();\
    dc40 = _mm_setzero_pd();\
    dc50 = _mm_setzero_pd();\
    dc60 = _mm_setzero_pd();\
    dc70 = _mm_setzero_pd();\
    for (k=k_start;k<K4;){\
        KERNEL_K1_2x8_avx512_intrinsics_packing\
        KERNEL_K1_2x8_avx512_intrinsics_packing\
        KERNEL_K1_2x8_avx512_intrinsics_packing\
        KERNEL_K1_2x8_avx512_intrinsics_packing\
    }\
    for (k=K4;k<k_end;){\
        KERNEL_K1_2x8_avx512_intrinsics_packing\
    }\
    _mm_storeu_pd(&C(i,j), _mm_add_pd(dc00,_mm_loadu_pd(&C(i,j))));\
    _mm_storeu_pd(&C(i,j+1), _mm_add_pd(dc10,_mm_loadu_pd(&C(i,j+1))));\
    _mm_storeu_pd(&C(i,j+2), _mm_add_pd(dc20,_mm_loadu_pd(&C(i,j+2))));\
    _mm_storeu_pd(&C(i,j+3), _mm_add_pd(dc30,_mm_loadu_pd(&C(i,j+3))));\
    _mm_storeu_pd(&C(i,j+4), _mm_add_pd(dc40,_mm_loadu_pd(&C(i,j+4))));\
    _mm_storeu_pd(&C(i,j+5), _mm_add_pd(dc50,_mm_loadu_pd(&C(i,j+5))));\
    _mm_storeu_pd(&C(i,j+6), _mm_add_pd(dc60,_mm_loadu_pd(&C(i,j+6))));\
    _mm_storeu_pd(&C(i,j+7), _mm_add_pd(dc70,_mm_loadu_pd(&C(i,j+7))));

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

#define KERNEL_K1_24x4_avx512_intrinsics_packing\
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
    ptr_packing_a+=24;ptr_packing_b+=4;k++;

#define macro_kernel_24xkx4_packing_avx512_v1\
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
    for (k=k_start;k<K4;){\
        KERNEL_K1_24x4_avx512_intrinsics_packing\
        KERNEL_K1_24x4_avx512_intrinsics_packing\
        KERNEL_K1_24x4_avx512_intrinsics_packing\
        KERNEL_K1_24x4_avx512_intrinsics_packing\
    }\
    for (k=K4;k<k_end;){\
        KERNEL_K1_24x4_avx512_intrinsics_packing\
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
    _mm512_storeu_pd(&C(i+16,j+3), _mm512_add_pd(c32,_mm512_loadu_pd(&C(i+16,j+3))));

#define KERNEL_K1_4x8_avx512_intrinsics_packing\
    a0 = _mm512_mul_pd(valpha, _mm512_load_pd(ptr_packing_a));\
    b0 = _mm512_set1_pd(*ptr_packing_b);\
    b1 = _mm512_set1_pd(*(ptr_packing_b+1));\
    c00 = _mm512_fmadd_pd(a0,b0,c00);\
    c10 = _mm512_fmadd_pd(a0,b1,c10);\
    b0 = _mm512_set1_pd(*(ptr_packing_b+2));\
    b1 = _mm512_set1_pd(*(ptr_packing_b+3));\
    c20 = _mm512_fmadd_pd(a0,b0,c20);\
    c30 = _mm512_fmadd_pd(a0,b1,c30);\
    ptr_packing_a+=8;ptr_packing_b+=4;k++;

#define macro_kernel_8xkx4_packing_avx512_v1\
    c00 = _mm512_setzero_pd();\
    c10 = _mm512_setzero_pd();\
    c20 = _mm512_setzero_pd();\
    c30 = _mm512_setzero_pd();\
    for (k=k_start;k<K4;){\
        KERNEL_K1_4x8_avx512_intrinsics_packing\
        KERNEL_K1_4x8_avx512_intrinsics_packing\
        KERNEL_K1_4x8_avx512_intrinsics_packing\
        KERNEL_K1_4x8_avx512_intrinsics_packing\
    }\
    for (k=K4;k<k_end;){\
        KERNEL_K1_4x8_avx512_intrinsics_packing\
    }\
    _mm512_storeu_pd(&C(i,j), _mm512_add_pd(c00,_mm512_loadu_pd(&C(i,j))));\
    _mm512_storeu_pd(&C(i,j+1), _mm512_add_pd(c10,_mm512_loadu_pd(&C(i,j+1))));\
    _mm512_storeu_pd(&C(i,j+2), _mm512_add_pd(c20,_mm512_loadu_pd(&C(i,j+2))));\
    _mm512_storeu_pd(&C(i,j+3), _mm512_add_pd(c30,_mm512_loadu_pd(&C(i,j+3))));

#define KERNEL_K1_2x4_avx512_intrinsics_packing\
    da0 = _mm_mul_pd(dvalpha, _mm_load_pd(ptr_packing_a));\
    db0 = _mm_set1_pd(*ptr_packing_b);\
    db1 = _mm_set1_pd(*(ptr_packing_b+1));\
    dc00 = _mm_fmadd_pd(da0,db0,dc00);\
    dc10 = _mm_fmadd_pd(da0,db1,dc10);\
    db0 = _mm_set1_pd(*(ptr_packing_b+2));\
    db1 = _mm_set1_pd(*(ptr_packing_b+3));\
    dc20 = _mm_fmadd_pd(da0,db0,dc20);\
    dc30 = _mm_fmadd_pd(da0,db1,dc30);\
    ptr_packing_a+=2;ptr_packing_b+=4;k++;

#define macro_kernel_2xkx4_packing_avx512_v1\
    dc00 = _mm_setzero_pd();\
    dc10 = _mm_setzero_pd();\
    dc20 = _mm_setzero_pd();\
    dc30 = _mm_setzero_pd();\
    for (k=k_start;k<K4;){\
        KERNEL_K1_2x4_avx512_intrinsics_packing\
        KERNEL_K1_2x4_avx512_intrinsics_packing\
        KERNEL_K1_2x4_avx512_intrinsics_packing\
        KERNEL_K1_2x4_avx512_intrinsics_packing\
    }\
    for (k=K4;k<k_end;){\
        KERNEL_K1_2x4_avx512_intrinsics_packing\
    }\
    _mm_storeu_pd(&C(i,j), _mm_add_pd(dc00,_mm_loadu_pd(&C(i,j))));\
    _mm_storeu_pd(&C(i,j+1), _mm_add_pd(dc10,_mm_loadu_pd(&C(i,j+1))));\
    _mm_storeu_pd(&C(i,j+2), _mm_add_pd(dc20,_mm_loadu_pd(&C(i,j+2))));\
    _mm_storeu_pd(&C(i,j+3), _mm_add_pd(dc30,_mm_loadu_pd(&C(i,j+3))));


void kernel_n_8_k10(double *a_buffer,double *b_buffer,double *c_ptr,int m,int K,int LDC,double alpha){
    int m_count,m_count_sub;
    int i,j,k;
    double *C=c_ptr;
    double sc0,sc1,sc2,sc3,sc4,sc5,sc6,sc7,sa,sb0,sb1,sb2,sb3,sb4,sb5,sb6,sb7;
    __m128d da,da0,da1,da2,db0,db1,db2,db3;
    __m128d dc00,dc10,dc20,dc30,dc40,dc50,dc60,dc70;
    __m512d valpha = _mm512_set1_pd(alpha);//broadcast alpha to a 512-bit vector
    __m128d dvalpha = _mm_set1_pd(alpha);//broadcast alpha to a 128-bit vector
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
        i=m_count;j=0;ptr_packing_a=a_buffer+m_count*K;ptr_packing_b=b_buffer;
        macro_kernel_8xkx8_packing_avx512_v1
    }
    for (;m_count_sub>1;m_count_sub-=2,m_count+=2){
        //call the micro kernel: m2n8;
        i=m_count;j=0;ptr_packing_a=a_buffer+m_count*K;ptr_packing_b=b_buffer;
        macro_kernel_2xkx8_packing_avx512_v1
    }
    for (;m_count_sub>0;m_count_sub-=1,m_count+=1){
        //call the micro kernel: m1n8;
        i=m_count;j=0;ptr_packing_a=a_buffer+m_count*K;ptr_packing_b=b_buffer;
        macro_packing_kernel_1xkx8
    }
}

void kernel_n_4_k10(double *a_buffer,double *b_buffer,double *c_ptr,int m,int K,int LDC,double alpha){
    int m_count,m_count_sub;
    int i,j,k;
    double *C=c_ptr;
    double sc0,sc1,sc2,sc3,sc4,sc5,sc6,sc7,sa,sb0,sb1,sb2,sb3,sb4,sb5,sb6,sb7;
    __m128d da,da0,da1,da2,db0,db1,db2,db3;
    __m128d dc00,dc10,dc20,dc30,dc40,dc50,dc60,dc70;
    __m512d valpha = _mm512_set1_pd(alpha);//broadcast alpha to a 512-bit vector
    __m128d dvalpha = _mm_set1_pd(alpha);//broadcast alpha to a 128-bit vector
    __m512d a,a0,a1,a2,b0,b1,b2,b3;
    __m512d c00,c01,c02,c10,c11,c12,c20,c21,c22,c30,c31,c32,c40,c41,c42,c50,c51,c52,c60,c61,c62,c70,c71,c72;
    __m512d c0,c1,c2,c3;
    double *ptr_packing_a,*ptr_packing_b;
    int k_start,k_end,K4;
    K4=K&-4;k_end=K;k_start=0;
    for (m_count_sub=m,m_count=0;m_count_sub>23;m_count_sub-=24,m_count+=24){
        //call the micro kernel: m24n4;
        i=m_count;j=0;ptr_packing_a=a_buffer+m_count*K;ptr_packing_b=b_buffer;
        macro_kernel_24xkx4_packing_avx512_v1
    }
    for (;m_count_sub>7;m_count_sub-=8,m_count+=8){
        //call the micro kernel: m8n4;
        i=m_count;j=0;ptr_packing_a=a_buffer+m_count*K;ptr_packing_b=b_buffer;
        macro_kernel_8xkx4_packing_avx512_v1
    }
    for (;m_count_sub>1;m_count_sub-=2,m_count+=2){
        //call the micro kernel: m2n4;
        i=m_count;j=0;ptr_packing_a=a_buffer+m_count*K;ptr_packing_b=b_buffer;
        macro_kernel_2xkx4_packing_avx512_v1
    }
    for (;m_count_sub>0;m_count_sub-=1,m_count+=1){
        //call the micro kernel: m1n4;
    }
}

void macro_kernel_k10(double *a_buffer,double *b_buffer,int m,int n,int k,double *C, int LDC,double alpha){
    int m_count,n_count,m_count_sub,n_count_sub;
    // printf("m= %d, n=%d, k = %d\n",m,n,k);
    for (n_count_sub=n,n_count=0;n_count_sub>7;n_count_sub-=8,n_count+=8){
        //call the m layer with n=8;
        kernel_n_8_k10(a_buffer,b_buffer+n_count*k,C+n_count*LDC,m,k,LDC,alpha);
    }
    for (;n_count_sub>3;n_count_sub-=4,n_count+=4){
        //call the m layer with n=4
        kernel_n_4_k10(a_buffer,b_buffer+n_count*k,C+n_count*LDC,m,k,LDC,alpha);
    }
    for (;n_count_sub>1;n_count_sub-=2,n_count+=2){
        //call the m layer with n=2
    }
    for (;n_count_sub>0;n_count_sub-=1,n_count+=1){
        //call the m layer with n=1
    }
}

void mydgemm_cpu_v10(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    
    if (beta != 1.0) scale_c_k10(C,M,N,LDC,beta);
    double *b_buffer = (double *)aligned_alloc(4096,K_BLOCKING*N_BLOCKING*sizeof(double));
    double *a_buffer = (double *)aligned_alloc(4096,K_BLOCKING*M_BLOCKING*sizeof(double));
    int m_count, n_count, k_count;
    int m_inc, n_inc, k_inc;
    for (n_count=0;n_count<N;n_count+=n_inc){
        n_inc = (N-n_count>N_BLOCKING)?N_BLOCKING:N-n_count;
        for (k_count=0;k_count<K;k_count+=k_inc){
            k_inc = (K-k_count>K_BLOCKING)?K_BLOCKING:K-k_count;
            packing_b_k10(B+k_count+n_count*LDB,b_buffer,LDB,k_inc,n_inc);
            for (m_count=0;m_count<M;m_count+=m_inc){
                m_inc = (M-m_count>M_BLOCKING)?M_BLOCKING:N-m_count;
                packing_a_k10(A+m_count+k_count*LDA,a_buffer,LDA,m_inc,k_inc);
                //macro kernel: to compute C += A_tilt * B_tilt
                macro_kernel_k10(a_buffer, b_buffer, m_inc, n_inc, k_inc, &C(m_count, n_count), LDC, alpha);
            }
        }
    }
    free(a_buffer);free(b_buffer);
}
