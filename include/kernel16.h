//#define TIMER 1
#include "immintrin.h"
#include <stdint.h>
#ifdef TIMER
  #include "utils.h"
#endif
#define A(i,j) A[(i)+(j)*LDA]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]
#define OUT_M_BLOCKING 1152
#define OUT_N_BLOCKING 9216
#define M_BLOCKING 192
#define N_BLOCKING 96
#define K_BLOCKING 384


void scale_c_k16(double *C,int M, int N, int LDC, double scalar){
    int i,j;
    for (i=0;i<M;i++){
        for (j=0;j<N;j++){
            C(i,j)*=scalar;
        }
    }
}

void packing_b_24x8_edge_version2_k16(double *src,double *dst,int leading_dim,int dim_first,int dim_second){
    //dim_first:K,dim_second:N
    double *tosrc1,*tosrc2,*todst;
    todst=dst;
    int count_first,count_second,count_sub=dim_second;
    for (count_second=0;count_sub>1;count_second+=2,count_sub-=2){
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

void packing_a_24x8_edge_k16(double alpha,double *src, double *dst, int leading_dim, int dim_first, int dim_second){
    //dim_first: M, dim_second: K
    double *tosrc,*todst;
    todst=dst;
    int count_first,count_second,count_sub=dim_first;
    __m512d valpha=_mm512_set1_pd(alpha);
    __m128d valpha_128=_mm_set1_pd(alpha);
    for (count_first=0;count_sub>23;count_first+=24,count_sub-=24){
        tosrc=src+count_first;
        for(count_second=0;count_second<dim_second;count_second++){
            _mm512_store_pd(todst,_mm512_mul_pd(_mm512_loadu_pd(tosrc),valpha));
            _mm512_store_pd(todst+8,_mm512_mul_pd(_mm512_loadu_pd(tosrc+8),valpha));
            _mm512_store_pd(todst+16,_mm512_mul_pd(_mm512_loadu_pd(tosrc+16),valpha));
            tosrc+=leading_dim;
            todst+=24;
        }
    }
    // edge case
    for (;count_sub>7;count_first+=8,count_sub-=8){
        tosrc=src+count_first;
        for(count_second=0;count_second<dim_second;count_second++){
            _mm512_store_pd(todst,_mm512_mul_pd(_mm512_loadu_pd(tosrc),valpha));
            tosrc+=leading_dim;
            todst+=8;
        }
    }
    for (;count_sub>1;count_first+=2,count_sub-=2){
        tosrc=src+count_first;
        for(count_second=0;count_second<dim_second;count_second++){
            _mm_store_pd(todst,_mm_mul_pd(_mm_loadu_pd(tosrc),valpha_128));
            tosrc+=leading_dim;
            todst+=2;
        }
    }
    for (;count_sub>0;count_first+=1,count_sub-=1){
        tosrc=src+count_first;
        for(count_second=0;count_second<dim_second;count_second++){
            *todst=(*tosrc)*alpha;
            tosrc+=leading_dim;
            todst++;
        }
    }
}

#define init_m24n1 \
  "vpxorq %%zmm8,%%zmm8,%%zmm8;vpxorq %%zmm9,%%zmm9,%%zmm9;vpxorq %%zmm10,%%zmm10,%%zmm10;"
#define init_m8n1 \
  "vpxorq %%zmm8,%%zmm8,%%zmm8;"
#define init_m2n1 \
  "vpxorq %%xmm8,%%xmm8,%%xmm8;"
#define init_m24n2 \
  init_m24n1 \
  "vpxorq %%zmm11,%%zmm11,%%zmm11;vpxorq %%zmm12,%%zmm12,%%zmm12;vpxorq %%zmm13,%%zmm13,%%zmm13;"
#define init_m8n2 \
  init_m8n1 \
  "vpxorq %%zmm11,%%zmm11,%%zmm11;"
#define init_m2n2 \
  init_m2n1 \
  "vpxorq %%xmm9,%%xmm9,%%xmm9;"
#define init_m24n4 \
  init_m24n2 \
  "vpxorq %%zmm14,%%zmm14,%%zmm14;vpxorq %%zmm15,%%zmm15,%%zmm15;vpxorq %%zmm16,%%zmm16,%%zmm16;"\
  "vpxorq %%zmm17,%%zmm17,%%zmm17;vpxorq %%zmm18,%%zmm18,%%zmm18;vpxorq %%zmm19,%%zmm19,%%zmm19;"
#define init_m8n4 \
  init_m8n2 \
  "vpxorq %%zmm14,%%zmm14,%%zmm14;"\
  "vpxorq %%zmm17,%%zmm17,%%zmm17;"
#define init_m2n4 \
  init_m2n2 \
  "vpxorq %%xmm10,%%xmm10,%%xmm10;"\
  "vpxorq %%xmm11,%%xmm11,%%xmm11;"
#define init_m24n6 \
  init_m24n4 \
  "vpxorq %%zmm20,%%zmm20,%%zmm20;vpxorq %%zmm21,%%zmm21,%%zmm21;vpxorq %%zmm22,%%zmm22,%%zmm22;"\
  "vpxorq %%zmm23,%%zmm23,%%zmm23;vpxorq %%zmm24,%%zmm24,%%zmm24;vpxorq %%zmm25,%%zmm25,%%zmm25;"
#define init_m8n6 \
  init_m8n4 \
  "vpxorq %%zmm20,%%zmm20,%%zmm20;"\
  "vpxorq %%zmm23,%%zmm23,%%zmm23;"
#define init_m2n6 \
  init_m2n4 \
  "vpxorq %%xmm12,%%xmm12,%%xmm12;"\
  "vpxorq %%xmm13,%%xmm13,%%xmm13;"
#define init_m24n8 \
  init_m24n6 \
  "vpxorq %%zmm26,%%zmm26,%%zmm26;vpxorq %%zmm27,%%zmm27,%%zmm27;vpxorq %%zmm28,%%zmm28,%%zmm28;"\
  "vpxorq %%zmm29,%%zmm29,%%zmm29;vpxorq %%zmm30,%%zmm30,%%zmm30;vpxorq %%zmm31,%%zmm31,%%zmm31;"
#define init_m8n8 \
  init_m8n6 \
  "vpxorq %%zmm26,%%zmm26,%%zmm26;"\
  "vpxorq %%zmm29,%%zmm29,%%zmm29;"
#define init_m2n8 \
  init_m2n6 \
  "vpxorq %%xmm14,%%xmm14,%%xmm14;"\
  "vpxorq %%xmm15,%%xmm15,%%xmm15;"
#define save_init_m24 \
  "movq %2,%3; addq $192,%2;"
#define save_init_m8 \
  "movq %2,%3; addq $64,%2;"
#define save_init_m2 \
  "movq %2,%3; addq $16,%2;"
#define save_init_m1 \
  "movq %2,%3; addq $8,%2;"
#define kernel_m24n2_1 \
  "vbroadcastsd (%1),%%zmm4;vfmadd231pd %%zmm1,%%zmm4,%%zmm8; vfmadd231pd %%zmm2,%%zmm4,%%zmm9; vfmadd231pd %%zmm3,%%zmm4,%%zmm10;"\
  "vbroadcastsd 8(%1),%%zmm5;vfmadd231pd %%zmm1,%%zmm5,%%zmm11; vfmadd231pd %%zmm2,%%zmm5,%%zmm12; vfmadd231pd %%zmm3,%%zmm5,%%zmm13;"
#define kernel_m8n2_1 \
  "vbroadcastsd (%1),%%zmm4;vfmadd231pd %%zmm1,%%zmm4,%%zmm8;"\
  "vbroadcastsd 8(%1),%%zmm5;vfmadd231pd %%zmm1,%%zmm5,%%zmm11;"
#define kernel_m2n2_1 \
  "vmovddup (%1),%%xmm4;vfmadd231pd %%xmm1,%%xmm4,%%xmm8;"\
  "vmovddup 8(%1),%%xmm5;vfmadd231pd %%xmm1,%%xmm5,%%xmm9;"
#define kernel_m24n2_2 \
  "vbroadcastsd (%1,%%r11,1),%%zmm4;vfmadd231pd %%zmm1,%%zmm4,%%zmm14; vfmadd231pd %%zmm2,%%zmm4,%%zmm15; vfmadd231pd %%zmm3,%%zmm4,%%zmm16;"\
  "vbroadcastsd 8(%1,%%r11,1),%%zmm5;vfmadd231pd %%zmm1,%%zmm5,%%zmm17; vfmadd231pd %%zmm2,%%zmm5,%%zmm18; vfmadd231pd %%zmm3,%%zmm5,%%zmm19;"
#define kernel_m8n2_2 \
  "vbroadcastsd (%1,%%r11,1),%%zmm4;vfmadd231pd %%zmm1,%%zmm4,%%zmm14;"\
  "vbroadcastsd 8(%1,%%r11,1),%%zmm5;vfmadd231pd %%zmm1,%%zmm5,%%zmm17;"
#define kernel_m2n2_2 \
  "vmovddup (%1,%%r11,1),%%xmm4;vfmadd231pd %%xmm1,%%xmm4,%%xmm10;"\
  "vmovddup 8(%1,%%r11,1),%%xmm5;vfmadd231pd %%xmm1,%%xmm5,%%xmm11;"
#define kernel_m24n2_3 \
  "vbroadcastsd (%1,%%r11,2),%%zmm4;vfmadd231pd %%zmm1,%%zmm4,%%zmm20; vfmadd231pd %%zmm2,%%zmm4,%%zmm21; vfmadd231pd %%zmm3,%%zmm4,%%zmm22;"\
  "vbroadcastsd 8(%1,%%r11,2),%%zmm5;vfmadd231pd %%zmm1,%%zmm5,%%zmm23; vfmadd231pd %%zmm2,%%zmm5,%%zmm24; vfmadd231pd %%zmm3,%%zmm5,%%zmm25;"
#define kernel_m8n2_3 \
  "vbroadcastsd (%1,%%r11,2),%%zmm4;vfmadd231pd %%zmm1,%%zmm4,%%zmm20;"\
  "vbroadcastsd 8(%1,%%r11,2),%%zmm5;vfmadd231pd %%zmm1,%%zmm5,%%zmm23;"
#define kernel_m2n2_3 \
  "vmovddup (%1,%%r11,2),%%xmm4;vfmadd231pd %%xmm1,%%xmm4,%%xmm12;"\
  "vmovddup 8(%1,%%r11,2),%%xmm5;vfmadd231pd %%xmm1,%%xmm5,%%xmm13;"
#define kernel_m24n2_4 \
  "vbroadcastsd (%%r12),%%zmm4;vfmadd231pd %%zmm1,%%zmm4,%%zmm26; vfmadd231pd %%zmm2,%%zmm4,%%zmm27; vfmadd231pd %%zmm3,%%zmm4,%%zmm28;"\
  "vbroadcastsd 8(%%r12),%%zmm5;vfmadd231pd %%zmm1,%%zmm5,%%zmm29; vfmadd231pd %%zmm2,%%zmm5,%%zmm30; vfmadd231pd %%zmm3,%%zmm5,%%zmm31;"
#define kernel_m8n2_4 \
  "vbroadcastsd (%%r12),%%zmm4;vfmadd231pd %%zmm1,%%zmm4,%%zmm26;"\
  "vbroadcastsd 8(%%r12),%%zmm5;vfmadd231pd %%zmm1,%%zmm5,%%zmm29;"
#define kernel_m2n2_4 \
  "vmovddup (%%r12),%%xmm4;vfmadd231pd %%xmm1,%%xmm4,%%xmm14;"\
  "vmovddup 8(%%r12),%%xmm5;vfmadd231pd %%xmm1,%%xmm5,%%xmm15;"
#define LOAD_A_COL_m24 \
  "vmovaps (%0),%%zmm1;vmovaps 64(%0),%%zmm2;vmovaps 128(%0),%%zmm3;addq $192,%0;"
#define LOAD_A_COL_m8 \
  "vmovaps (%0),%%zmm1;addq $64,%0;"
#define LOAD_A_COL_m2 \
  "vmovaps (%0),%%xmm1;addq $16,%0;"
#define KERNEL_m24n8 \
  LOAD_A_COL_m24 \
  kernel_m24n2_1 \
  kernel_m24n2_2 \
  kernel_m24n2_3 \
  kernel_m24n2_4 \
  "addq $16,%1;addq $16,%%r12;"
#define KERNEL_m8n8 \
  LOAD_A_COL_m8 \
  kernel_m8n2_1 \
  kernel_m8n2_2 \
  kernel_m8n2_3 \
  kernel_m8n2_4 \
  "addq $16,%1;addq $16,%%r12;"
#define KERNEL_m2n8 \
  LOAD_A_COL_m2 \
  kernel_m2n2_1 \
  kernel_m2n2_2 \
  kernel_m2n2_3 \
  kernel_m2n2_4 \
  "addq $16,%1;addq $16,%%r12;"
#define KERNEL_m24n4 \
  LOAD_A_COL_m24 \
  kernel_m24n2_1 \
  kernel_m24n2_2 \
  "addq $16,%1;"
#define KERNEL_m8n4 \
  LOAD_A_COL_m8 \
  kernel_m8n2_1 \
  kernel_m8n2_2 \
  "addq $16,%1;"
#define KERNEL_m2n4 \
  LOAD_A_COL_m2 \
  kernel_m2n2_1 \
  kernel_m2n2_2 \
  "addq $16,%1;"
#define save_m24n2_1 \
  "vaddpd (%3),%%zmm8,%%zmm8;vaddpd 64(%3),%%zmm9,%%zmm9;vaddpd 128(%3),%%zmm10,%%zmm10;"\
  "vmovups %%zmm8,(%3);vmovups %%zmm9,64(%3);vmovups %%zmm10,128(%3);"\
  "vaddpd (%3,%4,1),%%zmm11,%%zmm11;vaddpd 64(%3,%4,1),%%zmm12,%%zmm12;vaddpd 128(%3,%4,1),%%zmm13,%%zmm13;"\
  "vmovups %%zmm11,(%3,%4,1);vmovups %%zmm12,64(%3,%4,1);vmovups %%zmm13,128(%3,%4,1);leaq (%3,%4,2),%3;"
#define save_m8n2_1 \
  "vaddpd (%3),%%zmm8,%%zmm8;"\
  "vmovups %%zmm8,(%3);"\
  "vaddpd (%3,%4,1),%%zmm11,%%zmm11;"\
  "vmovups %%zmm11,(%3,%4,1);leaq (%3,%4,2),%3;"
#define save_m2n2_1 \
  "vaddpd (%3),%%xmm8,%%xmm8;"\
  "vmovups %%xmm8,(%3);"\
  "vaddpd (%3,%4,1),%%xmm9,%%xmm9;"\
  "vmovups %%xmm9,(%3,%4,1);leaq (%3,%4,2),%3;"
#define save_m24n2_2 \
  "vaddpd (%3),%%zmm14,%%zmm14;vaddpd 64(%3),%%zmm15,%%zmm15;vaddpd 128(%3),%%zmm16,%%zmm16;"\
  "vmovups %%zmm14,(%3);vmovups %%zmm15,64(%3);vmovups %%zmm16,128(%3);"\
  "vaddpd (%3,%4,1),%%zmm17,%%zmm17;vaddpd 64(%3,%4,1),%%zmm18,%%zmm18;vaddpd 128(%3,%4,1),%%zmm19,%%zmm19;"\
  "vmovups %%zmm17,(%3,%4,1);vmovups %%zmm18,64(%3,%4,1);vmovups %%zmm19,128(%3,%4,1);leaq (%3,%4,2),%3;"
#define save_m8n2_2 \
  "vaddpd (%3),%%zmm14,%%zmm14;"\
  "vmovups %%zmm14,(%3);"\
  "vaddpd (%3,%4,1),%%zmm17,%%zmm17;"\
  "vmovups %%zmm17,(%3,%4,1);leaq (%3,%4,2),%3;"
#define save_m2n2_2 \
  "vaddpd (%3),%%xmm10,%%xmm10;"\
  "vmovups %%xmm10,(%3);"\
  "vaddpd (%3,%4,1),%%xmm11,%%xmm11;"\
  "vmovups %%xmm11,(%3,%4,1);leaq (%3,%4,2),%3;"
#define save_m24n2_3 \
  "vaddpd (%3),%%zmm20,%%zmm20;vaddpd 64(%3),%%zmm21,%%zmm21;vaddpd 128(%3),%%zmm22,%%zmm22;"\
  "vmovups %%zmm20,(%3);vmovups %%zmm21,64(%3);vmovups %%zmm22,128(%3);"\
  "vaddpd (%3,%4,1),%%zmm23,%%zmm23;vaddpd 64(%3,%4,1),%%zmm24,%%zmm24;vaddpd 128(%3,%4,1),%%zmm25,%%zmm25;"\
  "vmovups %%zmm23,(%3,%4,1);vmovups %%zmm24,64(%3,%4,1);vmovups %%zmm25,128(%3,%4,1);leaq (%3,%4,2),%3;"
#define save_m8n2_3 \
  "vaddpd (%3),%%zmm20,%%zmm20;"\
  "vmovups %%zmm20,(%3);"\
  "vaddpd (%3,%4,1),%%zmm23,%%zmm23;"\
  "vmovups %%zmm23,(%3,%4,1);leaq (%3,%4,2),%3;"
#define save_m2n2_3 \
  "vaddpd (%3),%%xmm12,%%xmm12;"\
  "vmovups %%xmm12,(%3);"\
  "vaddpd (%3,%4,1),%%xmm13,%%xmm13;"\
  "vmovups %%xmm13,(%3,%4,1);leaq (%3,%4,2),%3;"
#define save_m24n2_4 \
  "vaddpd (%3),%%zmm26,%%zmm26;vaddpd 64(%3),%%zmm27,%%zmm27;vaddpd 128(%3),%%zmm28,%%zmm28;"\
  "vmovups %%zmm26,(%3);vmovups %%zmm27,64(%3);vmovups %%zmm28,128(%3);"\
  "vaddpd (%3,%4,1),%%zmm29,%%zmm29;vaddpd 64(%3,%4,1),%%zmm30,%%zmm30;vaddpd 128(%3,%4,1),%%zmm31,%%zmm31;"\
  "vmovups %%zmm29,(%3,%4,1);vmovups %%zmm30,64(%3,%4,1);vmovups %%zmm31,128(%3,%4,1);leaq (%3,%4,2),%3;"
#define save_m8n2_4 \
  "vaddpd (%3),%%zmm26,%%zmm26;"\
  "vmovups %%zmm26,(%3);"\
  "vaddpd (%3,%4,1),%%zmm29,%%zmm29;"\
  "vmovups %%zmm29,(%3,%4,1);leaq (%3,%4,2),%3;"
#define save_m2n2_4 \
  "vaddpd (%3),%%xmm14,%%xmm14;"\
  "vmovups %%xmm14,(%3);"\
  "vaddpd (%3,%4,1),%%xmm15,%%xmm15;"\
  "vmovups %%xmm15,(%3,%4,1);leaq (%3,%4,2),%3;"
#define SAVE_m24n8 \
  save_init_m24 \
  save_m24n2_1 \
  save_m24n2_2 \
  save_m24n2_3 \
  save_m24n2_4
#define SAVE_m8n8 \
  save_init_m8 \
  save_m8n2_1 \
  save_m8n2_2 \
  save_m8n2_3 \
  save_m8n2_4
#define SAVE_m2n8 \
  save_init_m2 \
  save_m2n2_1 \
  save_m2n2_2 \
  save_m2n2_3 \
  save_m2n2_4
#define SAVE_m24n4 \
  save_init_m24 \
  save_m24n2_1 \
  save_m24n2_2
#define SAVE_m8n4 \
  save_init_m8 \
  save_m8n2_1 \
  save_m8n2_2
#define SAVE_m2n4 \
  save_init_m2 \
  save_m2n2_1 \
  save_m2n2_2

#define COMPUTE_m24n8 \
    "movq %%r14,%1;leaq (%1,%%r11,2),%%r12;addq %%r11,%%r12;movq %5,%%r13;"\
    init_m24n8 \
    "cmpq $4,%%r13;jb 724782f;\n\t"\
    "724781:\n\t"\
    KERNEL_m24n8 \
    KERNEL_m24n8 \
    KERNEL_m24n8 "subq $4,%%r13;"\
    KERNEL_m24n8 \
    "cmpq $4,%%r13;jnb 724781b;\n\t"\
    "cmpq $0,%%r13;je 724783f;\n\t"\
    "724782:\n\t"\
    KERNEL_m24n8 \
    "testq %%r13,%%r13;jnz 724782b;\n\t"\
    "724783:\n\t"\
    SAVE_m24n8 

#define COMPUTE_m24n4 \
    "movq %%r14,%1;leaq (%1,%%r11,2),%%r12;addq %%r11,%%r12;movq %5,%%r13;"\
    init_m24n4 \
    "cmpq $4,%%r13;jb 724742f;\n\t"\
    "724741:\n\t"\
    KERNEL_m24n4 \
    KERNEL_m24n4 \
    KERNEL_m24n4 "subq $4,%%r13;"\
    KERNEL_m24n4 \
    "cmpq $4,%%r13;jnb 724741b;\n\t"\
    "cmpq $0,%%r13;je 724743f;\n\t"\
    "724742:\n\t"\
    KERNEL_m24n4 \
    "testq %%r13,%%r13;jnz 724742b;\n\t"\
    "724743:\n\t"\
    SAVE_m24n4 

#define COMPUTE_m8n8 \
    "movq %%r14,%1;leaq (%1,%%r11,2),%%r12;addq %%r11,%%r12;movq %5,%%r13;"\
    init_m8n8 \
    "cmpq $4,%%r13;jb 78782f;\n\t"\
    "78781:\n\t"\
    KERNEL_m8n8 \
    KERNEL_m8n8 \
    KERNEL_m8n8 "subq $4,%%r13;"\
    KERNEL_m8n8 \
    "cmpq $4,%%r13;jnb 78781b;\n\t"\
    "cmpq $0,%%r13;je 78783f;\n\t"\
    "78782:\n\t"\
    KERNEL_m8n8 \
    "testq %%r13,%%r13;jnz 78782b;\n\t"\
    "78783:\n\t"\
    SAVE_m8n8

#define COMPUTE_m2n8 \
    "movq %%r14,%1;leaq (%1,%%r11,2),%%r12;addq %%r11,%%r12;movq %5,%%r13;"\
    init_m2n8 \
    "cmpq $4,%%r13;jb 72782f;\n\t"\
    "72781:\n\t"\
    KERNEL_m2n8 \
    KERNEL_m2n8 \
    KERNEL_m2n8 "subq $4,%%r13;"\
    KERNEL_m2n8 \
    "cmpq $4,%%r13;jnb 72781b;\n\t"\
    "cmpq $0,%%r13;je 72783f;\n\t"\
    "72782:\n\t"\
    KERNEL_m2n8 \
    "testq %%r13,%%r13;jnz 72782b;\n\t"\
    "72783:\n\t"\
    SAVE_m2n8

#define COMPUTE_m8n4 \
    "movq %%r14,%1;leaq (%1,%%r11,2),%%r12;addq %%r11,%%r12;movq %5,%%r13;"\
    init_m8n4 \
    "cmpq $4,%%r13;jb 78742f;\n\t"\
    "78741:\n\t"\
    KERNEL_m8n4 \
    KERNEL_m8n4 \
    KERNEL_m8n4 "subq $4,%%r13;"\
    KERNEL_m8n4 \
    "cmpq $4,%%r13;jnb 78741b;\n\t"\
    "cmpq $0,%%r13;je 78743f;\n\t"\
    "78742:\n\t"\
    KERNEL_m8n4 \
    "testq %%r13,%%r13;jnz 78742b;\n\t"\
    "78743:\n\t"\
    SAVE_m8n4 

#define COMPUTE_m2n4 \
    "movq %%r14,%1;leaq (%1,%%r11,2),%%r12;addq %%r11,%%r12;movq %5,%%r13;"\
    init_m2n4 \
    "cmpq $4,%%r13;jb 72742f;\n\t"\
    "72741:\n\t"\
    KERNEL_m2n4 \
    KERNEL_m2n4 \
    KERNEL_m2n4 "subq $4,%%r13;"\
    KERNEL_m2n4 \
    "cmpq $4,%%r13;jnb 72741b;\n\t"\
    "cmpq $0,%%r13;je 72743f;\n\t"\
    "72742:\n\t"\
    KERNEL_m2n4 \
    "testq %%r13,%%r13;jnz 72742b;\n\t"\
    "72743:\n\t"\
    SAVE_m2n4 

#define macro_n8 {\
  __asm__ __volatile__(\
    "movq %6,%%r15; movq %1,%%r14; movq %5,%%r11; salq $4,%%r11;"\
    "cmpq $24,%%r15; jb 3243831f;"\
    "3243830:\n\t"\
    COMPUTE_m24n8 "subq $24,%%r15; cmpq $24,%%r15; jnb 3243830b;"\
    "3243831:\n\t"\
    "cmpq $8,%%r15; jb 3243833f;"\
    "3243832:\n\t"\
    COMPUTE_m8n8 "subq $8,%%r15; cmpq $8,%%r15; jnb 3243832b;"\
    "3243833:\n\t"\
    "cmpq $2,%%r15; jb 3243835f;"\
    "3243834:\n\t"\
    COMPUTE_m2n8 "subq $2,%%r15; cmpq $2,%%r15; jnb 3243834b;"\
    "3243835:\n\t"\
    "movq %%r14,%1;"\
  :"+r"(a_ptr),"+r"(b_ptr),"+r"(c_ptr),"+r"(c_tmp),"+r"(ldc_in_bytes):"m"(K),"m"(M):"r11","r12","r13","r14","r15","cc","memory",\
    "zmm0","zmm1","zmm2","zmm3","zmm4","zmm5","zmm6","zmm7","zmm8","zmm9","zmm10","zmm11","zmm12","zmm13","zmm14","zmm15",\
    "zmm16","zmm17","zmm18","zmm19","zmm20","zmm21","zmm22","zmm23","zmm24","zmm25","zmm26","zmm27","zmm28","zmm29","zmm30","zmm31");\
  a_ptr -= M * K; b_ptr += 8 * K; c_ptr += 8 * ldc - M;\
}

#define macro_n4 {\
  __asm__ __volatile__(\
    "movq %6,%%r15; movq %1,%%r14; movq %5,%%r11; salq $4,%%r11;"\
    "cmpq $24,%%r15; jb 3243431f;"\
    "3243430:\n\t"\
    COMPUTE_m24n4 "subq $24,%%r15; cmpq $24,%%r15; jnb 3243430b;"\
    "3243431:\n\t"\
    "cmpq $8,%%r15; jb 3243433f;"\
    "3243432:\n\t"\
    COMPUTE_m8n4 "subq $8,%%r15; cmpq $8,%%r15; jnb 3243432b;"\
    "3243433:\n\t"\
    "cmpq $2,%%r15; jb 3243435f;"\
    "3243434:\n\t"\
    COMPUTE_m2n4 "subq $2,%%r15; cmpq $2,%%r15; jnb 3243434b;"\
    "3243435:\n\t"\
    "movq %%r14,%1;"\
  :"+r"(a_ptr),"+r"(b_ptr),"+r"(c_ptr),"+r"(c_tmp),"+r"(ldc_in_bytes):"m"(K),"m"(M):"r11","r12","r13","r14","r15","cc","memory",\
    "zmm0","zmm1","zmm2","zmm3","zmm4","zmm5","zmm6","zmm7","zmm8","zmm9","zmm10","zmm11","zmm12","zmm13","zmm14","zmm15",\
    "zmm16","zmm17","zmm18","zmm19","zmm20","zmm21","zmm22","zmm23","zmm24","zmm25","zmm26","zmm27","zmm28","zmm29","zmm30","zmm31");\
  a_ptr -= M * K; b_ptr += 8 * K; c_ptr += 8 * ldc - M;\
}


void macro_kernel_k16(double *a_buffer,double *b_buffer,int m,int n,int k,double *C, int LDC,int k_inc,double alpha){
    int m_count,n_count,m_count_sub,n_count_sub;
    if (m==0||n==0||k==0) return;
    int64_t M=(int64_t)m,K=(int64_t)k,ldc_in_bytes=(int64_t)LDC*sizeof(double),ldc=(int32_t)LDC;
    double *a_ptr=a_buffer,*b_ptr=b_buffer,*c_ptr=C,*c_tmp=C;
    double ALPHA=alpha;
    // printf("m= %d, n=%d, k = %d\n",m,n,k);
    for (n_count_sub=n,n_count=0;n_count_sub>7;n_count_sub-=8,n_count+=8){
        //call the m layer with n=8;
        macro_n8
    }
    for (;n_count_sub>3;n_count_sub-=4,n_count+=4){
        //call the m layer with n=4
        macro_n4
    }
    for (;n_count_sub>1;n_count_sub-=2,n_count+=2){
        //call the m layer with n=2
    }
    for (;n_count_sub>0;n_count_sub-=1,n_count+=1){
        //call the m layer with n=1
    }
}


void mydgemm_cpu_v16(\
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
#ifdef TIMER
    double t_tot=0.,t_copy_a=0.,t_copy_b=0.,t_scale_c=0., t0, t1;
    t0=get_sec();
#endif
    if (beta != 1.0) scale_c_k16(C,M,N,LDC,beta);
#ifdef TIMER
    t1=get_sec();
    t_scale_c+=(t1-t0);
#endif
    if (alpha == 0.||K==0) return;
    int M4,N8=N&-8,K4;
    
    double *b_buffer = (double *)aligned_alloc(4096,K_BLOCKING*OUT_N_BLOCKING*sizeof(double));
    double *a_buffer = (double *)aligned_alloc(4096,K_BLOCKING*OUT_M_BLOCKING*sizeof(double));
    int second_m_count,second_n_count,second_m_inc,second_n_inc;
    int m_count,n_count,k_count;
    int m_inc,n_inc,k_inc;
    for (k_count=0;k_count<K;k_count+=k_inc){
        k_inc=(K-k_count>K_BLOCKING)?K_BLOCKING:K-k_count;
        for (n_count=0;n_count<N;n_count+=n_inc){
            n_inc=(N-n_count>OUT_N_BLOCKING)?OUT_N_BLOCKING:N-n_count;
#ifdef TIMER
            t0=get_sec();
#endif
            packing_b_24x8_edge_version2_k16(B+k_count+n_count*LDB,b_buffer,LDB,k_inc,n_inc);
#ifdef TIMER
            t1=get_sec();
            t_copy_b+=(t1-t0);
#endif
            //print_matrix(b_buffer,k_inc,n_inc);
            for (m_count=0;m_count<M;m_count+=m_inc){
                m_inc=(M-m_count>OUT_M_BLOCKING)?OUT_M_BLOCKING:M-m_count;
#ifdef TIMER
                t0=get_sec();
#endif
                packing_a_24x8_edge_k16(alpha,A+m_count+k_count*LDA,a_buffer,LDA,m_inc,k_inc);
#ifdef TIMER
                t1=get_sec();
                t_copy_a+=(t1-t0);
#endif
                //print_matrix(a_buffer,m_inc,k_inc);
                for (second_m_count=m_count;second_m_count<m_count+m_inc;second_m_count+=second_m_inc){
                    second_m_inc=(m_count+m_inc-second_m_count>M_BLOCKING)?M_BLOCKING:m_count+m_inc-second_m_count;
                    for (second_n_count=n_count;second_n_count<n_count+n_inc;second_n_count+=second_n_inc){
                        second_n_inc=(n_count+n_inc-second_n_count>N_BLOCKING)?N_BLOCKING:n_count+n_inc-second_n_count;
                        // printf("m=%d,n=%d,k=%d,m_dim=%d,n_dim=%d,k_dim=%d\n",second_m_count,second_n_count,k_count,second_m_inc,second_n_inc,k_inc);
#ifdef TIMER
                        t0=get_sec();
#endif
                        macro_kernel_k16(a_buffer+(second_m_count-m_count)*k_inc,b_buffer+(second_n_count-n_count)*k_inc,second_m_inc,second_n_inc,k_inc,&C(second_m_count,second_n_count),LDC,k_inc,alpha);
#ifdef TIMER
                        t1=get_sec();
                        t_tot+=(t1-t0);
#endif
                        //printf("m=%d,m_inc=%d,n=%d,n_inc=%d\n",second_m_count,second_m_inc,second_n_count,second_n_inc);
                    }
                }
            }
        }
    }
#ifdef TIMER
    printf("Time in major loop: %f, perf=%f GFLOPS\n", t_tot,2.*1e-9*M*N*K/t_tot);
    printf("Time in copy A: %f, perf = %f GFLOPS\n", t_copy_a, 2.*1e-9*M*K/t_copy_a);
    printf("Time in copy B: %f, perf = %f GFLOPS\n", t_copy_b, 2.*1e-9*N*K/t_copy_b);
    printf("Time in scaling C: %f, perf = %f GFLOPS\n", t_scale_c, 2.*1e-9*M*N/t_scale_c);
    printf("Total: %f, perf = %f GFLOPS\n", t_tot+t_copy_b+t_scale_c+t_copy_a, 2.*1e-9*M*N*K/(t_tot+t_copy_b+t_scale_c+t_copy_a));
#endif
    free(a_buffer);free(b_buffer);
}