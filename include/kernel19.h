#include "immintrin.h"
#include <stdint.h>
#include "../utils.h"
#define A(i,j) A[(i)+(j)*LDA]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]
#define M_BLOCKING 192
#define N_BLOCKING 8640
#define K_BLOCKING 384
#define MIN(a, b) ((a) < (b)) ? (a) : (b)
#define MAX(a, b) ((a) > (b)) ? (a) : (b)

#include "omp.h"

void scale_c_k19(double *C,int M, int N, int LDC, double scalar){
    int m_count,n_count;
    int M8=M&-8,N4=N&-4,LDC2=LDC<<1,LDC3=LDC2+LDC,LDC4=LDC<<2;
    __m512d vscalar = _mm512_set1_pd(scalar);
    double *c_ptr_base1 = C,*c_ptr_base2 = C+LDC,*c_ptr_base3 = C+LDC2,*c_ptr_base4 = C+LDC3;
    double *c_ptr_dyn1,*c_ptr_dyn2,*c_ptr_dyn3,*c_ptr_dyn4;
    for (n_count=0;n_count<N4;n_count+=4){
        c_ptr_dyn1 = c_ptr_base1;c_ptr_dyn2 = c_ptr_base2;c_ptr_dyn3 = c_ptr_base3;c_ptr_dyn4 = c_ptr_base4;
        for (m_count=0;m_count<M8;m_count+=8){
            _mm512_storeu_pd(c_ptr_dyn1,_mm512_mul_pd(_mm512_loadu_pd(c_ptr_dyn1),vscalar));
            _mm512_storeu_pd(c_ptr_dyn2,_mm512_mul_pd(_mm512_loadu_pd(c_ptr_dyn2),vscalar));
            _mm512_storeu_pd(c_ptr_dyn3,_mm512_mul_pd(_mm512_loadu_pd(c_ptr_dyn3),vscalar));
            _mm512_storeu_pd(c_ptr_dyn4,_mm512_mul_pd(_mm512_loadu_pd(c_ptr_dyn4),vscalar));
            c_ptr_dyn1+=8;c_ptr_dyn2+=8;c_ptr_dyn3+=8;c_ptr_dyn4+=8;
        }
        for (;m_count<M;m_count++){
            *c_ptr_dyn1 *= scalar; c_ptr_dyn1++;
            *c_ptr_dyn2 *= scalar; c_ptr_dyn2++;
            *c_ptr_dyn3 *= scalar; c_ptr_dyn3++;
            *c_ptr_dyn4 *= scalar; c_ptr_dyn4++;
        }
        c_ptr_base1 += LDC4;c_ptr_base2 += LDC4;c_ptr_base3 += LDC4;c_ptr_base4 += LDC4;
    }
    for (;n_count<N;n_count++){
        c_ptr_dyn1 = c_ptr_base1;
        for (m_count=0;m_count<M8;m_count+=8){
            _mm512_storeu_pd(c_ptr_dyn1,_mm512_mul_pd(_mm512_loadu_pd(c_ptr_dyn1),vscalar));
            c_ptr_dyn1+=8;
        }
        for (;m_count<M;m_count++){
            *c_ptr_dyn1 *= scalar; c_ptr_dyn1++;
        }
        c_ptr_base1 += LDC;
    }
}

void packing_b_k19(double *src,double *dst,int leading_dim,int dim_first,int dim_second){
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

void packing_a_k19(double alpha,double *src, double *dst, int leading_dim, int dim_first, int dim_second){
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
  "vbroadcastsd 8(%1),%%zmm5;vfmadd231pd %%zmm1,%%zmm5,%%zmm11; vfmadd231pd %%zmm2,%%zmm5,%%zmm12; vfmadd231pd %%zmm3,%%zmm5,%%zmm13;prefetcht0 384(%0);prefetcht0 64(%1);"
#define kernel_m8n2_1 \
  "vbroadcastsd (%1),%%zmm4;vfmadd231pd %%zmm1,%%zmm4,%%zmm8;"\
  "vbroadcastsd 8(%1),%%zmm5;vfmadd231pd %%zmm1,%%zmm5,%%zmm11;"
#define kernel_m2n2_1 \
  "vmovddup (%1),%%xmm4;vfmadd231pd %%xmm1,%%xmm4,%%xmm8;"\
  "vmovddup 8(%1),%%xmm5;vfmadd231pd %%xmm1,%%xmm5,%%xmm9;"
#define kernel_m24n2_2 \
  "vbroadcastsd (%1,%%r11,1),%%zmm4;vfmadd231pd %%zmm1,%%zmm4,%%zmm14; vfmadd231pd %%zmm2,%%zmm4,%%zmm15; vfmadd231pd %%zmm3,%%zmm4,%%zmm16;"\
  "vbroadcastsd 8(%1,%%r11,1),%%zmm5;vfmadd231pd %%zmm1,%%zmm5,%%zmm17; vfmadd231pd %%zmm2,%%zmm5,%%zmm18; vfmadd231pd %%zmm3,%%zmm5,%%zmm19;prefetcht0 448(%0);"
#define kernel_m8n2_2 \
  "vbroadcastsd (%1,%%r11,1),%%zmm4;vfmadd231pd %%zmm1,%%zmm4,%%zmm14;"\
  "vbroadcastsd 8(%1,%%r11,1),%%zmm5;vfmadd231pd %%zmm1,%%zmm5,%%zmm17;"
#define kernel_m2n2_2 \
  "vmovddup (%1,%%r11,1),%%xmm4;vfmadd231pd %%xmm1,%%xmm4,%%xmm10;"\
  "vmovddup 8(%1,%%r11,1),%%xmm5;vfmadd231pd %%xmm1,%%xmm5,%%xmm11;"
#define kernel_m24n2_3 \
  "vbroadcastsd (%1,%%r11,2),%%zmm4;vfmadd231pd %%zmm1,%%zmm4,%%zmm20; vfmadd231pd %%zmm2,%%zmm4,%%zmm21; vfmadd231pd %%zmm3,%%zmm4,%%zmm22;"\
  "vbroadcastsd 8(%1,%%r11,2),%%zmm5;vfmadd231pd %%zmm1,%%zmm5,%%zmm23; vfmadd231pd %%zmm2,%%zmm5,%%zmm24; vfmadd231pd %%zmm3,%%zmm5,%%zmm25;prefetcht0 512(%0);"
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
#define KERNEL_m24n2 \
  LOAD_A_COL_m24 \
  kernel_m24n2_1 \
  "addq $16,%1;"
#define KERNEL_m8n8 \
  LOAD_A_COL_m8 \
  kernel_m8n2_1 \
  kernel_m8n2_2 \
  kernel_m8n2_3 \
  kernel_m8n2_4 \
  "addq $16,%1;addq $16,%%r12;"
#define KERNEL_m8n2 \
  LOAD_A_COL_m8 \
  kernel_m8n2_1 \
  "addq $16,%1;"
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
#define KERNEL_m2n2 \
  LOAD_A_COL_m2 \
  kernel_m2n2_1 \
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
#define SAVE_m24n2 \
  save_init_m24 \
  save_m24n2_1
#define SAVE_m8n8 \
  save_init_m8 \
  save_m8n2_1 \
  save_m8n2_2 \
  save_m8n2_3 \
  save_m8n2_4
#define SAVE_m8n2 \
  save_init_m8 \
  save_m8n2_1
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
#define SAVE_m2n2 \
  save_init_m2 \
  save_m2n2_1

#define COMPUTE_m24n8 \
    "movq %%r14,%1;leaq (%1,%%r11,2),%%r12;addq %%r11,%%r12;movq %6,%%r13;"\
    init_m24n8 \
    "cmpq $4,%%r13;jb 724782f;\n\t"\
    "724781:\n\t"\
    KERNEL_m24n8 "subq $4,%%r13;testq $12,%%r13;movq $172,%%r10;cmovz %4,%%r10;"\
    KERNEL_m24n8 "prefetcht1 (%3);subq $129,%3;addq %%r10,%3;"\
    KERNEL_m24n8 "prefetcht1 (%5);addq $32,%5;cmpq $192,%%r13;cmoveq %2,%3;"\
    KERNEL_m24n8 \
    "cmpq $16,%%r13;jnb 724781b;\n\t"\
    "movq %2,%3;"\
    "cmpq $0,%%r13;je 724783f;\n\t"\
    "724782:\n\t"\
    "prefetcht0 (%3); prefetcht0 64(%3); prefetcht0 128(%3);decq %%r13;"\
    KERNEL_m24n8 \
    "addq %4,%3;testq %%r13,%%r13;jnz 724782b;\n\t"\
    "724783:\n\t"\
    SAVE_m24n8 

#define COMPUTE_m24n4 \
    "movq %%r14,%1;leaq (%1,%%r11,2),%%r12;addq %%r11,%%r12;movq %6,%%r13;"\
    init_m24n4 \
    "cmpq $4,%%r13;jb 724742f;\n\t"\
    "724741:\n\t"\
    KERNEL_m24n4 "subq $4,%%r13;testq $12,%%r13;movq $172,%%r10;cmovz %4,%%r10;"\
    KERNEL_m24n4 "prefetcht1 (%3);subq $129,%3;addq %%r10,%3;"\
    KERNEL_m24n4 "prefetcht1 (%5);addq $32,%5;cmpq $192,%%r13;cmoveq %2,%3;"\
    KERNEL_m24n4 \
    "cmpq $16,%%r13;jnb 724741b;\n\t"\
    "movq %2,%3;"\
    "cmpq $0,%%r13;je 724743f;\n\t"\
    "724742:\n\t"\
    "prefetcht0 (%3); prefetcht0 64(%3); prefetcht0 128(%3);decq %%r13;"\
    KERNEL_m24n4 \
    "addq %4,%3;testq %%r13,%%r13;jnz 724742b;\n\t"\
    "724743:\n\t"\
    SAVE_m24n4 

#define COMPUTE_m8n8 \
    "movq %%r14,%1;leaq (%1,%%r11,2),%%r12;addq %%r11,%%r12;movq %6,%%r13;"\
    init_m8n8 \
    "cmpq $4,%%r13;jb 78782f;\n\t"\
    "78781:\n\t"\
    KERNEL_m8n8 \
    KERNEL_m8n8 \
    KERNEL_m8n8 "subq $4,%%r13;"\
    KERNEL_m8n8 \
    "cmpq $4,%%r13;jnb 78781b;\n\t"\
    "movq %2,%3;"\
    "cmpq $0,%%r13;je 78783f;\n\t"\
    "78782:\n\t"\
    "prefetcht0 (%3); prefetcht0 64(%3); prefetcht0 128(%3);decq %%r13;"\
    KERNEL_m8n8 \
    "addq %4,%3;testq %%r13,%%r13;jnz 78782b;\n\t"\
    "78783:\n\t"\
    SAVE_m8n8

#define COMPUTE_m24n2 \
    "movq %%r14,%1;leaq (%1,%%r11,2),%%r12;addq %%r11,%%r12;movq %6,%%r13;"\
    init_m24n2 \
    "cmpq $4,%%r13;jb 724722f;\n\t"\
    "724721:\n\t"\
    KERNEL_m24n2 \
    KERNEL_m24n2 \
    KERNEL_m24n2 "subq $4,%%r13;"\
    KERNEL_m24n2 \
    "cmpq $4,%%r13;jnb 724721b;\n\t"\
    "movq %2,%3;"\
    "cmpq $0,%%r13;je 724723f;\n\t"\
    "724722:\n\t"\
    "prefetcht0 (%3); prefetcht0 64(%3); prefetcht0 128(%3);decq %%r13;"\
    KERNEL_m24n2 \
    "addq %4,%3;testq %%r13,%%r13;jnz 724722b;\n\t"\
    "724723:\n\t"\
    SAVE_m24n2

#define COMPUTE_m8n2 \
    "movq %%r14,%1;leaq (%1,%%r11,2),%%r12;addq %%r11,%%r12;movq %6,%%r13;"\
    init_m8n2 \
    "cmpq $4,%%r13;jb 78722f;\n\t"\
    "78721:\n\t"\
    KERNEL_m8n2 \
    KERNEL_m8n2 \
    KERNEL_m8n2 "subq $4,%%r13;"\
    KERNEL_m8n2 \
    "cmpq $4,%%r13;jnb 78721b;\n\t"\
    "movq %2,%3;"\
    "cmpq $0,%%r13;je 78723f;\n\t"\
    "78722:\n\t"\
    "prefetcht0 (%3); prefetcht0 64(%3); prefetcht0 128(%3);decq %%r13;"\
    KERNEL_m8n2 \
    "addq %4,%3;testq %%r13,%%r13;jnz 78722b;\n\t"\
    "78723:\n\t"\
    SAVE_m8n2

#define COMPUTE_m2n8 \
    "movq %%r14,%1;leaq (%1,%%r11,2),%%r12;addq %%r11,%%r12;movq %6,%%r13;"\
    init_m2n8 \
    "cmpq $4,%%r13;jb 72782f;\n\t"\
    "72781:\n\t"\
    KERNEL_m2n8 \
    KERNEL_m2n8 \
    KERNEL_m2n8 "subq $4,%%r13;"\
    KERNEL_m2n8 \
    "cmpq $4,%%r13;jnb 72781b;\n\t"\
    "movq %2,%3;"\
    "cmpq $0,%%r13;je 72783f;\n\t"\
    "72782:\n\t"\
    "prefetcht0 (%3); prefetcht0 64(%3); prefetcht0 128(%3);decq %%r13;"\
    KERNEL_m2n8 \
    "addq %4,%3;testq %%r13,%%r13;jnz 72782b;\n\t"\
    "72783:\n\t"\
    SAVE_m2n8

#define COMPUTE_m8n4 \
    "movq %%r14,%1;leaq (%1,%%r11,2),%%r12;addq %%r11,%%r12;movq %6,%%r13;"\
    init_m8n4 \
    "cmpq $4,%%r13;jb 78742f;\n\t"\
    "78741:\n\t"\
    KERNEL_m8n4 \
    KERNEL_m8n4 \
    KERNEL_m8n4 "subq $4,%%r13;"\
    KERNEL_m8n4 \
    "cmpq $4,%%r13;jnb 78741b;\n\t"\
    "movq %2,%3;"\
    "cmpq $0,%%r13;je 78743f;\n\t"\
    "78742:\n\t"\
    "prefetcht0 (%3); prefetcht0 64(%3); prefetcht0 128(%3);decq %%r13;"\
    KERNEL_m8n4 \
    "addq %4,%3;testq %%r13,%%r13;jnz 78742b;\n\t"\
    "78743:\n\t"\
    SAVE_m8n4 

#define COMPUTE_m2n2 \
    "movq %%r14,%1;leaq (%1,%%r11,2),%%r12;addq %%r11,%%r12;movq %6,%%r13;"\
    init_m2n2 \
    "cmpq $4,%%r13;jb 72722f;\n\t"\
    "72721:\n\t"\
    KERNEL_m2n2 \
    KERNEL_m2n2 \
    KERNEL_m2n2 "subq $4,%%r13;"\
    KERNEL_m2n2 \
    "cmpq $4,%%r13;jnb 72721b;\n\t"\
    "movq %2,%3;"\
    "cmpq $0,%%r13;je 72723f;\n\t"\
    "72722:\n\t"\
    "prefetcht0 (%3); prefetcht0 64(%3); prefetcht0 128(%3);decq %%r13;"\
    KERNEL_m2n2 \
    "addq %4,%3;testq %%r13,%%r13;jnz 72722b;\n\t"\
    "72723:\n\t"\
    SAVE_m2n2 

#define COMPUTE_m2n4 \
    "movq %%r14,%1;leaq (%1,%%r11,2),%%r12;addq %%r11,%%r12;movq %6,%%r13;"\
    init_m2n4 \
    "cmpq $4,%%r13;jb 72742f;\n\t"\
    "72741:\n\t"\
    KERNEL_m2n4 \
    KERNEL_m2n4 \
    KERNEL_m2n4 "subq $4,%%r13;"\
    KERNEL_m2n4 \
    "cmpq $4,%%r13;jnb 72741b;\n\t"\
    "movq %2,%3;"\
    "cmpq $0,%%r13;je 72743f;\n\t"\
    "72742:\n\t"\
    "prefetcht0 (%3); prefetcht0 64(%3); prefetcht0 128(%3);decq %%r13;"\
    KERNEL_m2n4 \
    "addq %4,%3;testq %%r13,%%r13;jnz 72742b;\n\t"\
    "72743:\n\t"\
    SAVE_m2n4 

#define macro_n8 {\
  b_pref = b_ptr + 8 * K;\
  __asm__ __volatile__(\
    "movq %7,%%r15; movq %1,%%r14; movq %6,%%r11; salq $4,%%r11;"\
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
  :"+r"(a_ptr),"+r"(b_ptr),"+r"(c_ptr),"+r"(c_tmp),"+r"(ldc_in_bytes),"+r"(b_pref):"m"(K),"m"(M):"r10","r11","r12","r13","r14","r15","cc","memory",\
    "zmm0","zmm1","zmm2","zmm3","zmm4","zmm5","zmm6","zmm7","zmm8","zmm9","zmm10","zmm11","zmm12","zmm13","zmm14","zmm15",\
    "zmm16","zmm17","zmm18","zmm19","zmm20","zmm21","zmm22","zmm23","zmm24","zmm25","zmm26","zmm27","zmm28","zmm29","zmm30","zmm31");\
  a_ptr -= M * K; b_ptr += 8 * K; c_ptr += 8 * ldc - M;\
}

#define macro_n4 {\
  b_pref = b_ptr + 4 * K;\
  __asm__ __volatile__(\
    "movq %7,%%r15; movq %1,%%r14; movq %6,%%r11; salq $4,%%r11;"\
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
  :"+r"(a_ptr),"+r"(b_ptr),"+r"(c_ptr),"+r"(c_tmp),"+r"(ldc_in_bytes),"+r"(b_pref):"m"(K),"m"(M):"r10","r11","r12","r13","r14","r15","cc","memory",\
    "zmm0","zmm1","zmm2","zmm3","zmm4","zmm5","zmm6","zmm7","zmm8","zmm9","zmm10","zmm11","zmm12","zmm13","zmm14","zmm15",\
    "zmm16","zmm17","zmm18","zmm19","zmm20","zmm21","zmm22","zmm23","zmm24","zmm25","zmm26","zmm27","zmm28","zmm29","zmm30","zmm31");\
  a_ptr -= M * K; b_ptr += 4 * K; c_ptr += 4 * ldc - M;\
}

#define macro_n2 {\
  b_pref = b_ptr + 2 * K;\
  __asm__ __volatile__(\
    "movq %7,%%r15; movq %1,%%r14; movq %6,%%r11; salq $4,%%r11;"\
    "cmpq $24,%%r15; jb 3243231f;"\
    "3243230:\n\t"\
    COMPUTE_m24n2 "subq $24,%%r15; cmpq $24,%%r15; jnb 3243230b;"\
    "3243231:\n\t"\
    "cmpq $8,%%r15; jb 3243233f;"\
    "3243232:\n\t"\
    COMPUTE_m8n2 "subq $8,%%r15; cmpq $8,%%r15; jnb 3243232b;"\
    "3243233:\n\t"\
    "cmpq $2,%%r15; jb 3243235f;"\
    "3243234:\n\t"\
    COMPUTE_m2n2 "subq $2,%%r15; cmpq $2,%%r15; jnb 3243234b;"\
    "3243235:\n\t"\
    "movq %%r14,%1;"\
  :"+r"(a_ptr),"+r"(b_ptr),"+r"(c_ptr),"+r"(c_tmp),"+r"(ldc_in_bytes),"+r"(b_pref):"m"(K),"m"(M):"r10","r11","r12","r13","r14","r15","cc","memory",\
    "zmm0","zmm1","zmm2","zmm3","zmm4","zmm5","zmm6","zmm7","zmm8","zmm9","zmm10","zmm11","zmm12","zmm13","zmm14","zmm15",\
    "zmm16","zmm17","zmm18","zmm19","zmm20","zmm21","zmm22","zmm23","zmm24","zmm25","zmm26","zmm27","zmm28","zmm29","zmm30","zmm31");\
  a_ptr -= M * K; b_ptr += 2 * K; c_ptr += 2 * ldc - M;\
}


void macro_kernel_k19(double *a_buffer,double *b_buffer,int m,int n,int k,double *C, int LDC){
    int m_count,n_count,m_count_sub,n_count_sub;
    if (m==0||n==0||k==0) return;
    int64_t M=(int64_t)m,K=(int64_t)k,ldc_in_bytes=(int64_t)LDC*sizeof(double),ldc=(int32_t)LDC;
    double *a_ptr=a_buffer,*b_ptr=b_buffer,*c_ptr=C,*b_pref=b_ptr,*c_tmp=C;
    // printf("m= %d, n=%d, k = %d\n",m,n,k);
    for (n_count_sub=n,n_count=0;n_count_sub>7;n_count_sub-=8,n_count+=8){
        //call the m layer with n=8;
        macro_n8
        //TODO: case when m is divisible by 1
    }
    for (;n_count_sub>3;n_count_sub-=4,n_count+=4){
        //call the m layer with n=4
        macro_n4
        //TODO: case when m is divisible by 1
    }
    for (;n_count_sub>1;n_count_sub-=2,n_count+=2){
        //call the m layer with n=2
        macro_n2
    }
    for (;n_count_sub>0;n_count_sub-=1,n_count+=1){
        //TODO:call the m layer with n=1
    }
}

/*
 * parallel functions
 * 
 */

static inline void partition_m_dim(const int ithr, const int nthrs, const int m, int *offset, int *block)
{
    int band = m / nthrs;
    int tail = m - (nthrs - 1) * band;
    if (tail > (band + 1)){
        band++;
    }
    tail = m - (nthrs - 1) * band;
    if (ithr < (nthrs - 1)){
        *block = band;
    }else{
        *block = tail;
    }
    *offset = ithr * band;
    if (*offset >= m) {
        *block = 0;
        *offset = 0;
    }else if ((*offset + *block) > m) {
        *block = m - *offset;
    }
}

static inline int div_up(int a, int b){
    return ((a + b - 1) / b);
}

static inline int rnd_up(int a, int b) {
    return (div_up(a, b) * b);
}


static inline int get_n_padd(int n, int un, int bn)
{
    return rnd_up(MIN(MAX(n, un), bn), un);
}

static inline int get_n_padd_parallel_a(int n, int nthr)
{
    int n_padd = get_n_padd(n, 8, N_BLOCKING);
    return n_padd;
}

void mydgemm_cpu_v19(\
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

    double *b_buffer_global = NULL;
    if (alpha == 0.||K==0) return;

    int nthr_spawn = 10;
    #pragma omp parallel num_threads(nthr_spawn) 
    {
        int nthr = nthr_spawn;
        int ithr = omp_get_thread_num();
        int m_offset = 0, m_block = 0;
        partition_m_dim(ithr, nthr, M, &m_offset, &m_block);

        if (beta != 1.0) scale_c_k19(&C(m_offset, 0),m_block,N,LDC,beta);

        int m_count, n_count, k_count;
        int m_inc, n_inc, k_inc;
        double *a_buffer_local = NULL;
        double *b_buffer_local = NULL;
        if (ithr == 0) {
            b_buffer_global = (double *)aligned_alloc(4096, sizeof(double) * (N_BLOCKING * K_BLOCKING));
        }
        #pragma omp barrier
        b_buffer_local = b_buffer_global;
        for (k_count = 0; k_count < K; k_count += k_inc){
            k_inc = (K - k_count > K_BLOCKING) ? K_BLOCKING : K - k_count;
            // parallel copy for A packing
            for (n_count = 0; n_count < N; n_count += n_inc) {
                n_inc = (N - n_count > N_BLOCKING) ? N_BLOCKING : N - n_count;
                int band = (n_inc + nthr - 1) / nthr;
                band = rnd_up(band, 8);
                int offset = band * ithr;
                if (offset > n_inc) {
                    offset = 0;
                    band = 0;
                }
                if (offset + band > n_inc) {
                    band = n_inc - offset;
                }
                if (band > 0) {
                    packing_b_k19(&B(k_count, n_count + offset), b_buffer_local + offset * k_inc, LDB, k_inc, band);
                }
                #pragma omp barrier
                if (!a_buffer_local) {
                    a_buffer_local = (double *)aligned_alloc(4096, sizeof(double) * (M_BLOCKING * k_inc));
                }

                for (m_count = 0; m_count < m_block; m_count += m_inc) {
                    m_inc = (m_block - m_count > M_BLOCKING) ? M_BLOCKING : m_block - m_count;
                    packing_a_k19(alpha, &A(m_offset + m_count, k_count), a_buffer_local, LDA, m_inc, k_inc);
                    double *b_buff_ptr = b_buffer_local + k_inc * n_count;
                    macro_kernel_k19(a_buffer_local, b_buff_ptr, m_inc, n_inc, k_inc, &C(m_count + m_offset, n_count), LDC);
                }

                #pragma omp barrier
            }
        }

        if (a_buffer_local) {
            free(a_buffer_local);
        }
    }

    if (b_buffer_global) {
        free(b_buffer_global);
    }
}