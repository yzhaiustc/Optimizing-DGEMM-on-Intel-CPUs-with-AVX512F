#include "immintrin.h"
#include <stdint.h>
#define A(i,j) A[(i)+(j)*LDA]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]
#define OUT_M_BLOCKING 1152
#define OUT_N_BLOCKING 9216
#define M_BLOCKING 192
#define N_BLOCKING 96
#define K_BLOCKING 384


void scale_c(double *C,int M, int N, int LDC, double scalar){
    int i,j;
    for (i=0;i<M;i++){
        for (j=0;j<N;j++){
            C(i,j)*=scalar;
        }
    }
}

void packing_b_24x8_edge_version2(double *src,double *dst,int leading_dim,int dim_first,int dim_second){
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

#define init_3zmm(c1,c2,c3) "vpxorq %%zmm"#c1",%%zmm"#c1",%%zmm"#c1";vpxorq %%zmm"#c2",%%zmm"#c2",%%zmm"#c2";vpxorq %%zmm"#c3",%%zmm"#c3",%%zmm"#c3";"

#define init_6zmm(c1,c2,c3,c4,c5,c6) init_3zmm(c1,c2,c3) init_3zmm(c4,c5,c6)
#define init_m24n1 init_3zmm(8,9,10)
#define init_m24n2 init_m24n1 init_3zmm(11,12,13)
#define init_m24n4 init_m24n2 init_6zmm(14,15,16,17,18,19)
#define init_m24n8 init_m24n4 init_6zmm(20,21,22,23,24,25) init_6zmm(26,27,28,29,30,31)

//r11 == r12
#define KERNEL_h_k1m24n1 \
  "vmovaps (%0),%%zmm1; vmovaps 64(%0),%%zmm2; vmovaps 128(%0),%%zmm3; addq $192,%0;"\
  "vbroadcastsd (%1),%%zmm4; vfmadd231pd %%zmm1,%%zmm4,%%zmm8; vfmadd231pd %%zmm2,%%zmm4,%%zmm9; vfmadd231pd %%zmm3,%%zmm4,%%zmm10;"
#define KERNEL_k1m24n1 KERNEL_h_k1m24n1 "addq $8,%1;"
#define KERNEL_h_k1m24n2 KERNEL_h_k1m24n1\
  "vbroadcastsd 8(%1),%%zmm5; vfmadd231pd %%zmm1,%%zmm5,%%zmm11; vfmadd231pd %%zmm2,%%zmm5,%%zmm12; vfmadd231pd %%zmm3,%%zmm5,%%zmm13;"
#define KERNEL_k1m24n2 KERNEL_h_k1m24n2 "addq $16,%1;"
#define unit_acc_m24n2(c1_no,c2_no,c3_no,c4_no,c5_no,c6_no,...)\
  "vbroadcastsd ("#__VA_ARGS__"),%%zmm4; vfmadd231pd %%zmm1,%%zmm4,%%zmm"#c1_no"; vfmadd231pd %%zmm2,%%zmm4,%%zmm"#c2_no"; vfmadd231pd %%zmm3,%%zmm4,%%zmm"#c3_no";"\
  "vbroadcastsd 8("#__VA_ARGS__"),%%zmm5; vfmadd231pd %%zmm1,%%zmm5,%%zmm"#c4_no"; vfmadd231pd %%zmm2,%%zmm5,%%zmm"#c5_no"; vfmadd231pd %%zmm3,%%zmm5,%%zmm"#c6_no";"
#define KERNEL_h_k1m24n4 KERNEL_h_k1m24n2 unit_acc_m24n2(14,15,16,17,18,19,%1,%%r9,1)
#define KERNEL_k1m24n4 KERNEL_h_k1m24n4 "addq $16,%1;"
#define KERNEL_k1m24n6 KERNEL_h_k1m24n4 unit_acc_m24n2(20,21,22,23,24,25,%1,%%r9,2) "addq $16,%1;"
#define KERNEL_h_k1m24n8 KERNEL_k1m24n6 unit_acc_m24n2(26,27,28,29,30,31,%%r13)
#define KERNEL_k1m24n8 KERNEL_h_k1m24n8 "addq $16,%%r13;"


#define save_init_m24 "movq %2,%%r14; addq $192,%2;"
#define SAVE_m24n1\
  "vaddpd (%2),%%zmm8,%%zmm8; vmovupd %%zmm8,(%2); vaddpd 64(%2),%%zmm9,%%zmm9; vmovupd %%zmm9,64(%2); vaddpd 128(%2),%%zmm10,%%zmm10; vmovupd %%zmm10,128(%2); addq $192,%2;"

#define unit_save_m24n2(c1_no,c2_no,c3_no,c4_no,c5_no,c6_no)\
  "vaddpd (%%r14),%%zmm"#c1_no",%%zmm"#c1_no"; vmovupd %%zmm"#c1_no",(%%r14); vaddpd 64(%%r14),%%zmm"#c2_no",%%zmm"#c2_no"; vmovupd %%zmm"#c2_no",64(%%r14); vaddpd 128(%%r14),%%zmm"#c3_no",%%zmm"#c3_no"; vmovupd %%zmm"#c3_no",128(%%r14);"\
  "vaddpd (%%r14,%3,1),%%zmm"#c4_no",%%zmm"#c4_no"; vmovupd %%zmm"#c4_no",(%%r14,%3,1); vaddpd 64(%%r14,%3,1),%%zmm"#c5_no",%%zmm"#c5_no"; vmovupd %%zmm"#c5_no",64(%%r14,%3,1); vaddpd 128(%%r14,%3,1),%%zmm"#c6_no",%%zmm"#c6_no"; vmovupd %%zmm"#c6_no",128(%%r14,%3,1); leaq (%%r14,%3,2),%%r14;"\

#define SAVE_m24n2 save_init_m24 unit_save_m24n2(8,9,10,11,12,13)
#define SAVE_m24n4 SAVE_m24n2 unit_save_m24n2(14,15,16,17,18,19)
#define SAVE_m24n6 SAVE_m24n4 unit_save_m24n2(20,21,22,23,24,25)
#define SAVE_m24n8 SAVE_m24n6 unit_save_m24n2(26,27,28,29,30,31)

// r14==r12,r12==r9,r15==r13
#define COMPUTE_m24n8 \
  init_m24n8 "movq %%r10,%4; movq %%r12,%1; leaq (%%r12,%%r9,2),%%r13;addq %%r9,%%r13;"\
  "cmpq $16,%4; jb 724783f; movq $16,%4;"\
  "724781:\n\t"\
  KERNEL_k1m24n8 "addq $4,%4;"\
  KERNEL_k1m24n8 \
  KERNEL_k1m24n8 \
  KERNEL_k1m24n8 "cmpq %4,%%r10; jnb 724781b;"\
  "negq %4; leaq 16(%%r10,%4,1),%4;"\
  "724783:\n\t"\
  "testq %4,%4; jz 724789f;"\
  "724785:\n\t"\
  KERNEL_k1m24n8 "decq %4;jnz 724785b;"\
  "724789:\n\t"\
  SAVE_m24n8


#define COMPUTE(ndim) {\
  __asm__ __volatile__(\
    "vbroadcastsd %6,%%zmm0; movq %4,%%r9; movq %4,%%r10; movq %5,%%r11; movq %1, %%r12; salq $4,%%r9;"\
    "cmpq $24,%%r11; jb "#ndim"33101f;"\
    #ndim"33100:\n\t"\
    COMPUTE_m24n##ndim "subq $24,%%r11; cmpq $24,%%r11; jnb "#ndim"33100b;"\
    #ndim"33101:\n\t"\
    "movq %%r10,%4;"\
  :"+r"(a_ptr),"+r"(b_ptr),"+r"(c_ptr),"+r"(ldc_in_bytes),"+r"(K)\
  :"m"(M),"m"(ALPHA)\
  :"r9","r10","r11","r12","r13","r14","r15","cc","memory",\
    "zmm0","zmm1","zmm2","zmm3","zmm4","zmm5","zmm6","zmm7","zmm8","zmm9","zmm10","zmm11","zmm12","zmm13","zmm14","zmm15",\
    "zmm16","zmm17","zmm18","zmm19","zmm20","zmm21","zmm22","zmm23","zmm24","zmm25","zmm26","zmm27","zmm28","zmm29","zmm30","zmm31");\
}

void macro_kernel(double *a_buffer,double *b_buffer,int m,int n,int k,double *C, int LDC,int k_inc,double alpha){
    int m_count,n_count,m_count_sub,n_count_sub;
    if (m==0||n==0||k==0) return;
    int64_t M=(int64_t)m,K=(int64_t)k,ldc_in_bytes=(int64_t)LDC*sizeof(double);
    double *a_ptr,*b_ptr=b_buffer,*c_ptr=C;
    double ALPHA=alpha;
    // printf("m= %d, n=%d, k = %d\n",m,n,k);
    for (n_count_sub=n,n_count=0;n_count_sub>7;n_count_sub-=8,n_count+=8){
        //call the m layer with n=8;
        a_ptr=a_buffer;b_ptr=b_buffer+n_count*k;c_ptr=C+n_count*LDC;
        COMPUTE(8)
    }
    for (;n_count_sub>3;n_count_sub-=4,n_count+=4){
        //call the m layer with n=4
        //kernel_n_4_v2(a_buffer,b_buffer+n_count*k_inc,C+n_count*LDC,m,k_inc,LDC,alpha);
    }
    for (;n_count_sub>1;n_count_sub-=2,n_count+=2){
        //call the m layer with n=2
    }
    for (;n_count_sub>0;n_count_sub-=1,n_count+=1){
        //call the m layer with n=1
    }
}


void dgemm_asm(\
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
    double *b_buffer = (double *)aligned_alloc(4096,K_BLOCKING*OUT_N_BLOCKING*sizeof(double));
    double *a_buffer = (double *)aligned_alloc(4096,K_BLOCKING*OUT_M_BLOCKING*sizeof(double));
    int second_m_count,second_n_count,second_m_inc,second_n_inc;
    int m_count,n_count,k_count;
    int m_inc,n_inc,k_inc;
    for (k_count=0;k_count<K;k_count+=k_inc){
        k_inc=(K-k_count>K_BLOCKING)?K_BLOCKING:K-k_count;
        for (n_count=0;n_count<N;n_count+=n_inc){
            n_inc=(N-n_count>OUT_N_BLOCKING)?OUT_N_BLOCKING:N-n_count;
            packing_b_24x8_edge_version2(B+k_count+n_count*LDB,b_buffer,LDB,k_inc,n_inc);
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