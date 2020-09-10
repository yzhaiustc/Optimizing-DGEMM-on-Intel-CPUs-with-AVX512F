#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "utils.h"
#include "mkl.h"
#include "kernels.h"
//#define verbose 1

#define MYDGEMM dgemm_packing_cache_blocking_reg_blocking_24x8_avx512_template_unrollx4_v4

int main(int argc, char *argv[]){
    if (argc != 4) {
        printf("Please enter matrix size [m], [n], [k].\n");
        exit(-1);
    }
    int m, n, k;
    m=atoi(argv[1]);n=atoi(argv[2]);k=atoi(argv[3]);
    if (m==0||n==0||k==0) {
        printf("Please not provide matrix sizes equal to ZERO.\n");
        exit(-2);
    }
    double *A=NULL,*B=NULL,*C=NULL,*C_ref=NULL;
    double alpha = 1, beta = 0;
    A=(double *)malloc(sizeof(double)*m*k);
    B=(double *)malloc(sizeof(double)*k*n);
    C=(double *)malloc(sizeof(double)*m*n);
    C_ref=(double *)malloc(sizeof(double)*m*n);
    if (!A||!B||!C||!C_ref) {
        printf("Malloc failed, exited.\n");
        exit(-3);
    }
#ifdef verbose
    printf("Malloc completed. start initializing...\n");
#endif
    randomize_matrix(A,m,k);randomize_matrix(B,k,n);randomize_matrix(C,m,n);
    copy_matrix(C,C_ref,m*n);
#ifdef verbose
    printf("Initialization completed.\nStart verifying correctness against Intel MKL...\n");
#endif
    MYDGEMM(m,n,k,alpha,A,m,B,k,beta,C,m);
    cblas_dgemm(CblasColMajor, CblasNoTrans,CblasNoTrans,m,n,k,alpha,A,m,B,k,beta,C_ref,m);
    // print_matrix(C,m,n);
    // print_matrix(C_ref,m,n);
    if (!verify_matrix(C_ref,C,m*n)) {
        printf("Incorrect. Exited.\n");
        exit(-4);
    }else{
#ifdef verbose
        printf("Correctness testing passed. Starting benchmarking performance.\n");
#endif
    }

    int n_count,N=10;
    double t0,t1;
#ifdef verbose
    printf("Start benchmarking MYDGEMM...\n");
#endif
    t0=get_sec();
    for (n_count=0;n_count<N;n_count++){
       MYDGEMM(m,n,k,alpha,A,m,B,k,beta,C,m);
    }
    t1=get_sec();
    printf("Average elasped time: %f second, performance: %f GFLOPS.\n", (t1-t0)/N,2.*1e-9*N*m*n*k/(t1-t0));
#ifdef verbose
    printf("Start benchmarking MKL DGEMM...\n");
#endif
    t0=get_sec();
    for (n_count=0;n_count<N;n_count++){
       cblas_dgemm(CblasColMajor, CblasNoTrans,CblasNoTrans,m,n,k,alpha,A,m,B,k,beta,C_ref,m);
    }
    t1=get_sec();
    printf("Average elasped time: %f second, performance: %f GFLOPS.\n", (t1-t0)/N,2.*1e-9*N*m*n*k/(t1-t0));
    free(A);free(B);free(C);free(C_ref);
    return 0;
}