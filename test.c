#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "utils.h"
#include "mkl.h"

int main(int argc, char *argv[]){
    if (argc != 2) {
        printf("Please select a kernel (range 0 - 19, here 0 is for Intel MKL).\n");
        exit(-1);
    }
    int SIZE[30]={100,200,300,400,500,600,700,800,900,1000,1100,\
                1200,1300,1400,1500,1600,1700,1800,1900,2000,\
                2100,2200,2300,2400,2500,2600,2700,2800,2900,3000};//testing 100-3000 square matrices
    int kernel_num=atoi(argv[1]);
    if (kernel_num<0||kernel_num>19) {
        printf("Please enter a valid kernel number (0-19).\n");
        exit(-2);
    }
    int m, n, k,max_size=3000;
    int n_count,N=3,upper_limit;
    if (kernel_num<=4&&kernel_num!=0) upper_limit=10;
    else upper_limit=30;
    double *A=NULL,*B=NULL,*C=NULL,*C_ref=NULL;
    double alpha = 2.0, beta = 0.;//two arbitary input parameters
    double t0,t1;
    A=(double *)malloc(sizeof(double)*max_size*max_size);
    B=(double *)malloc(sizeof(double)*max_size*max_size);
    C=(double *)malloc(sizeof(double)*max_size*max_size);
    C_ref=(double *)malloc(sizeof(double)*max_size*max_size);

    randomize_matrix(A,max_size,max_size);randomize_matrix(B,max_size,max_size);randomize_matrix(C,max_size,max_size);copy_matrix(C,C_ref,max_size*max_size);
    for (int i_count=0;i_count<upper_limit;i_count++){
        m=n=k=SIZE[i_count];
        printf("\nM=N=K=%d:\n",m);
        if (kernel_num != 0){//not an MKL implementation
            test_kernel(kernel_num,m,n,k,alpha,A,B,beta,C);
            cblas_dgemm(CblasColMajor, CblasNoTrans,CblasNoTrans,m,n,k,alpha,A,m,B,k,beta,C_ref,m);
            if (!verify_matrix(C_ref,C,m*n)) {
                printf("Failed to pass the correctness verification against Intel MKL. Exited.\n");
                exit(-3);
            }
        }
        t0=get_sec();
        for (n_count=0;n_count<N;n_count++){
            test_kernel(kernel_num,m,n,k,alpha,A,B,beta,C);
        }
        t1=get_sec();
        printf("Average elasped time: %f second, performance: %f GFLOPS.\n", (t1-t0)/N,2.*1e-9*N*m*n*k/(t1-t0));
        copy_matrix(C_ref,C,m*n);//sync C with Intel MKl to prepare for the next run
    }
    free(A);free(B);free(C);free(C_ref);
    return 0;
}