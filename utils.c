#include <stdio.h>
#include <stdbool.h>
#include "utils.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "kernels.h"
#include "mkl.h"
void print_vector(double *vec, int n){
    int i;
    for (i = 0; i < n; i++){
        printf(" %5.2f, ", vec[i]);
    }
    printf("\n");
}

void print_matrix_lda(const double *A, int lda, int m, int n)
{
    int i,j;
    double *a_curr_pos=A;
    printf("[");
    for (i = 0; i < n; i++){
        for (j=0;j<m;j++){
            printf("%5.2f, ",*a_curr_pos);
            a_curr_pos++;
        }
        a_curr_pos+=(lda-m);
    }
    printf("]\n");
}

void print_matrix(const double *A, int m, int n){
    int i;printf("[");
    for (i = 0; i < m * n; i++){
        if ((i + 1) % n == 0)
            printf("%5.2f ", A[i]);
        else
            printf("%5.2f, ", A[i]);
        if ((i + 1) % n == 0){
            if (i + 1 < m * n)
                printf(";\n");
        }
    }
    printf("]\n");
}

double get_sec(){
    struct timeval time;
    gettimeofday(&time, NULL);
    return (time.tv_sec + 1e-6 * time.tv_usec);
}

void randomize_matrix(double *A, int m, int n){
    srand(time(NULL));
    int i, j;
    for (i = 0; i < m; i++){
        for (j = 0; j < n; j++){
            A[i * n + j] = (double)(rand() % 100) + 0.01 * (rand() % 100);
            if (rand() % 2 == 0) A[i * n + j] *= 1.0;
        }
    }
}

void copy_matrix(double *src, double *dest, int n){
    int i;
    for (i = 0; src + i && dest + i && i < n; i++){
        *(dest + i) = *(src + i);
    }
    if (i != n){
        printf("copy failed at %d while there are %d elements in total.\n", i, n);
    }
}

bool verify_matrix(double *mat1, double *mat2, int n){
    double diff = 0.0;
    int i;
    for (i = 0; mat1 + i && mat2 + i && i < n; i++){
        diff = fabs(mat1[i] - mat2[i]);
        if (diff > 1e-2) {
            printf("error. %5.2f,%5.2f,%d\n", mat1[i],mat2[i],i);
            return false;
        }
    }
    return true;
}

void test_mkl(int m,int n,int k,double alpha,double *A,double *B,double beta,double *C){
    cblas_dgemm(CblasColMajor, CblasNoTrans,CblasNoTrans,m,n,k,alpha,A,m,B,k,beta,C,m);
}

void test_mydgemm_v1(int m,int n,int k,double alpha,double *A,double *B,double beta,double *C){
    mydgemm_cpu_v1(m,n,k,alpha,A,m,B,k,beta,C,m);
}

void test_mydgemm_v2(int m,int n,int k,double alpha,double *A,double *B,double beta,double *C){
    mydgemm_cpu_v2(m,n,k,alpha,A,m,B,k,beta,C,m);
}

void test_mydgemm_v3(int m,int n,int k,double alpha,double *A,double *B,double beta,double *C){
    mydgemm_cpu_v3(m,n,k,alpha,A,m,B,k,beta,C,m);
}

void test_mydgemm_v4(int m,int n,int k,double alpha,double *A,double *B,double beta,double *C){
    mydgemm_cpu_v4(m,n,k,alpha,A,m,B,k,beta,C,m);
}

void test_mydgemm_v5(int m,int n,int k,double alpha,double *A,double *B,double beta,double *C){
    mydgemm_cpu_v5(m,n,k,alpha,A,m,B,k,beta,C,m);
}

void test_mydgemm_v6(int m,int n,int k,double alpha,double *A,double *B,double beta,double *C){
    mydgemm_cpu_v6(m,n,k,alpha,A,m,B,k,beta,C,m);
}

void test_mydgemm_v7(int m,int n,int k,double alpha,double *A,double *B,double beta,double *C){
    mydgemm_cpu_v7(m,n,k,alpha,A,m,B,k,beta,C,m);
}

void test_mydgemm_v8(int m,int n,int k,double alpha,double *A,double *B,double beta,double *C){
    mydgemm_cpu_v8(m,n,k,alpha,A,m,B,k,beta,C,m);
}

void test_mydgemm_v9(int m,int n,int k,double alpha,double *A,double *B,double beta,double *C){
    mydgemm_cpu_v9(m,n,k,alpha,A,m,B,k,beta,C,m);
}

void test_mydgemm_v10(int m,int n,int k,double alpha,double *A,double *B,double beta,double *C){
    mydgemm_cpu_v10(m,n,k,alpha,A,m,B,k,beta,C,m);
}

void test_mydgemm_v11(int m,int n,int k,double alpha,double *A,double *B,double beta,double *C){
    mydgemm_cpu_v11(m,n,k,alpha,A,m,B,k,beta,C,m);
}

void test_mydgemm_v12(int m,int n,int k,double alpha,double *A,double *B,double beta,double *C){
    mydgemm_cpu_v12(m,n,k,alpha,A,m,B,k,beta,C,m);
}

void test_mydgemm_v13(int m,int n,int k,double alpha,double *A,double *B,double beta,double *C){
    mydgemm_cpu_v13(m,n,k,alpha,A,m,B,k,beta,C,m);
}

void test_mydgemm_v14(int m,int n,int k,double alpha,double *A,double *B,double beta,double *C){
    mydgemm_cpu_v14(m,n,k,alpha,A,m,B,k,beta,C,m);
}

void test_mydgemm_v15(int m,int n,int k,double alpha,double *A,double *B,double beta,double *C){
    mydgemm_cpu_v15(m,n,k,alpha,A,m,B,k,beta,C,m);
}

void test_mydgemm_v16(int m,int n,int k,double alpha,double *A,double *B,double beta,double *C){
    mydgemm_cpu_v16(m,n,k,alpha,A,m,B,k,beta,C,m);
}

void test_mydgemm_v17(int m,int n,int k,double alpha,double *A,double *B,double beta,double *C){
    mydgemm_cpu_v17(m,n,k,alpha,A,m,B,k,beta,C,m);
}

void test_mydgemm_v18(int m,int n,int k,double alpha,double *A,double *B,double beta,double *C){
    mydgemm_cpu_v18(m,n,k,alpha,A,m,B,k,beta,C,m);
}

void test_kernel(int kernel_num,int m,int n,int k,double alpha,double *A,double *B,double beta,double *C){
    switch (kernel_num){
        case 0: test_mkl(m,n,k,alpha,A,B,beta,C); break;
        case 1: test_mydgemm_v1(m,n,k,alpha,A,B,beta,C); break;
        case 2: test_mydgemm_v2(m,n,k,alpha,A,B,beta,C); break;
        case 3: test_mydgemm_v3(m,n,k,alpha,A,B,beta,C); break;
        case 4: test_mydgemm_v4(m,n,k,alpha,A,B,beta,C); break;
        case 5: test_mydgemm_v5(m,n,k,alpha,A,B,beta,C); break;
        case 6: test_mydgemm_v6(m,n,k,alpha,A,B,beta,C); break;
        case 7: test_mydgemm_v7(m,n,k,alpha,A,B,beta,C); break;
        case 8: test_mydgemm_v8(m,n,k,alpha,A,B,beta,C); break;
        case 9: test_mydgemm_v9(m,n,k,alpha,A,B,beta,C); break;
        case 10: test_mydgemm_v10(m,n,k,alpha,A,B,beta,C); break;
        case 11: test_mydgemm_v11(m,n,k,alpha,A,B,beta,C); break;
        case 12: test_mydgemm_v12(m,n,k,alpha,A,B,beta,C); break;
        case 13: test_mydgemm_v13(m,n,k,alpha,A,B,beta,C); break;
        case 14: test_mydgemm_v14(m,n,k,alpha,A,B,beta,C); break;
        case 15: test_mydgemm_v15(m,n,k,alpha,A,B,beta,C); break;
        case 16: test_mydgemm_v16(m,n,k,alpha,A,B,beta,C); break;
        case 17: test_mydgemm_v17(m,n,k,alpha,A,B,beta,C); break;
        case 18: test_mydgemm_v18(m,n,k,alpha,A,B,beta,C); break;
        default: break;
    }
}