#ifndef _UTIL_H_
#define _UTIL_H_
#include "sys/time.h"
#include <stdbool.h>

void randomize_matrix(double *A, int m, int n);
double get_sec();
void print_matrix(const double *A, int m, int n);
void print_vector(double *vec, int n);
void copy_matrix(double *src, double *dest, int n);
bool verify_matrix(double *mat1, double *mat2, int n);
void test_kernel(int kernel_num,int m,int n,int k,double alpha,double *A,double *B,double beta,double *C);
#endif // _UTIL_H_