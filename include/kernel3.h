#define A(i,j) A[(i)+(j)*LDA]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]

void scale_c_k3(double *C,int M, int N, int LDC, double scalar){
    int i,j;
    for (i=0;i<M;i++){
        for (j=0;j<N;j++){
            C(i,j)*=scalar;
        }
    }
}

void mydgemm_cpu_opt_k3(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    int i,j,k;
    if (beta != 1.0) scale_c_k3(C,M,N,LDC,beta);
    for (i=0;i<M;i++){
        for (j=0;j<N;j++){
            double tmp=C(i,j);
            for (k=0;k<K;k++){
                tmp += alpha*A(i,k)*B(k,j);
            }
            C(i,j) = tmp;
        }
    }
}

void mydgemm_cpu_v3(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    int i,j,k;
    if (beta != 1.0) scale_c_k3(C,M,N,LDC,beta);
    int M2=M&-2,N2=N&-2;
    for (i=0;i<M2;i+=2){
        for (j=0;j<N2;j+=2){
            double c00=C(i,j);
            double c01=C(i,j+1);
            double c10=C(i+1,j);
            double c11=C(i+1,j+1);
            for (k=0;k<K;k++){
                double a0 = alpha*A(i,k);
                double a1 = alpha*A(i+1,k);
                double b0 = B(k,j);
                double b1 = B(k,j+1);
                c00 += a0*b0;
                c01 += a0*b1;
                c10 += a1*b0;
                c11 += a1*b1;
            }
            C(i,j) = c00;
            C(i,j+1) = c01;
            C(i+1,j) = c10;
            C(i+1,j+1) = c11;
        }
    }
    if (M2==M&&N2==N) return;
    // boundary conditions
    if (M2!=M) mydgemm_cpu_opt_k3(M-M2,N,K,alpha,A+M2,LDA,B,LDB,1.0,&C(M2,0),LDC);
    if (N2!=N) mydgemm_cpu_opt_k3(M2,N-N2,K,alpha,A,LDA,&B(0,N2),LDB,1.0,&C(0,N2),LDC);
}