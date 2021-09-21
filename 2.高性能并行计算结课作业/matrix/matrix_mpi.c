#include <stdio.h>
#include <time.h>
#include<stdlib.h>
#include<mpi.h>
#define N 2000
int main(int argc, char * argv[])
{
    double t_start, t_end;
    int **A, **B, **C, **D;
    int rank,np;
    MPI_Status status;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    A=(int**)malloc(N*sizeof(int*));
    B=(int**)malloc(N*sizeof(int*));
    C=(int**)malloc(N*sizeof(int*));
    D=(int**)malloc(N*sizeof(int*));
    for(int i=0;i<N;i++){
        A[i]=(int*)malloc(sizeof(int)*N);
        B[i]=(int*)malloc(sizeof(int)*N);
        C[i]=(int*)malloc(sizeof(int)*N);
        D[i]=(int*)malloc(sizeof(int)*N);
    }
    srand(time(NULL));
    for(int i=0; i<N; i++)
    {
        for (int j=0; j<N; j++)
        {
            A[i][j] = rand()%10;
            B[i][j] = rand()%10;
        }
    }
    t_start = MPI_Wtime();
    for (int i=0; i<N; i++)
    {
        for (int j=0; j<N; j++)
        {
            C[i][j]=0;
            for(int k=rank; k<N; k+=np)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
        {
            MPI_Reduce(&C[i][j], &D[i][j], 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        }
    }
    t_end = MPI_Wtime();
    if(rank==0)
        printf("MPI时间为：%.2fs\n",t_end-t_start);
    MPI_Finalize();
    for(int i=0;i<N;i++){
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    free(A);free(B);free(C);
    return 0;
}