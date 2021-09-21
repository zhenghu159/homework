#include <stdio.h>
#include <time.h>
#include<stdlib.h>
#define N 2000
int main(int argc, char * argv[])
{
    int **A, **B, **C;
    A=(int**)malloc(N*sizeof(int*));
    B=(int**)malloc(N*sizeof(int*));
    C=(int**)malloc(N*sizeof(int*));
    for(int i=0;i<N;i++){
        A[i]=(int*)malloc(sizeof(int)*N);
        B[i]=(int*)malloc(sizeof(int)*N);
        C[i]=(int*)malloc(sizeof(int)*N);
    }
    clock_t t_start, t_end;
    srand(time(NULL));
    for(int i=0; i<N; i++)
    {
        for (int j=0; j<N; j++)
        {
            A[i][j] = rand()%10;
            B[i][j] = rand()%10;
        }
    }
    t_start = clock();
    for (int i=0; i<N; i++)
    {
        for (int j=0; j<N; j++)
        {
            for(int k=0; k<N; k++)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    t_end = clock();
    double time;
    time = (double)(t_end - t_start)/CLOCKS_PER_SEC;
    for(int i=0;i<N;i++){
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    free(A);free(B);free(C);
    printf("串行时间为：%.2fs\n",time);
    return 0;
}