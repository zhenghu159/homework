#include <stdio.h>
#include <mpi.h>
#include<time.h>
#define N 10000000

double main(int argc, char *argv[]){
    double x,pi=0.0,sum = 0.0;
    double step = 1.0 / N;
    int id,n;
    double start,end;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &n);
    start=MPI_Wtime();
    for(int i=id;i<N; i=i+n){
        x =(i-0.5)*step;
        sum+=4.0/(1.0+x*x);
    }
    MPI_Send(&sum, 1, MPI_DOUBLE, 0, 99, MPI_COMM_WORLD);
    if(id==0){
        for (int source=0;source<n;source++){
            MPI_Recv(&sum, 1, MPI_DOUBLE, source, 99, MPI_COMM_WORLD, &status);
            pi+=sum*step;
        }
    end=MPI_Wtime();
    double T=end-start;
    printf("MPI_dot结果 pi=%f\n",pi);
    }
    MPI_Finalize();
    return 0;
}
