#include<stdio.h>
#include<time.h>
int main()
{
    long N=10000000000;
    clock_t start, stop;
    double clock_Duration;
    long sum=0.0;
    long i, j, k;
    start = clock();
    for (i=1;i<N-1;i++){
        j = i+1; k = i+2;
        sum += (i + j + k)/2;
    }
    stop = clock();
    clock_Duration =(double)(stop-start)/CLOCKS_PER_SEC;
    printf("串行结果：\nN=%lld,K=%.1lld,clock_Duration=%fs\n",N,sum,clock_Duration);
    return clock_Duration;
}