#include<stdio.h>
#include<time.h>
static long num_steps = 1000000000;
double step;
clock_t start, stop;
double clock_Duration;
int main()
{
 start = clock();
 int i;
 double x,pi,sum=0.0;
 step = 1.0/(double)num_steps;
        for (i=1; i<=num_steps; i++){
                x=(i-0.5)*step;
                sum +=4.0/(1.0+x*x);
        }
        pi = step*sum;
        stop = clock();
        clock_Duration =(double)(stop-start)/CLOCKS_PER_SEC;
    printf("num_steps=%d,pi=%f,clock_Duration=%f\n",num_steps,pi,clock_Duration);
        return 0;
}