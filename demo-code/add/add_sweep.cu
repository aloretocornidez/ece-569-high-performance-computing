/*
load the cuda module:
$ module load cuda11/11.0

to compile: 
$ nvcc -o myadd add_sweep.cu

to execute: 
$ ./myadd


Introduce variables “bsize” and “psize” for block size array and problem size arrays
Initialize “bsize” array with 16, 32, 64, 128, 256, 512, 1024
Initialize “psize” array with 1000, 1000000,  10000000
Comment out printf from kernel and for loops printing 5 elements
Modify printf for total execution time so that it includes the block size and problem size
Discuss your findings 

*/


#include<cuda.h>
#include <cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>

// define the kernel 
__global__ void AddInts(int * k_a, int* k_b, int k_count){

int  tid;

tid =  blockDim.x* blockIdx.x + threadIdx.x;

if (tid < k_count) {
 // print thread id and blockid for the 16 blocks and 1 thread/block configuration
//  printf("my global id is %d and I am thread %d in block %d\n", tid, threadIdx.x, blockIdx.x);
 k_a[tid] = k_a[tid]+k_b[tid];
 }

}

int main()
{
int i,j;
int* d_a;
int* d_b;

int* h_a;
int* h_b;


cudaEvent_t startEvent, stopEvent;
float elapsedTime;

int count;

int bsize[] = {16, 32, 64, 128, 256, 512, 1024};
int psize[] = {1000,1000000, 10000000};
cudaEventCreate(&startEvent);
cudaEventCreate(&stopEvent);
for ( j=0;j<3; j++) {

count = psize[j];

h_a = (int*)malloc(count*sizeof(int));
h_b = (int*)malloc(count*sizeof(int));

for (int k=0;i<count;i++) {
  h_a[k] = rand()%1000;
  h_b[k] = rand()%1000;
}
//printf("before addition\n");
//for(i=0;i<5;i++)
//   printf("%d and %d\n",h_a[i],h_b[i]);





/* allocate memory on device, check for failure */
if (cudaMalloc((void **) &d_a, count*sizeof(int)) != cudaSuccess) {
 printf("malloc error for d_a\n");
 return 0;
 }
 
 if (cudaMalloc((void **) &d_b, count*sizeof(int)) != cudaSuccess) {
 printf("malloc error for d_b\n");
 cudaFree(d_a);
 return 0;
 }
 
 /* copy data to device, check for failure, free device if needed */
if (cudaMemcpy(d_a,h_a,count*sizeof(int),cudaMemcpyHostToDevice) != cudaSuccess){
  cudaFree(d_a);
  cudaFree(d_b);
  printf("data transfer error from host to device on d_a\n");
  return 0;
 }
if (cudaMemcpy(d_b,h_b,count*sizeof(int),cudaMemcpyHostToDevice) != cudaSuccess){
  cudaFree(d_a);
  cudaFree(d_b);
  printf("data transfer error from host to device on d_b\n");
  return 0;
 }
 

 for (i=0;i<7;i++) {
 
 



 int threads = bsize[i];


srand(time(NULL));







/* 
generic kernel launch: 
b: blocks
t: threads
shmem: amount of shard memory allocated per block, 0 if not defined

AddInts<<<dim3(bx,by,bz), dims(tx,ty,tz),shmem>>>(parameters)
dim3(w,1,1) = dim3(w) = w

AddInts<<<dim3(4,4,2),dim3(8,8)>>>(....)

How many blocks?
How many threads/blocks?
How many threads?

*/

/* 
Activities:
 1) set the grid size and block size with the dim3 structure and launch the kernel 
 intitially set the block size to 256 and determine the grid size 
 launch the kernel
 
 2) experiment with printing block ids for the configuration of
 16 blocks and 1 thread per block. For this second experiment insert printf statement 
 in the kernel. you will need cudaDeviceSynchronize() call after kernel launch to 
 flush the printfs. 

 3) run a sweeping experiment by using the following block sizes: 16, 32, 64, 128, 256, 512 and monitor
 the execution time. For this set the array size to 1000, then increase to 1,000,000, then to 10,000,000 (count value). 
 
*/


 
 
 dim3 mygrid(ceil(count/float(threads)));
 dim3 myblock(threads);


//dim3 mygrid(ceil(count/256.0));
//dim3 myblock(256);

//dim3 mygrid(16);
//dim3 myblock(2);
cudaEventRecord(startEvent, 0);
AddInts<<<mygrid,myblock>>>(d_a,d_b,count);
cudaEventRecord(stopEvent, 0);
cudaEventSynchronize(stopEvent);

//if printing from the kernel flush the printfs 
//cudaDeviceSynchronize();


// retrieve data from the device, check for error, free device if needed 
if (cudaMemcpy(h_a,d_a,count*sizeof(int),cudaMemcpyDeviceToHost) != cudaSuccess){
  cudaFree(d_a);
  cudaFree(d_b);
  printf("data transfer error from host to device on d_a\n");
  return 0;
 }
 

cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
printf("Total execution time data size %d block size %d %fms\n",count, threads, elapsedTime);
//for(i=0;i<5;i++)
//   printf("%d \n",h_a[i]);




}
}
cudaEventDestroy(startEvent);
cudaEventDestroy(stopEvent);
return 0;
}

