#include <wb.h>

#define wbCheck(stmt)                                                \
  do                                                                 \
  {                                                                  \
    cudaError_t err = stmt;                                          \
    if (err != cudaSuccess)                                          \
    {                                                                \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                    \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err)); \
      return -1;                                                     \
    }                                                                \
  } while (0)

#define TILE_WIDTH 16

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns)
{
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use tiling with shared memory for arbitrary size
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int column = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ float Ashared[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Bshared[TILE_WIDTH][TILE_WIDTH];

  // Boundary condition to make sure only threads that need to conduct calculations are participating.
  if (row < numCRows && column < numCColumns)
  {
    // Accumulator value
    float cValue = 0;

    for (int i = 0; i < (TILE_WIDTH + numAColumns - 1) / TILE_WIDTH; i++)
    {
      // Copying the matrices from global memory to shared memory.
      if (i * TILE_WIDTH + threadIdx.x < numAColumns && row < numARows)
        Ashared[threadIdx.y][threadIdx.x] = A[row * numAColumns + i * TILE_WIDTH + threadIdx.x];
      // If the thread is out of bounds, just set the value at that address to zero so that accumulation stays the same. (Saves conditions)
      else
        Ashared[threadIdx.y][threadIdx.x] = 0.0;

      // Copying the matrices from global memory to shared memory.
      if (i * TILE_WIDTH + threadIdx.y < numBRows && column < numBColumns)
        Bshared[threadIdx.y][threadIdx.x] = B[(i * TILE_WIDTH + threadIdx.y) * numBColumns + column];
      // If the thread is out of bounds, just set the value at that address to zero so that accumulation stays the same. (Saves conditions)
      else
        Bshared[threadIdx.y][threadIdx.x] = 0.0;

      __syncthreads();

      for (int i = 0; i < TILE_WIDTH; i++)
        cValue += Ashared[threadIdx.y][i] * Bshared[i][threadIdx.x];

      __syncthreads();
    }

    C[row * numCColumns + column] = cValue;
  }
}

int main(int argc, char **argv)
{
  wbArg_t args;
  float *hostA;    // The A matrix
  float *hostB;    // The B matrix
  float *hostC;    // The output C matrix
  float *deviceA;  // A matrix on device
  float *deviceB;  // B matrix on device
  float *deviceC;  // C matrix on device
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C(you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);

  //@@ Set numCRows and numCColumns
  numCRows = numARows;       // set to correct value
  numCColumns = numBColumns; // set to correct value
  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));


  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
  wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void **)&deviceA, numAColumns * numARows * sizeof(float));
  cudaMalloc((void **)&deviceB, numBColumns * numBRows * sizeof(float));
  cudaMalloc((void **)&deviceC, numCColumns * numCRows * sizeof(float));

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, sizeof(float) * numAColumns * numARows, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, sizeof(float) * numBColumns * numBRows, cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  // note that TILE_WIDTH is set to 16 on line number 13.
  const dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);

  const dim3 blocksPerGrid((numCColumns + TILE_WIDTH - 1) / TILE_WIDTH, (numCRows + TILE_WIDTH - 1) / TILE_WIDTH);
  // const dim3 blocksPerGrid(ceil(numCColumns / (TILE_WIDTH)), ceil(numCRows / TILE_WIDTH));

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<blocksPerGrid, threadsPerBlock>>>(deviceA, deviceB, deviceC,
                                                           numARows, numAColumns,
                                                           numBRows, numBColumns,
                                                           numCRows, numCColumns);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostA, deviceA, sizeof(float) * numARows * numAColumns, cudaMemcpyDeviceToHost);
  cudaMemcpy(hostB, deviceB, sizeof(float) * numBRows * numBColumns, cudaMemcpyDeviceToHost);
  cudaMemcpy(hostC, deviceC, sizeof(float) * numCRows * numCColumns, cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
