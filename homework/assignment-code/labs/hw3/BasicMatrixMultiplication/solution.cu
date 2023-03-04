
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

// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns)
{
  //@@ Insert code to implement basic matrix multiplication for
  //@@ arbitrary size using global memory.
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int column = blockIdx.x * blockDim.x + threadIdx.x;

  // Boundary condition to make sure only threads that need to conduct calculations are participating.
  if (row < numCRows && column < numCColumns)
  {
    // Accumulator value
    float cValue = 0;

    for (int i = 0; i < numBRows; i++)
    {
      
      float aValue = A[row * numAColumns + i];
      float bValue = B[i * numBColumns + column];

      cValue += aValue * bValue;
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
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);

  //@@Complete Set numCRows and numCColumns
  // If the dimensions for a matrix multiplication are not correct, then return with an error.
  if ((numAColumns != numBRows))
  {
    return -1;
  }

  // Set the correct number rows and columns for the output matrix.
  numCRows = numARows;       // set to correct value
  numCColumns = numBColumns; // set to correct value

  //@@(Complete) Allocate the hostC matrix
  hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));

  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
  wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@(Complete) Allocate GPU memory here for A, B and C
  cudaMalloc((void **)&deviceA, numAColumns * numARows * sizeof(float));
  cudaMalloc((void **)&deviceB, numBColumns * numBRows * sizeof(float));
  cudaMalloc((void **)&deviceC, numCColumns * numCRows * sizeof(float));

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@(Complete) Copy memory to the GPU here for A and B
  cudaMemcpy(deviceA, hostA, sizeof(float) * numAColumns * numARows, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, sizeof(float) * numBColumns * numBRows, cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@(Complete) Initialize the grid and block dimensions here
  // set block size to 16,16 and determine the grid dimensions
  // use dim3 structure for setting block and grid dimensions
  int blockSize = 16;
  const dim3 threadsPerBlock(blockSize, blockSize);
  const dim3 blocksPerGrid(ceil(numCColumns / (float)blockSize), ceil(numCRows / (float)blockSize));

  wbTime_start(Compute, "Performing CUDA computation");
  //@@(Complete) Launch the GPU Kernel here
  matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(deviceA, deviceB, deviceC,
                                                     numARows, numAColumns,
                                                     numBRows, numBColumns,
                                                     numCRows, numCColumns);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@(Complete) Copy the GPU memory back to the CPU here
  cudaMemcpy(hostA, deviceA, sizeof(float) * numARows * numAColumns, cudaMemcpyDeviceToHost);
  cudaMemcpy(hostB, deviceB, sizeof(float) * numBRows * numBColumns, cudaMemcpyDeviceToHost);
  cudaMemcpy(hostC, deviceC, sizeof(float) * numCRows * numCColumns, cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@(Complete) Free the GPU memory here
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
