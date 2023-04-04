// version 0
// global memory only interleaved version
// include comments describing your approach
__global__ void histogram_global_kernel(unsigned int *input, unsigned int *bins, unsigned int num_elements, unsigned int num_bins)
{

    // Insert your code here
    int thread = threadIdx.x + blockIdx.x * blockDim.x;

    // Stride
    int stride = blockDim.x * gridDim.x;

    while (thread < num_elements)
    {
        // Value of the data point
        int value = input[thread];

        // Adding the value to the corresponding bin.
        atomicAdd(&(bins[value]), 1);

        // Incrementing the stride value.
        thread += stride;
    }
}

// version 1
// shared memory privatized version
// include comments describing your approach
__global__ void histogram_shared_kernel(unsigned int *input, unsigned int *bins, unsigned int num_elements, unsigned int num_bins)
{
    // Initializng shared memory histogram.
    __shared__ unsigned int privateHistogram[4096];
    // Thread id.
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;

    // Stride
    unsigned int stride = blockDim.x * gridDim.x;

    // Initialize all values in the private histogram to zero.
    // While loop used because there may not be enough threads in a block to initialize all bins.
    unsigned int index = threadIdx.x;
    unsigned int initializationStride = blockDim.x;
    while (index < 4096)
    {
        privateHistogram[index] = 0;

        // This increases the index by the amount of threads in the block.
        index += initializationStride;
    }


    // Update the shared memory histogram.
    while (i < num_elements)
    {
        // Value of the data point
        unsigned int value = input[i];

        // Adding the value to the corresponding bin.
        atomicAdd(&(privateHistogram[value]), 1);

        // Incrementing the stride value.
        i += stride;
    }

    // Wait for all threads to finish their atomic add operations.
    __syncthreads();

    // Set all global histogram values
    index = threadIdx.x;
    while (index < 4096)
    {
        atomicAdd(&(bins[index]), privateHistogram[index]);

        // This increases the index by the amount of threads in the block.
        index += initializationStride;
    }
}

// version 2
// your method of optimization using shared memory
// include DETAILED comments describing your approach
// for competition you need to include description of the idea
// where you borrowed the idea from, and how you implmented
__global__ void histogram_shared_optimized(unsigned int *input, unsigned int *bins, unsigned int num_elements, unsigned int num_bins)
{

    // insert your code here
}

// clipping function
void clippingFunction()
{
}
// resets bins that have value larger than 127 to 127.
// that is if bin[i]>127 then bin[i]=127

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins)
{

    // insert your code here
}
