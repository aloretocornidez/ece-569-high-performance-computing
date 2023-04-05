
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
    // The number of elements is set to 4096 as determined by the assignment requirements.
    const unsigned int SIZE = 4096;

    // Initializng shared memory histogram.
    __shared__ unsigned int sharedHistogram[SIZE];

    // A local histogram is used to avoid multiple hits when completing atmoic adds. Each thread is to be assigned
    unsigned int localHistogram[SIZE];

    /*
     Initialize all values in the private histogram to zero.
    */
    // Initialized using a loop, each thread populates each index.
    // Stride is the size of the block dimension.
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < SIZE; i += blockDim.x)
    {
        sharedHistogram[i] = 0;
        localHistogram[i] = 0;
    }

    // Update the local histogram so that we can minimize atomic adds.
    // Each thread looks through the entire input.
    for (int i = 0; i < num_elements; i++)
    {
        // Adding the value to the corresponding bin.
        // atomicAdd(&(privateHistogram[input[i]]), 1);

        unsigned int value = input[i];
        if (value == threadIdx.x)
        {
            localHistogram[threadIdx.x]++;
        }
    }

    unsigned int thread = threadIdx.x + blockIdx.x * blockDim.x;

    unsigned int stride = blockDim.x * gridDim.x;
    // This atomic adds all of the local histograms into the shared memory.
    for (int i = thread; i < SIZE; i += stride)
    {
        // Adding the value to the corresponding bin.
        atomicAdd(&(sharedHistogram[i]), localHistogram[i]);

        // __syncthreads();
        // atomicAdd(&(bins[i]), sharedHistogram[i]);
    }

    // Wait for all threads to finish their atomic add operations.
    __syncthreads();

    // // Set all global histogram values
    // // This  is the same as the initialization loop.
    // // Increments the index by the amount of threads in the block.
    // for (int i = threadIdx.x; i < SIZE; i += blockDim.x)
    for (int i = thread; i < SIZE; i += stride)
    {
        atomicAdd(&(bins[i]), sharedHistogram[i]);
    }
}

// clipping function
// resets bins that have value larger than 127 to 127.
// that is if bin[i]>127 then bin[i]=127

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins)
{

    // int thread = threadIdx.x + threadIdx.x * blockDim.x;
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x)
    {

        if (i < num_bins)
        {
            if (bins[i] > 127)
            {
                bins[i] = 127;
            }
        }
    }
}
