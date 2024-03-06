#include <float.h>
#include <hip/hip_runtime.h>
#include <stdlib.h>

#include "kmeans.h"

#define WARP_SIZE 64
#define BLOCKSIZE 256
#define BLOCKSIZE_XL 1024
#define DIVUP(x, y) ((x + y - 1) / y)

#define HIPCHECK(err) __checkHipErrors(err, __FILE__, __LINE__)

inline void __checkHipErrors(hipError_t err, const char* file, const int line) {
    if (HIP_SUCCESS != err) {
        const char* errorStr = hipGetErrorString(err);
        fprintf(stderr,
                "checkHipErrors() HIP API error = %04d \"%s\" from file <%s>, "
                "line %i.\n",
                err, errorStr, file, line);
        exit(EXIT_FAILURE);
    }
}


__device__ __forceinline__ double warpReduceSum(double val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_xor(val, offset);
    return val;
}

__device__ __forceinline__ double blockReduceSum(double val) {
    static __shared__ double shared[WARP_SIZE];  // Shared mem for 32 partial sums
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warpReduceSum(val);  // Each warp performs partial reduction

    if (lane == 0)
        shared[wid] = val;  // Write reduced value to shared memory

    __syncthreads();  // Wait for all partial reductions

    // read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;

    if (wid == 0)
        val = warpReduceSum(val);  // Final reduce within first warp

    return val;
}


__global__ void assignment_kernel(int data_n, int class_n, Point* centroids, Point* data, int* partitioned) {
    const uint data_i = blockIdx.x * blockDim.x + threadIdx.x;

    // extern __shared__ Point shm_centroids[];

    const Point p = data[data_i];

    // Load centroids to shared memory
    // for (uint i = threadIdx.x; i < class_n; i += blockDim.x) {
    //     shm_centroids[i].x = centroids[i].x;
    //     shm_centroids[i].y = centroids[i].y;
    // }
    // __syncthreads();

    if (data_i < data_n) {
        double min_dist = DBL_MAX;

        #pragma unroll
        for (uint class_i = 0; class_i < class_n; class_i++) {
            double x = p.x - centroids[class_i].x;
            double y = p.y - centroids[class_i].y;

            double dist = x * x + y * y;

            if (dist < min_dist) {
                partitioned[data_i] = class_i;
                min_dist = dist;
            }
        }
    }
}

__global__ void update_kernel(int data_n, int class_n, Point* centroids, Point* data, int* partitioned) {
    int class_i = blockIdx.x;

    double sum_x = 0, sum_y = 0;
    int count = 0;

    for (int data_i = threadIdx.x; data_i < data_n; data_i+=blockDim.x) {
        if (partitioned[data_i] == class_i) {
            sum_x += data[data_i].x;
            sum_y += data[data_i].y;
            count++;
        }
    }

    sum_x = blockReduceSum(sum_x);
    sum_y = blockReduceSum(sum_y);
    count = blockReduceSum(count);

    if (threadIdx.x == 0) {
        centroids[class_i].x = sum_x / count;
        centroids[class_i].y = sum_y / count;

    }
}

void kmeans(int iteration_n, int class_n, int data_n, Point* centroids, Point* data, int* partitioned) {
    // Loop indices for iteration, data and class
    int i;

    // Allocate GPU memory
    Point *d_centroids, *d_data;
    int *d_partitioned;
    HIPCHECK(hipMalloc(&d_partitioned, sizeof(int) * data_n));
    HIPCHECK(hipMalloc(&d_centroids, sizeof(Point) * class_n));
    HIPCHECK(hipMalloc(&d_data, sizeof(Point) * data_n));

    // Copy data to GPU memory
    HIPCHECK(hipMemcpy(d_centroids, centroids, sizeof(Point) * class_n, hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(d_data, data, sizeof(Point) * data_n, hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(d_partitioned, partitioned, sizeof(int) * data_n, hipMemcpyHostToDevice));

    // Iterate through number of interations
    for (i = 0; i < iteration_n; i++) {
        // Assignment step
        // uint32_t shared_mem_size = class_n * sizeof(Point);
        hipLaunchKernelGGL(assignment_kernel, dim3(DIVUP(data_n, BLOCKSIZE)), dim3(BLOCKSIZE), 0, 0,
                           data_n, class_n, d_centroids, d_data, d_partitioned);
        HIPCHECK(hipGetLastError());
        // Update step
        hipLaunchKernelGGL(update_kernel, dim3(class_n), dim3(BLOCKSIZE_XL), 0, 0,
                           data_n, class_n, d_centroids, d_data, d_partitioned);
        HIPCHECK(hipGetLastError());
    }

    // Copy data back to CPU memory
    HIPCHECK(hipMemcpy(centroids, d_centroids, sizeof(Point) * class_n, hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(partitioned, d_partitioned, sizeof(int) * data_n, hipMemcpyDeviceToHost));
}
