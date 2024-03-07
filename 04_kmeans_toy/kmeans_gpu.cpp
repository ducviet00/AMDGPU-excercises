#include <float.h>
#include <hip/hip_runtime.h>
#include <stdlib.h>

#include "kmeans.h"

#define WARP_SIZE 64
#define BLOCKSIZE 256
#define BLOCKSIZE_XL 1024
#define DIVUP(x, y) ((x + y - 1) / y)

#define HIPCHECK(err) __checkHipErrors(err, __FILE__, __LINE__)

inline void __checkHipErrors(hipError_t err, const char *file, const int line) {
    if (HIP_SUCCESS != err) {
        const char *errorStr = hipGetErrorString(err);
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

__global__ void assignment_kernel(int data_n, int class_n, Point *centroids, Point *data, int *partitioned) {
    const uint data_i = blockIdx.x * blockDim.x + threadIdx.x;

    const Point p = data[data_i];
    // printf("Data %d: (%f, %f)\n", data_i, p.x, p.y);
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

__global__ void sum_count_kernel(int data_n, Point *centroids, Point *data, int *partitioned, int *count) {
    int class_i = blockIdx.x;

    double sum_x = 0, sum_y = 0;
    int c = 0;

    for (int data_i = threadIdx.x; data_i < data_n; data_i += blockDim.x) {
        if (partitioned[data_i] == class_i) {
            sum_x += data[data_i].x;
            sum_y += data[data_i].y;
            c++;
        }
    }

    sum_x = blockReduceSum(sum_x);
    sum_y = blockReduceSum(sum_y);
    c = blockReduceSum(c);

    if (threadIdx.x == 0) {
        centroids[class_i].x = sum_x;
        centroids[class_i].y = sum_y;
        count[class_i] = c;
    }
}

void kmeans(int iteration_n, int class_n, int data_n, Point *centroids, Point *data, int *partitioned) {
    // Loop indices for iteration, data and class
    int i;
    int NGPU;
    HIPCHECK(hipGetDeviceCount(&NGPU));
    NGPU = std::min(NGPU, 2);

    Point *h_centroids[NGPU];
    int *h_count[NGPU];
    for (int j = 0; j < NGPU; j++) {
        HIPCHECK(hipHostMalloc(&h_centroids[j], sizeof(Point) * class_n));
        HIPCHECK(hipHostMalloc(&h_count[j], sizeof(int) * class_n));
    }

    // Create streams
    hipStream_t streams[NGPU];
    for (int j = 0; j < NGPU; j++) {
        HIPCHECK(hipSetDevice(j));
        HIPCHECK(hipStreamCreate(&streams[j]));
    }

    // Allocate GPU memory
    Point *d_centroids[NGPU], *d_data[NGPU];
    int *d_count[NGPU];
    int *d_partitioned[NGPU];
    int s_d[NGPU], e_d[NGPU], n_d[NGPU];
    for (int j = 0; j < NGPU; j++) {
        HIPCHECK(hipSetDevice(j));

        HIPCHECK(hipMalloc(&d_centroids[j], sizeof(Point) * class_n));
        HIPCHECK(hipMalloc(&d_count[j], sizeof(int) * class_n));

        s_d[j] = j * DIVUP(data_n, NGPU);
        e_d[j] = std::min((j + 1) * DIVUP(data_n, NGPU), data_n);
        n_d[j] = e_d[j] - s_d[j];
        // printf("GPU %d: %d %d\n", j, s_d[j], n_d[j]);
        HIPCHECK(hipMalloc(&d_partitioned[j], sizeof(int) * n_d[j]));
        HIPCHECK(hipMalloc(&d_data[j], sizeof(Point) * n_d[j]));

        // Copy data to GPU memory
        HIPCHECK(hipMemcpyHtoDAsync(d_centroids[j], centroids, sizeof(Point) * class_n, streams[j]));
        HIPCHECK(hipMemcpyHtoDAsync(d_data[j], data + s_d[j], sizeof(Point) * n_d[j], streams[j]));
    }

    // Iterate through number of interations
    for (i = 0; i < iteration_n; i++) {
        for (int j = 0; j < NGPU; j++) {
            HIPCHECK(hipSetDevice(j));
            // Assignment step
            hipLaunchKernelGGL(assignment_kernel, dim3(DIVUP(n_d[j], BLOCKSIZE)), dim3(BLOCKSIZE), 0, streams[j],
                               n_d[j], class_n, d_centroids[j], d_data[j], d_partitioned[j]);
            HIPCHECK(hipGetLastError());
            // Sum up and count data for each class
            hipLaunchKernelGGL(sum_count_kernel, dim3(class_n), dim3(BLOCKSIZE_XL), 0, streams[j],
                               n_d[j], d_centroids[j], d_data[j], d_partitioned[j], d_count[j]);
            HIPCHECK(hipGetLastError());
            // Copy centroids back to CPU memory to update
            HIPCHECK(hipMemcpyDtoHAsync(h_centroids[j], d_centroids[j], sizeof(Point) * class_n, streams[j]));
            HIPCHECK(hipMemcpyDtoHAsync(h_count[j], d_count[j], sizeof(int) * class_n, streams[j]));
        }
        for (int j = 0; j < NGPU; j++) {
            HIPCHECK(hipSetDevice(j));
            HIPCHECK(hipDeviceSynchronize());
        }
        // Update centroids by averaging between GPU results
        for (int class_i = 0; class_i < class_n; class_i++) {
            double sum_x = 0, sum_y = 0;
            int count = 0;
            for (int j = 0; j < NGPU; j++) {
                sum_x += h_centroids[j][class_i].x;
                sum_y += h_centroids[j][class_i].y;
                count += h_count[j][class_i];
            }
            centroids[class_i].x = sum_x / count;
            centroids[class_i].y = sum_y / count;
        }
        for (int j = 0; j < NGPU; j++) {
            HIPCHECK(hipSetDevice(j));
            HIPCHECK(hipMemcpyHtoDAsync(d_centroids[j], centroids, sizeof(Point) * class_n, streams[j]));
        }
    }
    // Copy data back to CPU memory
    for (int j = 0; j < NGPU; j++) {
        HIPCHECK(hipSetDevice(j));
        HIPCHECK(hipMemcpyDtoHAsync(partitioned + s_d[j], d_partitioned[j], sizeof(int) * n_d[j], streams[j]));
    }
    for (int j = 0; j < NGPU; j++) {
        HIPCHECK(hipSetDevice(j));
        HIPCHECK(hipDeviceSynchronize());
    }
}
