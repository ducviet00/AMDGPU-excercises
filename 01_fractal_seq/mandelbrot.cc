#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "hip/hip_runtime.h"
#include "timers.h"

#define COUNT_MAX 5000
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

#ifdef SAVE_JPG
void save_jpeg_image(const char *filename, unsigned char *r, unsigned char *g, unsigned char *b, int image_width, int image_height);
#endif

#define HIPCHECK(err) __checkHipErrors(err, __FILE__, __LINE__)

inline void __checkHipErrors(hipError_t err, const char *file, const int line)
{
  if (HIP_SUCCESS != err)
  {
    const char *errorStr = hipGetErrorString(err);
    fprintf(stderr,
            "checkHipErrors() HIP API error = %04d \"%s\" from file <%s>, "
            "line %i.\n",
            err, errorStr, file, line);
    exit(EXIT_FAILURE);
  }
}

__global__ void mandelbrotKernel(unsigned char *r, unsigned char *g, unsigned char *b, int m, int n,
                                 float x_max, float x_min,
                                 float y_max, float y_min)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= m || j >= n)
    return;

  int c, k;
  float x, x1, x2, y, y1, y2, x1y1;

  int count_max = COUNT_MAX;

  x = ((float)(j - 1) * x_max + (float)(m - j) * x_min) / (float)(m - 1);

  y = ((float)(i - 1) * y_max + (float)(n - i) * y_min) / (float)(n - 1);

  int count = 0;

  x1 = x;
  y1 = y;

  for (k=1; k<=count_max; k++)
  {
    x2 = x1 * x1 - y1 * y1 + x;
    x1y1 = x1 * y1;
    y2 = x1y1 + x1y1 + y;

    if (x2 < -2.0 || 2.0 < x2 || y2 < -2.0 || 2.0 < y2)
    {
      count = k;
      break;
    }
    x1 = x2;
    y1 = y2;
  }

  if ((count % 2) == 1)
  {
    r[i * n + j] = 255;
    g[i * n + j] = 255;
    b[i * n + j] = 255;

    r[(m-1-i) * n + j] = 255;
    g[(m-1-i) * n + j] = 255;
    b[(m-1-i) * n + j] = 255;
  }
  else
  {
    c = (int)(255.0 * sqrtf(sqrtf(sqrtf(
                          ((float)(count) / (float)(count_max))))));
    r[i * n + j] = 3 * c / 5;
    g[i * n + j] = 3 * c / 5;
    b[i * n + j] = c;

    r[(m-1-i) * n + j] = 3 * c / 5;
    g[(m-1-i) * n + j] = 3 * c / 5;
    b[(m-1-i) * n + j] = c;
  }
}

void mandelbrot(int m, int n, char *output_filename)
{

  int count_max = COUNT_MAX;

#ifndef SAVE_JPG
  int jhi, jlo;
  FILE *output_unit;
#endif
  double wtime;

  float x_max = 1.25;
  float x_min = -2.25;

  float y_max = 1.75;
  float y_min = -1.75;

  unsigned char *r, *g, *b;
  HIPCHECK(hipHostMalloc(&r, sizeof(char) * m * n, hipHostMallocNumaUser));
  HIPCHECK(hipHostMalloc(&g, sizeof(char) * m * n, hipHostMallocNumaUser));
  HIPCHECK(hipHostMalloc(&b, sizeof(char) * m * n, hipHostMallocNumaUser));

  printf("  Sequential C version\n");
  printf("\n");
  printf("  Create an ASCII PPM image of the Mandelbrot set.\n");
  printf("\n");
  printf("  For each point C = X + i*Y\n");
  printf("  with X range [%g,%g]\n", x_min, x_max);
  printf("  and  Y range [%g,%g]\n", y_min, y_max);
  printf("  carry out %d iterations of the map\n", count_max);
  printf("  Z(n+1) = Z(n)^2 + C.\n");
  printf("  If the iterates stay bounded (norm less than 2)\n");
  printf("  then C is taken to be a member of the set.\n");
  printf("\n");
  printf("  An image of the set is created using\n");
  printf("    M = %d pixels in the X direction and\n", m);
  printf("    N = %d pixels in the Y direction.\n", n);

  unsigned char *r_gpu, *g_gpu, *b_gpu;
  HIPCHECK(hipMalloc(&r_gpu, sizeof(char) * m * n));
  HIPCHECK(hipMalloc(&g_gpu, sizeof(char) * m * n));
  HIPCHECK(hipMalloc(&b_gpu, sizeof(char) * m * n));

  timer_init();
  timer_start(0);

  // Carry out the iteration for each pixel, determining COUNT.
  dim3 gDim((m/2 + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (n + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
  dim3 bDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);

  hipLaunchKernelGGL(mandelbrotKernel, gDim, bDim, 0, 0, r_gpu, g_gpu, b_gpu, m, n, x_max, x_min, y_max, y_min);
  HIPCHECK(hipGetLastError());
  HIPCHECK(hipDeviceSynchronize());

  HIPCHECK(hipMemcpy(r, r_gpu, sizeof(char) * m * n, hipMemcpyDeviceToHost));
  HIPCHECK(hipMemcpy(g, g_gpu, sizeof(char) * m * n, hipMemcpyDeviceToHost));
  HIPCHECK(hipMemcpy(b, b_gpu, sizeof(char) * m * n, hipMemcpyDeviceToHost));
  HIPCHECK(hipDeviceSynchronize());

  timer_stop(0);
  wtime = timer_read(0);
  printf("\n");
  printf("  Time = %lf seconds.\n", wtime);

#ifdef SAVE_JPG
  // Write data to an JPEG file.
  save_jpeg_image(output_filename, r, g, b, n, m);
#else
  // Write data to an ASCII PPM file.
  output_unit = fopen(output_filename, "wt");

  fprintf(output_unit, "P3\n");
  fprintf(output_unit, "%d  %d\n", n, m);
  fprintf(output_unit, "%d\n", 255);
  int i, j;
  for (i = 0; i < m; i++)
  {
    for (jlo = 0; jlo < n; jlo = jlo + 4)
    {
      jhi = MIN(jlo + 4, n);
      for (j = jlo; j < jhi; j++)
      {
        fprintf(output_unit, "  %d  %d  %d", r[i * n + j], g[i * n + j], b[i * n + j]);
      }
      fprintf(output_unit, "\n");
    }
  }
  fclose(output_unit);
#endif

  printf("\n");
  printf("  Graphics data written to \"%s\".\n\n", output_filename);

  // Terminate.
  HIPCHECK(hipHostFree(r));
  HIPCHECK(hipHostFree(g));
  HIPCHECK(hipHostFree(b));
  HIPCHECK(hipFree(r_gpu));
  HIPCHECK(hipFree(b_gpu));
  HIPCHECK(hipFree(g_gpu));
}
