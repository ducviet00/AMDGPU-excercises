#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "hip/hip_runtime.h"
#include "timers.h"

#define COUNT_MAX 2000
#define NUM_CHANNELS 3
#define BLOCK_SIZE_X 64
#define BLOCK_SIZE_Y 4
#define MIN(x, y) ((x) < (y) ? (x) : (y))

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

typedef struct
{
  char r;
  char g;
  char b;
} RgbColor;

__device__ RgbColor HSVtoRGB(unsigned h, unsigned s, unsigned v);
__global__ void juliaKernel(unsigned char *r, unsigned char *g, unsigned char *b, int w, int h,
                            float cRe, float cIm,
                            float zoom, float moveX, float moveY)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  float newRe, newIm, oldRe, oldIm;
  // after how much iterations the function should stop
  int maxIterations = COUNT_MAX;
  // calculate the initial real and imaginary part of z,
  // based on the pixel location and zoom and position values
  newRe = 1.5 * (x - w / 2) / (0.5 * zoom * w) + moveX;
  newIm = (y - h / 2) / (0.5 * zoom * h) + moveY;

  // i will represent the number of iterations
  int i;
  // start the iteration process

  for (i = 0; i < maxIterations; i++)
  {
    // remember value of previous iteration
    oldRe = newRe;
    oldIm = newIm;

    // the actual iteration, the real and imaginary part are calculated
    newRe = oldRe * oldRe - oldIm * oldIm + cRe;
    newIm = (oldRe + oldRe) * oldIm + cIm;

    // if the point is outside the circle with radius 2: stop
    if ((newRe * newRe + newIm * newIm) > 4)
      break;
  }

  // use color model conversion to get rainbow palette,
  // make brightness black if maxIterations reached
  RgbColor color = HSVtoRGB(i % 256, 255, 255 * (i < maxIterations));
  r[y * w + x] = color.r;
  g[y * w + x] = color.g;
  b[y * w + x] = color.b;
}

// Main part of the below code is originated from Lode Vandevenne's code.
// Please refer to http://lodev.org/cgtutor/juliamandelbrot.html
void julia(int w, int h, char *output_filename)
{
  // each iteration, it calculates: new = old*old + c,
  // where c is a constant and old starts at current pixel

  // real and imaginary part of the constant c
  // determinate shape of the Julia Set
  float cRe, cIm;

  // you can change these to zoom and change position
  float zoom = 1, moveX = 0, moveY = 0;

#ifndef SAVE_JPG
  FILE *output_unit;
#endif

  double wtime;

  // pick some values for the constant c
  // this determines the shape of the Julia Set
  cRe = -0.7;
  cIm = 0.27015;

  // int *r = (int *)calloc(w * h, sizeof(int));
  // int *g = (int *)calloc(w * h, sizeof(int));
  // int *b = (int *)calloc(w * h, sizeof(int));

  unsigned char *r, *g, *b;
  HIPCHECK(hipHostMalloc(&r, sizeof(char) * w * h, hipHostMallocNumaUser));
  HIPCHECK(hipHostMalloc(&g, sizeof(char) * w * h, hipHostMallocNumaUser));
  HIPCHECK(hipHostMalloc(&b, sizeof(char) * w * h, hipHostMallocNumaUser));

  printf("  Sequential C version\n");
  printf("\n");
  printf("  Create an ASCII PPM image of the Julia set.\n");
  printf("\n");
  printf("  An image of the set is created using\n");
  printf("    W = %d pixels in the X direction and\n", w);
  printf("    H = %d pixels in the Y direction.\n", h);

  unsigned char *r_gpu, *g_gpu, *b_gpu;
  HIPCHECK(hipMalloc(&r_gpu, sizeof(char) * w * h));
  HIPCHECK(hipMalloc(&g_gpu, sizeof(char) * w * h));
  HIPCHECK(hipMalloc(&b_gpu, sizeof(char) * w * h));
  HIPCHECK(hipDeviceSynchronize());
  // HIPCHECK(hipMemset(r_gpu, 1, sizeof(int) * w * h));

  timer_init();
  timer_start(0);

  dim3 gDim((w + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (h + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
  dim3 bDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
  // jkernel<<<gDim, bDim>>>(r_gpu, g_gpu, b_gpu, w, h, cRe, cIm, zoom, moveX, moveY);
  hipLaunchKernelGGL(juliaKernel, gDim, bDim, 0, 0, r_gpu, g_gpu, b_gpu, w, h, cRe, cIm, zoom, moveX, moveY);
  HIPCHECK(hipGetLastError());
  HIPCHECK(hipDeviceSynchronize());

  HIPCHECK(hipMemcpy(r, r_gpu, sizeof(char) * w * h, hipMemcpyDeviceToHost));
  HIPCHECK(hipMemcpy(g, g_gpu, sizeof(char) * w * h, hipMemcpyDeviceToHost));
  HIPCHECK(hipMemcpy(b, b_gpu, sizeof(char) * w * h, hipMemcpyDeviceToHost));
  HIPCHECK(hipDeviceSynchronize());

  timer_stop(0);
  wtime = timer_read(0);

  printf("\n");
  printf("  Time = %lf seconds.\n", wtime);

#ifdef SAVE_JPG
  save_jpeg_image(output_filename, r, g, b, w, h);
#else
  // Write data to an ASCII PPM file.
  output_unit = fopen(output_filename, "wt");

  fprintf(output_unit, "P3\n");
  fprintf(output_unit, "%d  %d\n", h, w);
  fprintf(output_unit, "%d\n", 255);
  for (int i = 0; i < h; i++)
  {
    for (int jlo = 0; jlo < w; jlo = jlo + 4)
    {
      int jhi = MIN(jlo + 4, w);
      for (int j = jlo; j < jhi; j++)
      {
        fprintf(output_unit, "  %d  %d  %d", r[i * w + j], g[i * w + j], b[i * w + j]);
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

__device__ RgbColor HSVtoRGB(unsigned h, unsigned s, unsigned v)
{
  RgbColor rgb;
  unsigned char region, remainder, p, q, t;

  if (s == 0)
  {
    rgb.r = v;
    rgb.g = v;
    rgb.b = v;
    return rgb;
  }

  region = h / 43;
  remainder = (h - (region * 43)) * 6;

  p = (v * (255 - s)) >> 8;
  q = (v * (255 - ((s * remainder) >> 8))) >> 8;
  t = (v * (255 - ((s * (255 - remainder)) >> 8))) >> 8;

  switch (region)
  {
  case 0:
    rgb.r = v;
    rgb.g = t;
    rgb.b = p;
    break;
  case 1:
    rgb.r = q;
    rgb.g = v;
    rgb.b = p;
    break;
  case 2:
    rgb.r = p;
    rgb.g = v;
    rgb.b = t;
    break;
  case 3:
    rgb.r = p;
    rgb.g = q;
    rgb.b = v;
    break;
  case 4:
    rgb.r = t;
    rgb.g = p;
    rgb.b = v;
    break;
  default:
    rgb.r = v;
    rgb.g = p;
    rgb.b = q;
    break;
  }

  return rgb;
}
