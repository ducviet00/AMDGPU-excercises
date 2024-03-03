#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>

#include <jpeglib.h>
#include <jerror.h>

#include <hip/hip_runtime.h>

using namespace std;

#define NSTREAM 4

#define BLOCKSIZE 16

#define OFFSET 1

#define DIVUP(x, y) ((x + y - 1) / y)

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



int timespec_subtract(struct timespec *result, struct timespec *x, struct timespec *y)
{
  /* Perform the carry for the later subtraction by updating y. */
  if (x->tv_nsec < y->tv_nsec)
  {
    int nsec = (y->tv_nsec - x->tv_nsec) / 1000000000 + 1;
    y->tv_nsec -= 1000000000 * nsec;
    y->tv_sec += nsec;
  }
  if (x->tv_nsec - y->tv_nsec > 1000000000)
  {
    int nsec = (x->tv_nsec - y->tv_nsec) / 1000000000;
    y->tv_nsec += 1000000000 * nsec;
    y->tv_sec -= nsec;
  }

  /* Compute the time remaining to wait.
     tv_nsec is certainly positive. */
  result->tv_sec = x->tv_sec - y->tv_sec;
  result->tv_nsec = x->tv_nsec - y->tv_nsec;

  /* Return 1 if result is negative. */
  return x->tv_sec < y->tv_sec;
}

struct Pixel
{
  unsigned char r, g, b;
};

struct Image
{
  Pixel *pixels;
  int width, height;

  __host__ __device__ Pixel &getPixel(int x, int y)
  {
    return pixels[y * width + x];
  }

  void copy(Image &out)
  {
    out.width = width;
    out.height = height;
    // out.pixels = (Pixel *)malloc(width * height * sizeof(Pixel));
    HIPCHECK(hipHostMalloc(&out.pixels, width * height * sizeof(Pixel), hipHostMallocNumaUser));
    memcpy(out.pixels, pixels, width * height * sizeof(Pixel));
  }
};

int NGPU;
Image **inputImageGPU, **outputImageGPU;
hipStream_t **streams;
hipEvent_t **h2d;

const int filterX[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
const int filterY[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

// Function to read an image as a JPG file
bool readJPEGImage(const string &filename, Image &img)
{
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  JSAMPROW row_pointer;
  FILE *fp;

  if ((fp = fopen(filename.c_str(), "rb")) == NULL)
  {
    fprintf(stderr, "can't open %s\n", filename.c_str());
    return false;
  }

  cinfo.err = jpeg_std_error(&jerr);

  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, fp);

  jpeg_read_header(&cinfo, TRUE);
  jpeg_start_decompress(&cinfo);

  if (cinfo.num_components != 3)
  {
    fprintf(stderr, "JPEG file with 3 channels is only supported\n");
    fprintf(stderr, "%s has %d channels\n", filename.c_str(), cinfo.num_components);
    return false;
  }

  img.width = cinfo.output_width;
  img.height = cinfo.output_height;

  // img.pixels = (Pixel *)malloc(img.width * img.height * sizeof(Pixel));
  HIPCHECK(hipHostMalloc(&img.pixels, img.width * img.height * sizeof(Pixel)));
  for (int i = 0; i < img.height; i++)
  {
    row_pointer = (JSAMPROW)&img.pixels[i * img.width];
    jpeg_read_scanlines(&cinfo, &row_pointer, 1);
  }

  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);

  fclose(fp);
  return true;
}

// Function to save an image as a JPG file
bool saveJPEGImage(const string &filename, const Image &img)
{
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  JSAMPROW row_pointer;
  FILE *fp;

  cinfo.err = jpeg_std_error(&jerr);

  if ((fp = fopen(filename.c_str(), "wb")) == NULL)
  {
    fprintf(stderr, "can't open %s\n", filename.c_str());
    return false;
  }

  jpeg_create_compress(&cinfo);
  jpeg_stdio_dest(&cinfo, fp);

  cinfo.image_width = img.width;
  cinfo.image_height = img.height;
  cinfo.input_components = 3;
  cinfo.in_color_space = JCS_RGB;

  jpeg_set_defaults(&cinfo);
  jpeg_start_compress(&cinfo, TRUE);

  for (int i = 0; i < img.height; i++)
  {
    row_pointer = (JSAMPROW)&img.pixels[i * img.width];
    jpeg_write_scanlines(&cinfo, &row_pointer, 1);
  }

  jpeg_finish_compress(&cinfo);
  fclose(fp);

  return true;
}

// Function to apply the Sobel filter for edge detection
void applySobelFilter(Image &input, Image &output, const int filterX[3][3], const int filterY[3][3])
{

  for (int y = 1; y < input.height - 1; ++y)
  {
    for (int x = 1; x < input.width - 1; ++x)
    {
      int gx = 0, gy = 0;
      for (int i = -1; i <= 1; ++i)
      {
        for (int j = -1; j <= 1; ++j)
        {
          Pixel p = input.getPixel(x + j, y + i);
          gx += (p.r + p.g + p.b) / 3 * filterX[i + 1][j + 1];
          gy += (p.r + p.g + p.b) / 3 * filterY[i + 1][j + 1];
        }
      }
      int magnitude = static_cast<int>(sqrt(gx * gx + gy * gy));
      magnitude = min(max(magnitude, 0), 255);
      output.getPixel(x, y) = {static_cast<unsigned char>(magnitude),
                               static_cast<unsigned char>(magnitude),
                               static_cast<unsigned char>(magnitude)};
    }
  }
}

__global__ void applySobelFilter_hip(Pixel *input, Pixel *output, int w, int h)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < 1 || x >= w - 1 || y < 1 || y >= h - 1)
    return;

  int gx = 0, gy = 0;
  #pragma unroll
  for (int i = -1; i <= 1; ++i)
  {
    #pragma unroll
    for (int j = -1; j <= 1; ++j)
    {
      Pixel p = input[(y + i) * w + x + j];
      gx += (p.r + p.g + p.b) / 3 * filterX[i + 1][j + 1];
      gy += (p.r + p.g + p.b) / 3 * filterY[i + 1][j + 1];
    }
  }
  int magnitude = static_cast<int>(sqrt(gx * gx + gy * gy));
  magnitude = min(max(magnitude, 0), 255);
  output[y * w + x] = {static_cast<unsigned char>(magnitude),
                           static_cast<unsigned char>(magnitude),
                           static_cast<unsigned char>(magnitude)};
}

void initialize(int width, int height)
{
  inputImageGPU = (Image **)malloc(NGPU * sizeof(Image *));
  outputImageGPU = (Image **)malloc(NGPU * sizeof(Image *));

  streams = (hipStream_t **)malloc(NGPU * sizeof(hipStream_t *));

  h2d = (hipEvent_t **)malloc(NGPU * sizeof(hipEvent_t *));

  for (int j = 0; j < NGPU; j++)
  {

    HIPCHECK(hipSetDevice(j));
    HIPCHECK(hipMalloc(&inputImageGPU[j], sizeof(Image)));
    inputImageGPU[j]->width = width;
    inputImageGPU[j]->height = height;
    HIPCHECK(hipMalloc(&inputImageGPU[j]->pixels, width * height * sizeof(Pixel)));

    HIPCHECK(hipMalloc(&outputImageGPU[j], sizeof(Image)));
    outputImageGPU[j]->width = width;
    outputImageGPU[j]->height = height;
    HIPCHECK(hipMalloc(&outputImageGPU[j]->pixels, width * height * sizeof(Pixel)));

    streams[j]  = (hipStream_t *)malloc(NSTREAM * sizeof(hipStream_t));
    h2d[j]      = (hipEvent_t *)malloc(NSTREAM * sizeof(hipEvent_t));
    for (size_t i = 0; i < NSTREAM; i++)
    {
      HIPCHECK(hipStreamCreate(&streams[j][i]));
      HIPCHECK(hipEventCreate(&h2d[j][i]));
    }
  }
}

void cleanup()
{
  for (size_t j = 0; j < NGPU; j++)
  {
    HIPCHECK(hipSetDevice(j));
    HIPCHECK(hipFree(inputImageGPU[j]->pixels));
    HIPCHECK(hipFree(outputImageGPU[j]->pixels));
    HIPCHECK(hipFree(inputImageGPU[j]));
    HIPCHECK(hipFree(outputImageGPU[j]));

    for (size_t i = 0; i < NSTREAM; i++)
    {
      HIPCHECK(hipStreamDestroy(streams[j][i]));
      HIPCHECK(hipEventDestroy(h2d[j][i]));
    }

    free(streams[j]);
    free(h2d[j]);
  }
  free(inputImageGPU);
  free(outputImageGPU);
  free(streams);
  free(h2d);
}

int main(int argc, char **argv)
{
  string inputFilename;
  string outputFilename;
  string outputFilename_hip;
  int verify = 0;

  if (argc < 4)
  {
    fprintf(stderr, "$> edge <input_filename> <output_filename_seq> <output_filename_hip> <verification:0|1>\n");
    return 1;
  }
  else
  {
    inputFilename = argv[1];
    outputFilename = argv[2];
    outputFilename_hip = argv[3];
    if (argc > 4)
    {
      verify = atoi(argv[4]);
    }
  }


  struct timespec start, end, spent;

  Image inputImage;
  Image outputImage;
  Image outputImage_hip;

  if (!readJPEGImage(inputFilename, inputImage))
  {
    return -1;
  }

  inputImage.copy(outputImage);     // Copy input image properties to output image
  inputImage.copy(outputImage_hip); // Copy input image properties to output image

  clock_gettime(CLOCK_MONOTONIC, &start);
  applySobelFilter(inputImage, outputImage, filterX, filterY);
  clock_gettime(CLOCK_MONOTONIC, &end);
  timespec_subtract(&spent, &end, &start);
  printf("CPU Time spent: %ld.%09ld\n", spent.tv_sec, spent.tv_nsec);

  clock_gettime(CLOCK_MONOTONIC, &start);

  // You may modify this code part
  //{

  int width, height;
  width = inputImage.width;
  height = inputImage.height;

  HIPCHECK(hipGetDeviceCount(&NGPU));

  long long threshold = 49152 / 2048 * 49152;

  if (width / 2048 * height > threshold)
    NGPU = min(NGPU, 2);
  else
    NGPU = min(NGPU, 1);

  initialize(width, height);

  const int TOTAL_BUFFER = NGPU * NSTREAM;
  for (int j = 0; j < NGPU; j++)
  {
    HIPCHECK(hipSetDevice(j));

    for (int i = 0; i < NSTREAM; i++)
    {
      // example height = 21
      // j = 0, s_row = 0,  e_row = 12, num_rows = 12
      // j = 1, s_row = 10, e_row = 21, num_rows = 11
      int id = j * NSTREAM + i;
      int s_row = max(   id     * DIVUP(height, TOTAL_BUFFER) - OFFSET, 0     );
      int e_row = min((id + 1)  * DIVUP(height, TOTAL_BUFFER) + OFFSET, height);
      int num_rows = e_row - s_row;


      HIPCHECK(hipMemcpyHtoDAsync(inputImageGPU[j]->pixels  + width * s_row,
                                  inputImage.pixels         + width * s_row,
                                  width * num_rows * sizeof(Pixel),
                                  streams[j][i]));

      // if (i > 0) HIPCHECK(hipStreamWaitEvent(streams[j][i], h2d[j][i-1], 0));
      HIPCHECK(hipMemcpyDtoDAsync(outputImageGPU[j]->pixels + width * (s_row + OFFSET),
                                  inputImageGPU[j]->pixels  + width * (s_row + OFFSET),
                                  width * (num_rows - 2 * OFFSET) * sizeof(Pixel),
                                  streams[j][i]));


      dim3 bs(DIVUP(width, BLOCKSIZE), DIVUP(num_rows, BLOCKSIZE));
      dim3 ts(BLOCKSIZE, BLOCKSIZE);
      hipLaunchKernelGGL(applySobelFilter_hip, bs, ts, 0, streams[j][i],
                        inputImageGPU[j]->pixels  + width * s_row,
                        outputImageGPU[j]->pixels + width * s_row,
                        width, num_rows);

      HIPCHECK(hipGetLastError());

      // j = 0, s_row = 1,  num_rows = 10
      // j = 1, s_row = 11, num_rows = 9
      // if
      HIPCHECK(hipMemcpyDtoHAsync(outputImage_hip.pixels    + width * (s_row + OFFSET),
                                  outputImageGPU[j]->pixels + width * (s_row + OFFSET),
                                  width * (num_rows - 2 * OFFSET) * sizeof(Pixel),
                                  streams[j][i]));
      // HIPCHECK(hipEventRecord(h2d[j][i], streams[j][i]));
    }
  }
  for (int j = 0; j < NGPU; j++)
  {
    HIPCHECK(hipSetDevice(j));
    HIPCHECK(hipDeviceSynchronize());
  }
  cleanup();

  clock_gettime(CLOCK_MONOTONIC, &end);
  timespec_subtract(&spent, &end, &start);
  printf("GPU Time spent: %ld.%09ld\n", spent.tv_sec, spent.tv_nsec);
  //}

  // Save the output image
  saveJPEGImage(outputFilename, outputImage);
  saveJPEGImage(outputFilename_hip, outputImage_hip);

  // verfication (CPU vs GPU)
  if (verify == 1)
  {
    // Verification
    bool pass = true;
    int count = 0;
    for (int i = 0; i < outputImage.width * outputImage.height; i++)
    {
      if (outputImage.pixels[i].r != outputImage_hip.pixels[i].r)
      {
        outputImage_hip.pixels[i].r = 255;
        printf("[%d] r=%d vs %d : %d\n", i, outputImage.pixels[i].r, outputImage_hip.pixels[i].r, inputImage.pixels[i].r);
        pass = false;
        count++;
      }
      if (outputImage.pixels[i].g != outputImage_hip.pixels[i].g)
      {
        printf("[%d] g=%d vs %d : %d\n", i, outputImage.pixels[i].g, outputImage_hip.pixels[i].g, inputImage.pixels[i].g);
        outputImage_hip.pixels[i].g = 255;
        pass = false;
        count++;
      }
      if (outputImage.pixels[i].b != outputImage_hip.pixels[i].b)
      {
        outputImage_hip.pixels[i].b = 255;
        printf("[%d] b=%d vs %d : %d\n", i, outputImage.pixels[i].b, outputImage_hip.pixels[i].b, inputImage.pixels[i].b);
        pass = false;
        count++;
      }
    }
    if (pass)
    {
      printf("Verification Pass!\n");
    }
    else
    {
      printf("Verification Failed! (%d)\n", count);
    }
  }


  HIPCHECK(hipHostFree(inputImage.pixels));
  HIPCHECK(hipHostFree(outputImage.pixels));
  HIPCHECK(hipHostFree(outputImage_hip.pixels));

  return 0;
}