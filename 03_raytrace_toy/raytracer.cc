// A very basic raytracer example.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <vector>
#include <iostream>
#include <cassert>

#include <jpeglib.h>
#include <hip/hip_runtime.h>

#define BLOCKSIZE 16
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

#ifndef M_PI
#define M_PI 3.141592653589793
#endif

#ifndef INFINITY
#define INFINITY 1e8
#endif

int timespec_subtract (struct timespec* result, struct timespec *x, struct timespec *y)
{
  /* Perform the carry for the later subtraction by updating y. */
  if (x->tv_nsec < y->tv_nsec) {
    int nsec = (y->tv_nsec - x->tv_nsec) / 1000000000 + 1;
    y->tv_nsec -= 1000000000 * nsec;
    y->tv_sec += nsec;
  }
  if (x->tv_nsec - y->tv_nsec > 1000000000) {
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

template<typename T>
class Vec3
{
  public:
    T x, y, z;
    __host__ __device__
    Vec3() : x(T(0)), y(T(0)), z(T(0)) {}
    __host__ __device__
    Vec3(T xx) : x(xx), y(xx), z(xx) {}
    __host__ __device__
    Vec3(T xx, T yy, T zz) : x(xx), y(yy), z(zz) {}
    __host__ __device__
    Vec3& normalize()
    {
      T nor2 = length2();
      if (nor2 > 0) {
        T invNor = 1.0f / sqrtf(nor2);
        x *= invNor, y *= invNor, z *= invNor;
      }
      return *this;
    }
    __host__ __device__
    Vec3<T> operator * (const T &f) const { return Vec3<T>(x * f, y * f, z * f); }
    __host__ __device__
    Vec3<T> operator * (const Vec3<T> &v) const { return Vec3<T>(x * v.x, y * v.y, z * v.z); }
    __host__ __device__
    T dot(const Vec3<T> &v) const { return x * v.x + y * v.y + z * v.z; }
    __host__ __device__
    Vec3<T> operator - (const Vec3<T> &v) const { return Vec3<T>(x - v.x, y - v.y, z - v.z); }
    __host__ __device__
    Vec3<T> operator + (const Vec3<T> &v) const { return Vec3<T>(x + v.x, y + v.y, z + v.z); }
    __host__ __device__
    Vec3<T>& operator += (const Vec3<T> &v) { x += v.x, y += v.y, z += v.z; return *this; }
    __host__ __device__
    Vec3<T>& operator *= (const Vec3<T> &v) { x *= v.x, y *= v.y, z *= v.z; return *this; }
    __host__ __device__
    Vec3<T> operator - () const { return Vec3<T>(-x, -y, -z); }
    __host__ __device__
    T length2() const { return x * x + y * y + z * z; }
    __host__ __device__
    T length() const { return sqrtf(length2()); }
    friend std::ostream & operator << (std::ostream &os, const Vec3<T> &v)
    {
      os << "[" << v.x << " " << v.y << " " << v.z << "]";
      return os;
    }
};

typedef Vec3<float> Vec3f;

class Sphere
{
  public:
    Vec3f center;                           /// position of the sphere
    float radius, radius2;                  /// sphere radius and radius^2
    Vec3f surfaceColor, emissionColor;      /// surface color and emission (light)
    float transparency, reflection;         /// surface transparency and reflectivity
    __host__ __device__
    Sphere(
        const Vec3f &c,
        const float &r,
        const Vec3f &sc,
        const float &refl = 0,
        const float &transp = 0,
        const Vec3f &ec = 0) :
      center(c), radius(r), radius2(r * r), surfaceColor(sc), emissionColor(ec),
      transparency(transp), reflection(refl)
  { /* empty */ }

    // Compute a ray-sphere intersection using the geometric solution
    __host__ __device__
    bool intersect(const Vec3f &rayorig, const Vec3f &raydir, float &t0, float &t1) const
    {
      Vec3f l = center - rayorig;
      float tca = l.dot(raydir);
      if (tca < 0) return false;
      float d2 = l.dot(l) - tca * tca;
      if (d2 > radius2) return false;
      float thc = sqrt(radius2 - d2);
      t0 = tca - thc;
      t1 = tca + thc;

      return true;
    }
};

// This variable controls the maximum recursion depth
#define MAX_RAY_DEPTH 5

__host__ __device__
float mix(const float &a, const float &b, const float &mix)
{
  return b * mix + a * (1 - mix);
}

// This is the main trace function. It takes a ray as argument (defined by its origin
// and direction). We test if this ray intersects any of the geometry in the scene.
// If the ray intersects an object, we compute the intersection point, the normal
// at the intersection point, and shade this point using this information.
// Shading depends on the surface property (is it transparent, reflective, diffuse).
// The function returns a color for the ray. If the ray intersects an object that
// is the color of the object at the intersection point, otherwise it returns
// the background color.
Vec3f trace(
    const Vec3f &rayorig,
    const Vec3f &raydir,
    const std::vector<Sphere> &spheres,
    const int &depth)
{
  //if (raydir.length() != 1) std::cerr << "Error " << raydir << std::endl;
  float tnear = INFINITY;
  const Sphere* sphere = NULL;
  // find intersection of this ray with the sphere in the scene
  for (unsigned i = 0; i < spheres.size(); ++i) {
    float t0 = INFINITY, t1 = INFINITY;
    if (spheres[i].intersect(rayorig, raydir, t0, t1)) {
      if (t0 < 0) t0 = t1;
      if (t0 < tnear) {
        tnear = t0;
        sphere = &spheres[i];
      }
    }
  }
  // if there's no intersection return black or background color
  if (!sphere) return Vec3f(2);
  Vec3f surfaceColor = 0; // color of the ray/surfaceof the object intersected by the ray
  Vec3f phit = rayorig + raydir * tnear; // point of intersection
  Vec3f nhit = phit - sphere->center; // normal at the intersection point
  nhit.normalize(); // normalize normal direction
  // If the normal and the view direction are not opposite to each other
  // reverse the normal direction. That also means we are inside the sphere so set
  // the inside bool to true. Finally reverse the sign of IdotN which we want
  // positive.
  float bias = 1e-4; // add some bias to the point from which we will be tracing
  bool inside = false;
  if (raydir.dot(nhit) > 0) nhit = -nhit, inside = true;
  if ((sphere->transparency > 0 || sphere->reflection > 0) && depth < MAX_RAY_DEPTH) {
    float facingratio = -raydir.dot(nhit);
    // change the mix value to tweak the effect
    float fresneleffect = mix(pow(1 - facingratio, 3), 1, 0.1);
    // compute reflection direction (not need to normalize because all vectors
    // are already normalized)
    Vec3f refldir = raydir - nhit * 2 * raydir.dot(nhit);
    refldir.normalize();
    Vec3f reflection = trace(phit + nhit * bias, refldir, spheres, depth + 1);
    Vec3f refraction = 0;
    // if the sphere is also transparent compute refraction ray (transmission)
    if (sphere->transparency) {
      float ior = 1.1, eta = (inside) ? ior : 1 / ior; // are we inside or outside the surface?
      float cosi = -nhit.dot(raydir);
      float k = 1 - eta * eta * (1 - cosi * cosi);
      Vec3f refrdir = raydir * eta + nhit * (eta *  cosi - sqrt(k));
      refrdir.normalize();
      refraction = trace(phit - nhit * bias, refrdir, spheres, depth + 1);
    }
    // the result is a mix of reflection and refraction (if the sphere is transparent)
    surfaceColor = (
        reflection * fresneleffect +
        refraction * (1 - fresneleffect) * sphere->transparency) * sphere->surfaceColor;
  }
  else {
    // it's a diffuse object, no need to raytrace any further
    for (unsigned i = 0; i < spheres.size(); ++i) {
      if (spheres[i].emissionColor.x > 0) {
        // this is a light
        Vec3f transmission = 1;
        Vec3f lightDirection = spheres[i].center - phit;
        lightDirection.normalize();
        for (unsigned j = 0; j < spheres.size(); ++j) {
          if (i != j) {
            float t0, t1;
            if (spheres[j].intersect(phit + nhit * bias, lightDirection, t0, t1)) {
              transmission = 0;
              break;
            }
          }
        }
        surfaceColor += sphere->surfaceColor * transmission *
          std::max(float(0), nhit.dot(lightDirection)) * spheres[i].emissionColor;
      }
    }
  }

  return surfaceColor + sphere->emissionColor;
}

// Main rendering function. We compute a camera ray for each pixel of the image
// trace it and return a color. If the ray hits a sphere, we return the color of the
// sphere at the intersection point, else we return the background color.
Vec3f* render_cpu(const std::vector<Sphere> &spheres, size_t width, size_t height)
{
  Vec3f *image = new Vec3f[width * height], *pixel = image;
  float invWidth = 1 / float(width), invHeight = 1 / float(height);
  float fov = 30, aspectratio = width / float(height);
  float angle = tan(M_PI * 0.5 * fov / 180.);
  // Trace rays
  for (unsigned y = 0; y < height; ++y) {
    for (unsigned x = 0; x < width; ++x, ++pixel) {
      float xx = (2 * ((x + 0.5) * invWidth) - 1) * angle * aspectratio;
      float yy = (1 - 2 * ((y + 0.5) * invHeight)) * angle;
      Vec3f raydir(xx, yy, -1);
      raydir.normalize();
      *pixel = trace(Vec3f(0), raydir, spheres, 0);
    }
  }

  return image;
}

// This is the ported trace function for GPU
__device__ Vec3f trace_gpu(
    const Vec3f &rayorig,
    const Vec3f &raydir,
    const Sphere *spheres,
    const size_t spheres_size,
    const int &depth)
{
  //if (raydir.length() != 1) std::cerr << "Error " << raydir << std::endl;
  float tnear = INFINITY;
  const Sphere* sphere = NULL;
  // find intersection of this ray with the sphere in the scene
  for (unsigned i = 0; i < spheres_size; ++i) {
    float t0 = INFINITY, t1 = INFINITY;
    if (spheres[i].intersect(rayorig, raydir, t0, t1)) {
      if (t0 < 0) t0 = t1;
      if (t0 < tnear) {
        tnear = t0;
        sphere = &spheres[i];
      }
    }
  }
  // if there's no intersection return black or background color
  if (!sphere) return Vec3f(2);
  Vec3f surfaceColor = 0; // color of the ray/surfaceof the object intersected by the ray
  Vec3f phit = rayorig + raydir * tnear; // point of intersection
  Vec3f nhit = phit - sphere->center; // normal at the intersection point
  nhit.normalize(); // normalize normal direction
  // If the normal and the view direction are not opposite to each other
  // reverse the normal direction. That also means we are inside the sphere so set
  // the inside bool to true. Finally reverse the sign of IdotN which we want
  // positive.
  float bias = 1e-4; // add some bias to the point from which we will be tracing
  bool inside = false;
  if (raydir.dot(nhit) > 0) nhit = -nhit, inside = true;
  if ((sphere->transparency > 0 || sphere->reflection > 0) && depth < MAX_RAY_DEPTH) {
    float facingratio = -raydir.dot(nhit);
    // change the mix value to tweak the effect
    float fresneleffect = mix(powf(1 - facingratio, 3), 1, 0.1);
    // compute reflection direction (not need to normalize because all vectors
    // are already normalized)
    Vec3f refldir = raydir - nhit * 2 * raydir.dot(nhit);
    refldir.normalize();
    Vec3f reflection = trace_gpu(phit + nhit * bias, refldir, spheres, spheres_size, depth + 1);
    Vec3f refraction = 0;
    // if the sphere is also transparent compute refraction ray (transmission)
    if (sphere->transparency) {
      float ior = 1.1, eta = (inside) ? ior : 1 / ior; // are we inside or outside the surface?
      float cosi = -nhit.dot(raydir);
      float k = 1 - eta * eta * (1 - cosi * cosi);
      Vec3f refrdir = raydir * eta + nhit * (eta *  cosi - sqrtf(k));
      refrdir.normalize();
      refraction = trace_gpu(phit - nhit * bias, refrdir, spheres, spheres_size, depth + 1);
    }
    // the result is a mix of reflection and refraction (if the sphere is transparent)
    surfaceColor = (
        reflection * fresneleffect +
        refraction * (1.0f - fresneleffect) * sphere->transparency) * sphere->surfaceColor;
  }
  else {
    // it's a diffuse object, no need to raytrace any further
    for (unsigned i = 0; i < spheres_size; ++i) {
      if (spheres[i].emissionColor.x > 0) {
        // this is a light
        Vec3f transmission = 1;
        Vec3f lightDirection = spheres[i].center - phit;
        lightDirection.normalize();
        for (unsigned j = 0; j < spheres_size; ++j) {
          if (i != j) {
            float t0, t1;
            if (spheres[j].intersect(phit + nhit * bias, lightDirection, t0, t1)) {
              transmission = 0;
              break;
            }
          }
        }
        surfaceColor += sphere->surfaceColor * transmission *
          std::max(float(0), nhit.dot(lightDirection)) * spheres[i].emissionColor;
      }
    }
  }

  return surfaceColor + sphere->emissionColor;
}

__global__ void render_gpu_kernel(Vec3f *image, size_t width, size_t height,
                                  Sphere *spheres, size_t spheres_size,
                                  float invWidth, float invHeight,
                                  float angle, float aspectratio, int s_row)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y + s_row;
  if (x >= width || y >= height) return;

  float xx = (2.0f * ((x + 0.5f) * invWidth) - 1) * angle * aspectratio;
  float yy = (1 - 2.0f * ((y + 0.5f) * invHeight)) * angle;
  Vec3f raydir(xx, yy, -1);
  raydir.normalize();
  image[(y - s_row) * width + x] = trace_gpu(Vec3f(0), raydir, spheres, spheres_size, 0);
}

Vec3f* render_gpu(const std::vector<Sphere> &spheres, size_t width, size_t height)
{
  //TODO:
  int NGPU;
  HIPCHECK(hipGetDeviceCount(&NGPU));
  int threshold = 16384 / 2048 * 16384;
  if (width / 2048 * height > threshold)
    NGPU = std::min(NGPU, 2);
  else
    NGPU = std::min(NGPU, 1);

  size_t spheres_size = spheres.size();
  Sphere  **spheres_d;
  spheres_d = (Sphere **)malloc(NGPU * sizeof(Sphere *));

  for (int j = 0; j < NGPU; j++)
  {
    HIPCHECK(hipSetDevice(j));
    HIPCHECK(hipMalloc(&spheres_d[j], spheres_size * sizeof(Sphere)));
    HIPCHECK(hipMemcpyHtoD(spheres_d[j], (void *)spheres.data(), spheres_size * sizeof(Sphere)));
  }

  float invWidth = 1.0f / float(width), invHeight = 1.0f / float(height);
  float fov = 30, aspectratio = width / float(height);
  float angle = tan(M_PI * 0.5f * fov / 180.0f);

  // Trace rays
  const int TOTAL_BUFFER = NGPU;
  Vec3f *image;
  HIPCHECK(hipHostMalloc(&image, width * height * sizeof(Vec3f), hipHostMallocNonCoherent));

  for (int j = 0; j < NGPU; j++)
  {
    HIPCHECK(hipSetDevice(j));
    int s_row = j * DIVUP(height, NGPU);
    int e_row = std::min((j + 1)  * DIVUP(height, TOTAL_BUFFER), height);
    int num_rows = e_row - s_row;

    dim3 bs(DIVUP(width, BLOCKSIZE), DIVUP(num_rows, BLOCKSIZE));
    dim3 ts(BLOCKSIZE, BLOCKSIZE);
    hipLaunchKernelGGL(render_gpu_kernel, bs, ts, 0, 0,
                      image + width * s_row, width, height, spheres_d[j], spheres_size, invWidth, invHeight, angle, aspectratio, s_row);
    HIPCHECK(hipGetLastError());
  }
  for (size_t j = 0; j < NGPU; j++)
  {
    HIPCHECK(hipSetDevice(j));
    HIPCHECK(hipDeviceSynchronize());
    HIPCHECK(hipFree(spheres_d[j]));
  }
  free(spheres_d);
  return image;
}




void save_jpeg_image(const char* filename, Vec3f* image, int image_width, int image_height);
// In the main function, we will create the scene which is composed of 5 spheres
// and 1 light (which is also a sphere). Then, once the scene description is complete
// we render that scene, by calling the render() function.
int main(int argc, char **argv)
{
  size_t width;
  size_t height;
  char* filename = NULL;
  int verification = 0;

  struct timespec start, end, spent;
  clock_gettime(CLOCK_MONOTONIC, &start);

  if(argc < 3 || argc > 5) {
    fprintf(stderr, "$ ./raytracer <width> <height> <verification:(optional)0|1> <filename:(optional)>\n");
    return 1;
  }
  else {
    width = atoi(argv[1]);
    height = atoi(argv[2]);
    if(argc >= 3) {
      verification = atoi(argv[3]);
    }
    if(argc >= 4) {
      filename = argv[4];
    }
  }

  std::vector<Sphere> spheres;
  // position, radius, surface color, reflectivity, transparency, emission color
  spheres.push_back(Sphere(Vec3f( 0.0, -10004, -20), 10000, Vec3f(0.20, 0.20, 0.20), 0, 0.0));
  spheres.push_back(Sphere(Vec3f( 0.0,      0, -20),     4, Vec3f(1.00, 0.32, 0.36), 1, 0.5));
  spheres.push_back(Sphere(Vec3f( 5.0,     -1, -15),     2, Vec3f(0.90, 0.76, 0.46), 1, 0.0));
  spheres.push_back(Sphere(Vec3f( 5.0,      0, -25),     3, Vec3f(0.65, 0.77, 0.97), 1, 0.0));
  spheres.push_back(Sphere(Vec3f(-5.5,      0, -15),     3, Vec3f(0.90, 0.90, 0.90), 1, 0.0));
  // light
  spheres.push_back(Sphere(Vec3f( 0.0,     20, -30),     3, Vec3f(0.00, 0.00, 0.00), 0, 0.0, Vec3f(3)));

  Vec3f *image_gpu = render_gpu(spheres, width, height);


  float tolerance = 0.35;
  float diff = 0.0;
  size_t diff_cnt = 0;
  size_t total_cnt = 0;
  if(verification > 0) {
    printf("\n=========Verification=========\n");

    Vec3f* image_cpu = render_cpu(spheres, width, height);
    for (unsigned i = 0; i < width * height; ++i) {
      total_cnt++;
      diff = abs(image_gpu[i].x - image_cpu[i].x) + abs(image_gpu[i].y - image_cpu[i].y) + abs(image_gpu[i].z - image_cpu[i].z);
      if(diff > tolerance) {
        printf("%d: diff(%f > %f), gpu(%f,%f,%f) cpu(%f,%f,%f)\n", i, diff, tolerance, image_gpu[i].x, image_gpu[i].y, image_gpu[i].z, image_cpu[i].x, image_cpu[i].y, image_cpu[i].z);
        diff_cnt++;
      }
    }

    double diff_rate = (double)diff_cnt/(double)total_cnt;
    printf("diff_cnt = %zu, correctness = %lf%%\n", diff_cnt, ((double)1.0-diff_rate)*100);
    if(diff_rate < 0.00001) {
      fprintf(stdout, "Verification Pass!\n");
    }
    else {
      fprintf(stdout, "Verification Failed\n");
    }
    printf("===============================\n\n");

    delete [] image_cpu;
  }

  if(filename != NULL) {
#if 0
    // Save result to a PPM image (keep these flags if you compile under Windows)
    std::ofstream ofs(filename, std::ios::out | std::ios::binary);
    ofs << "P6\n" << width << " " << height << "\n255\n";
    for (unsigned i = 0; i < width * height; ++i) {
      ofs << (unsigned char)(std::min(float(1), image_gpu[i].x) * 255) <<
        (unsigned char)(std::min(float(1), image_gpu[i].y) * 255) <<
        (unsigned char)(std::min(float(1), image_gpu[i].z) * 255);
    }
    ofs.close();
#else
    save_jpeg_image(filename, image_gpu, width, height);
#endif
  }
  HIPCHECK(hipHostFree(image_gpu));

  clock_gettime(CLOCK_MONOTONIC, &end);
  timespec_subtract(&spent, &end, &start);
  printf("Elapsed Time: %ld.%09ld\n", spent.tv_sec, spent.tv_nsec);

  return 0;
}



typedef struct _RGB
{
  unsigned char r;
  unsigned char g;
  unsigned char b;
}RGB;


void save_jpeg_image(const char* filename, Vec3f* image, int image_width, int image_height)
{
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  JSAMPROW row_pointer;

  RGB* rgb = (RGB*)malloc(sizeof(RGB)*image_width*image_height);
  for (int i = 0; i < image_width * image_height; ++i) {
    rgb[i].r = (unsigned char)(std::min((float)1, image[i].x) * 255);
    rgb[i].g = (unsigned char)(std::min((float)1, image[i].y) * 255);
    rgb[i].b = (unsigned char)(std::min((float)1, image[i].z) * 255);
  }

  int i;
  FILE* fp;

  cinfo.err = jpeg_std_error(&jerr);

  fp = fopen(filename, "wb");
  if( fp == NULL )
  {
    printf("Cannot open file to save jpeg image: %s\n", filename);
    exit(0);
  }

  jpeg_create_compress(&cinfo);

  jpeg_stdio_dest(&cinfo, fp);

  cinfo.image_width = image_width;
  cinfo.image_height = image_height;
  cinfo.input_components = 3;
  cinfo.in_color_space = JCS_RGB;

  jpeg_set_defaults(&cinfo);

  jpeg_start_compress(&cinfo, TRUE);

  for(i = 0; i < image_height; i++ )
  {
    row_pointer = (JSAMPROW)&rgb[i*image_width];
    jpeg_write_scanlines(&cinfo, &row_pointer, 1);
  }

  jpeg_finish_compress(&cinfo);
  fclose(fp);

  free(rgb);
}
