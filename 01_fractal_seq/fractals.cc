#include <stdio.h>
#include <stdlib.h>

#ifdef SAVE_JPG
#include <jpeglib.h>
#endif

#define DEFAULT_M   2048
#define DEFAULT_N   2048

void mandelbrot(int m, int n, char *output_filename);
void julia(int w, int h, char *output_filename);

#ifdef SAVE_JPG
typedef struct _RGB
{
	unsigned char r;
	unsigned char g;
	unsigned char b;
}RGB;

void save_jpeg_image(const char* filename, unsigned char* r, unsigned char* g, unsigned char* b, int image_width, int image_height)
{
	struct jpeg_compress_struct cinfo;
	struct jpeg_error_mgr jerr;
	JSAMPROW row_pointer;

	RGB* rgb = (RGB*)malloc(sizeof(RGB)*image_width*image_height);
	for (int i = 0; i < image_width * image_height; ++i) {
		rgb[i].r = (unsigned char)r[i];
		rgb[i].g = (unsigned char)g[i];
		rgb[i].b = (unsigned char)b[i];
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
}
#endif

void error_usage() {
    fprintf(stderr, "$> fractals <mode> <output filename> <size=w=h>\n");
    fprintf(stderr, "<mode>: 1=MANDELBROT SET, 2=JULIA SET\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
  int m = DEFAULT_M;
  int n = DEFAULT_N;
  int mode = 0;

  char* output_filename;

  if(argc < 4) {
    error_usage();
  }

  mode = atoi(argv[1]);
  output_filename = argv[2];
  m = n = atoi(argv[3]);

  if(mode == 1) {
    printf("1. MANDELBROT SET\n\n");
    mandelbrot(m, n, output_filename);
    printf("\n");
  }
  else if(mode ==2){
    printf("2. JULIA SET\n\n");
    julia(m, n, output_filename);
  }
  else {
    error_usage();
  }

  return EXIT_SUCCESS;
}

