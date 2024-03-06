CXX=hipcc
CXXFLAGS=-Wall --offload-arch=gfx908

LIBS = -lrt
LDFLAGS = ${LIBS}


all: seq gpu

.PHONY: all seq opencl clean


seq: kmeans_seq

kmeans_seq: kmeans_seq.o kmeans_main.o
	${CXX} $^ -o $@ ${LDFLAGS}


gpu: kmeans_gpu

kmeans_gpu: kmeans_gpu.o kmeans_main.o
	${CXX} $^ -o $@ ${LDFLAGS}


clean:
	rm -f kmeans_seq kmeans_gpu kmeans_main.o kmeans_seq.o kmeans_gpu.o
