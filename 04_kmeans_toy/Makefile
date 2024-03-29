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

verify:
	./kmeans_seq centroids.pt data.pt result_seq.pt final_centroids_seq.pt 1024
	srun -p EM --gres=gpu:2 ./kmeans_gpu centroids.pt data.pt result_gpu.pt final_centroids_gpu.pt 1024
	diff result_seq.pt result_gpu.pt

plot:
	./plot_data.py result final_centroids_seq.pt data.pt result_seq.pt output_seq.png
	./plot_data.py result final_centroids_gpu.pt data.pt result_gpu.pt output_gpu.png

gendata:
	./gen_data.py centroid 161 centroids.pt
	./gen_data.py data 169213 data.pt 161

run:
	srun -p EM --gres=gpu:2 ./kmeans_gpu centroids128.pt data8M.pt result_gpu.pt final_centroids_gpu.pt 1024

prof:
	srun -p EM --gres=gpu:2 rocprof --hip-trace --stats --basenames on ./kmeans_gpu centroids128.pt data8M.pt result_gpu.pt final_centroids_gpu.pt 1024

debug: kmeans_gpu.o kmeans_main.o
	${CXX} -Og -ggdb $^ -o $@ ${LDFLAGS}
	srun -p EM --gres=gpu:1 rocgdb --args ./kmeans_gpu centroids.pt data.pt result_gpu.pt final_centroids_gpu.pt 1024