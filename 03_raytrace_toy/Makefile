CC=gcc
CXX=hipcc
INCLUDES=
CFLAGS=-O2 -Wall --offload-arch=gfx908
LDFLAGS=-lm -ljpeg
TARGET=raytracer
OBJS=raytracer.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

.c.o:
	$(CC) $(CFLAGS) -c $< -o $@
.cc.o:
	$(CXX) $(CFLAGS) -c $< -o $@

debug:
	srun -p EM --gres=gpu:1 --pty rocgdb --args ./raytracer 512 512 0

clean:
	rm -rf $(TARGET) $(OBJS)

run: $(TARGET)
	./$(TARGET)
