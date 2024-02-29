CC=gcc
CXX=hipcc
INCLUDES=
CFLAGS=-O2 -Wall --offload-arch=gfx908
LDFLAGS=-ljpeg
TARGET=edge
OBJS=edge.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

.c.o:
	$(CC) $(CFLAGS) -c $< -o $@

.cc.o:
	$(CXX) -c $< -o $@ $(CFLAGS)

clean:
	rm -rf $(TARGET) $(OBJS)

run: $(TARGET)
	./$(TARGET)
