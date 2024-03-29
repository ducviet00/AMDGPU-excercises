#ifndef __KMEANS_H__
#define __KMEANS_H__

struct Point {
    double x, y;
};


// Kmean algorighm
void kmeans(int iteration_n, int class_n, int data_n, Point* centroids, Point* data, int* clsfy_result);

#endif // __KMEANS_H__
