#include <algorithm>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <vector>

#include "mtx_to_csr.h"

double *convertCOOtoDense(const COO &cooMatrix) {
    int rows = cooMatrix.rows;
    int cols = cooMatrix.cols;

    // Allocate memory on the heap
    double *denseMatrix = new double[rows * cols]();

    // Populate the dense matrix
    for (int i = 0; i < cooMatrix.nnz; i++) {
        int row = cooMatrix.row_indices[i];
        int col = cooMatrix.col_indices[i];
        double value = cooMatrix.values[i];

        // Row-major order: index = row * cols + col
        denseMatrix[row * cols + col] += value;
    }

    return denseMatrix;
}

double *mv(const double *denseMatrix, const double *x, int rows, int cols,
           double *y) {
    // Perform matrix-vector multiplication
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            y[i] += denseMatrix[i * cols + j] * x[j];
        }
    }
    return y;
}

// Example usage
int main(int argc, char *argv[]) {

    struct timespec t1;
    struct timespec t2;

    // read the filename from the command line
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <filename> <num_threads> <num_runs> <dense_vec_path>"
                  << std::endl;
        exit(1);
    }
    std::string filename = argv[1];
    int num_threads = std::stoi(argv[2]);
    int NUMBER_OF_RUNS = std::stoi(argv[3]);
    const char *dense_vec_path = argv[4];

// set the number of threads
#ifdef OPENMP
    omp_set_num_threads(num_threads);
#endif

    COO cooMatrix = readMTXtoCOO(filename);
    double *denseMatrix = convertCOOtoDense(cooMatrix);

    FILE *file2 = fopen(dense_vec_path, "r");
    if (file2 == NULL) {
        printf("Error opening file2");
        return 1;
    }

    // create an array with the same size as the number of cols in the csrMatrix
    double *x = new double[cooMatrix.cols];
    int x_size = 0;
    while (x_size < cooMatrix.cols && fscanf(file2, "%lf,", &x[x_size]) == 1) {
        x_size++;
    }

    // create an array with the same size as the number of rows in the csrMatrix
    double *y = new double[cooMatrix.rows];

    // calculate the matrix-vector product and time it
    float *exec_time = new float[NUMBER_OF_RUNS];
    mv(denseMatrix, x, cooMatrix.rows, cooMatrix.cols, y);
    for (int i = 0; i < NUMBER_OF_RUNS; i++) {
        memset(y, 0, sizeof(double) * cooMatrix.rows);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        mv(denseMatrix, x, cooMatrix.rows, cooMatrix.cols, y);
        clock_gettime(CLOCK_MONOTONIC, &t2);
        exec_time[i] =
            (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);
    }

    // get the median of the execution time
    std::sort(exec_time, exec_time + NUMBER_OF_RUNS);
    const int mid_point = NUMBER_OF_RUNS / 2;
    std::cout << "Time: " << exec_time[mid_point] << " us" << std::endl;
    for (int i = 0; i < cooMatrix.rows; i++) {
        std::cout << y[i] << std::endl;
    }

    delete[] denseMatrix;
    delete[] x;
    delete[] y;
    delete[] exec_time;

    return 0;
}
