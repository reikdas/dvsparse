#include <algorithm>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <mkl.h>
#include <mkl_spblas.h>
#include <set>
#include <sstream>
#include <vector>

#include "mtx_to_csr.h"

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
    mkl_set_num_threads(num_threads);

    COO cooMatrix = readMTXtoCOO(filename);
    CSR csrMatrix = coo_to_csr(cooMatrix);

    FILE *file2 = fopen(dense_vec_path, "r");
    if (file2 == NULL) {
        printf("Error opening file2");
        return 1;
    }

    // create an array with the same size as the number of cols in the csrMatrix
    double *x = new double[csrMatrix.cols];
    int x_size = 0;
    while (x_size < csrMatrix.cols && fscanf(file2, "%lf,", &x[x_size]) == 1) {
        x_size++;
    }
    if (x_size != csrMatrix.cols) {
        std::cerr << "Error: Dense vector size mismatch!" << std::endl;
        exit(1);
    }

    // create an array with the same size as the number of rows in the csrMatrix
    double *y = new double[csrMatrix.rows];

    MKL_INT n = csrMatrix.rows; // Size of the matrix (n x n)
    MKL_INT m = csrMatrix.cols; // Number of non-zero elements

    MKL_INT *ia = new MKL_INT[csrMatrix.rows + 1];
    MKL_INT *ja = new MKL_INT[csrMatrix.nnz];

    std::copy(csrMatrix.row_ptrs, csrMatrix.row_ptrs + csrMatrix.rows + 1, ia);
    std::copy(csrMatrix.col_indices, csrMatrix.col_indices + csrMatrix.nnz, ja);

    double *a = csrMatrix.values;

    // Create a sparse matrix handle
    sparse_matrix_t A;
    sparse_status_t status = mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO,
                                                     n, m, ia, ia + 1, ja, a);
    if (status != SPARSE_STATUS_SUCCESS) {
        std::cerr << "Error creating sparse matrix (status: " << status << ")"
                  << std::endl;
        exit(1);
    }

    // Perform sparse matrix-vector multiplication (y = A * x)
    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL; // General matrix (no symmetry)

    struct timespec start, end;
    uint64_t times[NUMBER_OF_RUNS];

    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, descr, x, 0.0, y);
    for (int i = 0; i < NUMBER_OF_RUNS; i++) {
        memset(y, 0, sizeof(double) * csrMatrix.rows);
        clock_gettime(CLOCK_MONOTONIC, &start); // Start timer
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, descr, x, 0.0,
                        y);
        clock_gettime(CLOCK_MONOTONIC, &end); // End timer
        uint64_t delta_ns = (end.tv_sec - start.tv_sec) * 1000000000 +
                            (end.tv_nsec - start.tv_nsec);
        times[i] = delta_ns;
    }

    // sort the times
    std::sort(times, times + NUMBER_OF_RUNS);

    const int mid_point = NUMBER_OF_RUNS / 2;
    std::cout << "Time: " << times[mid_point] << " us" << std::endl;
    for (int i = 0; i < csrMatrix.rows; i++) {
        std::cout << y[i] << std::endl;
    }

    delete[] x;
    delete[] y;
    mkl_sparse_destroy(A);

    return 0;
}
