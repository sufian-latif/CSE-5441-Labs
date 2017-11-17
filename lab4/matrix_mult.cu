#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void init(double **A, double **C1, double **C2, int dim) {
    int i, j;
    int size = dim * dim * sizeof(double);

    *A = (double *) malloc(size);

    srand(time(NULL));
    for (i = 0; i < dim; i++) {
        for (j = 0; j < dim; j++) {
            (*A)[i * dim + j] = 1.0 + 1.0 * rand() / RAND_MAX;
        }
    }

    *C1 = (double *) malloc(size);
    memset(*C1, 0, size);
    *C2 = (double *) malloc(size);
    memset(*C2, 0, size);
}

void cleanup(double **A, double **C1, double **C2) {
    free(*A);
    free(*C1);
    free(*C2);
}

double multiplyHost(double *A, double *C, int dim) {
    int i, j, k;

    clock_t clockStart;
    clockStart = clock();

    for (i = 0; i < dim; i++) {
        for (j = 0; j < dim; j++) {
            for (k = 0; k < dim; k++) {
                C[i * dim + j] += A[k * dim + i] * A[k * dim + j];
            }
        }
    }

    double clockDiff = clock() - clockStart;
    return clockDiff / 1000;
}

__global__ void multiplyKernel(double *A, double *C, int dim) {
    int i, j, k;

    i = blockIdx.x;
    j = threadIdx.x;
    for (k = 0; k < dim; k++) {
        C[i * dim + j] += A[k * dim + i] * A[k * dim + j];
    }
}

double multiplyDevice(double *A, double *C, int dim) {
    double *dA, *dC;
    int size = dim * dim * sizeof(double);

    cudaMalloc((void **) &dA, size);
    cudaMalloc((void **) &dC, size);

    cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C, size, cudaMemcpyHostToDevice);

    dim3 dimGrid(dim);
    dim3 dimBlock(dim);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    multiplyKernel<<<dimGrid, dimBlock>>>(dA, dC, dim);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float diff = 0;
    cudaEventElapsedTime(&diff, start, stop);

    cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dC);

    return diff;
}

int match(double *C1, double *C2, int dim) {
    int i, j, nError = 0;

    for (i = 0; i < dim; i++) {
        for (j = 0; j < dim; j++) {
            if(fabs(C1[i * dim + j] - C2[i * dim + j]) > 0.000001) {
                nError++;
            }
        }
    }

    return nError;
}

int main() {
    int dim = 1024;
    double *A, *CHost, *CDevice;
    double nOp = 2.0 * dim * dim * dim;

    init(&A, &CHost, &CDevice, dim);

    double timeHost = multiplyHost(A, CHost, dim);
    double flopsHost = 1000 * nOp / timeHost;
    printf("Time taken on host (ms) = %lf\n", timeHost);
    printf("FLOPS on host = %lf\n", flopsHost);

    double timeDevice = multiplyDevice(A, CDevice, dim);
    double flopsDevice = 1000 * nOp / timeDevice;
    printf("Time taken on device (ms) = %lf\n", timeDevice);
    printf("FLOPS on device = %lf\n", flopsDevice);

    int err = match(CHost, CDevice, dim);
    if(err > 0) {
        printf("Results did not match. %d mismatches\n", err);
    } else {
        printf("Results matched.\n");
    }

    cleanup(&A, &CHost, &CDevice);
}
