#include <stdio.h>
#include <stdlib.h>

extern "C" {
#include "read_bmp.h"
}

void sobelSerial(uint8_t *bmpIn, uint32_t width, uint32_t height, uint8_t *bmpOut) {
    int i, j;
    int threshold = 0;
    int percentBlackCells = 0;

    clock_t clockStart;
    clockStart = clock();

    while (percentBlackCells < 75) {
        percentBlackCells = 0;
        threshold++;

        for (i = 1; i < height - 1; i++) {
            for (j = 1; j < width - 1; j++) {
                uint32_t sum1 = bmpIn[(i - 1) * width + (j + 1)] - bmpIn[(i - 1) * width + (j - 1)]
                       + 2 * bmpIn[(i) * width + (j + 1)] - 2 * bmpIn[(i) * width + (j - 1)]
                       + bmpIn[(i + 1) * width + (j + 1)] - bmpIn[(i + 1) * width + (j - 1)];

                uint32_t sum2 = bmpIn[(i - 1) * width + (j - 1)] + 2 * bmpIn[(i - 1) * width + (j)]
                       + bmpIn[(i - 1) * width + (j + 1)] - bmpIn[(i + 1) * width + (j - 1)]
                       - 2 * bmpIn[(i + 1) * width + (j)] - bmpIn[(i + 1) * width + (j + 1)];

                if (sum1 * sum1 + sum2 * sum2 > threshold * threshold) {
                    bmpOut[i * width + j] = 255;
                }
                else {
                    bmpOut[i * width + j] = 0;
                    percentBlackCells++;
                }
            }
        }
        percentBlackCells = (percentBlackCells * 100) / (width * height);
    }

    double diff = (clock() - clockStart) / 1000;
    printf("Time taken for serial sobel operation: %lf ms\n", diff);
    printf("Threshold during convergence: %d\n\n", threshold);
}

__global__ void sobelKernel(uint8_t *bmpIn, uint32_t width, uint32_t height, uint8_t *bmpOut, uint32_t threshold, uint32_t *percentBlackCells) {
    int i, j;

    for (i = blockIdx.x + 1; i < height - 1; i += gridDim.x) {
        for (j = threadIdx.x + 1; j < width - 1; j += blockDim.x) {
            uint32_t sum1 = bmpIn[(i - 1) * width + (j + 1)] - bmpIn[(i - 1) * width + (j - 1)]
                            + 2 * bmpIn[(i) * width + (j + 1)] - 2 * bmpIn[(i) * width + (j - 1)]
                            + bmpIn[(i + 1) * width + (j + 1)] - bmpIn[(i + 1) * width + (j - 1)];

            uint32_t sum2 = bmpIn[(i - 1) * width + (j - 1)] + 2 * bmpIn[(i - 1) * width + (j)]
                            + bmpIn[(i - 1) * width + (j + 1)] - bmpIn[(i + 1) * width + (j - 1)]
                            - 2 * bmpIn[(i + 1) * width + (j)] - bmpIn[(i + 1) * width + (j + 1)];

            if (sum1 * sum1 + sum2 * sum2 > threshold * threshold) {
                bmpOut[i * width + j] = 255;
            }
            else {
                bmpOut[i * width + j] = 0;
                atomicAdd(percentBlackCells, 1);
            }
        }
    }
}

void sobelCUDA(uint8_t *bmpIn, uint32_t width, uint32_t height, uint8_t *bmpOut) {
    int threshold = 0;
    int percentBlackCells = 0;
    uint8_t *dBmpIn, *dBmpOut;
    uint32_t *dPercentBlackCells;

    cudaMalloc((void **) &dBmpIn, get_num_pixel());
    cudaMemcpy(dBmpIn, bmpIn, get_num_pixel(), cudaMemcpyHostToDevice);

    cudaMalloc((void **) &dPercentBlackCells, sizeof(uint32_t));
    cudaMemcpy(dPercentBlackCells, &percentBlackCells, sizeof(uint32_t), cudaMemcpyHostToDevice);

    cudaMalloc((void **) &dBmpOut, get_num_pixel());

    dim3 dimGrid(2048);
    dim3 dimBlock(1024);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    while (percentBlackCells < 75) {
        percentBlackCells = 0;
        cudaMemcpy(dPercentBlackCells, &percentBlackCells, sizeof(uint32_t), cudaMemcpyHostToDevice);
        threshold++;

        sobelKernel<<<dimGrid, dimBlock>>>(dBmpIn, width, height, dBmpOut, threshold, dPercentBlackCells);
        cudaDeviceSynchronize();

        cudaMemcpy(&percentBlackCells, dPercentBlackCells, sizeof(uint32_t), cudaMemcpyDeviceToHost);

        percentBlackCells = (percentBlackCells * 100) / (width * height);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float diff = 0;
    cudaEventElapsedTime(&diff, start, stop);

    cudaMemcpy(bmpOut, dBmpOut, get_num_pixel(), cudaMemcpyDeviceToHost);

    cudaFree(dBmpIn);
    cudaFree(dPercentBlackCells);
    cudaFree(dBmpOut);

    printf("Time taken for CUDA sobel operation: %lf ms\n", diff);
    printf("Threshold during convergence: %d\n", threshold);
}

int main(int argc, char **argv) {
    FILE *inFile = fopen(argv[1], "rb");
    FILE *outFileSerial = fopen(argv[2], "wb");
    FILE *outFileCuda = fopen(argv[3], "wb");

    uint8_t *bmpData = (uint8_t *) read_bmp_file(inFile);
    uint32_t width = get_image_width();
    uint32_t height = get_image_height();
    uint8_t *bmpSobel = (uint8_t *) malloc(get_num_pixel());

    printf("********************************************************************");

    sobelSerial(bmpData, width, height, bmpSobel);
    write_bmp_file(outFileSerial, bmpSobel);

    bmpSobel = (uint8_t *) malloc(get_num_pixel());
    sobelCUDA(bmpData, width, height, bmpSobel);
    write_bmp_file(outFileCuda, bmpSobel);

    printf("********************************************************************");

    free(bmpData);
}