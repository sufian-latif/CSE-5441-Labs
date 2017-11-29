#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <time.h>
#include "read_bmp.h"

int min(int a, int b) {
    return a < b ? a : b;
}

int sobel(uint8_t *bmpIn, uint32_t height, uint32_t width, int threshold, uint8_t *bmpOut, int nThreads) {
    int i, j;
    int nBlackCells = 0;

    for (i = 1; i < height - 1; i++) {
        #pragma omp parallel num_threads(nThreads)
        #pragma omp for
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
                nBlackCells++;
            }
        }
    }

    return nBlackCells;
}

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    FILE *inFile;
    uint8_t *bmpData;
    uint8_t *slice;
    uint32_t width;
    uint32_t height;
    uint32_t sliceHeight;
    uint32_t sliceSize;
    uint8_t *bmpSobel;
    uint8_t *sliceSobel;
    clock_t clockStart;

    if(rank == 0) {
        printf("********************************************************************\n");
        inFile = fopen(argv[1], "rb");
        bmpData = (uint8_t *) read_bmp_file(inFile);
        width = get_image_width();
        height = get_image_height();

        sliceHeight = 2 + (height - 2) / size + ((height - 2) % size != 0 ? 1 : 0);
        sliceSize = width * sliceHeight;
        slice = (uint8_t *) malloc(sliceSize);
        memcpy(slice, bmpData, sliceSize);

        int i;
        for (i = 1; i < size; i++) {
            MPI_Send(&width, 1, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD);
            MPI_Send(&height, 1, MPI_UNSIGNED, i, 1, MPI_COMM_WORLD);

            int start = (height - 2) / size * i + min(i, (height - 2) % size);
            int nRow = 2 + (height - 2) / size + (i < (height - 2) % size ? 1 : 0);
            int tmpSize = width * nRow;

            MPI_Send(&nRow, 1, MPI_UNSIGNED, i, 2, MPI_COMM_WORLD);
            MPI_Send(bmpData + (start - 1) * width, tmpSize, MPI_UNSIGNED_CHAR, i, 3, MPI_COMM_WORLD);

        }

        free(bmpData);
    } else {
        MPI_Recv(&width, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&height, 1, MPI_UNSIGNED, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&sliceHeight, 1, MPI_UNSIGNED, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        sliceSize = width * sliceHeight;
        slice = (uint8_t *) malloc(sliceSize);

        MPI_Recv(slice, sliceSize, MPI_UNSIGNED_CHAR, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    sliceSobel = (uint8_t *) malloc(sliceSize);

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0) {
        clockStart = clock();
    }

    int threshold = 0;
    int nBlackCells = 0;
    int percentBlackCells = 0;
    int nThreads = sscanf(argv[3], "%d", &nThreads);
    int totalBlackCells;

    while(percentBlackCells < 75) {
        MPI_Barrier(MPI_COMM_WORLD);
        threshold++;
        nBlackCells = sobel(slice, sliceHeight, width, threshold, sliceSobel, nThreads);
        MPI_Allreduce(&nBlackCells, &totalBlackCells, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        percentBlackCells = totalBlackCells * 100 / (width * height);
    }

    if(rank == 0) {
        bmpSobel = (uint8_t *) malloc(width * height);
        memcpy(bmpSobel + width, sliceSobel + width, width * (sliceHeight - 2));

        int i;
        for(i = 1; i < size; i++) {
            int startRow = 1 + (height - 2) / size * i + min(i, (height - 2) % size);
            int nRow = (height - 2) / size + (i < (height - 2) % size ? 1 : 0);
            MPI_Recv(bmpSobel + width * (startRow - 1), width * nRow, MPI_UNSIGNED_CHAR, i, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        FILE *outFile = fopen(argv[2], "wb");
        write_bmp_file(outFile, bmpSobel);

        double diff = (clock() - clockStart) / 1000;
        printf("Time taken for MPI operation: %lf ms\n", diff);
        printf("Threshold during convergence: %d\n\n", threshold);
        printf("********************************************************************\n");
    } else {
        MPI_Send(sliceSobel + width, width * (sliceHeight - 2), MPI_UNSIGNED_CHAR, 0, 4, MPI_COMM_WORLD);
    }

    MPI_Finalize();
}