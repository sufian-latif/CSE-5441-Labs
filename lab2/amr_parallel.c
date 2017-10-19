//
// Created by sufianlatif on 10/17/17.
//

#include "amr_parallel.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define max(x, y) ((x) > (y) ? (x) : (y))
#define min(x, y) ((x) < (y) ? (x) : (y))
#define abs(x) (x < 0 ? -(x) : x)
#define NS_PER_US 1000

double AFFECT_RATE;
double EPSILON;
int N_THREADS;
int nBox;
Box *boxes;
double maxTemp, minTemp;

char *testInput[] = {
        "testgrid_0",
        "testgrid_1",
        "testgrid_2",
        "testgrid_50_78",
        "testgrid_50_201",
        "testgrid_200_1166",
        "testgrid_400_1636",
        "testgrid_400_12206"
};

int main(int argc, char **argv) {
    sscanf(argv[1], "%lf", &AFFECT_RATE);
    sscanf(argv[2], "%lf", &EPSILON);
    freopen(testInput[7], "r", stdin);
    readInput();

    int i, j;
    for (i = 0; i < nBox; i++) {
        updateNeighborInfo(i);
    }

    struct timespec chronoStart, chronoEnd;
    clock_t clockStart, clockEnd;
    time_t timeStart, timeEnd;

    clockStart = clock();
    time(&timeStart);
    clock_gettime(CLOCK_REALTIME, &chronoStart);

    int iter = runConvergenceLoop();

    clockEnd = clock();
    time(&timeEnd);
    clock_gettime(CLOCK_REALTIME, &chronoEnd);

    int clockDiff = clockEnd - clockStart;
    int timeDiff = timeEnd - timeStart;
    double chronoDiff = (double) (((chronoEnd.tv_sec - chronoStart.tv_sec) * CLOCKS_PER_SEC)
                                  + ((chronoEnd.tv_nsec - chronoStart.tv_nsec) / NS_PER_US));

    printf("\n**************************************************************************\n");
    printf("dissipation converged in %d iterations,\n", iter);
    printf("    with max DSV = %.7lf and min DSV = %.7lf\n", maxTemp, minTemp);
    printf("    affect rate  = %lf; epsilon = %lf\n", AFFECT_RATE, EPSILON);
    printf("elapsed convergence loop time  (clock): %d\n", clockDiff);
    printf("elapsed convergence loop time   (time): %d\n", timeDiff);
    printf("elapsed convergence loop time (chrono): %lf\n", chronoDiff);
    printf("\n**************************************************************************\n");

    cleanup();

    return 0;
}

void readInput() {
    scanf("%d %*d %*d", &nBox);
    boxes = (Box *) malloc(nBox * sizeof(Box));

    int i;
    for (i = 0; i < nBox; i++) {
        readBoxInfo(&boxes[i]);
    }

    scanf("%*d");
}

void readBoxInfo(Box *box) {
    scanf("%d", &box->id);
    scanf("%d %d %d %d", &box->upperLeftY, &box->upperLeftX, &box->height, &box->width);
    box->perimeter = 2 * (box->height + box->width);

    int i, j;

    for (i = 0; i < 4; i++) {
        scanf("%d", &box->nNeighbors[i]);
        box->neighbors[i] = (int *) malloc(box->nNeighbors[i] * sizeof(int));

        for (j = 0; j < box->nNeighbors[i]; j++) {
            scanf("%d", &box->neighbors[i][j]);
        }
    }

    scanf("%lf", &box->temp);
}

void updateNeighborInfo(int n) {
    int i, j;
    boxes[n].nTotalNeighbors = 0;

    for (i = 0; i < 4; i++) {
        boxes[n].nTotalNeighbors += boxes[n].nNeighbors[i];
    }

    int k = 0;
    boxes[n].allNeighbors = (Neighbor *) malloc(boxes[n].nTotalNeighbors * sizeof(Neighbor));
    boxes[n].nonSharedEdgeLength = 2 * (boxes[n].height + boxes[n].width);

    for (i = 0; i < 4; i++) {
        for (j = 0; j < boxes[n].nNeighbors[i]; j++) {
            int sharedEdge;
            Box neighbor = boxes[boxes[n].neighbors[i][j]];

            if (i < 2) {
                sharedEdge = abs(max(boxes[n].upperLeftX, neighbor.upperLeftX)
                                 - min(boxes[n].upperLeftX + boxes[n].width, neighbor.upperLeftX + neighbor.width));
            } else {
                sharedEdge = abs(max(boxes[n].upperLeftY, neighbor.upperLeftY)
                                 - min(boxes[n].upperLeftY + boxes[n].height, neighbor.upperLeftY + neighbor.height));
            }

            boxes[n].allNeighbors[k].id = neighbor.id;
            boxes[n].allNeighbors[k].sharedEdgeLength = sharedEdge;
            boxes[n].nonSharedEdgeLength -= sharedEdge;
            k++;
        }
    }
}

double calcNewTemp(int n) {
    int i;
    double waat = boxes[n].temp * boxes[n].nonSharedEdgeLength;

    for (i = 0; i < boxes[n].nTotalNeighbors; i++) {
        waat += boxes[n].allNeighbors[i].sharedEdgeLength * boxes[boxes[n].allNeighbors[i].id].temp;
    }

    waat /= boxes[n].perimeter;

    return boxes[n].temp + (waat - boxes[n].temp) * AFFECT_RATE;
}

int checkConvergence() {
    int i;

    maxTemp = minTemp = boxes[0].temp;

    for (i = 1; i < nBox; i++) {
        if (boxes[i].temp > maxTemp) {
            maxTemp = boxes[i].temp;
        }

        if (boxes[i].temp < minTemp) {
            minTemp = boxes[i].temp;
        }
    }

    return maxTemp - minTemp > maxTemp * EPSILON ? 0 : 1;
}

void cleanup() {
    int i, j;

    for (i = 0; i < nBox; i++) {
        for (j = 0; j < 4; j++) {
            free(boxes[i].neighbors[j]);
        }

        free(boxes[i].allNeighbors);
    }

    free(boxes);
}
