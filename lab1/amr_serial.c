//
// Created by sufianlatif on 9/16/2017.
//

#include "amr_serial.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define max(x, y) ((x) > (y) ? (x) : (y))
#define min(x, y) ((x) < (y) ? (x) : (y))
#define abs(x) (x < 0 ? -(x) : x)

double AFFECT_RATE;
double EPSILON;
int nBox;
Box *boxes;
double maxTemp, minTemp;

char *testInput[] = {
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
    freopen(testInput[6], "r", stdin);
    readInput();

    int i, j;
    for (i = 0; i < nBox; i++) {
        updateNeighborInfo(&boxes[i]);
    }

//    struct timespec start, end;
//    double diff;

//    clock_gettime(CLOCK_REALTIME, &start);

    for (i = 0; checkConvergence() == 0; i++) {
        for (j = 0; j < nBox; j++) {
            calcNewTemp(&boxes[j]);
        }

        for (j = 0; j < nBox; j++) {
            boxes[j].temp = boxes[j].newTemp;
        }
    }
//    clock_gettime(CLOCK_REALTIME, &end);
//    diff = (double)(((end.tv_sec - start.tv_sec) * CLOCKS_PER_SEC) + ((end.tv_nsec - start.tv_nsec) / NS_PER_US));

    printf("Converged after %d iterations\n", i);
    printf("max DSV = %lf and min DSV = %lf\n", maxTemp, minTemp);
//    printf("time: %lf\n", diff);

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

void updateNeighborInfo(Box *box) {
    int i, j;
    box->nTotalNeighbors = 0;

    for (i = 0; i < 4; i++) {
        box->nTotalNeighbors += box->nNeighbors[i];
    }

    int k = 0;
    box->allNeighbors = (Neighbor *) malloc(box->nTotalNeighbors * sizeof(Neighbor));
    box->nonSharedEdgeLength = 2 * (box->height + box->width);

    for (i = 0; i < 4; i++) {
        for (j = 0; j < box->nNeighbors[i]; j++) {
            int sharedEdge;
            Box neighbor = boxes[box->neighbors[i][j]];

            if (i < 2) {
                sharedEdge = abs(max(box->upperLeftX, neighbor.upperLeftX)
                                 - min(box->upperLeftX + box->width, neighbor.upperLeftX + neighbor.width));
            } else {
                sharedEdge = abs(max(box->upperLeftY, neighbor.upperLeftY)
                                 - min(box->upperLeftY + box->height, neighbor.upperLeftY + neighbor.height));
            }

            box->allNeighbors[k].id = neighbor.id;
            box->allNeighbors[k].sharedEdgeLength = sharedEdge;
            box->nonSharedEdgeLength -= sharedEdge;
            k++;
        }
    }
}

void calcNewTemp(Box *box) {
    int i;
    double waat = box->temp * box->nonSharedEdgeLength;

    for (i = 0; i < box->nTotalNeighbors; i++) {
        waat += box->allNeighbors[i].sharedEdgeLength * boxes[box->allNeighbors[i].id].temp;
    }

    waat /= box->perimeter;

    box->newTemp = box->temp + (waat - box->temp) * AFFECT_RATE;
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
            free(boxes[i].neighbors);
        }

        free(boxes[i].allNeighbors);
    }

    free(boxes);
}
