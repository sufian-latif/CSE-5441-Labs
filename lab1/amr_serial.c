//
// Created by sufianlatif on 9/16/2017.
//

#include "amr_serial.h"
#include <stdio.h>
#include <stdlib.h>

#define max(x, y) ((x) > (y) ? (x) : (y))
#define min(x, y) ((x) < (y) ? (x) : (y))
#define abs(x) (x < 0 ? -(x) : x)

double AFFECT_RATE;
double EPSILON;
int nBox;
Box *boxes;

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
    freopen(testInput[0], "r", stdin);
    readInput();
    updateAllNeighborInfo();

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
    box->nonSharedEdgeLength = 2 * (box->height + box->width);

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

void updateAllNeighborInfo() {
    int i;

    for (i = 0; i < nBox; i++) {
        updateNeighborInfo(&boxes[i]);
    }
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

