//
// Created by sufianlatif on 9/16/2017.
//

#include "amr_serial.h"
#include <stdio.h>
#include <stdlib.h>

double AFFECT_RATE;
double EPSILON;
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

int main() {
    freopen(testInput[0], "r", stdin);
    readInput();
    return 0;
}

void readInput() {
    int nBox;
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
        box->neighbors[i] = (int*) malloc (box->nNeighbors[i] * sizeof(int));

        for (j = 0; j < box->nNeighbors[i]; j++) {
            scanf("%d", &box->neighbors[i][j]);
        }
    }

    scanf("%lf", &box->temp);
}
