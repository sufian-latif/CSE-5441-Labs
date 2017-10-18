//
// Created by sufianlatif on 10/17/17.
//

#ifndef AMR_PARALLEL_H
#define AMR_PARALLEL_H

extern double AFFECT_RATE;
extern double EPSILON;
extern double N_THREADS;

typedef struct {
    int id;
    int sharedEdgeLength;
} Neighbor;

typedef struct {
    int id;
    int upperLeftX, upperLeftY;
    int height, width;
    double temp;
    int nNeighbors[4];
    int *neighbors[4];
    int nTotalNeighbors;
    Neighbor *allNeighbors;
    int nonSharedEdgeLength;
    int perimeter;
} Box;

extern int nBox;
extern Box *boxes;
extern double *newTemp;

void readInput();
void readBoxInfo(Box *box);
void updateNeighborInfo(int n);
void calcNewTemp(int n);
int checkConvergence();
void cleanup();

char *testInput[] = {
        "testgrid_1",
        "testgrid_2",
        "testgrid_50_78",
        "testgrid_50_201",
        "testgrid_200_1166",
        "testgrid_400_1636",
        "testgrid_400_12206"
};

#endif //AMR_PARALLEL_H
