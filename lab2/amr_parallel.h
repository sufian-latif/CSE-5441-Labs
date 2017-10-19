//
// Created by sufianlatif on 10/17/17.
//

#ifndef AMR_PARALLEL_H
#define AMR_PARALLEL_H

extern double AFFECT_RATE;
extern double EPSILON;
extern int N_THREADS;

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

int runConvergenceLoop();
void readInput();
void readBoxInfo(Box *box);
void updateNeighborInfo(int n);
double calcNewTemp(int n);
int checkConvergence();
void cleanup();

#endif //AMR_PARALLEL_H
