//
// Created by sufianlatif on 9/16/2017.
//

#ifndef AMR_SERIAL_H
#define AMR_SERIAL_H

extern double AFFECT_RATE;
extern double EPSILON;

enum {
    TOP = 0, BOTTOM, LEFT, RIGHT
};

typedef struct {
    int id;
    int sharedEdgeLength;
} Neighbor;

typedef struct {
    int id;
    int upperLeftX, upperLeftY;
    int height, width;
    double temp, newTemp;
    int nNeighbors[4];
    int *neighbors[4];
    int nTotalNeighbors;
    Neighbor *allNeighbors;
    int nonSharedEdgeLength;
} Box;

extern Box *boxes;

void readInput();
void readBoxInfo(Box *box);

#endif //AMR_SERIAL_H
