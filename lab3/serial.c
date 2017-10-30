//
// Created by sufianlatif on 10/18/17.
//

#include "amr_parallel.h"

int runConvergenceLoop() {
    int iter, i;
    double *newTemp = (double *) malloc(nBox * sizeof(double));

    for (iter = 0; checkConvergence() == 0; iter++) {
        for (i = 0; i < nBox; i++) {
            newTemp[i] = calcNewTemp(i);
        }

        for (i = 0; i < nBox; i++) {
            boxes[i].temp = newTemp[i];
        }
    }

    free(newTemp);
    return iter;
}
