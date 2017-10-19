//
// Created by sufianlatif on 10/18/17.
//

#include "amr_parallel.h"

int runConvergenceLoop() {
    int iter, i;
    double *newTemp = (double *) malloc(nBox * sizeof(double));

    printf("%lf %lf\n", boxes[0].temp, boxes[1].temp);
    for (iter = 0; checkConvergence() == 0; iter++) {
        for (i = 0; i < nBox; i++) {
            newTemp[i] = calcNewTemp(i);
        }

        for (i = 0; i < nBox; i++) {
            boxes[i].temp = newTemp[i];
        }
        printf("%lf %lf\n", boxes[0].temp, boxes[1].temp);
    }

    free(newTemp);

    return iter;
}
