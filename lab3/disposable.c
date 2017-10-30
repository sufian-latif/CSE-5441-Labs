//
// Created by sufianlatif on 10/29/17.
//

#include "amr_parallel.h"
#include <stdlib.h>
#include <omp.h>

int runConvergenceLoop() {
    int iter, i;
    double *newTemp = (double *) malloc(nBox * sizeof(double));
    int nThread;

    for (iter = 0; checkConvergence() == 0; iter++) {

        #pragma omp parallel num_threads(N_THREADS)
        {
            int id = omp_get_thread_num();
            if(id == 0) {
                nThread = omp_get_num_threads();
            }

            #pragma omp for
            for (i = 0; i < nBox; i++) {
                newTemp[i] = calcNewTemp(i);
            }
        }

        for (i = 0; i < nBox; i++) {
            boxes[i].temp = newTemp[i];
        }
    }

    free(newTemp);
    return iter;
}
