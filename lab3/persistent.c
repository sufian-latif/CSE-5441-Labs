//
// Created by sufianlatif on 10/29/17.
//

#include "amr_parallel.h"
#include <stdlib.h>
#include <omp.h>

int runConvergenceLoop() {
    int iter = 0, stop = 0, i;
    double *newTemp = (double *) malloc(nBox * sizeof(double));
    int nThread;

    #pragma omp parallel num_threads(N_THREADS)
    {

        int id = omp_get_thread_num();
        if (id == 0) {
            nThread = omp_get_num_threads();
        }

        while (stop == 0) {
            #pragma omp for
            for (i = 0; i < nBox; i++) {
                newTemp[i] = calcNewTemp(i);
            }

            #pragma omp master
            {
                for (i = 0; i < nBox; i++) {
                    boxes[i].temp = newTemp[i];
                }
                stop = checkConvergence();
                iter++;
            }

            #pragma omp barrier
        }
    }

    free(newTemp);
    return iter;
}
