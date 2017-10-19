//
// Created by sufianlatif on 10/18/17.
//

#include "amr_parallel.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

void *threadFunc(void *param) {
    int i;
    ThreadParams tp = *(ThreadParams *) param;
    for (i = 0; i < tp.nBox; i++) {
        tp.newTemp[i] = calcNewTemp(tp.start + i);
    }

    return NULL;
}

int runConvergenceLoop() {
    int iter, i, th;
    double *newTemp = (double *) malloc(nBox * sizeof(double));
    void *th_status;

    for (iter = 0; checkConvergence() == 0; iter++) {
        pthread_t threads[N_THREADS];
        ThreadParams tp[N_THREADS];
        int start = 0;

        for (th = 0; th < N_THREADS; th++) {
            tp[th].tid = th;
            tp[th].start = start;
            tp[th].nBox = nBox / N_THREADS + (th < nBox % N_THREADS ? 1 : 0);
            tp[th].newTemp = (double *) malloc(tp[th].nBox * sizeof(double));

            start += tp[th].nBox;

            pthread_create(&threads[th], NULL, threadFunc, (void *) &tp[th]);
        }

        for (th = 0; th < N_THREADS; th++) {
            pthread_join(threads[th], &th_status);
        }

        for(th = 0; th < N_THREADS; th++) {
            for (i = 0; i < tp[th].nBox; i++) {
                boxes[tp[th].start + i].temp = tp[th].newTemp[i];
            }
            free(tp[th].newTemp);
        }
    }

    free(newTemp);
    return iter;
}
