//
// Created by sufianlatif on 10/19/17.
//

#include "amr_parallel.h"
#include <pthread.h>

pthread_barrier_t barr1, barr2, barr3;
int stop = 0;

void *threadFunc(void *param) {
    int i;
    ThreadParams tp = *(ThreadParams *) param;

    while (stop == 0) {
        pthread_barrier_wait(&barr1);

        for (i = 0; i < tp.nBox; i++) {
            tp.target[tp.start + i] = calcNewTemp(tp.start + i);
        }

        pthread_barrier_wait(&barr2);
    }
    return NULL;
}

int runConvergenceLoop() {
    int iter, i, th;
    double *newTemp = (double *) malloc(nBox * sizeof(double));
    pthread_t threads[N_THREADS];
    ThreadParams tp[N_THREADS];
    void *th_status;

    pthread_barrier_init(&barr1, NULL, N_THREADS + 1);
    pthread_barrier_init(&barr2, NULL, N_THREADS + 1);

    int start = 0;
    for (th = 0; th < N_THREADS; th++) {
        tp[th].start = start;
        tp[th].nBox = nBox / N_THREADS + (th < nBox % N_THREADS ? 1 : 0);
        tp[th].target = newTemp;

        start += tp[th].nBox;
        pthread_create(&threads[th], NULL, threadFunc, (void *) &tp[th]);
    }

    for (iter = 0; checkConvergence() == 0; iter++) {
        pthread_barrier_wait(&barr1);
        pthread_barrier_destroy(&barr1);
        pthread_barrier_init(&barr1, NULL, N_THREADS + 1);
        pthread_barrier_wait(&barr2);
        pthread_barrier_destroy(&barr2);
        pthread_barrier_init(&barr2, NULL, N_THREADS + 1);

        for (i = 0; i < nBox; i++) {
            boxes[i].temp = newTemp[i];
        }
    }

    stop = 1;

    pthread_barrier_wait(&barr1);
    pthread_barrier_destroy(&barr1);
    pthread_barrier_wait(&barr2);
    pthread_barrier_destroy(&barr2);

    for (th = 0; th < N_THREADS; th++) {
        pthread_join(threads[th], &th_status);
    }

    free(newTemp);
    return iter;
}
