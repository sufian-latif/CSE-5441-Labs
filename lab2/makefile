GCC = gcc
FLAGS = -O3
LIBS = -lrt -pthread

serial:
	$(GCC) $(FLAGS) $(LIBS) serial.c amr_parallel.c

disposable:
	$(GCC) $(FLAGS) $(LIBS) disposable.c amr_parallel.c

clean:
	-rm *.o *.out
