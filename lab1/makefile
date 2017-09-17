OBJS = amr_serial.o
GCC = gcc
DEBUG = -g
CFLAGS = -Wall -c -l -r -t -O3 $(DEBUG)
LFLAGS = -Wall $(DEBUG)

all : $(OBJS)
	$(GCC) $(LFLAGS) $(OBJS)

amr_serial.o : amr_serial.h
	$(GCC) $(CFLAGS) amr_serial.c

clean:
	del *.o *.exe
