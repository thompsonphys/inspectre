PREFIX = $(CONDA_PREFIX)
GCC = $(PREFIX)/bin/gcc
CFLAGS  = -g -Wall -std=gnu99 -I../ -I$(PREFIX)/include
LDFLAGS = -L$(PREFIX)/lib -Wl,-rpath,$(PREFIX)/lib
LIBS = -lm -lgsl -lgslcblas -lfftw3
MAKE = /usr/bin/make

EFFSRCDIR = $(PWD)/lib/effectivesource
EFFSRCTESTDIR = $(EFFSRCDIR)/test
KERRGEODIR = $(PWD)/lib/kerrgeodesics

all : inspector

inspector : inspector.o 
	$(GCC) $(CFLAGS) -o ../bin/inspector inspector.o $(KERRGEODIR)/korb.o $(LDFLAGS) $(LIBS)

inspector.o : inspector.c 
	$(GCC) $(CFLAGS) -O3 -c -o inspector.o $(KERRGEODIR)/korb.o inspector.c

clean :
	rm -rf *.o

.PHONY : clean all