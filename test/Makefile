PREFIX = $(CONDA_PREFIX)
GCC = $(PREFIX)/bin/gcc
CFLAGS  = -g -Wall -std=gnu99 -I../ -I$(PREFIX)/include
LDFLAGS = -L$(PREFIX)/lib -Wl,-rpath,$(PREFIX)/lib
LIBS = -lm -lgsl -lgslcblas -lfftw3
MAKE = /usr/bin/make

EFFSRCDIR = ../lib/effectivesource
EFFSRCTESTDIR = $(EFFSRCDIR)/test
KERRGEODIR = ../lib/kerrgeodesics
INSPDIR = ../src/

all : example

example : example.o 
	$(GCC) $(CFLAGS) -o ../bin/example example.o $(KERRGEODIR)/korb.o $(EFFSRCTESTDIR)/kerr-equatorial-dtcoeffs.o $(EFFSRCTESTDIR)/kerr-equatorial-dttcoeffs.o $(EFFSRCTESTDIR)/kerr-equatorial-coeffs.o $(EFFSRCTESTDIR)/kerr-equatorial.o $(LDFLAGS) $(LIBS)

example.o : example.c 
	$(GCC) $(CFLAGS) -O3 -c -o example.o example.c

clean :
	rm -rf example
	rm -rf *.o

.PHONY : clean all