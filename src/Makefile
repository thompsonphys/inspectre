PREFIX = $(CONDA_PREFIX)
GCC = $(PREFIX)/bin/gcc
CFLAGS  = -g -Wall -std=gnu99 -I../ -I$(PREFIX)/include
LDFLAGS = -L$(PREFIX)/lib -Wl,-rpath,$(PREFIX)/lib
LIBS = -lm -lgsl -lgslcblas -lfftw3
MAKE = /usr/bin/make

EFFSRCDIR = $(PWD)/lib/effectivesource
EFFSRCTESTDIR = $(EFFSRCDIR)/test
KERRGEODIR = $(PWD)/lib/kerrgeodesics

all : inspectre

inspectre : inspectre.o 
	$(GCC) $(CFLAGS) -o ../bin/inspectre inspectre.o $(KERRGEODIR)/korb.o $(EFFSRCTESTDIR)/kerr-equatorial-dtcoeffs.o $(EFFSRCTESTDIR)/kerr-equatorial-dttcoeffs.o $(EFFSRCTESTDIR)/kerr-equatorial-coeffs.o $(EFFSRCTESTDIR)/kerr-equatorial.o $(LDFLAGS) $(LIBS)

inspectre.o : inspectre.c 
	$(GCC) $(CFLAGS) -O3 -c -o inspectre.o $(KERRGEODIR)/korb.o inspectre.c

clean :
	rm -rf inspectre
	rm -rf *.o

.PHONY : clean all