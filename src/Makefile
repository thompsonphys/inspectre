PREFIX = $(CONDA_PREFIX)
GCC = $(PREFIX)/bin/gcc
CFLAGS  = -g -Wall -std=gnu99 -I../ -I$(PREFIX)/include
LDFLAGS = -L$(PREFIX)/lib -Wl,-rpath,$(PREFIX)/lib
LIBS = -lm -lgsl -lgslcblas -lfftw3
MAKE = /usr/bin/make

EFFSRCDIR = ../lib/effectivesource
EFFSRCTESTDIR = $(EFFSRCDIR)/test
KERRGEODIR = ../lib/kerrgeodesics

all : inspectre

adaptive : inspectre_adaptive

inspectre : inspectre.o 
	$(GCC) $(CFLAGS) -o ../bin/inspectre inspectre.o $(KERRGEODIR)/korb.o $(EFFSRCTESTDIR)/kerr-equatorial-dtcoeffs.o $(EFFSRCTESTDIR)/kerr-equatorial-dttcoeffs.o $(EFFSRCTESTDIR)/kerr-equatorial-coeffs.o $(EFFSRCTESTDIR)/kerr-equatorial.o $(LDFLAGS) $(LIBS)

inspectre.o : inspectre.c 
	$(GCC) $(CFLAGS) -O3 -c -o inspectre.o inspectre.c

inspectre_adaptive : inspectre_adaptive.o 
	$(GCC) $(CFLAGS) -o ../bin/inspectre_adaptive inspectre_adaptive.o $(KERRGEODIR)/korb.o $(EFFSRCTESTDIR)/kerr-equatorial-dtcoeffs.o $(EFFSRCTESTDIR)/kerr-equatorial-dttcoeffs.o $(EFFSRCTESTDIR)/kerr-equatorial-coeffs.o $(EFFSRCTESTDIR)/kerr-equatorial.o $(LDFLAGS) $(LIBS)

inspectre_adaptive.o : inspectre_adaptive.c 
	$(GCC) $(CFLAGS) -O3 -c -o inspectre_adaptive.o inspectre_adaptive.c

clean :
	rm -rf inspectre inspectre_adaptive
	rm -rf *.o

.PHONY : clean all adaptive