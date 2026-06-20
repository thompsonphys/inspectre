PREFIX = $(CONDA_PREFIX)
GCC = $(PREFIX)/bin/gcc
CFLAGS  = -g -Wall -std=gnu99 -I../ -I$(PREFIX)/include
LDFLAGS = -L$(PREFIX)/lib -Wl,-rpath,$(PREFIX)/lib
LIBS = -lm -lgsl -lgslcblas -lfftw3
MAKE = /usr/bin/make

EFFSRCDIR = ../../effectivesource
EFFSRCTESTDIR = $(EFFSRCDIR)/test
KERRGEODIR = ../../kerrgeodesic

CTXOBJS = kerr-equatorial-ctx.o kerr-equatorial-coeffs-ctx.o kerr-equatorial-dtcoeffs-ctx.o kerr-equatorial-dttcoeffs-ctx.o
EFFSRCOBJS = $(EFFSRCTESTDIR)/kerr-equatorial-dtcoeffs.o $(EFFSRCTESTDIR)/kerr-equatorial-dttcoeffs.o $(EFFSRCTESTDIR)/kerr-equatorial-coeffs.o $(EFFSRCTESTDIR)/kerr-equatorial.o $(CTXOBJS)
INSPOBJS = inspectre_lib.o $(KERRGEODIR)/korb.o $(EFFSRCOBJS)

all : inspectre

adaptive : inspectre_adaptive

inspectre : inspectre.o inspectre_lib.o $(CTXOBJS)
	$(GCC) $(CFLAGS) -o ../bin/inspectre inspectre.o $(INSPOBJS) $(LDFLAGS) $(LIBS)

inspectre.o : inspectre.c
	$(GCC) $(CFLAGS) -O3 -c -o inspectre.o inspectre.c

inspectre_adaptive : inspectre_adaptive.o inspectre_lib.o $(CTXOBJS)
	$(GCC) $(CFLAGS) -o ../bin/inspectre_adaptive inspectre_adaptive.o $(INSPOBJS) $(LDFLAGS) $(LIBS)

inspectre_adaptive.o : inspectre_adaptive.c
	$(GCC) $(CFLAGS) -O3 -c -o inspectre_adaptive.o inspectre_adaptive.c

inspectre_lib.o : inspectre_lib.c ../include/inspectre.h
	$(GCC) $(CFLAGS) -O3 -c -o inspectre_lib.o inspectre_lib.c

kerr-equatorial-ctx.o : $(EFFSRCDIR)/kerr-equatorial-ctx.c
	$(GCC) $(CFLAGS) -O0 -c -o kerr-equatorial-ctx.o $(EFFSRCDIR)/kerr-equatorial-ctx.c

kerr-equatorial-coeffs-ctx.o : $(EFFSRCDIR)/kerr-equatorial-coeffs-ctx.c
	$(GCC) $(CFLAGS) -O0 -c -o kerr-equatorial-coeffs-ctx.o $(EFFSRCDIR)/kerr-equatorial-coeffs-ctx.c

kerr-equatorial-dtcoeffs-ctx.o : $(EFFSRCDIR)/kerr-equatorial-dtcoeffs-ctx.c
	$(GCC) $(CFLAGS) -O0 -c -o kerr-equatorial-dtcoeffs-ctx.o $(EFFSRCDIR)/kerr-equatorial-dtcoeffs-ctx.c

kerr-equatorial-dttcoeffs-ctx.o : $(EFFSRCDIR)/kerr-equatorial-dttcoeffs-ctx.c
	$(GCC) $(CFLAGS) -O0 -c -o kerr-equatorial-dttcoeffs-ctx.o $(EFFSRCDIR)/kerr-equatorial-dttcoeffs-ctx.c

clean :
	rm -rf inspectre inspectre_adaptive
	rm -rf *.o

.PHONY : clean all adaptive