PREFIX = $(CONDA_PREFIX)
GCC = $(PREFIX)/bin/gcc
CFLAGS  = -g -Wall -std=gnu99 -I../ -I$(PREFIX)/include
LDFLAGS = -L$(PREFIX)/lib -Wl,-rpath,$(PREFIX)/lib
LIBS = -lm -lgsl -lgslcblas -lfftw3
MAKE = /usr/bin/make

EFFSRCDIR = ../../effectivesource
EFFSRCTESTDIR = $(EFFSRCDIR)/test
KERRGEODIR = ../../kerrgeodesic
INSPDIR = ../src/

# inspectre_lib + the kerr-equatorial context objects (built under src/) needed
# by anything that calls the inspectre_lib eval/integration routines.
INSPOBJS = $(INSPDIR)/inspectre_lib.o \
  $(INSPDIR)/kerr-equatorial-ctx.o $(INSPDIR)/kerr-equatorial-coeffs-ctx.o \
  $(INSPDIR)/kerr-equatorial-dtcoeffs-ctx.o $(INSPDIR)/kerr-equatorial-dttcoeffs-ctx.o
EFFSRCOBJS = $(EFFSRCTESTDIR)/kerr-equatorial-dtcoeffs.o $(EFFSRCTESTDIR)/kerr-equatorial-dttcoeffs.o $(EFFSRCTESTDIR)/kerr-equatorial-coeffs.o $(EFFSRCTESTDIR)/kerr-equatorial.o

all : example numtest verify_lib scantest

example : example.o
	$(GCC) $(CFLAGS) -o ../bin/example example.o $(KERRGEODIR)/korb.o $(EFFSRCOBJS) $(LDFLAGS) $(LIBS)

example.o : example.c
	$(GCC) $(CFLAGS) -O3 -c -o example.o example.c

numtest : numtest.o
	$(GCC) $(CFLAGS) -o ../bin/numtest numtest.o $(KERRGEODIR)/korb.o $(EFFSRCOBJS) $(INSPOBJS) $(LDFLAGS) $(LIBS)

numtest.o : numtest.c ../include/inspectre.h
	$(GCC) $(CFLAGS) -O3 -c -o numtest.o numtest.c

verify_lib : verify_lib.o
	$(GCC) $(CFLAGS) -o ../bin/verify_lib verify_lib.o $(KERRGEODIR)/korb.o $(EFFSRCOBJS) $(INSPOBJS) $(LDFLAGS) $(LIBS)

verify_lib.o : verify_lib.c ../include/inspectre.h
	$(GCC) $(CFLAGS) -O3 -c -o verify_lib.o verify_lib.c

scantest : scantest.o
	$(GCC) $(CFLAGS) -o ../bin/scantest scantest.o $(KERRGEODIR)/korb.o $(EFFSRCOBJS) $(INSPOBJS) $(LDFLAGS) $(LIBS)

scantest.o : scantest.c ../include/inspectre.h
	$(GCC) $(CFLAGS) -O3 -c -o scantest.o scantest.c

clean :
	rm -rf example numtest verify_lib scantest
	rm -rf *.o

.PHONY : clean all