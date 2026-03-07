PREFIX = $(CONDA_PREFIX)
GCC = $(PREFIX)/bin/gcc
CFLAGS  = -g -Wall -std=gnu99 -I../ -I$(PREFIX)/include
LDFLAGS = -L$(PREFIX)/lib -Wl,-rpath,$(PREFIX)/lib
LIBS = -lm -lgsl -lgslcblas
MAKE = /usr/bin/make

EFFSRCDIR = $(PWD)/lib/effectivesource
EFFSRCTESTDIR = $(EFFSRCDIR)/test
KERRGEODIR = $(PWD)/lib/kerrgeodesics

all : subdirs

subdirs :
	$(MAKE) -C $(EFFSRCTESTDIR)
	$(MAKE) -C $(KERRGEODIR)

clean :
	-rm -rf *.o
	$(MAKE) -C $(EFFSRCTESTDIR) clean
	$(MAKE) -C $(KERRGEODIR) clean

.PHONY : clean subdirs all