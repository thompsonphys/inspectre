PREFIX = $(CONDA_PREFIX)
GCC = $(PREFIX)/bin/gcc
CFLAGS  = -g -Wall -std=gnu99 -I../ -I$(PREFIX)/include
LDFLAGS = -L$(PREFIX)/lib -Wl,-rpath,$(PREFIX)/lib
LIBS = -lm -lgsl -lgslcblas
MAKE = /usr/bin/make

EFFSRCDIR = $(PWD)/lib/effectivesource
EFFSRCTESTDIR = $(EFFSRCDIR)/test
KERRGEODIR = $(PWD)/lib/kerrgeodesics
SRCDIR = $(PWD)/src
TESTDIR = $(PWD)/test

all : subdirs

adaptive : subdirs-adaptive

subdirs :
	$(MAKE) -C $(EFFSRCTESTDIR)
	$(MAKE) -C $(KERRGEODIR)
	$(MAKE) -C $(SRCDIR)
	
test : subdirs
	$(MAKE) -C $(TESTDIR)

subdirs-adaptive :
	$(MAKE) -C $(EFFSRCTESTDIR)
	$(MAKE) -C $(KERRGEODIR)
	$(MAKE) -C $(SRCDIR) adaptive

clean :
	rm -rf *.o
	rm -rf bin/*
	$(MAKE) -C $(EFFSRCTESTDIR) clean
	$(MAKE) -C $(KERRGEODIR) clean
	$(MAKE) -C $(SRCDIR) clean
	$(MAKE) -C $(TESTDIR) clean

.PHONY : clean subdirs test all
