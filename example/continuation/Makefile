INCDIR=-I$(HOME)/local/include/
LIBDIR=-L$(HOME)/local/lib/
LIBS=-lgsl -lgslcblas -lconfig++ -lergopack -lgomp
CONFIG=../cfg/readConfig

CC=g++
WARN=-Wall -Wformat=0
CFLAGS=$(WARN) -pedantic-errors -O3
LDFLAGSSTD=-undefined dynamic_lookup
LDFLAGSGDB=$(LDFLAGSSTD) -gdb

LDFLAGS=$(LDFLAGSSTD)
#LDFLAGS=$(LDFLAGSGDB)

ifeq ($(EXE),)
EXE=FPCont
#EXE=FPContLoop
#EXE=POCont
endif

all:$(EXE).out
	mkdir -p ../results/continuation/
	mkdir -p ../results/continuation/fpState/
	mkdir -p ../results/continuation/fpEigVal/
	mkdir -p ../results/continuation/fpEigVec/
	mkdir -p ../results/continuation/poState/
	mkdir -p ../results/continuation/poExp/
	mkdir -p ../results/continuation/poVecLeft/
	mkdir -p ../results/continuation/poVecRight/
	mkdir -p ../results/continuation/phase/

$(EXE).o:$(EXE).cpp
	$(CC) -c $(CFLAGS) $(INCDIR) -o $@ $<

$(CONFIG).o:$(CONFIG).cpp
	$(CC) -c $(CFLAGS) $(INCDIR) -o $@ $<

$(EXE).out:$(EXE).o $(CONFIG).o 
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBDIR) $(LIBS)

clean:
	rm -f $(EXE).o $(EXE).out

cleanall:
	rm -f $(EXE).o $(CONFIG).o  $(EXE).out
