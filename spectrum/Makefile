INCDIR=
LIBDIR=
LIBS=-lgsl -lconfig++ -lopenblas -larpack

CC=g++
WARN=-Wall -Wformat=0
CFLAGS=$(WARN) -pedantic-errors -O3
LDFLAGSSTD=-undefined dynamic_lookup
LDFLAGSGDB=$(LDFLAGSSTD) -gdb

LDFLAGS=$(LDFLAGSSTD)
#LDFLAGS=$(LDFLAGSGDB)

EXE=spectrumCZ

all:$(EXE).out

$(EXE).o:$(EXE).cpp 
	$(CC) -c $(CFLAGS) $(INCDIR) -o $@ $<

$(EXE).out:$(EXE).o
	$(CC) $(LDFLAGS) -o $(EXE).out $(LIBDIR) $(EXE).o $(LIBS)

clean:
	rm -f $(EXE).o $(EXE).out

cleanall:
	rm -f $(EXE).o $(EXE).out
