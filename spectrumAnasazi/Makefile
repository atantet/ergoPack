# Import trilinos variables
include $(TRILINOS_BUILD)/Makefile.export.Trilinos

TRILINOS_LIB=-llocathyra -llocaepetra -llocalapack -lloca -lnoxepetra -lnoxlapack -lnox -lteko -lstratimikos -lstratimikosbelos -lstratimikosaztecoo -lstratimikosamesos -lstratimikosml -lstratimikosifpack -lifpack2-adapters -lifpack2 -lanasazitpetra -lModeLaplace -lanasaziepetra -lanasazi -lkomplex -lamesos2 -lbelostpetra -lbelosepetra -lbelos -lml -lifpack -lzoltan2 -lpamgen_extras -lpamgen -lamesos -lgaleri-xpetra -lgaleri-epetra -laztecoo -ldpliris -lisorropia -lxpetra-sup -lxpetra -lthyratpetra -lthyraepetraext -lthyraepetra -lthyracore -lthyratpetra -lthyraepetraext -lthyraepetra -lthyracore -lepetraext -ltpetraext -ltpetrainout -ltpetra -lkokkostsqr -ltpetrakernels -ltpetraclassiclinalg -ltpetraclassicnodeapi -ltpetraclassic -ltpetraext -ltpetrainout -ltpetra -lkokkostsqr -ltpetrakernels -ltpetraclassiclinalg -ltpetraclassicnodeapi -ltpetraclassic -ltriutils -lzoltan -lsimpi -lepetra -lrtop -lteuchoskokkoscomm -lteuchoskokkoscompat -lteuchosremainder -lteuchosnumerics -lteuchoscomm -lteuchosparameterlist -lteuchoscore -lteuchoskokkoscomm -lteuchoskokkoscompat -lteuchosremainder -lteuchosnumerics -lteuchoscomm -lteuchosparameterlist -lteuchoscore -lkokkosalgorithms -lkokkoscontainers -lkokkoscore -lkokkosalgorithms -lkokkoscontainers -lkokkoscore

LIBS=-lgsl -lconfig++ -lopenblas -larpack $(TRILINOS_LIB)

#CC=g++
CC=$(Trilinos_CXX_COMPILER)
#WARN=-Wall -Wformat=0
#CFLAGS=$(WARN) -pedantic-errors -O3
CFLAGS=$(Trilinos_CXX_COMPILER_FLAGS) 
LDFLAGSSTD=-undefined dynamic_lookup
LDFLAGSGDB=$(LDFLAGSSTD) -gdb

#LDFLAGS=$(LDFLAGSSTD)
LDFLAGS=$(LDFLAGSGDB)

EXE=spectrumAnasazi

all:$(EXE).out

$(EXE).o:$(EXE).cpp 
	$(CC) -c $(CFLAGS) -o $@ $<

$(EXE).out:$(EXE).o
	$(CC) $(LDFLAGS) -o $(EXE).out $(EXE).o $(LIBS)

clean:
	rm -f $(EXE).o $(EXE).out
