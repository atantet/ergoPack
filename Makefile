INCDIR=include/
SRCDIR=src/
PYDIR=site-packages/
INC=-I$(HOME)/.local/include/ -I/opt/local/include/
LIBS=-L$(HOME)/.local/lib/ -L/opt/local/lib/
PREFIX=$(HOME)/.local/
PYTHONPKG=$(HOME)/.local/miniconda3/lib/python3.7/site-packages
#PREFIX=/usr/local/

CC=g++ -g
WARN=-Wall -Wno-reorder -Wformat=0
CFLAGS=$(WARN)  -O3
WITH_OMP=0

CPP_FILES := $(wildcard $(SRCDIR)/*.cpp)
OBJ_CPP_FILES := $(addprefix $(SRCDIR)/,$(notdir $(CPP_FILES:.cpp=.o)))

# Use OpenMP?
ifeq ($(WITH_OMP),1)
  LIBS +=-L/opt/local/lib/libgcc/ -lgomp
  CFLAGS +=-fopenmp -DWITH_OMP=$(WITH_OMP) -std=gnu++2a
endif

EXE=libergopack

all:$(EXE).a

$(SRCDIR)/%.o:$(SRCDIR)/%.cpp
	$(CC) -c $(CFLAGS) -I$(INCDIR) $(INC) -o $@ $<

$(EXE).a:$(OBJ_CPP_FILES)
	ar -r $(EXE).a $^

clean:
	rm -f $(OBJ_CPP_FILES) $(EXE).a

install:
	cp $(INCDIR)/* $(PREFIX)/include/
	cp $(EXE).a $(PREFIX)/lib/
	mkdir -p $(PYTHONPKG)/ergoPack
	cp $(PYDIR)/*.py $(PYTHONPKG)/ergoPack/
