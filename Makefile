INCDIR=include/
SRCDIR=src/
PYDIR=site-packages/
INC=-I$(HOME)/.local/include/
PREFIX=$(HOME)/.local/
PYTHONPKG=$(HOME)/.local/lib/python3.6/site-packages/
#PREFIX=/usr/local/

CC=g++
WARN=-Wall -Wno-reorder -Wformat=0
CFLAGS=$(WARN) -pedantic-errors -O3
WITH_OMP=1

CPP_FILES := $(wildcard $(SRCDIR)/*.cpp)
OBJ_CPP_FILES := $(addprefix $(SRCDIR)/,$(notdir $(CPP_FILES:.cpp=.o)))

# Use OpenMP?
ifeq ($(WITH_OMP),1)
  LIBS +=-L/opt/local/lib/libgcc/ -lgomp
  CFLAGS +=-fopenmp -DWITH_OMP=$(WITH_OMP)
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
