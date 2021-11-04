# change to CC for Cray systems
CXX = mpicxx

OPTFLAGS = -g -O3 -fopenmp -DDONT_CREATE_DIAG_FILES #-DDEBUG_PRINTF -DCHECK_COLORING_CONFLICTS
# use export ASAN_OPTIONS=verbosity=1 to check ASAN output
SNTFLAGS = -std=c++11 -fopenmp -fsanitize=address -O1 -fno-omit-frame-pointer
CXXFLAGS = -std=c++11 $(OPTFLAGS) #-DUSE_MPI_COLLECTIVES #-DUSE_32_BIT_GRAPH  #-DDEBUG_PRINTF

ENABLE_HPCLINK=0
ifeq ($(ENABLE_HPCLINK),1)
HPCLINKER = /projects/Tools/hpctoolkit/pkgs-theta/hpctoolkit-2018-08-09/bin/hpclink
# Static LLVM OpenMP
OMPT= -Wl,-rpath=/projects/Tools/hpctoolkit/pkgs-theta/llvm-openmp/lib -L/projects/Tools/hpctoolkit/pkgs-theta/llvm-openmp/lib -lomp
LDFLAGS = $(OMPT) -lm
endif

ENABLE_NETWORKIT=0
ifeq ($(ENABLE_NETWORKIT),1)
    NETWORKIT_DIR = $(HOME)/sources/NetworKit
    CXXFLAGS += -I$(NETWORKIT_DIR)/include
    NOBJFILES = generators/nkit-er.o generators/nkit-rgg.o generators/nkit-ba.o generators/nkit-hg.o generators/gen-io.o
    NTARGET = $(BIN)/graphGenerator
    LDFLAGS = -L$(NETWORKIT_DIR) -lNetworKit
endif

GOBJFILES = main.o rebuild.o distgraph.o louvain.o coloring.o compare.o
FOBJFILES = converters/convert.o converters/matrix-market.o converters/dimacs.o converters/metis.o converters/simple2.o converters/simple.o converters/snap.o converters/shards.o utils.o
POBJFILES = parallel-converters/parallel-converter.o parallel-converters/parallel-shards.o utils.o 
ALLOBJFILES = $(GOBJFILES) $(FOBJFILES) $(NOBJFILES) $(POBJFILES)

BIN = bin

GTARGET = $(BIN)/graphClustering
FTARGET = $(BIN)/fileConvert
PTARGET = $(BIN)/parallelFileConvert

ALLTARGETS = $(GTARGET) $(FTARGET) $(NTARGET) $(PTARGET) 

all: bindir $(ALLTARGETS)

bindir: $(BIN)
	
$(BIN): 
	mkdir -p $(BIN)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $^

$(GTARGET):  $(GOBJFILES)
	$(CXX) $^ $(OPTFLAGS) -o $@ -lstdc++

$(FTARGET):  $(FOBJFILES)
	$(CXX) $^ $(OPTFLAGS) -o $@ $(LDFLAGS)

$(NTARGET): utils.o $(NOBJFILES)
	$(CXX) $^ $(OPTFLAGS) -o $@ $(LDFLAGS)

$(PTARGET): $(POBJFILES)
	$(CXX) $^ $(OPTFLAGS) -o $@ $(LDFLAGS)

.PHONY: bindir clean

clean:
	rm -rf *~ $(ALLOBJFILES) $(ALLTARGETS) $(BIN) dat.out.* check.out.*
