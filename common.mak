################################################################################

# Cancel version control implicit rules
%:: %,v
%:: RCS/%
%:: RCS/%,v
%:: s.%
%:: SCCS/s.%
# Delete default suffixes
.SUFFIXES:
# Define suffixes of interest
.SUFFIXES: .o .c .cc .cu .cpp .h .hpp .cuh .d .mak .ld

SHELL := bash
.DEFAULT_GOAL := all

# Add comma separated list of defines to DEFS
ifdef D
  SEP := ,
  DEFS += $(patsubst %,-D%,$(subst $(SEP), ,$(D)))
endif

# Location of the CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda

# Libraries
HTK_DIR = ../libhtk
HTK_LIB = $(HTK_DIR)/lib/libhtk.a

# Compilers
CXX  ?= g++
NVCC ?= nvcc # $(CUDA_PATH)/bin/nvcc -ccbin $(CXX)

# Flags
OPT       ?= -O3
CPPFLAGS  += $(DEFS) -I$(HTK_DIR)
CFLAGS    += $(OPT)
CXXFLAGS  += -std=c++11 $(OPT)
NVCCFLAGS := -std=c++11 $(OPT) --threads 0 -rdc=true -m64
LDFLAGS   += -L$(HTK_DIR)/lib
LDLIBS    += -lhtk

ifneq ($(findstring -g,$(OPT)),)
  NVCCFLAGS += -G
endif

ifdef OMP
  OPT += -fopenmp
  LDFLAGS += -fopenmp
endif

################################################################################

# Gencode arguments
#SMS ?= 35 37 50 52 60 61 70 75 80 86
SMS ?= 86

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

NVCCFLAGS += $(GENCODE_FLAGS)

################################################################################

# Implicit rules
%.o: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -c $< -o $@

################################################################################

# Explicit rules
$(HTK_LIB):
	$(MAKE) -C $(HTK_DIR)

ifneq ($(TARGET),)
$(TARGET): $(OBJECTS) | $(HTK_LIB)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@
endif

ifneq ($(NVTARGET),)
$(NVTARGET): $(OBJECTS) | $(HTK_LIB)
	$(NVCC) $(LDFLAGS) $(NVCCFLAGS) $^ $(LDLIBS) -o $@
endif

################################################################################

# Phony targets
.PHONY: all
all: $(TARGET) $(NVTARGET)

.PHONY: clean
clean:
	rm -rf $(wildcard *.o) $(TARGET) $(NVTARGET) run_log.txt *.png

.PHONY: vars
vars:
	@echo TARGET: $(TARGET)
	@echo NVTARGET: $(NVTARGET)
	@echo MODULES: $(MODULES)
	@echo OBJECTS: $(OBJECTS)
	@echo DEFS: $(DEFS)
	@echo CPPFLAGS: $(CPPFLAGS)
	@echo CFLAGS: $(CFLAGS)
	@echo CXXFLAGS: $(CXXFLAGS)
	@echo NVCCFLAGS: $(NVCCFLAGS)
	@echo LDFLAGS: $(LDFLAGS)
	@echo LDLIBS: $(LDLIBS)

.PHONY: submit
submit: run
	@rm -f ../submit-$(notdir $(CURDIR)).zip
	zip -quR ../submit-$(notdir $(CURDIR)).zip \* -x \*.o solution
