################################################################################

# Sources and targets
# use TARGET for host only programs (no GPU)
# use NVTARGET for GPU programs
# TARGET = solution
NVTARGET = solution
MODULES = $(if $(wildcard solution.*),solution,template)
OBJECTS = $(addsuffix .o,$(MODULES))

################################################################################

include common.mak
LDFLAGS += -L$(CUDA_PATH)/lib64
CPPFLAGS += -I$(CUDA_PATH)/include
CPPFLAGS += -I/usr/local/include/opencv4
LDLIBS += -lnppc -lnppidei -lnppif -lnppist -lnppisu -lnppitc -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio -lopencv_video


video_name ?= video_1.mp4

filter_name ?= blur


# Include "data" as an order-only prerequisite to generate data
# e.g. run: all | data
.PHONY: run
run: all
	rm -f run_log.txt
	./solution video_name filter_name\
	| tee -a run_log.txt; \

run_video: all
	rm -f run_log.txt
	./solution $(video_name) $(filter_name)\
	| tee -a run_log.txt; \
