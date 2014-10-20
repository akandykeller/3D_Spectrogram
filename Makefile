################################################################################
### CHANGE THESE LINES TO MATCH YOUR SYSTEM                                  ###
### COMPILER PATH                                                            ###
CC = /usr/bin/g++
### CUDA FOLDER PATH                                                         ###
CUDA_PATH       ?= /Developer/NVIDIA/CUDA-5.5
# CUDA code generation flags                                                 ###
GENCODE_FLAGS   := -gencode arch=compute_30,code=sm_30
# library flags -- on linux, this may look like -lgl -lglut.                 ###
#                  on mac, it would look like -framework OpenGL              ###
#                  -framework GLUT                                           ###
LD_FLAGS = -framework OpenGL -framework GLUT 
# includes for some helper functions -- if this doesn't work, you may not    ###
# have downloaded the CUDA SDK.                                              ###
CC_INCLUDE = -I$(CUDA_PATH)/samples/common/inc
################################################################################

CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
CUDA_LIB_PATH   ?= $(CUDA_PATH)/lib

# Common binaries
NVCC            ?= $(CUDA_BIN_PATH)/nvcc

# OS-specific build flags
ifneq ($(DARWIN),)
      LDFLAGS   := -Xlinker -rpath $(CUDA_LIB_PATH) -L$(CUDA_LIB_PATH) -lcudart -lGLEW
      CCFLAGS   := -arch $(OS_ARCH)
else
  ifeq ($(OS_SIZE),32)
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart -framework OpenGL -framework GLUT
      CCFLAGS   := -m32
  else
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart -framework OpenGL -framework GLUT -lsfml-audio -lsfml-system -lcufft
      CCFLAGS   := -m64
  endif
endif

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
      NVCCFLAGS := -m32
else
      NVCCFLAGS := -m64
endif

TARGETS = spectrogram

all: $(TARGETS)

spectrogram: spectrogram.cc spectrogram_kernel.o
	$(CC) $< -o $@ spectrogram_kernel.o -O3 -I$(CUDA_INC_PATH) $(CC_INCLUDE) $(LDFLAGS) $(LD_FLAGS) -Wall

spectrogram_kernel.o: spectrogram_kernel.cu
	$(NVCC) $(NVCCFLAGS) -O3 $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) $(CC_INCLUDE) -o $@ -c $<


clean:
	rm -f *.o $(TARGETS)

again: clean $(TARGETS)

# export DYLD_LIBRARY_PATH=/Developer/NVIDIA/CUDA-5.5/lib:$DYLD_LIBRARY_PATH

