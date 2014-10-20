// -*- C++ -*-
#ifndef MAIN1_CUDA_CUH
#define MAIN1_CUDA_CUH

#include <cuda_runtime.h>
#ifndef GL_GLEXT_PROTOTYPES
#define GL_GLEXT_PROTOTYPES
#endif
#include <cuda_gl_interop.h>
#include <time.h>
#include <stdlib.h>     /* srand, rand */

#define NUM_BODIES 1024

typedef float2 Complex;

void runCuda(GLuint *vbo, Complex* complex_signal, int sample_size, float elapsed);
void createVBOs(GLuint* vbo);
void deleteVBOs(GLuint* vbo);
int getPingpong();
int getNumBodies();
void setNumBodies(int n);

#endif // MAIN1_CUDA_CUH
