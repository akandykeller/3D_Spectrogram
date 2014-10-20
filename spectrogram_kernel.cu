#include "spectrogram_kernel.cuh"
#include <helper_math.h>
#include <stdio.h>
#include <math.h>
#include <cufft.h>
#include <helper_cuda.h>

#include <SFML/Audio.hpp>

////////////////////////////////////////////////////////////////////////////////
// constants & defines
// Number of threads in a block.
#define BLOCK_SIZE 512
// Size of section of audio-sample which is analyzed at each timestep 
#define WINDOW_SIZE 1000.0
// Damping factor to use on amplitudes to avoid verticies leaving the visable space
#define DAMPING_FACTOR 0.000001

// macro for error-handling
#define gpuErrchk(ans) { gpuAssert((ans), (char*)__FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char* file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

#define WRAP(x,m) ((x)<(m)?(x):((x)-(m)))

// Flag for pingpong;
int pingpong = 0;

unsigned int numBodies;     // Number particles; determined at runtime.

////////////////////////////////////////////////////////////////////////////////
//! Window sample in place for gien time input
////////////////////////////////////////////////////////////////////////////////
__global__ void hannWindow(float4* newPos, Complex* sample, 
                            int size, int numBodies, float t, 
                            int row_size) {

    int i = WRAP(threadIdx.x + blockDim.x * blockIdx.x, size);

    while (i < size) {
        // First we also clear newPos for the next kernel.
        if (i < numBodies)
            newPos[i].y = 0.0;

        // Then apply the window function to our audio sample
        sample[i].x = (float) sample[i].x * 0.5 * 
                        (1.0 - cosf((float) (2.0 * M_PI * (i - t * size) / 
                            (float) ((float)size/ WINDOW_SIZE - 1.0))));
       
        // additionally use a hard cutoff to avoid some problems I was experiencing with the above
        if (i < t * size - ((float) size / WINDOW_SIZE) || i > t * size + ((float) size / WINDOW_SIZE))
            sample[i].x = 0;

        i += blockDim.x * gridDim.x;
    }

    syncthreads();
}

////////////////////////////////////////////////////////////////////////////////
//! Compute squared magnitudes in place
////////////////////////////////////////////////////////////////////////////////
__global__ void squareMagnitudes(float4* newPos, float4* oldPos, 
                                    Complex* d_signal, int sample_size, int numBodies, int row_size) {

    int i = WRAP(threadIdx.x + blockDim.x * blockIdx.x, sample_size);

    // Number of frequencied summed into each vertex
    int freqs_per_bucket = (int) floorf((float) sample_size / (float) row_size);
    int j;

    while (i < sample_size) {
        j = (int) floorf((float) i / (float) freqs_per_bucket);

        // Compute squared magnitude of transfored sample
        d_signal[i].x = d_signal[i].x * d_signal[i].x + d_signal[i].y * d_signal[i].y;
        
        // Add in the damped average value for the given vertex for
        // all applicable frequencies
        newPos[j].y += (d_signal[i].x / (float) freqs_per_bucket) * DAMPING_FACTOR;

        // Shift over old time values
        if (i >= row_size && i < numBodies + row_size) {
            newPos[i].y = oldPos[i - row_size].y;
        }

            // Damp the low end more
        if (i <= 75 || (i >= row_size - 75 && i <= row_size)) 
            newPos[i].y *= 0.0;


        i += blockDim.x * gridDim.x;
    }

    syncthreads();

    // Now map/chop the values so that they can be seen on the display
    i = WRAP(threadIdx.x + blockDim.x * blockIdx.x, sample_size);
    
    if (i < numBodies) {
        while (newPos[i].y > 25.0) {
            newPos[i].y = log(newPos[i].y);
        }
    }


}


////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda(GLuint *vbo, Complex* complex_signal, int sample_size, float fraction_elapsed)
{
    // map OpenGL buffer object for writing from CUDA
    float4* oldPos;
    float4* newPos;

    unsigned int blocks = min((float)50, ceil(sample_size/(float)BLOCK_SIZE));

    // Map opengl buffers to CUDA.
    cudaGLMapBufferObject((void**)&oldPos, vbo[pingpong]);
    cudaGLMapBufferObject((void**)&newPos, vbo[!pingpong]);

    // Create space on gpu for the signal
    Complex *d_signal;
    gpuErrchk(cudaMalloc((void **)&d_signal, sample_size * sizeof(Complex)));

    gpuErrchk(cudaMemcpy((void *) d_signal, (void *) complex_signal, sample_size * sizeof(Complex),
                               cudaMemcpyHostToDevice));

    /* First window the sample for the given time */
    hannWindow<<<blocks, BLOCK_SIZE>>>(newPos, d_signal, sample_size, numBodies, fraction_elapsed, (int) sqrt(numBodies));

    /* Compute fft on windowed sample */
    cufftHandle plan;
    checkCudaErrors(cufftPlan1d(&plan, sample_size, CUFFT_C2C, 1));

    // Transform signal
    checkCudaErrors(cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_FORWARD));

    // Set the y_pos of each vertex to the magnitude of the corresponding
    // value from the stft
    squareMagnitudes<<<blocks, BLOCK_SIZE>>>(newPos, oldPos, d_signal, sample_size, numBodies, sqrt(numBodies));

    int pos = ceil(25 * fraction_elapsed);
    printf("Progress: [+");
    for (int i = 0; i < 25; i++){
        if (i < pos)
            printf("=");
        else if (i == pos)
            printf(">");
        else
            printf(" ");
    }

    printf("]   %d %%\r", pos*4);

    // unmap buffer objects from cuda.
    cudaGLUnmapBufferObject(vbo[0]);
    cudaGLUnmapBufferObject(vbo[1]);

    //Destroy CUFFT context
    checkCudaErrors(cufftDestroy(plan));

    gpuErrchk(cudaFree(d_signal));

    //Switch buffers between old/new
    pingpong = !pingpong;
}


////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBOs(GLuint* vbo)
{
    // create buffer object
    glGenBuffers(2, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);

    // initialize buffer object; this will be used as 'oldPos' initially
    unsigned int size = numBodies * 4 * sizeof(float);

    unsigned int plane_dim = ceil(sqrt(numBodies));

    float4* temppos = (float4*)malloc(size);

    for(int i = 0; i < numBodies; ++i)
    {
        temppos[i].x = ((i % plane_dim) - plane_dim / 2.) * 0.05;
        temppos[i].y = 0.;
        temppos[i].z = (floor(i / plane_dim) - plane_dim / 2.) * 0.05;
        temppos[i].w = 1.;
    }

    // Notice only vbo[0] has initial data!
    glBufferData(GL_ARRAY_BUFFER, size, temppos, GL_DYNAMIC_DRAW);

    free(temppos);

    // Create initial 'newPos' buffer
    glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
    glBufferData(GL_ARRAY_BUFFER, size, temppos, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register buffer objects with CUDA
    gpuErrchk(cudaGLRegisterBufferObject(vbo[0]));
    gpuErrchk(cudaGLRegisterBufferObject(vbo[1]));
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBOs(GLuint* vbo)
{
    glBindBuffer(1, vbo[0]);
    glDeleteBuffers(1, &vbo[0]);
    glBindBuffer(1, vbo[1]);
    glDeleteBuffers(1, &vbo[1]);

    gpuErrchk(cudaGLUnregisterBufferObject(vbo[0]));
    gpuErrchk(cudaGLUnregisterBufferObject(vbo[1]));

    *vbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Returns the value of pingpong
////////////////////////////////////////////////////////////////////////////////
int getPingpong()
{
  return pingpong;
}

////////////////////////////////////////////////////////////////////////////////
//! Gets/sets the number of bodies
////////////////////////////////////////////////////////////////////////////////
int getNumBodies()
{
  return numBodies;
}

void setNumBodies(int n)
{
  numBodies = n;
}

