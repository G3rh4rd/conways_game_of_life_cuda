/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// Utilities and system includes

#include <helper_cuda.h>

/* we need these includes for CUDA's random number stuff */
#include <curand.h>
#include <curand_kernel.h>


/**
 * Execute a CUDA call and print out any errors
 * @return the original cudaError_t result
 * @ingroup cudaError
 */
#define CUDA(x)				cudaCheckError((x), #x, __FILE__, __LINE__)

/**
 * Evaluates to true on success
 * @ingroup cudaError
 */
#define CUDA_SUCCESS(x)			(CUDA(x) == cudaSuccess)

/**
 * Evaluates to true on failure
 * @ingroup cudaError
 */
#define CUDA_FAILED(x)			(CUDA(x) != cudaSuccess)

/**
 * Return from the boolean function if CUDA call fails
 * @ingroup cudaError
 */
#define CUDA_VERIFY(x)			if(CUDA_FAILED(x))	return false;

/**
 * LOG_CUDA string.
 * @ingroup cudaError
 */
#define LOG_CUDA "[cuda]   "

/*
 * define this if you want all cuda calls to be printed
 * @ingroup cudaError
 */
//#define CUDA_TRACE



/**
 * cudaCheckError
 * @ingroup cudaError
 */
inline cudaError_t cudaCheckError(cudaError_t retval, const char* txt, const char* file, int line )
{
#if !defined(CUDA_TRACE)
	if( retval == cudaSuccess)
		return cudaSuccess;
#endif

	//int activeDevice = -1;
	//cudaGetDevice(&activeDevice);

	//Log("[cuda]   device %i  -  %s\n", activeDevice, txt);
	
	if( retval == cudaSuccess )
		printf(LOG_CUDA "%s\n", txt);
	else
		printf(LOG_CUDA "%s\n", txt);

	if( retval != cudaSuccess )
	{
		printf(LOG_CUDA "   %s (error %u) (hex 0x%02X)\n", cudaGetErrorString(retval), retval, retval);
		printf(LOG_CUDA "   %s:%i\n", file, line);	
	}

	return retval;
}


/**
 * Check for non-NULL pointer before freeing it, and then set the pointer to NULL.
 * @ingroup cudaError
 */
#define CUDA_FREE(x) 		if(x != NULL) { cudaFree(x); x = NULL; }

/**
 * Check for non-NULL pointer before freeing it, and then set the pointer to NULL.
 * @ingroup cudaError
 */
#define CUDA_FREE_HOST(x)	if(x != NULL) { cudaFreeHost(x); x = NULL; }

/**
 * Check for non-NULL pointer before deleting it, and then set the pointer to NULL.
 * @ingroup util
 */
#define SAFE_DELETE(x) 		if(x != NULL) { delete x; x = NULL; }


/**
 * If a / b has a remainder, round up.  This function is commonly using when launching 
 * CUDA kernels, to compute a grid size inclusive of the entire dataset if it's dimensions
 * aren't evenly divisible by the block size.
 *
 * For example:
 *
 *    const dim3 blockDim(8,8);
 *    const dim3 gridDim(iDivUp(imgWidth,blockDim.x), iDivUp(imgHeight,blockDim.y));
 *
 * Then inside the CUDA kernel, there is typically a check that thread index is in-bounds.
 *
 * Without the use of iDivUp(), if the data dimensions weren't evenly divisible by the
 * block size, parts of the data wouldn't be covered by the grid and not processed.
 *
 * @ingroup cuda
 */
inline __device__ __host__ int iDivUp( int a, int b )  		{ return (a % b != 0) ? (a / b + 1) : (a / b); }







static int _current_color = 0;

__global__ void _gpuRandom(
    uint8_t* output,
    int width,
    int height,
    int color) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if( x >= width || y >= height )
        return;
    
    // if(color == 0) output[y * width * 4 + (x * 4)] = 255;
    // else output[y * width * 4 + (x * 4)] = 0;

    // if(color == 1) output[y * width * 4 + (x * 4) + 1] = 255;
    // else output[y * width * 4 + (x * 4) + 1] = 0;

    // if(color == 1) output[y * width * 4 + (x * 4) + 2] = 255;
    // else output[y * width * 4 + (x * 4) + 2] = 0;

    /* CUDA's random number library uses curandState_t to keep track of the seed value
     we will store a random state for every thread  */
    curandState_t state;

    // /* we have to initialize the state */
    // curand_init(0, /* the seed controls the sequence of random values that are produced */
    //             0, /* the sequence number is only important with multiple cores */
    //             0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
    //             &state);
    curand_init(clock64(), x, 0, &state);

    /* curand works like rand - except that it takes a state as a parameter */
    output[y * width * 4 + (x * 4)] = curand(&state) % 255;
    output[y * width * 4 + (x * 4) + 1] = curand(&state) % 255;
    output[y * width * 4 + (x * 4) + 2] = curand(&state) % 255;
}

cudaError_t _cudaRandom(
    uint8_t* output,
    int width,
    int height)
{
    if( !output )
        return cudaErrorInvalidDevicePointer;

    if( width == 0 || height == 0 )
        return cudaErrorInvalidValue;


    // launch kernel
    const dim3 blockDim(32, 32);
    const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));
    
    _gpuRandom<<<gridDim, blockDim>>>( output, width, height, _current_color);

    _current_color++;

    if(_current_color > 2) _current_color = 0;

    return CUDA(cudaGetLastError());
}

extern "C" void
cudaRandom(
    uint8_t* output,
    int width,
    int height)
{
    _cudaRandom(output, width, height);
}


__global__ void _gpuInitRandomStates(
    uint8_t* output,
    int width,
    int height,
    int min,
    int max) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if( x >= width || y >= height )
        return;
    curandState_t state;

    curand_init(clock64(), x, 0, &state);

    output[y * width + x] = curand(&state) % (max - min) + min;
}

cudaError_t _cudaInitRandomStates(
    uint8_t* output,
    int width,
    int height,
    int min,
    int max)
{
    if( !output )
        return cudaErrorInvalidDevicePointer;

    if( width == 0 || height == 0 )
        return cudaErrorInvalidValue;


    // launch kernel
    const dim3 blockDim(32, 32);
    const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));
    
    _gpuInitRandomStates<<<gridDim, blockDim>>>( output, width, height, min, max);

    return CUDA(cudaGetLastError());
}

extern "C" void
cudaInitRandomStates(
    uint8_t* output,
    int width,
    int height,
    int min,
    int max)
{
    _cudaInitRandomStates(output, width, height, min, max);
}





















__global__ void _gpuConwayNextGeneration(
    uint8_t* input_state,
    uint8_t* output_state,
    uint8_t* img_rgba,
    int width,
    int height,
    int color_alive,
    int color_dead) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if( x >= width || y >= height )
        return;
    
    uint8_t current_cell_state = input_state[y * width + x];

    int alive_neighbours_count = 0;
    if(x > 0 && y > 0 && x < (width - 1) && (y < height -1)) {
        // iterate over all neighbours
        for(int x_i = x - 1; x_i <= x + 1; x_i++) {
            for(int y_i = y - 1; y_i <= y + 1; y_i++) {
                // don't count the current cell itself
                if(x_i != x && y_i != y) {
                    // increase count, if neighbour is alive
                    if(input_state[y_i * width + x_i] > 0) alive_neighbours_count++;
                }
            }
        }
    }

    // apply conway rules
    if( (current_cell_state == 0 && alive_neighbours_count == 3)
        || current_cell_state == 1 && (alive_neighbours_count == 2 || alive_neighbours_count == 3) ) {
        output_state[y * width + x] = 1;    // live
        current_cell_state = 1;
    } else if(current_cell_state == 1 && (alive_neighbours_count < 2 || alive_neighbours_count > 3)) {
        output_state[y * width + x] = 0;    // die
        current_cell_state = 0;
    } else {
        // just keep current state
    }


    int color_index = y * width * 4 + (x * 4);
    if(current_cell_state) {
        img_rgba[color_index] = color_alive >> 24;
        img_rgba[color_index + 1] = color_alive >> 16 && 0xFF;
        img_rgba[color_index + 2] = color_alive >> 8 && 0xFF;
        img_rgba[color_index + 3] = 255;
    } else {
        img_rgba[color_index] = color_dead >> 24;
        img_rgba[color_index + 1] = color_dead >> 16 && 0xFF;
        img_rgba[color_index + 2] = color_dead >> 8 && 0xFF;
        img_rgba[color_index + 3] = 255;
    }
}

cudaError_t _cudaConwayNextGeneration(
    uint8_t* input_state,
    uint8_t* output_state,
    uint8_t* img_rgba,
    int width,
    int height,
    int color_alive,
    int color_dead)
{
    if( !input_state || !output_state || !img_rgba )
        return cudaErrorInvalidDevicePointer;

    if( width == 0 || height == 0 )
        return cudaErrorInvalidValue;

    // launch kernel
    const dim3 blockDim(32, 32);
    const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));
    
    _gpuConwayNextGeneration<<<gridDim, blockDim>>>(input_state, output_state, img_rgba, width, height, color_alive, color_dead);

    return CUDA(cudaGetLastError());
}

extern "C" void
cudaConwayNextGeneration(
    uint8_t* input_state,
    uint8_t* output_state,
    uint8_t* img_rgba,
    int width,
    int height,
    int color_alive,
    int color_dead)
{
    _cudaConwayNextGeneration(input_state, output_state, img_rgba, width, height, color_alive, color_dead);
}