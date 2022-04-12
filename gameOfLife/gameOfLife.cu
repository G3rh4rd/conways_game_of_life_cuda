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
    int width,
    int height,
    uint8_t alive_state,
    uint8_t dead_state) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if( x >= width || y >= height )
        return;

    int cell_idx = y * width + x;
    
    uint8_t current_cell_state = input_state[cell_idx];

    int alive_neighbours_count = 0;
    // TODO: this ignores border pixels, the border need some special treatment
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
        output_state[cell_idx] = alive_state;    // live
        current_cell_state = 1;
    } else if(current_cell_state == 1 && (alive_neighbours_count < 2 || alive_neighbours_count > 3)) {
        output_state[cell_idx] = dead_state;    // die
    } else {
        // just keep current state
    }
}

cudaError_t _cudaConwayNextGeneration(
    uint8_t* input_state,
    uint8_t* output_state,
    int width,
    int height,
    uint8_t alive_state,
    uint8_t dead_state)
{
    if( !input_state || !output_state )
        return cudaErrorInvalidDevicePointer;

    if( width == 0 || height == 0 )
        return cudaErrorInvalidValue;

    // launch kernel
    const dim3 blockDim(32, 32);
    const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));
    
    _gpuConwayNextGeneration<<<gridDim, blockDim>>>(input_state, output_state, width, height, alive_state, dead_state);

    return CUDA(cudaGetLastError());
}

extern "C" void
cudaConwayNextGeneration(
    uint8_t* input_state,
    uint8_t* output_state,
    int width,
    int height,
    uint8_t alive_state,
    uint8_t dead_state)
{
    _cudaConwayNextGeneration(input_state, output_state, width, height, alive_state, dead_state);
}





__global__ void _gpuDrawConwayGeneration(
    uint8_t* input_state,
    uint32_t* img_rgba,
    int width,
    int height,
    uint8_t alive_state,
    uint8_t dead_state,
    int color_alive,
    int color_dead) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if( x >= width || y >= height )
        return;
    
    int cell_idx = y * width + x;
    int color_idx = y * width * 4 + (x * 4);

    bool is_cell_alive = (alive_state == input_state[cell_idx]);
    
    if(is_cell_alive) {
        img_rgba[color_idx] = color_alive;
    } else {
        img_rgba[color_idx] = color_dead;
    }
}

cudaError_t _cudaDrawConwayGeneration(
    uint8_t* input_state,
    uint32_t* img_rgba,
    int width,
    int height,
    uint8_t alive_state,
    uint8_t dead_state,
    int color_alive,
    int color_dead)
{
    if( !input_state || !img_rgba )
        return cudaErrorInvalidDevicePointer;

    if( width == 0 || height == 0 )
        return cudaErrorInvalidValue;

    // launch kernel
    const dim3 blockDim(32, 32);
    const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));
    
    _gpuDrawConwayGeneration<<<gridDim, blockDim>>>(input_state, img_rgba, width, height, alive_state, dead_state, color_alive, color_dead);

    return CUDA(cudaGetLastError());
}

extern "C" void
cudaDrawConwayGeneration(
    uint8_t* input_state,
    uint32_t* img_rgba,
    int width,
    int height,
    uint8_t alive_state,
    uint8_t dead_state,
    int color_alive,
    int color_dead)
{
    _cudaDrawConwayGeneration(input_state, img_rgba, width, height, alive_state, dead_state, color_alive, color_dead);
}