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


/*
    This example demonstrates the use of CUDA/OpenGL interoperability
    to post-process an image of a 3D scene generated in OpenGL.

    The basic steps are:
    1 - render the scene to the framebuffer
    2 - map the color texture so that its memory is accessible from CUDA
    4 - run CUDA to process the image, writing to memory
        a- either mapped from a second PBO
        b- or allocated through CUDA
    6 - copy result
        a- from the PBO to a texture with glTexSubImage2D()
        b- or map the target texture and do a cuda memory copy
    7 - display the texture with a fullscreen quad

    The example also provides two solutions for the format of the image:
      - RGBA16F : more bytes involved but easier to handle because
        compatible with regular fragment shader
      - RGBA8UI : 32bytes, but the teapot color must be scaled by 255 (so we needed GLSL code)
    How about RGBA8?  The CUDA driver does not have consistent interoperability with this format.
    Older GPUs may not store data the same way compared with newer GPUs, resulting in a swap of R and B components
    On older HW, this will need workarounds.

    Press space to toggle the CUDA processing on/off.
    Press 'a' to toggle animation.
    Press '+' and '-' to increment and decrement blur radius
*/

// this mode is "old fashion" : use glTexSubImage2D() to update the final result
// commenting it will make the sample use the other way :
// map a texture in CUDA and blit the result into it
#define USE_TEXSUBIMAGE2D

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#pragma warning(disable:4996)
#endif

// OpenGL Graphics includes
#include <helper_gl.h>
#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
// Sorry for Apple : unsigned int sampler is not available to you, yet...
// Let's switch to the use of PBO and glTexSubImage
#define USE_TEXSUBIMAGE2D
#else
#include <GL/freeglut.h>
#endif

// CUDA includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA utilities and system includes
#include <helper_cuda.h>

#include <helper_functions.h>
#include <rendercheck_gl.h>

// Shared Library Test Functions
#define MAX_EPSILON 10
#define REFRESH_DELAY     10 //ms

const char *sSDKname = "postProcessGL";

unsigned int g_TotalErrors = 0;

// CheckFBO/BackBuffer class objects
// CheckRender *g_CheckRender = NULL;

////////////////////////////////////////////////////////////////////////////////
// constants / global variables
unsigned int window_width = 512;
unsigned int window_height = 512;
unsigned int image_width = 512;
unsigned int image_height = 512;
int iGLUTWindowHandle = 0;          // handle to the GLUT window

// pbo and fbo variables
#ifdef USE_TEXSUBIMAGE2D
GLuint pbo_dest;
struct cudaGraphicsResource *cuda_pbo_dest_resource;
#else
unsigned int *cuda_dest_resource;
GLuint shDrawTex;  // draws a texture
struct cudaGraphicsResource *cuda_tex_result_resource;
#endif
// extern cudaTextureObject_t inTexObject;

GLuint fbo_source;
struct cudaGraphicsResource *cuda_tex_screen_resource;

unsigned int size_tex_data;
unsigned int num_texels;
unsigned int num_values;

// (offscreen) render target fbo variables
GLuint framebuffer;     // to bind the proper targets
GLuint depth_buffer;    // for proper depth test while rendering the scene
GLuint tex_screen;      // where we render the image
GLuint tex_cudaResult;  // where we will copy the CUDA result

int   *pArgc = NULL;
char **pArgv = NULL;


// Timer
static int fpsCount = 0;
static int fpsLimit = 1;
StopWatchInterface *timer = NULL;

static uint8_t *current_generation, *next_generation;


extern "C" void
cudaInitRandomStates(uint8_t* output, int width, int height, int min, int max);

extern "C" void
cudaConwayNextGeneration( uint8_t* input_state, uint8_t* output_state, int width, int height,
    uint8_t alive_state, uint8_t dead_state);

extern "C" void
cudaDrawConwayGeneration( uint8_t* input_state, uint32_t* img_rgba, int width, int height,
    uint8_t alive_state, uint8_t dead_state, int color_alive, int color_dead);

// Forward declarations
void runStdProgram(int argc, char **argv);
void FreeResource();
void Cleanup(int iExitCode);

// GL functionality
bool initGL(int *argc, char **argv);

#ifdef USE_TEXSUBIMAGE2D
void createPBO(GLuint *pbo, struct cudaGraphicsResource **pbo_resource);
void deletePBO(GLuint *pbo);
#endif

void createTextureDst(GLuint *tex_cudaResult, unsigned int size_x, unsigned int size_y);
void createTextureSrc(GLuint *tex_screen, unsigned int size_x, unsigned int size_y);
void deleteTexture(GLuint *tex);
void createDepthBuffer(GLuint *depth, unsigned int size_x, unsigned int size_y);
void deleteDepthBuffer(GLuint *depth);
void createFramebuffer(GLuint *fbo, GLuint color, GLuint depth);
void deleteFramebuffer(GLuint *fbo);

// rendering callbacks
void display();
void idle();
void keyboard(unsigned char key, int x, int y);
void reshape(int w, int h);
void mainMenu(int i);

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void process(int width, int height/* , int radius */)
{
    cudaArray *in_array;
    unsigned int *out_data;

    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_dest_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&out_data, &num_bytes, cuda_pbo_dest_resource));

    cudaConwayNextGeneration( current_generation, next_generation, width, height, 1, 0);

    uint8_t* temp = current_generation;
    current_generation = next_generation;
    next_generation = temp;

    // color format abgr
    cudaDrawConwayGeneration( current_generation, (uint32_t*)out_data, width, height, 1, 0, 0xFFFFFFFF, 0xFF000000);
    // printf("process: %d\n", num_bytes);
#ifdef USE_TEXSUBIMAGE2D
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_dest_resource, 0));
#endif
}

#ifdef USE_TEXSUBIMAGE2D
////////////////////////////////////////////////////////////////////////////////
//! Create PBO
////////////////////////////////////////////////////////////////////////////////
void
createPBO(GLuint *pbo, struct cudaGraphicsResource **pbo_resource)
{
    // set up vertex data parameter
    num_texels = image_width * image_height;
    num_values = num_texels * 4;
    size_tex_data = sizeof(GLubyte) * num_values;
    void *data = malloc(size_tex_data);

    // create buffer object
    glGenBuffers(1, pbo);
    glBindBuffer(GL_ARRAY_BUFFER, *pbo);
    glBufferData(GL_ARRAY_BUFFER, size_tex_data, data, GL_DYNAMIC_DRAW);
    free(data);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(pbo_resource, *pbo, cudaGraphicsMapFlagsNone));

    SDK_CHECK_ERROR_GL();
}

void
deletePBO(GLuint *pbo)
{
    glDeleteBuffers(1, pbo);
    SDK_CHECK_ERROR_GL();
    *pbo = 0;
}
#endif

// copy image and process using CUDA
void processImage()
{
    // run the Cuda kernel
    process(image_width, image_height/* , blur_radius */);

    // CUDA generated data in cuda memory or in a mapped PBO made of BGRA 8 bits
    // 2 solutions, here :
    // - use glTexSubImage2D(), there is the potential to loose performance in possible hidden conversion
    // - map the texture and blit the result thanks to CUDA API
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo_dest);

    glBindTexture(GL_TEXTURE_2D, tex_cudaResult);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                    image_width, image_height,
                    GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    SDK_CHECK_ERROR_GL();
    glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
}

// display image to the screen as textured quad
void displayImage(GLuint texture)
{
    glBindTexture(GL_TEXTURE_2D, texture);
    glEnable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glViewport(0, 0, window_width, window_height);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0);
    glVertex3f(-1.0, -1.0, 0.5);
    glTexCoord2f(1.0, 0.0);
    glVertex3f(1.0, -1.0, 0.5);
    glTexCoord2f(1.0, 1.0);
    glVertex3f(1.0, 1.0, 0.5);
    glTexCoord2f(0.0, 1.0);
    glVertex3f(-1.0, 1.0, 0.5);
    glEnd();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glDisable(GL_TEXTURE_2D);

#ifndef USE_TEXSUBIMAGE2D
    glUseProgram(0);
#endif
    SDK_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void
display()
{
    sdkStartTimer(&timer);
    
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, framebuffer);

    processImage();
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
    displayImage(tex_cudaResult);

    // NOTE: I needed to add this call so the timing is consistent.
    // Need to investigate why
    cudaDeviceSynchronize();
    sdkStopTimer(&timer);

    // flip backbuffer
    glutSwapBuffers();

    // Update fps counter, fps/title display and log
    if (++fpsCount == fpsLimit)
    {
        char cTitle[256];
        float fps = 1000.0f / sdkGetAverageTimerValue(&timer);
        sprintf(cTitle, "CUDA GL Post Processing (%d x %d): %.1f fps", window_width, window_height, fps);
        glutSetWindowTitle(cTitle);
        //printf("%s\n", cTitle);
        fpsCount = 0;
        fpsLimit = (int)((fps > 1.0f) ? fps : 1.0f);
        sdkResetTimer(&timer);
    }
}

void timerEvent(int value)
{
    glutPostRedisplay();
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
}

////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void
keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
        case (27) :
            Cleanup(EXIT_SUCCESS);
            break;

        case ' ':
            cudaInitRandomStates(current_generation, window_width, window_height, 0, 2);
            break;

        case 'a':
            break;

        case '=':
        case '+':
            break;

        case '-':
            break;
    }
}

void reshape(int w, int h)
{
    window_width = w;
    window_height = h;
}

void mainMenu(int i)
{
    keyboard((unsigned char) i, 0, 0);
}

////////////////////////////////////////////////////////////////////////////////
//!
////////////////////////////////////////////////////////////////////////////////
void
createTextureSrc(GLuint *tex_screen, unsigned int size_x, unsigned int size_y)
{
    // create a texture
    glGenTextures(1, tex_screen);
    glBindTexture(GL_TEXTURE_2D, *tex_screen);

    // set basic parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // buffer data
#ifndef USE_TEXTURE_RGBA8UI
    printf("Creating a Texture render target GL_RGBA16F_ARB\n");
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F_ARB, size_x, size_y, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
#else
    printf("Creating a Texture render target GL_RGBA8UI_EXT\n");
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI_EXT, size_x, size_y, 0, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_BYTE, NULL);
#endif
    SDK_CHECK_ERROR_GL();
    // register this texture with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterImage(&cuda_tex_screen_resource, *tex_screen,
                                                GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly));
}

////////////////////////////////////////////////////////////////////////////////
//!
////////////////////////////////////////////////////////////////////////////////
void
createTextureDst(GLuint *tex_cudaResult, unsigned int size_x, unsigned int size_y)
{
    // create a texture
    glGenTextures(1, tex_cudaResult);
    glBindTexture(GL_TEXTURE_2D, *tex_cudaResult);

    // set basic parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, size_x, size_y, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    SDK_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//!
////////////////////////////////////////////////////////////////////////////////
void
deleteTexture(GLuint *tex)
{
    glDeleteTextures(1, tex);
    SDK_CHECK_ERROR_GL();

    *tex = 0;
}

////////////////////////////////////////////////////////////////////////////////
//!
////////////////////////////////////////////////////////////////////////////////
void
createDepthBuffer(GLuint *depth, unsigned int size_x, unsigned int size_y)
{
    // create a renderbuffer
    glGenRenderbuffersEXT(1, depth);
    glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, *depth);

    // allocate storage
    glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT24, size_x, size_y);

    // clean up
    glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0);

    SDK_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//!
////////////////////////////////////////////////////////////////////////////////
void
deleteDepthBuffer(GLuint *depth)
{
    glDeleteRenderbuffersEXT(1, depth);
    SDK_CHECK_ERROR_GL();

    *depth = 0;
}

////////////////////////////////////////////////////////////////////////////////
//!
////////////////////////////////////////////////////////////////////////////////
void
createFramebuffer(GLuint *fbo, GLuint color, GLuint depth)
{
    // create and bind a framebuffer
    glGenFramebuffersEXT(1, fbo);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, *fbo);

    // attach images
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, color, 0);
    //glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_RENDERBUFFER_EXT, color);
    glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, depth);

    // clean up
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

    SDK_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//!
////////////////////////////////////////////////////////////////////////////////
void
deleteFramebuffer(GLuint *fbo)
{
    glDeleteFramebuffersEXT(1, fbo);
    SDK_CHECK_ERROR_GL();

    *fbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
#if defined(__linux__)
    char *Xstatus = getenv("DISPLAY");
    if (Xstatus == NULL)
    {
        printf("Waiving execution as X server is not running\n");
        exit(EXIT_WAIVED);
    }
    setenv ("DISPLAY", ":0", 0);
#endif

    int bytes = sizeof(uint8_t) * window_width * window_height;
    cudaMalloc(&current_generation, bytes);
    cudaMalloc(&next_generation, bytes);

    cudaInitRandomStates(current_generation, window_width, window_height, 0, 2);

    printf("%s Starting...\n\n", argv[0]);

    runStdProgram(argc, argv);

    cudaFree(current_generation);
    cudaFree(next_generation);

    exit(EXIT_SUCCESS);
}

////////////////////////////////////////////////////////////////////////////////
//!
////////////////////////////////////////////////////////////////////////////////
void FreeResource()
{
    sdkDeleteTimer(&timer);

    // unregister this buffer object with CUDA
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_tex_screen_resource));
#ifdef USE_TEXSUBIMAGE2D
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_dest_resource));
    deletePBO(&pbo_dest);
#else
    cudaFree(cuda_dest_resource);
#endif
    deleteTexture(&tex_screen);
    deleteTexture(&tex_cudaResult);
    deleteDepthBuffer(&depth_buffer);
    deleteFramebuffer(&framebuffer);

    if (iGLUTWindowHandle)
    {
        glutDestroyWindow(iGLUTWindowHandle);
    }

    // finalize logs and leave
    printf("Exiting...\n");
}

void Cleanup(int iExitCode)
{
    FreeResource();
    exit(EXIT_SUCCESS);
}

////////////////////////////////////////////////////////////////////////////////
//!
////////////////////////////////////////////////////////////////////////////////
void initGLBuffers()
{
    // create pbo
#ifdef USE_TEXSUBIMAGE2D
    createPBO(&pbo_dest, &cuda_pbo_dest_resource);
#endif
    // create texture that will receive the result of CUDA
    createTextureDst(&tex_cudaResult, image_width, image_height);

    // create texture for blitting onto the screen
    createTextureSrc(&tex_screen, image_width, image_height);
    //createRenderBuffer(&tex_screen, image_width, image_height); // Doesn't work

    SDK_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Run standard demo loop with or without GL verification
////////////////////////////////////////////////////////////////////////////////
void
runStdProgram(int argc, char **argv)
{
    // First initialize OpenGL context, so we can properly set the GL for CUDA.
    // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
    if (false == initGL(&argc, argv))
    {
        return;
    }

    // Now initialize CUDA context 
    findCudaDevice(argc, (const char **)argv);

    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    // register callbacks
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutReshapeFunc(reshape);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

    // create menu
    glutCreateMenu(mainMenu);
    glutAddMenuEntry("Create random generation", ' ');
    glutAddMenuEntry("Quit (esc)", '\033');
    glutAttachMenu(GLUT_RIGHT_BUTTON);

    initGLBuffers();
#ifndef USE_TEXSUBIMAGE2D
    initCUDABuffers();
#endif

    printf("\n"
           "\tControls\n"
           "\t(right click mouse button for Menu)\n"
           "\t[space] create random generation\n"
           "\t[esc] - Quit\n\n"
          );

    // start rendering mainloop
    glutMainLoop();

    // Normally unused return path
    Cleanup(EXIT_SUCCESS);
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool
initGL(int *argc, char **argv)
{
    // Create GL context
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(window_width, window_height);
    iGLUTWindowHandle = glutCreateWindow("CUDA OpenGL post-processing");

    // initialize necessary OpenGL extensions
    if (! isGLVersionSupported(2,0) ||
        ! areGLExtensionsSupported (
            "GL_ARB_pixel_buffer_object "
            "GL_EXT_framebuffer_object"
        ))
    {
        printf("ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

    // default initialization
#ifndef USE_TEXTURE_RGBA8UI
    glClearColor(0.5, 0.5, 0.5, 1.0);
#else
    glClearColorIuiEXT(128,128,128,255);
#endif
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1f, 10.0f);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glEnable(GL_LIGHT0);
    float red[] = { 1.0f, 0.1f, 0.1f, 1.0f };
    float white[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, red);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, white);
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 60.0f);

    SDK_CHECK_ERROR_GL();

    return true;
}
