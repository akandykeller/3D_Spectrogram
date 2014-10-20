#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#include <SFML/Audio.hpp>


#include "spectrogram_kernel.cuh"

#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

// window
int window_width = 512;
int window_height = 512;

int refreshes = 0;

// vbo variables
GLuint vbo_pos[2];

// Sound buffer samples
const sf::Int16* audio_samples;
Complex* complex_signal;
int sample_size;
sf::Time duration;

sf::Clock timer;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -30.0;

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runParticles(int argc, char** argv);

GLvoid initGL();

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    if(argc != 2)
    {
        printf("Usage: ./spectrogram /path/to/audio_file.xxx\n");
        printf("Acceptible file types include: ogg, wav, flac,\n"
               " aiff, au, raw, paf, svx, nist, voc, ircam, w64,\n"
               " mat4, mat5 pvf, htk, sds, avr, sd2, caf, wve,\n"
               " mpc2k, rf64.\n");
        exit(1);
    }

    // Use 1024 verticies since it's a square and a multiple of 512 as desired.
    setNumBodies(262144/4);
    
    runParticles(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run spectrogram program using vbo's
////////////////////////////////////////////////////////////////////////////////
void runParticles(int argc, char** argv)
{
    // Create GL context
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow(" Spectrogram ");

    // Init random number generator
    srand(time(NULL));

    // initialize GL
    initGL();

    // register callbacks
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);

    // create VBO
    createVBOs(vbo_pos);

    sf::SoundBuffer audio_buffer;
    // Load it from a file
    if (!audio_buffer.loadFromFile(argv[1]))
    {
        printf("Error: Could not load audio file\n.");
        return;
    }

    sample_size = audio_buffer.getSampleCount();
    audio_samples = new sf::Int16[sample_size];
    printf("Audio sample size = %d \n", sample_size);
    audio_samples = audio_buffer.getSamples();

    duration = audio_buffer.getDuration();

    if (sample_size >= 10000000) {
        printf("Audio sample too large to analyze on GPU, taking first 10 million points.\n");
        sample_size = 10000000;
    }

    complex_signal = new Complex[sample_size];

    for (int i = 0; i < sample_size; i++) {
        complex_signal[i].x = (float) audio_samples[i];
        complex_signal[i].y = 0.0;
    }

    // Create a sound source and bind it to the buffer
    sf::Sound sound;
    sound.setBuffer(audio_buffer);
    // Play the sound
    sound.play();

    // start rendering mainloop
    glutMainLoop();
}


////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
GLvoid initGL()
{
    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    
    // TODO (maybe) :: depending on your parameters, you may need to change
    // near and far view distances (1, 500), to better see the simulation.
    // If you do this, probably also change translate_z initial value at top.
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 1, 500.0);
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
    refreshes++;

    sf::Time elapsed = timer.getElapsedTime();

    //printf("frames/sec: %f \n", (float) refreshes / (float) elapsed.asSeconds());

    runCuda(vbo_pos, complex_signal, sample_size, elapsed.asSeconds() / duration.asSeconds());

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

    // render from the vbo with newPos
    glBindBuffer(GL_ARRAY_BUFFER, vbo_pos[getPingpong()]);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(1.0, 0.0, 0.0);
    glDrawArrays(GL_POINTS, 0, getNumBodies());
    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();
    glutPostRedisplay();
}

////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch(key) {
    case(27) :
        printf("\nExiting.\n\n");
        deleteVBOs(vbo_pos);
        exit(0);
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN) {
        mouse_buttons |= 1<<button;
    } else if (state == GLUT_UP) {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
    glutPostRedisplay();
}

void motion(int x, int y)
{
    float dx, dy;
    dx = x - mouse_old_x;
    dy = y - mouse_old_y;

    if (mouse_buttons & 1) {
        rotate_x += dy * 0.2;
        rotate_y += dx * 0.2;
    } else if (mouse_buttons & 4) {
        translate_z += dy * 0.01;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}
