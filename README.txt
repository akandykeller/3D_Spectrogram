Thomas Andy Keller
June 6th, 2014

CS 179 Final Project
====================

This project is a basic implementation of a spectrogram 
(https://en.wikipedia.org/wiki/Spectrogram), which is essentially a 3D
plot of frequency over time. The program takes in an audio file as input, 
and begins playing the file while simultaneously showing the amplitude of
the frequencies present in the currently playing time sample. In the plot, 
the x-axis corresponds to frequency, the z-axis corresponds to time, and 
the y-axis corresponds to amplitude. There are some parameters that need 
to be messed around with (such as DAMPING_FACTOR and WINDOW_SIZE) in order
to get the best appearance from the plot, but I've included some sample audio
files that show the functionality to some extent with the given parameters. 

Additionally, there are some memory problems when trying to use cuFFT with 
large music files such as songs. By this I mean that cuda was running out
of memory when trying to create the FFT plan. Feel free to try it, but I've 
had difficulty with it coming out nicely as well as performing nicely.

Also, the time resolution and frequency resolution are pretty limited as 
there was a bunch of averaging and windowing that had to go on to fit all
the information into the vertex buffer. So if the frequencies don't quite 
seem to make sense then that's sometimes expected, however, you can definitely
notice when a tone changes or a new instrument enters. I guess it's more of
a visualizer than a real spectrogram, but I think it's kinda cool.


Compilation instructions, external libs, usage:
You will need to have OpenGL as well as SMFL (http://www.sfml-dev.org/) 
installed to be able to run this. SFML handles the audio importation and
playback while OpenGL/CUDA perform the analysis and visualization.

I've included the makefile that I used to compile. The important part is
including the nessecary libraries with the flags: 
-lsfml-audio -lsfml-system -lcufft

To run the program after it is compiled, simply type ./spectrogram followed
by the name of the audio file that you wish the analyze. I've given sample 
files such as tones.aiff which work well.

$ ./spectrogram tones.aiff


Why does GPU help here?
The GPU is the only thing that is allowing this to run at somewhere near
real-time speeds. It allows us to "window" every point essentially 
simultaneously, a procedure which would have otherwise been iterative
using a CPU. This is nessecary when performing a Short Term Fourier Transform
as is done here, allowing us to plot the time axis as well. Additionally, 
the GPU & CUDA provide a +10x speedup when computing the fourier transform,
something which is done at every single timestep. Furthermore, after
computing the transform, another CUDA kernel is used to reduce and 
set the vertex buffer objects to the correct height. Overall, there 
would be no way to do all of these things simultaneously on the CPU 
while playing back a music file and hoping that the visuals matched up.


What does one thread do per kernel call?
In hannWindow<<<>>>, each thread represents some number of points from 
the audio sample. These points are then scaled based on whether they fit
in the current time-window that we are looking at. 

In squareMagnitudes<<<>>>, each thread again represents some number of
points from the audio sample. However, at this point, the trasform has 
already been performed, so each thread computes the squared magnitude of
it's points, representing the amplitudes at the given frequencies. These
frequencies are then sorted into buckets which represent the verticies, 
and each thread adds it's computed magnitudes to some verticies. Additionally,
the threads help shift the old data back to show the flow of time.


What sorts of considerations did you make regarding memory?
I decided to keep the entire signal in device memory while the computations
were being performed to avoid having to continually copy back and forth. This
is why I immediately turn the audio sample into a complex vector, so that it can
be manipulated by the kernels and passed into cuFFT without need for more 
manipulation. 


Benchmarking:
Below are some of the FPS rates for various files given their relative sizes:

silence.aiff (1.8 MB): 882,000 samples ==> 55 fps

tones.aiff (1.8 MB): 882,000 samples ==> ~50 fps

nice_music.ogg (154 KB): 830,442 samples ==> ~12 fps

Caroline_Street.ogg (4.1 MB): 27,398,110 samples, reduced to 10,000,000
							==> 11.6 fps

CHVRCHES - The Mother We Share.wav (33.7 MB): 16,872,094 samples, reduced to 10,000,000
									==> ~11-12 fps

We can see that although sample counts are important (as the form the basis for the duration
of the fourier transform), they are not the only important factor. I believe that the cuFFT 
function is optimized to work with multiples of certain values, which may explain why nice_music.ogg
performed so poorly. I was not able to run this on my cpu as a comparison since I didn't have time to 
convert the functions and find a FFT library, but its clear that it would be very much slower if not
impossible. There are numerous optimization would could be made to make it run faster and cleaner, such 
as perform windowing / transforms ahead of time but these also have a space tradeoff. 

















