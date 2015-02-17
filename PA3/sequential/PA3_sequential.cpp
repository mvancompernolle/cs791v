#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <time.h>
#include <iomanip>
#include <limits.h>
#include <math.h>

using namespace std;

#define NUM_ITERATIONS 5

struct complex{
	float real;
	float imag;
};

int cal_pixel(complex c, int maxIterations);
void processFlags(int argc, char **argv, int& size, int& iterations);
bool write_image(unsigned char **charImage, int size);
long long int todiff(struct timeval *tod1, struct timeval *tod2);

int main(int argc, char *argv[]){
	struct timeval tod1, tod2;
	int size = 2000, maxIterations = 1024;
	float scale_real, scale_imag;
	float real_max = 2.0f, real_min = -2.0f, imag_min = -2.0f, imag_max = 2.0f;
	unsigned int color;
	complex c;
	double totalTime = 0.0f;
	unsigned char **charImage;
	ofstream fout;

	processFlags(argc, argv, size, maxIterations);

	cout << "Sequential Run Times: " << endl;

	// create pixel array
	charImage = new unsigned char*[size];
	for(int n=0; n<size; n++){ 
		charImage[n] = new unsigned char[size];
	}

	// calculate the scaling for the image
	scale_real = (real_max - real_min)/size;
	scale_imag = (imag_max - imag_min)/size;

	// calculate time for the number of iterations and average
	for(int iteration=0; iteration < NUM_ITERATIONS; iteration++){

		// get time right before pixel calculations start
		gettimeofday(&tod1, NULL);

		// calculate all the pixels
		for(int x=0; x < size; x++){

			c.real = real_min + ((float) x * scale_real);
			for(int y=0; y < size; y++){

				c.imag = imag_min + ((float) y * scale_imag);
				color = cal_pixel(c, maxIterations);
				charImage[x][y]=(unsigned char)color; 

			}
		}

		// get time as soon as pixels are done calculating
		gettimeofday(&tod2, NULL);

		// calculate total time and max time in order to calc avg time later
		totalTime += todiff(&tod2, &tod1);

	}

	cout << setprecision(3) << setiosflags(ios::fixed) << setiosflags(ios::showpoint) << "Avg Time Passed for size "
		<< size  << " X " << size << " = " << (totalTime/NUM_ITERATIONS)/1000 << " ms" << endl;

	// append results to the end of file
	fout.open("sequential_results.txt", ios::app);
	fout << size << ", " << (totalTime/NUM_ITERATIONS)/1000 << endl;

	// reset total and avg times
	totalTime = 0.0f;

	// write the image to a file
	write_image(charImage,size);

	// delete image 2d array
	for(int i=0; i<size; i++){
		delete []charImage[i];
	}
	delete [] charImage;

}

int cal_pixel(complex c, int maxIterations){
	int count;
	complex z;
	float temp, lengthsq;
	z.real = 0;
	z.imag = 0;
	count = 0;

	do{
		temp = z.real * z.real - z.imag * z.imag + c.real;
		z.imag = 2 * z.real * z.imag + c.imag;
		z.real = temp;
		lengthsq = z.real * z.real + z.imag * z.imag;
		count++;
	}while ((lengthsq < 4.0) && (count < maxIterations));
	
	return count;
}

void processFlags(int argc, char **argv, int& size, int& iterations){
	//cout << argc;
	if(argc > 2){

		for(int i=1; i<argc; i++){
			std::string arg = argv[i];
			if(arg == "-s")
				size = atoi(argv[i+1]);
			if(arg == "-i")
				iterations = atoi(argv[i+1]);
		}

	}
}

long long int todiff(struct timeval *tod1, struct timeval *tod2)
{
	long long t1, t2;
	t1 = tod1->tv_sec * 1000000 + tod1->tv_usec;
	t2 = tod2->tv_sec * 1000000 + tod2->tv_usec;
	return t1 - t2;
}

bool write_image(unsigned char **image, int size){
    /* http://stackoverflow.com/questions/4346831/saving-numerical-2d-array-to-image */
    FILE *f = fopen("Image_Sequential.ppm", "wb");
    fprintf(f, "P6\n%i %i 255\n", size, size);

	unsigned int r = 0, g = 0, b = 0;
    for (int y=0; y<size; y++)
        for (int x=0; x<size; x++)
        {
			// shade mandelbrot as red
			b = image[x][y];
            fputc(r, f);   // 0 .. 255
            fputc(g, f); // 0 .. 255
            fputc(b, f);  // 0 .. 255
        }
    fclose(f);
    
    return true;
}
