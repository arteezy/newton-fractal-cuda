#include <GLUT/glut.h>
#include <math.h>
#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>

struct complex {
	float real;
	float imag;	
};

const int width = 1600;

float *z = (float*) malloc(width * width * sizeof(float));

__host__ __device__ float abs(complex x) {
	return sqrt(x.real * x.real + x.imag * x.imag);
}

__host__ __device__ complex operator*(complex x, complex y) {
	complex c = {
	(x.real * y.real - x.imag * y.imag),
	(x.imag * y.real + x.real * y.imag)
	};
	return c;
}

__host__ __device__ complex operator*(complex x, float dy) {
	complex y = {dy, 0.0};
	return x*y;
}

__host__ __device__ complex operator/(complex x, complex y) {
	complex c = {
	(x.real * y.real + x.imag * y.imag) / (y.real * y.real + y.imag * y.imag),
	(x.imag * y.real - x.real * y.imag) / (y.real * y.real + y.imag * y.imag)
	};
	return c;
}

__host__ __device__ complex operator+(complex x, complex y) {
	complex c = {
	(x.real + y.real),
	(x.imag + y.imag)
	};
	return c;
}

__host__ __device__ complex operator-(complex x, complex y) {
	complex c = {
	(x.real - y.real),
	(x.imag - y.imag)
	};
	return c;
}

__host__ __device__ complex operator-(complex x, float dy) {
	complex y = {dy, 0.0};
	return x-y;
}

__host__ __device__ complex f(complex x) {
	complex d = x*x*x*x*x*x*x*x + x*x*x*x*x*x*15.0 - 16.0;
	return d;
}

__host__ __device__ complex df(complex x) {
  	complex d = x*x*x*x*x*x*x*8.0 + x*x*x*x*x*90.0;
	return d;
}

__host__ __device__ float newton(complex x0, float eps, int maxiter) {
  complex x = x0;
  int iter = 0;
  while (abs(f(x)) > eps && iter <= maxiter) {
    iter++;
    x = x - f(x)/df(x);
  }  
  return iter;
}


void MathCPU() {
   float xmin = -2, xmax = 2;
   float ymin = -2, ymax = 2;

   int xsteps = width, ysteps = width;
   float 	hx = (xmax - xmin) / xsteps,
				hy = (ymax - ymin) / ysteps; 

   float eps = 0.0001;
   int maxiter = 255;

   float x, y;
   y = ymin;
   for(int i = 0; i < ysteps; i++) {
		x = xmin;
		for(int j = 0; j < xsteps; j++) {
			complex xy = {x,y};
			z[i*width + j] = newton(xy, eps, maxiter);
			x += hx;
		}
		y += hy;
   }		
	
}

__global__ void MathGPUKernel(float *zD) {
	float xi = blockIdx.x * blockDim.x + threadIdx.x;
  	float yi = blockIdx.y * blockDim.y + threadIdx.y;

	float eps = 0.0001;
   int maxiter = 255;

	complex xy = {xi/width*4 - 2,yi/width * 4 - 2};
	zD[(int)yi*width + (int)xi] = newton(xy, eps, maxiter);
}

void MathGPU() {
	const int block_width = 16;
	
	int size = width * width * sizeof(float);
	float *zD; 
	
	cudaMalloc(&zD, size); 
	
	dim3 dimGrid(width / block_width, width / block_width); 
	dim3 dimBlock(block_width, block_width); 
	MathGPUKernel <<<dimGrid, dimBlock>>> (zD);
	
	cudaMemcpy(z, zD, size, cudaMemcpyDeviceToHost);
		
	cudaFree(zD);
}

void Display() { 
	glClear(GL_COLOR_BUFFER_BIT);
  	glBegin(GL_POINTS);

   float xmin = -2, xmax = 2;
   float ymin = -2, ymax = 2;

   int xsteps = width, ysteps = width;
   float 	hx = (xmax - xmin) / xsteps,
				hy = (ymax - ymin) / ysteps;
				
	float x, y;
   
  	float max = z[0];
  	float min = z[0];

  	for(int i = 0; i < width; i++) {
    	for(int j = 0; j < width; j++) {
      	if(z[i * width + j] >= max) max = z[i * width + j];
      	if(z[i * width + j] <= min) min = z[i * width + j];
    	}
	}
				
   y = ymin; 
   for(int i = 0; i < width; i++) {
		x = xmin;
		for(int j = 0; j < width; j++) {
			float color = (z[i*width + j] - min) / (max - min);
  			 	glColor3d(	fmod(color * 13, 1.0f),
								fmod(color * 33, 1.0f),
								fmod(color * 49, 1.0f));
			   glVertex2d(x, y);
			x += hx;
		}
		y += hy;
   }
	
	glEnd();
  	glutSwapBuffers();
}

int main(int argc, char **argv) {
  	glutInit(&argc, argv);
  	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
  	glutInitWindowSize(700, 700);
  	glutInitWindowPosition(0, 0);
  	glutCreateWindow("Newton");
  	glClearColor(1.0, 1.0, 1.0, 1.0);
  	glMatrixMode(GL_PROJECTION);
  	glLoadIdentity();
  	glOrtho(-2, 2, 2, -2, 1, -1);

	struct timeval tv;
	double st, end;
	gettimeofday(&tv, NULL);
	st = tv.tv_sec + tv.tv_usec / 1000000.0;
	
	//MathCPU();
	MathGPU();
	
	gettimeofday(&tv, NULL);
	end = tv.tv_sec + tv.tv_usec / 1000000.0;
	printf("Execution time: %.5f с\n", end - st);
	
  	glutDisplayFunc(Display);
  	glutMainLoop();
}