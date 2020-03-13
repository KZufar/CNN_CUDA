#pragma comment (lib, "cublas.lib")
#include "stdio.h"
#include <cuda.h>
using namespace std;
#include <ctime>
#include "cuda_runtime.h"
#include "curand_kernel.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

#include <string>
#include <iomanip>
#include <time.h>
#include <iostream>
#include <cmath>
#include <math.h>

//#include "global.cuh"
#define TRAIN_NUM 60000
#define TEST_NUM 10000
#define ROW 28
#define COL 28
#define CONV_SIZE 24
#define POOL_SIZE 12
#define FC1_SIZE 45
#define FC2_SIZE 10
#define CONV_W_SIZE 5
#define CONV_W_NUM 6

__constant__ float _alpha;
__constant__ int _minibatch;
__constant__ int _epochs;

__device__ int _correct_cnt;
__device__ float _avg_error;

__device__ float _train_image[TRAIN_NUM][ROW][COL];
__device__ int _train_label[TRAIN_NUM];
__device__ float _test_image[TEST_NUM][ROW][COL];
__device__ int _test_label[TEST_NUM];

__device__ float _conv_w[CONV_W_NUM][CONV_W_SIZE][CONV_W_SIZE];
__device__ float _conv_b[CONV_W_NUM];
__device__ float _fc1_b[FC1_SIZE];
__device__ float _fc1_w[FC1_SIZE][CONV_W_NUM][POOL_SIZE][POOL_SIZE];
__device__ float _fc2_b[FC2_SIZE];
__device__ float _fc2_w[FC2_SIZE][FC1_SIZE];

__device__ float _input[ROW][COL];
__device__ float _conv_z[CONV_W_NUM][CONV_SIZE][CONV_SIZE];
__device__ float _conv_a[CONV_W_NUM][CONV_SIZE][CONV_SIZE];
__device__ int _pool_pos[CONV_W_NUM][POOL_SIZE][POOL_SIZE];
__device__ float _pool[CONV_W_NUM][POOL_SIZE][POOL_SIZE];
__device__ float _fc1_z[FC1_SIZE];
__device__ float _fc1_a[FC1_SIZE];
__device__ float _fc2_z[FC2_SIZE];
__device__ float _fc2_a[FC2_SIZE];
__device__ float _output[FC2_SIZE];
__device__ int _answer[FC2_SIZE];

__device__ float _conv_dw[CONV_W_NUM][CONV_W_SIZE][CONV_W_SIZE];
__device__ float _conv_db[CONV_W_NUM];
__device__ float _fc1_db[FC1_SIZE];
__device__ float _fc1_dw[FC1_SIZE][CONV_W_NUM][POOL_SIZE][POOL_SIZE];
__device__ float _fc2_db[FC2_SIZE];
__device__ float _fc2_dw[FC2_SIZE][FC1_SIZE];
__device__ float _C[FC2_SIZE];
__device__ float _fc2_delta[FC2_SIZE];
__device__ float _fc1_delta[FC1_SIZE];
__device__ float _conv_sigma_delta[CONV_W_NUM];
__device__ float _conv_delta[CONV_W_NUM][POOL_SIZE][POOL_SIZE];

__device__ int tmp;


float alpha = 0.2;
int epochs = 5;
int minibatch = 1;

float train_image[TRAIN_NUM][ROW][COL];
int train_label[TRAIN_NUM];
float test_image[TEST_NUM][ROW][COL];
int test_label[TEST_NUM];

float conv_w[CONV_W_NUM][CONV_W_SIZE][CONV_W_SIZE];
float conv_b[CONV_W_NUM];
float fc1_b[FC1_SIZE];
float fc1_w[FC1_SIZE][CONV_W_NUM][POOL_SIZE][POOL_SIZE];
float fc2_b[FC2_SIZE];
float fc2_w[FC2_SIZE][FC1_SIZE];

float input[ROW][COL];
float conv_z[CONV_W_NUM][CONV_SIZE][CONV_SIZE];
float conv_a[CONV_W_NUM][CONV_SIZE][CONV_SIZE];
int pool_pos[CONV_W_NUM][POOL_SIZE][POOL_SIZE];
float pool[CONV_W_NUM][POOL_SIZE][POOL_SIZE];
float fc1_z[FC1_SIZE];
float fc1_a[FC1_SIZE];
float fc2_z[FC2_SIZE];
float fc2_a[FC2_SIZE];
float output[FC2_SIZE];
int answer[FC2_SIZE];

float conv_dw[CONV_W_NUM][CONV_W_SIZE][CONV_W_SIZE];
float conv_db[CONV_W_NUM];
float fc1_db[FC1_SIZE];
float fc1_dw[FC1_SIZE][CONV_W_NUM][POOL_SIZE][POOL_SIZE];
float fc2_db[FC2_SIZE];
float fc2_dw[FC2_SIZE][FC1_SIZE];
float C[FC2_SIZE];
float fc2_delta[FC2_SIZE];
float fc1_delta[FC1_SIZE];
float conv_sigma_delta[CONV_W_NUM];
float conv_delta[CONV_W_NUM][POOL_SIZE][POOL_SIZE];

#define CHECK(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
}
int swap_endian(int val)
{
	unsigned char c1, c2, c3, c4;
	c1 = val & 255;
	c2 = (val >> 8) & 255;
	c3 = (val >> 16) & 255;
	c4 = (val >> 24) & 255;
	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

float get_rand(float fan_in)
{
	float sum = 0;
	for (int i = 0;i < 12;i++)
		sum += (float)rand() / RAND_MAX;
	sum -= 6;
	sum *= 1 / sqrt(fan_in);
	return sum;
}
void initDevice(int devNum)
{
	int dev = devNum;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("Using device %d: %s\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));
}

__device__ float _get_rand(int _rand, float fan_in)
{
	float sum = 0;
	for (int i = 0;i < 12;i++)
		sum += (float)_rand / RAND_MAX;
	sum -= 6;
	sum *= 1 / sqrt(fan_in);
	return sum;
}

__device__ float _sigmoid(float x)
{
	return (1 / (1 + exp(-1 * x)));
}

//#include "io.cuh"
void load_data()
{
	FILE* f_images = fopen("D:\\\\Zufar\\\\CUDA-CNN\\\\CudaCNN2\\\\CudaCNN2\\\\data\\\\train-images.idx3-ubyte", "rb");
	FILE* f_labels = fopen("D:\\\\Zufar\\\\CUDA-CNN\\\\CudaCNN2\\\\CudaCNN2\\\\data\\\\train-labels.idx1-ubyte", "rb");

	int tmp;

	int magic_num;
	fread(&magic_num, sizeof(int), 1, f_images);
	fread(&magic_num, sizeof(int), 1, f_labels);

	// printf("debug:%d\n",swap_endian(magic_num));

	int train_size;
	fread(&train_size, sizeof(int), 1, f_images);
	fread(&train_size, sizeof(int), 1, f_labels);
	train_size = swap_endian(train_size);

	// printf("debug:%d\n",swap_endian(train_size));

	int rows, cols;
	fread(&rows, sizeof(int), 1, f_images);
	fread(&cols, sizeof(int), 1, f_images);
	rows = swap_endian(rows);
	cols = swap_endian(cols);

	// printf("debug:%d\n",swap_endian(rows));
	// printf("debug:%d\n",swap_endian(cols));

	for (int i = 0;i < train_size;i++)
	{
		fread(&train_label[i], 1, 1, f_labels);
		if (i % 1000 == 0)
			printf("Training labels : Already read %5d labels\r", i);
		// printf("%d:debug:%d\r",i,train_label[i]);
		// system("pause");
	}
	printf("Training labels : Already read %5d labels\n", train_size);

	for (int i = 0;i < train_size;i++)
	{
		for (int j = 0;j < rows;j++)
			for (int k = 0;k < cols;k++)
			{
				tmp = 0;
				fread(&tmp, 1, 1, f_images);
				train_image[i][j][k] = tmp;
				train_image[i][j][k] /= 255;
				// printf("%d %d %d debug: %f\n",i,j,k,train_image[i][j][k]);
				// system("pause");
			}
		if (i % 1000 == 0)
			printf("Training images : Already read %5d images\r", i);
	}
	printf("Training images : Already read %5d images\n", train_size);

	fclose(f_images);
	fclose(f_labels);

	f_images = fopen("D:\\\\Zufar\\\\CUDA-CNN\\\\CudaCNN2\\\\CudaCNN2\\\\data\\\\t10k-images.idx3-ubyte", "rb");
	f_labels = fopen("D:\\\\Zufar\\\\CUDA-CNN\\\\CudaCNN2\\\\CudaCNN2\\\\data\\\\t10k-labels.idx1-ubyte", "rb");

	fread(&magic_num, sizeof(int), 1, f_images);
	fread(&magic_num, sizeof(int), 1, f_labels);

	int test_size;
	fread(&test_size, sizeof(int), 1, f_images);
	fread(&test_size, sizeof(int), 1, f_labels);
	test_size = swap_endian(test_size);

	fread(&rows, sizeof(int), 1, f_images);
	fread(&cols, sizeof(int), 1, f_images);
	rows = swap_endian(rows);
	cols = swap_endian(cols);

	for (int i = 0;i < test_size;i++)
	{
		fread(&test_label[i], 1, 1, f_labels);
		if (i % 1000 == 0)
			printf("Testing labels : Already read %5d labels\r", i);
	}
	printf("Testing labels : Already read %5d labels\n", test_size);

	for (int i = 0;i < test_size;i++)
	{
		for (int j = 0;j < rows;j++)
			for (int k = 0;k < cols;k++)
			{
				tmp = 0;
				fread(&tmp, 1, 1, f_images);
				test_image[i][j][k] = tmp;
				test_image[i][j][k] /= 255;
			}
		if (i % 1000 == 0)
			printf("Testing images : Already read %5d images\r", i);
	}
	printf("Testing images : Already read %5d images\n\n", test_size);

	fclose(f_images);
	fclose(f_labels);
}

void export_params()
{
	FILE* f_params = fopen("./params.txt", "w");

	fprintf(f_params, "6\n");

	fprintf(f_params, "conv1bias 0 6 ");
	for (int i = 0;i < CONV_W_NUM;i++)
		fprintf(f_params, "%X ", *(int*)& conv_b[i]);
	fprintf(f_params, "\n");

	fprintf(f_params, "conv1filter 0 150 ");
	for (int i = 0;i < CONV_W_NUM;i++)
		for (int j = 0;j < CONV_W_SIZE;j++)
			for (int k = 0;k < CONV_W_SIZE;k++)
				fprintf(f_params, "%X ", *(int*)& conv_w[i][j][k]);
	fprintf(f_params, "\n");

	fprintf(f_params, "ip1bias 0 45 ");
	for (int i = 0;i < FC1_SIZE;i++)
		fprintf(f_params, "%X ", *(int*)& fc1_b[i]);
	fprintf(f_params, "\n");

	fprintf(f_params, "ip1filter 0 38880 ");
	for (int i = 0;i < FC1_SIZE;i++)
		for (int j = 0;j < CONV_W_NUM;j++)
			for (int k = 0;k < POOL_SIZE;k++)
				for (int l = 0;l < POOL_SIZE;l++)
					fprintf(f_params, "%X ", *(int*)& fc1_w[i][j][k][l]);
	fprintf(f_params, "\n");

	fprintf(f_params, "ip2bias 0 10 ");
	for (int i = 0;i < FC2_SIZE;i++)
		fprintf(f_params, "%X ", *(int*)& fc2_b[i]);
	fprintf(f_params, "\n");

	fprintf(f_params, "ip2filter 0 450 ");
	for (int i = 0;i < FC2_SIZE;i++)
		for (int j = 0;j < FC1_SIZE;j++)
			fprintf(f_params, "%X ", *(int*)& fc2_w[i][j]);

	fclose(f_params);

}

//#include "global_gpu.cuh"
//#include "utils_gpu.cuh"
//#include "init_gpu.cuh"

void init_data_gpu()
{
	CHECK(cudaMemcpyToSymbol(_train_image, train_image, TRAIN_NUM * ROW * COL * sizeof(float)));
	CHECK(cudaMemcpyToSymbol(_train_label, train_label, sizeof(train_label)));
	CHECK(cudaMemcpyToSymbol(_test_image, test_image, TEST_NUM * ROW * COL * sizeof(float)));
	CHECK(cudaMemcpyToSymbol(_test_label, test_label, sizeof(test_label)));
}

__global__ void init_conv_b(int seed)
{
	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	curandState state;
	curand_init(seed, ix, 0, &state);
	float rn = _get_rand(abs((int)curand(&state)) % RAND_MAX, CONV_W_SIZE * CONV_W_SIZE);
	if (ix < CONV_W_NUM)
		_conv_b[ix] = rn;
}

__global__ void init_conv_w(int seed)
{
	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int iy = threadIdx.y + blockDim.y * blockIdx.y;
	int iz = threadIdx.z + blockDim.z * blockIdx.z;
	int idx = ix + iy * CONV_W_SIZE + iz * CONV_W_SIZE * CONV_W_SIZE;
	curandState state;
	curand_init(seed, idx, 0, &state);
	float rn = _get_rand(abs((int)curand(&state)) % RAND_MAX, CONV_W_SIZE * CONV_W_SIZE);
	if (ix < CONV_W_NUM && iy < CONV_W_SIZE && iz < CONV_W_SIZE)
		_conv_w[ix][iy][iz] = rn;
}

__global__ void init_fc1_b(int seed)
{
	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	curandState state;
	curand_init(seed, ix, 0, &state);
	float rn = _get_rand(abs((int)curand(&state)) % RAND_MAX, POOL_SIZE * POOL_SIZE * CONV_W_NUM);
	if (ix < FC1_SIZE)
		_fc1_b[ix] = rn;
}

__global__ void init_fc1_w(int seed, int i)
{
	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int iy = threadIdx.y + blockDim.y * blockIdx.y;
	int iz = threadIdx.z + blockDim.z * blockIdx.z;
	int idx = ix + iy * POOL_SIZE + iz * POOL_SIZE * POOL_SIZE;
	curandState state;
	curand_init(seed, idx, 0, &state);
	float rn = _get_rand(abs((int)curand(&state)) % RAND_MAX, POOL_SIZE * POOL_SIZE * CONV_W_NUM);
	if (ix < CONV_W_NUM && iy < POOL_SIZE && iz < POOL_SIZE)
		_fc1_w[i][ix][iy][iz] = rn;
}

__global__ void init_fc2_b(int seed)
{
	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	curandState state;
	curand_init(seed, ix, 0, &state);
	float rn = _get_rand(abs((int)curand(&state)) % RAND_MAX, FC1_SIZE);
	if (ix < FC2_SIZE)
		_fc2_b[ix] = rn;
}

__global__ void init_fc2_w(int seed)
{
	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int iy = threadIdx.y + blockDim.y * blockIdx.y;
	int idx = ix + iy * FC1_SIZE;
	curandState state;
	curand_init(seed, idx, 0, &state);
	float rn = _get_rand(abs((int)curand(&state)) % RAND_MAX, FC1_SIZE);
	if (ix < FC2_SIZE && iy < FC1_SIZE)
		_fc2_w[ix][iy] = rn;
}

void init_params_gpu()
{
	srand((unsigned)time(NULL));

	dim3 block1(32);
	dim3 grid1((CONV_W_NUM - 1) / block1.x + 1);
	dim3 block2(32, 32, 32);
	dim3 grid2((CONV_W_NUM - 1) / block2.x + 1, (CONV_W_SIZE - 1) / block2.y + 1, (CONV_W_SIZE - 1) / block2.z + 1);
	dim3 block3(32);
	dim3 grid3((FC1_SIZE - 1) / block3.x + 1);
	dim3 block4(32, 32, 32);
	dim3 grid4((CONV_W_NUM - 1) / block4.x + 1, (POOL_SIZE - 1) / block4.y + 1, (POOL_SIZE - 1) / block4.z + 1);
	dim3 block5(32);
	dim3 grid5((FC2_SIZE - 1) / block5.x + 1);
	dim3 block6(32, 32);
	dim3 grid6((FC2_SIZE - 1) / block6.x + 1, (FC1_SIZE - 1) / block6.y + 1);

	init_conv_b << <block1, grid1 >> > (rand());
	init_conv_w << <block2, grid2 >> > (rand());
	init_fc1_b << <block3, grid3 >> > (rand());

#pragma omp parallel for
	for (int i = 0;i < FC1_SIZE;i++)
		init_fc1_w << <block4, grid4 >> > (rand(), i);
	init_fc2_b << <block5, grid5 >> > (rand());
	init_fc2_w << <block6, grid6 >> > (rand());
	cudaDeviceSynchronize();
}
//#include "test_gpu.cuh"
__global__ void test_gpu()
{
	printf("%f %d %d\n", _alpha, _epochs, _minibatch);
	printf("%d\n", tmp);
	tmp = 18;
	printf("%d\n", tmp);
}

__global__ void test_gpu1()
{
	printf("====\n");
	printf("%d\n", tmp);
	tmp = 19;
	printf("%d\n", tmp);
}
//#include "fp_gpu.cuh"

__global__ void _set_input_train(int idx)
{
	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int iy = threadIdx.y + blockDim.y * blockIdx.y;
	if (ix < ROW && iy < COL)
	{
		_input[ix][iy] = _train_image[idx][ix][iy];
	}
}

__global__ void _set_input_test(int idx)
{
	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int iy = threadIdx.y + blockDim.y * blockIdx.y;
	if (ix < ROW && iy < COL)
	{
		_input[ix][iy] = _test_image[idx][ix][iy];
	}
}

void set_input_gpu_train(int idx)
{
	dim3 block(32, 32);
	dim3 grid((ROW - 1) / block.x + 1, (COL - 1) / block.y + 1);
	_set_input_train << <block, grid >> > (idx);
	cudaDeviceSynchronize();
}

void set_input_gpu_test(int idx)
{
	dim3 block(32, 32);
	dim3 grid((ROW - 1) / block.x + 1, (COL - 1) / block.y + 1);
	_set_input_test << <block, grid >> > (idx);
	cudaDeviceSynchronize();
}

__global__ void _input_conv()
{
	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int iy = threadIdx.y + blockDim.y * blockIdx.y;
	int iz = threadIdx.z + blockDim.z * blockIdx.z;
	if (ix < CONV_W_NUM && iy < CONV_SIZE && iz < CONV_SIZE)
	{
		_conv_z[ix][iy][iz] = 0;
		// #pragma unroll
		for (int l = 0;l < CONV_W_SIZE;l++)
			for (int m = 0;m < CONV_W_SIZE;m++)
				_conv_z[ix][iy][iz] += _input[iy + l][iz + m] * _conv_w[ix][l][m];
		_conv_z[ix][iy][iz] += _conv_b[ix];
		_conv_a[ix][iy][iz] = _sigmoid(_conv_z[ix][iy][iz]);
	}
}

void input_conv_gpu()
{
	dim3 block(8, 8, 8);
	dim3 grid((CONV_W_NUM - 1) / block.x + 1, (CONV_SIZE - 1) / block.y + 1, (CONV_SIZE - 1) / block.z + 1);
	_input_conv << <block, grid >> > ();
	cudaDeviceSynchronize();
}

__global__ void _conv_pool()
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	int k = threadIdx.z + blockDim.z * blockIdx.z;
	if (i < CONV_W_NUM && j < POOL_SIZE && k < POOL_SIZE)
	{
		float _max = _conv_a[i][j * 2][k * 2];
		_pool_pos[i][j][k] = 0;
		if (_conv_a[i][j * 2][k * 2 + 1] > _max)
		{
			_max = _conv_a[i][j * 2][k * 2 + 1];
			_pool_pos[i][j][k] = 1;
		}
		if (_conv_a[i][j * 2 + 1][k * 2] > _max)
		{
			_max = _conv_a[i][j * 2 + 1][k * 2];
			_pool_pos[i][j][k] = 2;
		}
		if (_conv_a[i][j * 2 + 1][k * 2 + 1] > _max)
		{
			_max = _conv_a[i][j * 2 + 1][k * 2 + 1];
			_pool_pos[i][j][k] = 3;
		}
		_pool[i][j][k] = _max;
	}
}

void conv_pool_gpu()
{
	dim3 block(8, 8, 8);
	dim3 grid((CONV_W_NUM - 1) / block.x + 1, (POOL_SIZE - 1) / block.y + 1, (POOL_SIZE - 1) / block.z + 1);
	_conv_pool << <block, grid >> > ();
	cudaDeviceSynchronize();
}

__global__ void _pool_fc1()
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < FC1_SIZE)
	{
		_fc1_z[i] = 0;
		for (int j = 0;j < CONV_W_NUM;j++)
			for (int k = 0;k < POOL_SIZE;k++)
				for (int l = 0;l < POOL_SIZE;l++)
					_fc1_z[i] += _pool[j][k][l] * _fc1_w[i][j][k][l];
		_fc1_z[i] += _fc1_b[i];
		_fc1_a[i] = _sigmoid(_fc1_z[i]);
	}
}

void pool_fc1_gpu()
{
	dim3 block(32);
	dim3 grid((FC1_SIZE - 1) / block.x + 1);
	_pool_fc1 << <block, grid >> > ();
	cudaDeviceSynchronize();
}

__global__ void _fc1_fc2()
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < FC2_SIZE)
	{
		_fc2_z[i] = 0;
		for (int j = 0;j < FC1_SIZE;j++)
			_fc2_z[i] += _fc1_a[j] * _fc2_w[i][j];
		_fc2_z[i] += _fc2_b[i];
		_fc2_a[i] = _sigmoid(_fc2_z[i]);
	}
}

void fc1_fc2_gpu()
{
	dim3 block(32);
	dim3 grid((FC2_SIZE - 1) / block.x + 1);
	_fc1_fc2 << <block, grid >> > ();
	cudaDeviceSynchronize();
}

__global__ void _set_answer_train(int idx)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < FC2_SIZE)
	{
		_output[i] = _fc2_a[i];
		_answer[i] = (_train_label[idx] == i) ? 1 : 0;
	}
}

__global__ void _set_answer_test(int idx)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < FC2_SIZE)
	{
		_output[i] = _fc2_a[i];
		_answer[i] = (_test_label[idx] == i) ? 1 : 0;
	}
}

void set_answer_gpu_train(int idx)
{
	dim3 block(32);
	dim3 grid((FC2_SIZE - 1) / block.x + 1);
	_set_answer_train << <block, grid >> > (idx);
	cudaDeviceSynchronize();
}

void set_answer_gpu_test(int idx)
{
	dim3 block(32);
	dim3 grid((FC2_SIZE - 1) / block.x + 1);
	_set_answer_test << <block, grid >> > (idx);
	cudaDeviceSynchronize();
}

__global__ void _check_answer_get_error()
{
	float _max = _output[0];
	int max_pos = 0;
	for (int i = 0;i < FC2_SIZE;i++)
	{
		if (_max < _output[i])
		{
			_max = _output[i];
			max_pos = i;
		}
	}
	if (_answer[max_pos])
		_correct_cnt++;
	for (int i = 0;i < FC2_SIZE;i++)
	{
		_C[i] = _output[i] - _answer[i];
		_avg_error += _C[i] * _C[i] * 0.5;
	}
}

void check_answer_get_error_gpu()
{
	_check_answer_get_error << <1, 1 >> > ();
	cudaDeviceSynchronize();
}
//#include "bp_gpu.cuh"

__global__ void _update_fc2_b()
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < FC2_SIZE)
	{
		_fc2_delta[i] = _alpha * _C[i] * (_fc2_a[i] * (1.0 - _fc2_a[i]));
		_fc2_db[i] += _fc2_delta[i];
	}
}

void update_fc2_b_gpu()
{
	dim3 block(32);
	dim3 grid((FC2_SIZE - 1) / block.x + 1);
	_update_fc2_b << <block, grid >> > ();
	cudaDeviceSynchronize();
}

__global__ void _update_fc2_w()
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i < FC2_SIZE && j < FC1_SIZE)
		_fc2_dw[i][j] += _fc2_delta[i] * _fc1_a[j];
}

void update_fc2_w_gpu()
{
	dim3 block(32, 32);
	dim3 grid((FC2_SIZE - 1) / block.x + 1, (FC1_SIZE - 1) / block.x + 1);
	_update_fc2_w << <block, grid >> > ();
	cudaDeviceSynchronize();
}

__global__ void _update_fc1_b()
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < FC1_SIZE)
	{
		float error = 0;
		for (int j = 0;j < FC2_SIZE;j++)
			error += _fc2_delta[j] * _fc2_w[j][i];
		_fc1_delta[i] = error * (_fc1_a[i] * (1.0 - _fc1_a[i]));
		_fc1_db[i] += _fc1_delta[i];
	}
}

void update_fc1_b_gpu()
{
	dim3 block(32);
	dim3 grid((FC1_SIZE - 1) / block.x + 1);
	_update_fc1_b << <block, grid >> > ();
	cudaDeviceSynchronize();
}

__global__ void _update_fc1_w(int j)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int k = threadIdx.y + blockDim.y * blockIdx.y;
	int l = threadIdx.z + blockDim.z * blockIdx.z;
	if (i < FC1_SIZE && k < POOL_SIZE && l < POOL_SIZE)
		_fc1_dw[i][j][k][l] += _fc1_delta[i] * _pool[j][k][l];
}

void update_fc1_w_gpu()
{
	dim3 block(8, 8, 8);
	dim3 grid((FC1_SIZE - 1) / block.x + 1, (POOL_SIZE - 1) / block.y + 1, (POOL_SIZE - 1) / block.z + 1);

	// #pragma omp parallel for
	for (int j = 0;j < CONV_W_NUM;j++)
		_update_fc1_w << <block, grid >> > (j);
	cudaDeviceSynchronize();
}

__global__ void _update_conv_b()
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < CONV_W_NUM)
	{
		_conv_sigma_delta[i] = 0;
		for (int j = 0;j < POOL_SIZE;j++)
			for (int k = 0;k < POOL_SIZE;k++)
			{
				float error = 0;
				_conv_delta[i][j][k] = 0;
				for (int l = 0;l < FC1_SIZE;l++)
					error += _fc1_delta[l] * _fc1_w[l][i][j][k];
				_conv_delta[i][j][k] = error * (_pool[i][j][k] * (1.0 - _pool[i][j][k]));
				_conv_sigma_delta[i] += error * (_pool[i][j][k] * (1.0 - _pool[i][j][k]));
			}
		_conv_db[i] += _conv_sigma_delta[i];
	}
}

void update_conv_b_gpu()
{
	dim3 block(32);
	dim3 grid((CONV_W_NUM - 1) / block.x + 1);
	_update_conv_b << <block, grid >> > ();
	cudaDeviceSynchronize();
}

__global__ void _update_conv_w()
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	int k = threadIdx.z + blockDim.z * blockIdx.z;
	if (i < CONV_W_NUM && j < CONV_W_SIZE && k < CONV_W_SIZE)
	{
		float error = 0;
		for (int m = 0;m < POOL_SIZE;m++)
			for (int n = 0;n < POOL_SIZE;n++)
			{
				int x = _pool_pos[i][m][n] / 2;
				int y = _pool_pos[i][m][n] % 2;
				error += _conv_delta[i][m][n] * _input[2 * m + j + x][2 * n + k + y];
			}
		_conv_dw[i][j][k] += error;
	}
}

void update_conv_w_gpu()
{
	dim3 block(8, 8, 8);
	dim3 grid((CONV_W_NUM - 1) / block.x + 1, (CONV_W_SIZE - 1) / block.y + 1, (CONV_W_SIZE - 1) / block.z + 1);
	_update_conv_w << <block, grid >> > ();
	cudaDeviceSynchronize();
}

__global__ void assign_fc2_b()
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < FC2_SIZE)
	{
		_fc2_b[i] -= (_fc2_db[i] / _minibatch);
		_fc2_db[i] = 0;
	}
}

__global__ void assign_fc2_w()
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i < FC2_SIZE && j < FC1_SIZE)
	{
		_fc2_w[i][j] -= (_fc2_dw[i][j] / _minibatch);
		_fc2_dw[i][j] = 0;
	}
}

__global__ void assign_fc1_b()
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < FC1_SIZE)
	{
		_fc1_b[i] -= (_fc1_db[i] / _minibatch);
		_fc1_db[i] = 0;
	}
}

__global__ void assign_fc1_w(int j)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int k = threadIdx.y + blockDim.y * blockIdx.y;
	int l = threadIdx.z + blockDim.z * blockIdx.z;
	if (i < FC1_SIZE && k < POOL_SIZE && l < POOL_SIZE)
	{
		_fc1_w[i][j][k][l] -= (_fc1_dw[i][j][k][l] / _minibatch);
		_fc1_dw[i][j][k][l] = 0;
	}
}

__global__ void assign_conv_b()
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < CONV_W_NUM)
	{
		_conv_b[i] -= (_conv_db[i] / _minibatch);
		_conv_db[i] = 0;
	}
}

__global__ void assign_conv_w()
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int l = threadIdx.y + blockDim.y * blockIdx.y;
	int m = threadIdx.z + blockDim.z * blockIdx.z;
	if (i < CONV_W_NUM && l < CONV_W_SIZE && m < CONV_W_SIZE)
	{
		_conv_w[i][l][m] -= (_conv_dw[i][l][m] / _minibatch);
		_conv_dw[i][l][m] = 0;
	}
}

void assign_grads_gpu()
{
	dim3 block1(32);
	dim3 grid1((FC2_SIZE - 1) / block1.x + 1);
	assign_fc2_b << <block1, grid1 >> > ();

	dim3 block2(32, 32);
	dim3 grid2((FC2_SIZE - 1) / block2.x + 1, (FC1_SIZE - 1) / block2.y + 1);
	assign_fc2_w << <block2, grid2 >> > ();

	dim3 block3(32);
	dim3 grid3((FC1_SIZE - 1) / block3.x + 1);
	assign_fc1_b << <block3, grid3 >> > ();

	dim3 block4(8, 8, 8);
	dim3 grid4((FC1_SIZE - 1) / block4.x + 1, (POOL_SIZE - 1) / block4.y + 1, (POOL_SIZE - 1) / block4.z + 1);
	for (int j = 0;j < CONV_W_NUM;j++)
		assign_fc1_w << <block4, grid4 >> > (j);

	dim3 block5(32);
	dim3 grid5((CONV_W_NUM - 1) / block5.x + 1);
	assign_conv_b << <block5, grid5 >> > ();

	dim3 block6(8, 8, 8);
	dim3 grid6((CONV_W_NUM - 1) / block6.x + 1, (CONV_W_SIZE - 1) / block6.y + 1, (CONV_W_SIZE - 1) / block6.z + 1);
	assign_conv_w << <block6, grid6 >> > ();

	cudaDeviceSynchronize();
}

int correct_cnt;
float avg_error;
float max_acc;

__global__ void _test()
{
	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int iy = threadIdx.y + blockDim.y * blockIdx.y;
	int iz = threadIdx.z + blockDim.z * blockIdx.z;

	for (int i = 5000;i < 5001;i++)
		for (int j = 0;j < ROW;j++)
		{
			for (int k = 0;k < COL;k++)
				printf("%f ", _test_image[i][j][k]);
			printf("\n");
		}
	printf("%d", _test_label[5000]);

	// printf("%f ",_test_image[ix][iy][iz]);
}

void test()
{
	puts("");
	puts("debug1");
	dim3 block(1, 1, 1);
	dim3 grid(1, 1, 1);
	_test << <block, grid >> > ();
	puts("debug2");
	cudaDeviceSynchronize();
	puts("debug3");
}
#define BASE_TYPE int
#define N 1000
#define M 64
__global__ void scalMult(const BASE_TYPE * A, const BASE_TYPE * B, BASE_TYPE * C) {
	BASE_TYPE sum = 0;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	sum = A[idx] * B[idx];
	atomicAdd(C, sum);
}

void scal(int* dev_a, int* dev_b, int* dev_c, dim3 blocksPerGrid) {
	scalMult << <blocksPerGrid, M >> > (dev_a, dev_b, dev_c);
}
int main2(int argc, char* argv[])
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int host_a[N], host_b[N];
	int* host_c = (int*)malloc(sizeof(int));
	int* dev_a, * dev_b, * dev_c, * dev_res;
	cout << "a" << "  " << "b" << endl;
	for (int i = 0; i < N; i++)
	{
		host_a[i] = rand() % 10;
		host_b[i] = rand() % 10;
		//cout << host_a[i] << " " << host_b[i] << endl;
	}
	cudaMalloc((void**)& dev_a, N * sizeof(int));
	cudaMalloc((void**)& dev_b, N * sizeof(int));
	cudaMalloc((void**)& dev_c, sizeof(int));
	cudaMemcpy(dev_a, host_a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, host_b, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemset(dev_c, 0, sizeof(int));
	//dim3 threadsPerBlock = dim3(BS, BS);
	dim3 blocksPerGrid = dim3(N / M);
	cudaEventRecord(start, 0);
	scal(dev_a, dev_b, dev_c, blocksPerGrid);

	//
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float KernelTime;
	cudaEventElapsedTime(&KernelTime, start, stop);
	printf("KernelTme: %.2f millseconds\n", KernelTime);
	cudaMemcpy(host_c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
	printf("Result: %d", host_c[0]);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("====== aininot260 gh@ysucloud.com ======\n");
	printf("         Processor used : %s\n", argv[1]);
	printf("         Learning rate  : %.2f\n", alpha);
	printf("         Epochs         : %d\n", epochs);
	printf("         Batch size     : %d\n", minibatch);
	printf("========================================\n");
	printf("\n");

	load_data();

	clock_t t = clock();

	//initDevice(0);
	CHECK(cudaMemcpyToSymbol(_alpha, &alpha, sizeof(float)));
	CHECK(cudaMemcpyToSymbol(_minibatch, &minibatch, sizeof(int)))
		CHECK(cudaMemcpyToSymbol(_epochs, &epochs, sizeof(int)));
	init_data_gpu();
	set_input_gpu_train(1);
	init_params_gpu();

	for (int i = 1;i <= epochs;i++)
	{

		int value1 = 0;
		float value2 = 0;
		cudaMemcpy((void*)& _correct_cnt, &value1, sizeof(int), cudaMemcpyHostToDevice);
		CHECK(cudaMemcpyToSymbol(_correct_cnt, &value1, sizeof(int)));
		cudaMemcpy((void*)& _avg_error, &value2, sizeof(int), cudaMemcpyHostToDevice);
		CHECK(cudaMemcpyToSymbol(_avg_error, &value2, sizeof(float)));
		//cudaMemcpyToSymbol(_correct_cnt, &value1, sizeof(int));
		//cudaMemcpyToSymbol(_avg_error, &value2, sizeof(float));
		cudaDeviceSynchronize();

		for (int j = 0;j < TRAIN_NUM;j++)
		{
			set_input_gpu_train(j);
			input_conv_gpu();
			conv_pool_gpu();
			pool_fc1_gpu();
			fc1_fc2_gpu();
			set_answer_gpu_train(j);
			check_answer_get_error_gpu();

			update_fc2_b_gpu();
			update_fc2_w_gpu();
			update_fc1_b_gpu();
			update_fc1_w_gpu();
			update_conv_b_gpu();
			update_conv_w_gpu();
			if ((j + 1) % minibatch == 0)
				assign_grads_gpu();

			if (j && j % 100 == 0)
			{

				cudaMemcpyFromSymbol(&correct_cnt, _correct_cnt, sizeof(int));
				cudaMemcpyFromSymbol(&avg_error, _avg_error, sizeof(float));
				printf("Training  Time spent : %.0fs Image count : %d Accuracy : %0.4f%% Error : %0.4f%% Epoch : %d \r", floor(((float)(clock() - t)) / CLOCKS_PER_SEC), j, ((float)correct_cnt / j) * 100, (avg_error / j) * 100, i);
			}
		}

		cudaMemcpyFromSymbol(&correct_cnt, _correct_cnt, sizeof(int));
		cudaMemcpyFromSymbol(&avg_error, _avg_error, sizeof(float));
		printf("Training  Time spent : %.0fs Image count : %d Accuracy : %0.4f%% Error : %0.4f%% Epoch : %d \n", floor(((float)(clock() - t)) / CLOCKS_PER_SEC), TRAIN_NUM, ((float)correct_cnt / TRAIN_NUM) * 100, (avg_error / TRAIN_NUM) * 100, i);

		correct_cnt = 0;
		avg_error = 0;
		cudaMemcpyToSymbol(_correct_cnt, &correct_cnt, sizeof(int));
		cudaMemcpyToSymbol(_avg_error, &avg_error, sizeof(float));

		for (int j = 0;j < TEST_NUM;j++)
		{
			set_input_gpu_test(j);
			input_conv_gpu();
			conv_pool_gpu();
			pool_fc1_gpu();
			fc1_fc2_gpu();
			set_answer_gpu_test(j);
			check_answer_get_error_gpu();

			if (j && j % 100 == 0)
			{
				cudaMemcpyFromSymbol(&correct_cnt, _correct_cnt, sizeof(int));
				cudaMemcpyFromSymbol(&avg_error, _avg_error, sizeof(float));
				printf("Testing   Time spent : %.0fs Image count : %d Accuracy : %0.4f%% Error : %0.4f%% \r", floor(((float)(clock() - t)) / CLOCKS_PER_SEC), j, ((float)correct_cnt / j) * 100, (avg_error / j) * 100);
			}
		}
		cudaMemcpyFromSymbol(&correct_cnt, _correct_cnt, sizeof(int));
		cudaMemcpyFromSymbol(&avg_error, _avg_error, sizeof(float));
		printf("Testing   Time spent : %.0fs Image count : %d Accuracy : %0.4f%% Error : %0.4f%% \n", floor(((float)(clock() - t)) / CLOCKS_PER_SEC), TEST_NUM, ((float)correct_cnt / TEST_NUM) * 100, (avg_error / TEST_NUM) * 100);

		if ((float)correct_cnt / TEST_NUM * 100 > max_acc)
		{
			max_acc = (float)correct_cnt / TEST_NUM * 100;
			//export_params();
			printf("The new model has been exported.Accuracy has reached to %0.5f%%\n\n", max_acc);
		}
		else
		{
			alpha = alpha - (alpha / 3);
			cudaMemcpyToSymbol(_alpha, &alpha, sizeof(float));
			printf("Learning rate has been reduced to %f\n\n", alpha);
		}
	}
	return 0;
}