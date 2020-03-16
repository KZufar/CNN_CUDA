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
//#include "global_gpu.cuh"
#define TRAIN_NUM 2
#define TEST_NUM 10000
#define ROW 4
#define COL 4
#define FC1_SIZE 2
#define FC2_SIZE 3
#define BATCH_SIZE 2
float fc1_b[FC1_SIZE];
//float fc1_w[FC1_SIZE][ROW][COL];
float fc2_b[FC2_SIZE];
//float fc2_w[FC2_SIZE][FC1_SIZE];

float train_image[2][ROW][COL] = {
	{{ 3, 1, 2, 4 },
	{ 2, 4, 3, 1 },
	{ 1, 5, 2, 3 },
	{ 2, 3, 4, 1 }},

	{{ 5, 4, 5, 1 },
	{ 3, 2, 2, 3 },
	{ 2, 4, 7, 2 },
	{ 7, 1, 2, 5 }} };

float fc1_w[FC1_SIZE][ROW][COL] = { {
{1, 2, 3, 4},
{4, 3, 1, 1},
{1, 2, 4, 3},
{1, 3, 2, 1}},

{{4, 2, 5, 7},
{2, 3, 1, 3},
{1, 2, 3, 1},
{4, 2, 5, 7}} };


float fc2_w[FC2_SIZE][FC1_SIZE] = {
{1, 6},
{4, 3},
{2, 5} };

__constant__ float _alpha=0.2;
__constant__ int _minibatch=1;
__constant__ int _epochs;

__device__ int _correct_cnt;
__device__ float _avg_error;

int correct_cnt = 3;
float avg_error = 2;
float max_acc;

float alpha = 0.2;
int epochs = 5;
int minibatch = 1;

//float train_image[TRAIN_NUM][ROW][COL];
int train_label[TRAIN_NUM];
float test_image[TEST_NUM][ROW][COL];
int test_label[TEST_NUM];

float input[ROW][COL];
float fc1_z[FC1_SIZE];
float fc1_a[FC1_SIZE];
float fc2_z[FC2_SIZE];
float fc2_a[FC2_SIZE];
float output[FC2_SIZE];
int answer[FC2_SIZE];


float fc1_db[FC1_SIZE];
float fc1_dw[FC1_SIZE][ROW][COL];
float fc2_db[FC2_SIZE];
float fc2_dw[FC2_SIZE][FC1_SIZE];
float C[FC2_SIZE];
float fc2_delta[FC2_SIZE];
float fc1_delta[FC1_SIZE];

__device__ float _train_image[TRAIN_NUM][ROW][COL];
__device__ int _train_label[TRAIN_NUM];
__device__ float _test_image[TEST_NUM][ROW][COL];
__device__ int _test_label[TEST_NUM];

__device__ float _fc1_b[FC1_SIZE];
__device__ float _fc1_w[FC1_SIZE][ROW][COL];
__device__ float _fc2_b[FC2_SIZE];
__device__ float _fc2_w[FC2_SIZE][FC1_SIZE];

__device__ float _input[BATCH_SIZE][ROW][COL];
__device__ float _fc1_z[BATCH_SIZE][FC1_SIZE];
__device__ float _fc1_a[BATCH_SIZE][FC1_SIZE];
__device__ float _fc2_z[BATCH_SIZE][FC2_SIZE];
__device__ float _fc2_a[BATCH_SIZE][FC2_SIZE];
__device__ float _output[BATCH_SIZE][FC2_SIZE];
__device__ int _answer[BATCH_SIZE][FC2_SIZE];

__device__ float _fc1_db[FC1_SIZE];
__device__ float _fc1_dw[FC1_SIZE][ROW][COL];
__device__ float _fc2_db[FC2_SIZE];
__device__ float _fc2_dw[FC2_SIZE][FC1_SIZE];
__device__ float _C[BATCH_SIZE][FC2_SIZE];
__device__ float _fc2_delta[BATCH_SIZE][FC2_SIZE];
__device__ float _fc1_delta[BATCH_SIZE][FC1_SIZE];

__device__ int tmp;

int swap_endian(int val)
{
	unsigned char c1, c2, c3, c4;
	c1 = val & 255;
	c2 = (val >> 8) & 255;
	c3 = (val >> 16) & 255;
	c4 = (val >> 24) & 255;
	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}
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
__device__ float _sigmoid(float x)
{
	//return (1 / (1 + exp(-1 * x)));
	return x;
}

//__global__ void _set_input_train(int idx)
//{
//	int ix = threadIdx.x + blockDim.x * blockIdx.x;
//	int iy = threadIdx.y + blockDim.y * blockIdx.y;
//	if (ix < ROW && iy < COL)
//	{
//		_input[ix][iy] = _train_image[idx][ix][iy];
//	}
//}
//
//__global__ void _set_input_test(int idx)
//{
//	int ix = threadIdx.x + blockDim.x * blockIdx.x;
//	int iy = threadIdx.y + blockDim.y * blockIdx.y;
//	if (ix < ROW && iy < COL)
//	{
//		_input[ix][iy] = _test_image[idx][ix][iy];
//	}
//}
//
//void set_input_gpu_train(int idx)
//{
//	dim3 block(32, 32);
//	dim3 grid((ROW - 1) / block.x + 1, (COL - 1) / block.y + 1);
//	_set_input_train << <block, grid >> > (idx);
//	cudaDeviceSynchronize();
//}
//
//void set_input_gpu_test(int idx)
//{
//	dim3 block(32, 32);
//	dim3 grid((ROW - 1) / block.x + 1, (COL - 1) / block.y + 1);
//	_set_input_test << <block, grid >> > (idx);
//	cudaDeviceSynchronize();
//}
float sigmoid(float x)
{
	//return (1 / (1 + exp(-1 * x)));
	return x;
}

void set_input(int idx, float image[TRAIN_NUM][ROW][COL])
{
	for (int i = 0;i < ROW;i++)
		for (int j = 0;j < COL;j++)
			input[i][j] = image[idx][i][j];
}

float data1[FC1_SIZE][ROW][COL];
__device__ float _data1[FC1_SIZE][ROW][COL];

void input_fc1()
{
	for (int ib = 0;ib < FC1_SIZE;ib++) {
		fc1_z[ib] = 0;
		for (int l = 0;l < ROW;l++)
			for (int m = 0;m < COL;m++) {
				data1[ib][l][m]= input[l][m] * fc1_w[ib][l][m];
				fc1_z[ib] += input[l][m] * fc1_w[ib][l][m];
			}
		fc1_z[ib] += fc1_b[ib];
		fc1_a[ib] = sigmoid(fc1_z[ib]);
	}
	int f;
}
void fc1_fc2()
{
	for (int i = 0;i < FC2_SIZE;i++)
	{
		fc2_z[i] = 0;
		for (int j = 0;j < FC1_SIZE;j++)
			fc2_z[i] += fc1_a[j] * fc2_w[i][j];
		fc2_z[i] += fc2_b[i];
		fc2_a[i] = sigmoid(fc2_z[i]);
	}
}
void set_answer(int idx, int label[TRAIN_NUM])
{
	for (int i = 0;i < FC2_SIZE;i++)
	{
		output[i] = fc2_a[i];
		answer[i] = (label[idx] == i) ? 1 : 0;
	}
}

void check_answer(int& correct_cnt)
{
	float _max = output[0];
	int max_pos = 0;
	for (int i = 0;i < FC2_SIZE;i++)
	{
		if (_max < output[i])
		{
			_max = output[i];
			max_pos = i;
		}
	}
	if (answer[max_pos])
		correct_cnt++;
}

void get_error(float& avg_error)
{
	for (int i = 0;i < FC2_SIZE;i++)
	{
		C[i] = output[i] - answer[i];
		avg_error += C[i] * C[i] * 0.5;
	}
}
void update_fc2_b()
{
	for (int i = 0;i < FC2_SIZE;i++)
	{
		fc2_delta[i] = alpha * C[i] * (fc2_a[i] * (1.0 - fc2_a[i]));
		fc2_db[i] += fc2_delta[i];
	}
}

void update_fc2_w()
{
	for (int i = 0;i < FC2_SIZE;i++)
		for (int j = 0;j < FC1_SIZE;j++)
			fc2_dw[i][j] += fc2_delta[i] * fc1_a[j];
}

void update_fc1_b()
{
	for (int i = 0;i < FC1_SIZE;i++)
	{
		float error = 0;
		for (int j = 0;j < FC2_SIZE;j++)
			error += fc2_delta[j] * fc2_w[j][i];
		fc1_delta[i] = error * (fc1_a[i] * (1.0 - fc1_a[i]));
		fc1_db[i] += fc1_delta[i];
	}
}

void update_fc1_w(int m)
{
	for (int i = 0;i < FC1_SIZE;i++)
		for (int j = 0;j < ROW;j++)
			for (int k = 0;k < COL;k++)
				fc1_dw[i][j][k]+= fc1_delta[i] * input[j][k];
}
__global__ void _input_fc1(int j)
{
	//int ib = blockIdx.x;
	int ib = blockIdx.y;
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	//int ind = gridDim.x * blockIdx.y + blockIdx.x;
	int ind = blockIdx.x;
	_input[ind][iy][ix] = _train_image[BATCH_SIZE * j + ind][iy][ix];
	__shared__ float data[ROW * COL];
	int tid = blockDim.x * threadIdx.y + threadIdx.x;
	data[tid] = _input[ind][iy][ix] * _fc1_w[ib][iy][ix];
	_data1[ind][iy][ix] = _input[ind][iy][ix] * _fc1_w[ib][iy][ix];
	_data1[ind][iy][ix] = data[tid];
	/*float b = data[648];*/
	//data[tid] = 1;
	__syncthreads();
	for (int s = ROW * COL / 2; s > 0; s >>= 1) {
		if (tid < s)
			data[tid] += data[tid + s];
		__syncthreads();
	}
	/*int m = ROW*COL / 2;

	while (m != 0) {
		if (tid < m)
			data[tid] += data[tid + m];
		__syncthreads();
		m /= 2;
	}*/
	if (tid == 0) {
		_fc1_z[ind][ib] = data[0];
		data[0] += _fc1_b[ib];
		_fc1_a[ind][ib] = _sigmoid(data[0]);
	}
}

__global__ void _fc1_fc2()
{
	int ind = blockIdx.x;
	int ib = blockIdx.y;
	int ix = threadIdx.x;
	__shared__ float data[FC1_SIZE];
	int tid = threadIdx.x;
	data[tid] = _fc1_a[ind][ix]* _fc2_w[ib][ix];
	__syncthreads();
	for (int s = FC1_SIZE/ 2; s > 0; s >>= 1) {
		if (tid < s)
			data[tid] += data[tid + s];
		__syncthreads();
	}
	if (tid == 0) {
		_fc2_z[ind][ib] = data[0];
		data[0] += _fc2_b[ib];
		_fc2_a[ind][ib] = _sigmoid(data[0]);
	}
}

__global__ void _set_answer_train(int idx)
{
	int ind = blockIdx.x;
	int i = threadIdx.x;
	_output[ind][i] = _fc2_a[ind][i];
	_answer[ind][i] = (_train_label[BATCH_SIZE*idx+ind] == i) ? 1 : 0;
}

__global__ void _set_answer_test(int idx)
	{
		int ind = gridDim.x * blockIdx.y + blockIdx.x;

	int i = threadIdx.x;
	_output[ind][i] = _fc2_a[ind][i];
	_answer[ind][i] = (_test_label[ind * idx + ind] == i) ? 1 : 0;
}

__global__ void _check_answer_get_error()
{
	//int ind = gridDim.x * blockIdx.y + blockIdx.x;
	int ind = threadIdx.x;

	float _max = _output[ind][0];
	int max_pos = 0;
	for (int i = 0;i < FC2_SIZE;i++)
	{
		if (_max < _output[ind][i])
		{
			_max = _output[ind][i];
			max_pos = i;
		}
	}
	__shared__ int data[BATCH_SIZE];
	if (_answer[ind][max_pos])
		data[ind] = 1;
	else
		data[ind] = 0;
	__syncthreads();

	for (int s = BATCH_SIZE / 2; s > 0; s >>= 1) {
		if (ind < s)
			data[ind] += data[ind + s];
		__syncthreads();
	}
	if (ind == 0) {
		_correct_cnt += data[0];
	}
	for (int i = 0;i < FC2_SIZE;i++)
	{
		_C[ind][i] = _output[ind][i] - _answer[ind][i];
		//_avg_error += _C[ind][i] * _C[ind][i] * 0.5;
	}
	/*if (j && j % 100 == 0)
	{
		printf("Accuracy : %0.4f%% Error : %0.4f%% \r", ((float)_correct_cnt / j) * 100, (_avg_error / j) * 100);
	}*/
}

__device__ float _fc2_del[BATCH_SIZE][FC2_SIZE];
__device__ float _fc2_del_w[BATCH_SIZE][FC2_SIZE][FC1_SIZE];
__device__ float _fc1_del[BATCH_SIZE][FC1_SIZE];
__device__ float _fc1_del_w[BATCH_SIZE][FC1_SIZE][ROW][COL];

__global__ void _calcDelta() {
	int ind = gridDim.x * blockIdx.y + blockIdx.x;
	int i = threadIdx.x;
	
	_fc2_del[ind][i] = _alpha * _C[ind][i] * (_fc2_a[ind][i] * (1.0 - _fc2_a[ind][i]));

	if (i < FC1_SIZE) {
		float error = 0;
		for (int j = 0;j < FC2_SIZE;j++)
			error += _fc2_del[ind][j] * _fc2_w[j][i];
		_fc1_del[ind][i] = error * (_fc1_a[ind][i] * (1.0 - _fc1_a[ind][i]));
	}

	if (i < FC2_SIZE) {
		if (ind == 0) {
			for (int k = 0;k < BATCH_SIZE;k++) {
				_fc2_db[i] += _fc2_del[k][i];
			}
		}
		/*_fc2_b[i] -= _fc2_db[i] / BATCH_SIZE;
		_fc2_db[i] = 0;*/
		for (int f = 0;f < FC1_SIZE;f++) {
			_fc2_del_w[ind][i][f] = _fc2_del[ind][i] * _fc1_a[ind][f];
			__syncthreads();
			if (ind == 0) {
				for (int m = 0;m < BATCH_SIZE;m++) {
					_fc2_dw[i][f] += _fc2_del_w[m][i][f];
				}
			}
			/*_fc2_w[i][f] -= _fc2_dw[i][f] / BATCH_SIZE;
			_fc2_dw[i][f] = 0;*/
		}
	}
	if (i < FC1_SIZE) {
		if (ind == 0) {
			for (int k = 0;k < BATCH_SIZE;k++) {
				_fc1_db[i] += _fc1_del[k][i];
			}
		}
		/*_fc1_b[i] -= _fc1_db[i] / BATCH_SIZE;
		_fc1_db[i] = 0;*/
		for (int k = 0;k < ROW;k++) {
			for (int l = 0;l < COL;l++) {
				_fc1_del_w[ind][i][k][l] = _fc1_del[ind][i] * _input[ind][k][l];
				__syncthreads();
				if (ind == 0) {
					for (int m = 0;m < BATCH_SIZE;m++) {
						_fc1_dw[i][k][l] += _fc1_del_w[m][i][k][l];
					}
				}
				/*_fc1_w[i][k][l] -= _fc1_dw[i][k][l] / BATCH_SIZE;
				_fc1_dw[i][k][l] = 0;*/
			}
		}
	}
	/*__syncthreads();
	for (int s = FC2_SIZE / 2; s > 0; s >>= 1) {
		if (j < s)
			data_bias[j] += data_bias[j + s];
		__syncthreads();
	}*/
	/*for (int k = 0;k < BATCH_SIZE;k++) {
		_fc2_db[i] += _fc2_delta[k][i];
		for (int j = 0;j < FC1_SIZE;j++) {
			_fc2_dw[i][j] += _fc2_delta[k][i] * _fc1_a[ind][j];
		}
	}*/
}
__global__ void _update_weights_HO() {
	/*int ind = gridDim.x * blockIdx.y + blockIdx.x;
	int i = blockIdx.x;
	int j = threadIdx.x;
	__shared__ float data_bias[FC2_SIZE];
	__shared__ float data_weight[FC1_SIZE];
	if (!i&&j < FC2_SIZE) {
		data_bias[j] = _fc2_delta[j][i];
		__syncthreads();
		for (int s = FC2_SIZE/ 2; s > 0; s >>= 1) {
			if (j < s)
				data_bias[j] += data_bias[j + s];
			__syncthreads();
		}
	}
	if (!i) {
		data_weight[j] = _fc2_delta[j][i] * _fc1_a[i][j];
		__syncthreads();
		for (int s = FC1_SIZE / 2; s > 0; s >>= 1) {
			if (j < s)
				data_weight[j] += data_weight[j + s];
			__syncthreads();
		}
		if (j == 0) {
			_fc2_db[i] = data_bias[0];
			_fc2_dw[i][j] = data_weight[0];
		}
	}
	if (j == 0) {
		_fc2_w[i][j] -= data_weight[0] / BATCH_SIZE;
		_fc2_w[i][j] -= 0;
		_fc2_b[i] -= data_weight[0] / BATCH_SIZE;
		_fc2_b[i] -= 0;
	}*/
}
__global__ void _update_weights_IH(int m) {
	/*int i = blockIdx.x;
	int j = threadIdx.x;
	int h = threadIdx.y;
	int ix = threadIdx.x * blockIdx.x + threadIdx.y;
	__shared__ float data_bias[FC1_SIZE];
	__shared__ float data_weight[ROW][COL];
	if (!i && j < ROW&&h<COL) {
		data_bias[j] = _fc1_delta[ix][i];
		__syncthreads();
		for (int s = FC1_SIZE / 2; s > 0; s >>= 1) {
			if (j < s)
				data_bias[j] += data_bias[j + s];
			__syncthreads();
		}
	}
	if (!i) {
		data_weight[j][h] = _fc2_delta[j][i] * _train_image[BATCH_SIZE*m+m][i][j];
		__syncthreads();
		for (int s = FC1_SIZE / 2; s > 0; s >>= 1) {
			if (j < s)
				data_weight[j] += data_weight[j + s];
			__syncthreads();
		}
		if (j == 0) {
			_fc2_db[i] = data_bias[0];
			_fc2_dw[i][j] = data_weight[0];
		}
	}
	if (j == 0) {
		_fc2_w[i][j] -= data_weight[0] / BATCH_SIZE;
		_fc2_w[i][j] -= 0;
		_fc2_b[i] -= data_weight[0] / BATCH_SIZE;
		_fc2_b[i] -= 0;
	}*/
}
//
//__global__ void _update_fc2_b()
//{
//	int ind = gridDim.x * blockIdx.y + blockIdx.x;
//	int i = threadIdx.x;
//	_fc2_delta[ind][i] = _alpha * _C[ind][i] * (_fc2_a[ind][i] * (1.0 - _fc2_a[ind][i]));
//	_fc2_db[i] += _fc2_delta[ind][i];
//}
//
//__global__ void _update_fc2_w()
//{
//	int ind = gridDim.x * blockIdx.y + blockIdx.x;
//	int i = threadIdx.x + blockDim.x * blockIdx.x;
//	int j = threadIdx.y + blockDim.y * blockIdx.y;
//	if (i < FC2_SIZE && j < FC1_SIZE)
//		_fc2_dw[ind][i][j] += _fc2_delta[ind][i] * _fc1_a[ind][j];
//}
//
//__global__ void _update_fc1_b()
//{
//	int i = threadIdx.x + blockDim.x * blockIdx.x;
//	if (i < FC1_SIZE)
//	{
//		float error = 0;
//		for (int j = 0;j < FC2_SIZE;j++)
//			error += _fc2_delta[j] * _fc2_w[j][i];
//		_fc1_delta[i] = error * (_fc1_a[i] * (1.0 - _fc1_a[i]));
//		_fc1_db[i] += _fc1_delta[i];
//	}
//}
//
//__global__ void _update_fc1_w()
//{
//	int i = blockIdx.x;
//	int k = threadIdx.y + blockDim.y * blockIdx.y;
//	int l = threadIdx.z + blockDim.z * blockIdx.z;
//	//_fc1_del_w[ind][i] += _fc1_del[ind][i] * _input[ind][k][l];
//
//	_fc1_dw[i][k][l] += _fc1_delta[i] * _input[k][l];
//}
//
//
//__global__ void assign_fc2_b()
//{
//	int i = threadIdx.x + blockDim.x * blockIdx.x;
//	if (i < FC2_SIZE)
//	{
//		_fc2_b[i] -= (_fc2_db[i] / _minibatch);
//		_fc2_db[i] = 0;
//	}
//}
//
//__global__ void assign_fc2_w()
//{
//	int i = threadIdx.x + blockDim.x * blockIdx.x;
//	int j = threadIdx.y + blockDim.y * blockIdx.y;
//	if (i < FC2_SIZE && j < FC1_SIZE)
//	{
//		_fc2_w[i][j] -= (_fc2_dw[i][j] / _minibatch);
//		_fc2_dw[i][j] = 0;
//	}
//}
//
//__global__ void assign_fc1_b()
//{
//	int i = threadIdx.x + blockDim.x * blockIdx.x;
//	if (i < FC1_SIZE)
//	{
//		_fc1_b[i] -= (_fc1_db[i] / _minibatch);
//		_fc1_db[i] = 0;
//	}
//}
//
//__global__ void assign_fc1_w()
//{
//	int i = threadIdx.x + blockDim.x * blockIdx.x;
//	int k = threadIdx.y + blockDim.y * blockIdx.y;
//	int l = threadIdx.z + blockDim.z * blockIdx.z;
//	_fc1_w[blockIdx.x][k][l] -= (_fc1_dw[blockIdx.x][k][l] / _minibatch);
//	_fc1_dw[blockIdx.x][k][l] = 0;
//	
//}
//
//
//void assign_grads_gpu()
//{
//	dim3 block1(32);
//	dim3 grid1((FC2_SIZE - 1) / block1.x + 1);
//	assign_fc2_b << <block1, grid1 >> > ();
//
//	dim3 block2(32, 32);
//	dim3 grid2((FC2_SIZE - 1) / block2.x + 1, (FC1_SIZE - 1) / block2.y + 1);
//	assign_fc2_w << <block2, grid2 >> > ();
//
//	dim3 block3(32);
//	dim3 grid3((FC1_SIZE - 1) / block3.x + 1);
//	assign_fc1_b << <block3, grid3 >> > ();
//
//	dim3 block4(8, 8, 8);
//	//dim3 grid4((FC1_SIZE - 1) / block4.x + 1, (POOL_SIZE - 1) / block4.y + 1, (POOL_SIZE - 1) / block4.z + 1);
//	assign_fc1_w << <block4, dim3(1, 28, 28) >> > ();
//
//
//	cudaDeviceSynchronize();
//}
void init_data_gpu()
{
	cudaMemcpyToSymbol(_train_image, train_image, TRAIN_NUM * ROW * COL * sizeof(float));
	cudaMemcpyToSymbol(_train_label, train_label, sizeof(train_label));
	cudaMemcpyToSymbol(_test_image, test_image, TEST_NUM * ROW * COL * sizeof(float));
	cudaMemcpyToSymbol(_test_label, test_label, sizeof(test_label));
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
void init_params()
{
	/*for (int i = 0;i < CONV_W_NUM;i++)
	{
		for (int j = 0;j < CONV_W_SIZE;j++)
			for (int k = 0;k < CONV_W_SIZE;k++)
				conv_w[i][j][k] = get_rand(CONV_W_SIZE * CONV_W_SIZE);
		conv_b[i] = get_rand(CONV_W_SIZE * CONV_W_SIZE);
	}

	for (int i = 0;i < FC1_SIZE;i++)
	{
		for (int j = 0;j < CONV_W_NUM;j++)
			for (int k = 0;k < POOL_SIZE;k++)
				for (int l = 0;l < POOL_SIZE;l++)
					fc1_w[i][j][k][l] = get_rand(POOL_SIZE * POOL_SIZE * CONV_W_NUM);
		fc1_b[i] = get_rand(POOL_SIZE * POOL_SIZE * CONV_W_NUM);
	}*/

	for (int i = 0;i < FC1_SIZE;i++)
	{
		for (int j = 0;j < ROW;j++)
			for (int k = 0;k < COL;k++)
				fc1_w[i][j][k] = get_rand(ROW*COL);
		fc1_b[i] = get_rand(ROW*COL);
	}

	for (int i = 0;i < FC2_SIZE;i++)
	{
		for (int j = 0;j < FC1_SIZE;j++)
			fc2_w[i][j] = get_rand(FC1_SIZE);
		fc2_b[i] = get_rand(FC1_SIZE);
	}
}

__global__  void _test() {
	_correct_cnt = _correct_cnt + 10;
	_avg_error = _avg_error + 16.35;
	/*float _max = _output[0];
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
	}*/
}
float matrixMult(float a[ROW][COL], float b[FC1_SIZE][ROW][COL], int k) {
	float c=0;
	for (int i = 0;i < ROW;i++) {
		for (int j = 0;j < COL;j++) {
			c += a[i][j] * b[k][i][j];
		}
	}
	return c;
}
int main() {
	//load_data();
	//clock_t t = clock();
	//cudaMemcpyToSymbol(_alpha, &alpha, sizeof(float));
	//cudaMemcpyToSymbol(_minibatch, &minibatch, sizeof(int));
	//cudaMemcpyToSymbol(_epochs, &epochs, sizeof(int));
	//init_data_gpu();
	//init_params();
	//cudaMemcpyToSymbol(_fc1_b, fc1_b, FC1_SIZE* sizeof(float));
	//cudaMemcpyToSymbol(_fc1_w, fc1_w, FC1_SIZE*COL*ROW* sizeof(float));
	//cudaMemcpyToSymbol(_fc2_b, fc2_b, FC2_SIZE* sizeof(float));
	//cudaMemcpyToSymbol(_fc2_w, fc2_w, FC1_SIZE * FC2_SIZE * sizeof(float));
	////dim3 block_set_input(1, 1);
	////dim3 grid_set_input((ROW - 1) / block_set_input.x + 1, (COL - 1) / block_set_input.y + 1);
	//dim3 grid_set_input(28,28);
	//dim3 block_input(8, 8, 8);
	////dim3 grid((CONV_W_NUM - 1) / block.x + 1, (CONV_SIZE - 1) / block.y + 1, (CONV_SIZE - 1) / block.z + 1);
	//dim3 grid_input(1,24,24);

	//for (int i = 1;i <= epochs;i++)
	//{
	//	int value1 = 1;
	//	float value2 = 1;
	//	cudaMemcpyToSymbol(_correct_cnt, &value1, sizeof(int));
	//	cudaMemcpyToSymbol(_avg_error, &value2, sizeof(float));
	//	cudaDeviceSynchronize();
	//	//for (int j = 1;j < TRAIN_NUM; j++)
	//	//{
	//	//	set_input(j, train_image);
	//	//	//input_fc1();
	//	//	if (j && j % 100 == 0)
	//	//	{
	//	//		printf("Training  Time spent : %.0fs Image count : %d Accuracy : %0.4f%% Error : %0.4f%% Epoch : %d \r", floor(((float)(clock() - t)) / CLOCKS_PER_SEC), j, ((float)correct_cnt / j) * 100, (avg_error / j) * 100, i);
	//	//	}
	//	//}

	//	for (int j = 0;j < TRAIN_NUM/BATCH_SIZE; j++)
	//	{
	//		//_set_input_train << <1, dim3(ROW, COL) >> > (j);
	//		set_input(j, train_image);
	//		input_fc1();
	//		_input_fc1 << <dim3(BATCH_SIZE, FC1_SIZE), dim3(ROW, COL) >> > (j);
	//		//float testOut[BATCH_SIZE][FC1_SIZE];
	//		float testOut[FC1_SIZE][ROW][COL];
	//		//cudaMemcpyFromSymbol(&testOut, _fc1_z, BATCH_SIZE*FC1_SIZE * sizeof(float));
	//		cudaMemcpyFromSymbol(&testOut, _data1, FC1_SIZE*ROW*COL * sizeof(float));
	//		for (int k = 0;k < FC1_SIZE;k++) {
	//			for (int m = 0;m < ROW;m++) {
	//				for (int l = 0;l < COL;l++) {
	//					if (data1[k][m][l] != testOut[k][m][l])
	//						cout << "No equals!!!";
	//				}
	//			}
	//		}

	//	/*	_fc1_fc2 << <dim3(BATCH_SIZE, FC2_SIZE), FC1_SIZE>> > ();
	//		_set_answer_train << < BATCH_SIZE, FC2_SIZE >> > (j);
	//		_check_answer_get_error << <1, BATCH_SIZE >> > ();
	//		*/
	//		/*_update_fc2_b << <BATCH_SIZE, FC2_SIZE >> > ();
	//		_update_fc2_w << <10, FC1_SIZE >> > ();
	//		_update_weights_HO << <FC2_SIZE, FC1_SIZE >> > ();
	//		_update_weights_IH << <FC1_SIZE, dim3(ROW,COL)>> > ();
	//		_update_fc1_b << <1, FC1_SIZE >> > ();
	//		_update_fc1_w << <FC1_SIZE, dim3(1, 28, 28) >> > ();
	//		if ((j + 1) % minibatch == 0)
	//			assign_grads_gpu();*/
	//		//_calcDelta << <BATCH_SIZE, FC1_SIZE >> > ();

	//		if (j && j % 100 == 0)
	//		{
	//			cudaMemcpyFromSymbol(&correct_cnt, _correct_cnt, sizeof(int));
	//			cudaMemcpyFromSymbol(&avg_error, _avg_error, sizeof(float));
	//			printf("Training  Time spent : %.0fs Image count : %d Accuracy : %0.4f%% Error : %0.4f%% Epoch : %d \r", floor(((float)(clock() - t)) / CLOCKS_PER_SEC), j*BATCH_SIZE, ((float)correct_cnt / (j*BATCH_SIZE)) * 100, (avg_error / j) * 100, i);
	//		}
	//	}
	//

	//	cudaMemcpyFromSymbol(&correct_cnt, _correct_cnt, sizeof(int));
	//	cudaMemcpyFromSymbol(&avg_error, _avg_error, sizeof(float));
	//	printf("Training  Time spent : %.0fs Image count : %d Accuracy : %0.4f%% Error : %0.4f%% Epoch : %d \n", floor(((float)(clock() - t)) / CLOCKS_PER_SEC), TRAIN_NUM, ((float)correct_cnt / TRAIN_NUM) * 100, (avg_error / TRAIN_NUM) * 100, i);

	//	correct_cnt = 0;
	//	avg_error = 0;
	//	cudaMemcpyToSymbol(_correct_cnt, &correct_cnt, sizeof(int));
	//	cudaMemcpyToSymbol(_avg_error, &avg_error, sizeof(float));

	//	//for (int j = 0;j < TEST_NUM;j++)
	//	//{
	//	//	_set_input_test << <1, dim3(28, 28) >> > (j);
	//	//	_input_fc1 << <400, dim3(28, 28) >> > ();
	//	//	_fc1_fc2 << <10, 400 >> > ();
	//	//	set_answer_gpu_test(j);
	//	//	check_answer_get_error_gpu();

	//	//	if (j && j % 100 == 0)
	//	//	{
	//	//		cudaMemcpyFromSymbol(&correct_cnt, _correct_cnt, sizeof(int));
	//	//		cudaMemcpyFromSymbol(&avg_error, _avg_error, sizeof(float));
	//	//		printf("Testing   Time spent : %.0fs Image count : %d Accuracy : %0.4f%% Error : %0.4f%% \r", floor(((float)(clock() - t)) / CLOCKS_PER_SEC), j, ((float)correct_cnt / j) * 100, (avg_error / j) * 100);
	//	//	}
	//	//}
	//	//cudaMemcpyFromSymbol(&correct_cnt, _correct_cnt, sizeof(int));
	//	//cudaMemcpyFromSymbol(&avg_error, _avg_error, sizeof(float));
	//	//printf("Testing   Time spent : %.0fs Image count : %d Accuracy : %0.4f%% Error : %0.4f%% \n", floor(((float)(clock() - t)) / CLOCKS_PER_SEC), TEST_NUM, ((float)correct_cnt / TEST_NUM) * 100, (avg_error / TEST_NUM) * 100);

	//	//if ((float)correct_cnt / TEST_NUM * 100 > max_acc)
	//	//{
	//	//	max_acc = (float)correct_cnt / TEST_NUM * 100;
	//	//	//export_params();
	//	//	printf("The new model has been exported.Accuracy has reached to %0.5f%%\n\n", max_acc);
	//	//}
	//	//else
	//	//{
	//	//	alpha = alpha - (alpha / 3);
	//	//	cudaMemcpyToSymbol(_alpha, &alpha, sizeof(float));
	//	//	printf("Learning rate has been reduced to %f\n\n", alpha);
	//	//}
	//}
	

	

	cout << matrixMult(train_image[0], fc1_w, 0) << endl;
	cout << matrixMult(train_image[0], fc1_w, 1) << endl;
	cout << matrixMult(train_image[1], fc1_w, 0) << endl;
	cout << matrixMult(train_image[1], fc1_w, 1) << endl;

	


	/*float train_image[ROW][COL] = {
	{ 3, 1, 2, 4, 3, 3 },
	{ 2, 4, 3, 1, 1, 4 },
	{ 1, 5, 2, 3, 2, 5 },
	{ 2, 3, 4, 1, 4, 1 },
	{ 1, 4, 2, 1, 2, 3 },
	{ 2, 3, 6, 5, 4, 1 }, };
	float conv_w[CONV_W_NUM][CONV_W_SIZE][CONV_W_SIZE] = { {
	{1, 2, 3},
	{4, 3, 1},
	{1, 2, 4}},
	{{4, 2, 5},
	{2, 3, 1},
	{1, 2, 3}} };*/

	//float conv_z[CONV_W_NUM][CONV_SIZE][CONV_SIZE];
	float conv_z[2][2][2];
	int train_label[2] = { 2, 0 };

	cudaMemcpyToSymbol(_train_image, train_image, TRAIN_NUM * ROW * COL * sizeof(float));
	cudaMemcpyToSymbol(_train_label, train_label, TRAIN_NUM * sizeof(int));
	cudaMemcpyToSymbol(_fc1_w, fc1_w, FC1_SIZE*ROW*COL*sizeof(float));
	cudaMemcpyToSymbol(_fc2_w, fc2_w, FC2_SIZE*FC1_SIZE*sizeof(float));
	//cudaMemcpyToSymbol(_conv_w, conv_w, CONV_W_NUM * CONV_W_SIZE * CONV_W_SIZE * sizeof(float));
	//cudaMemcpy(_train_label, train_label, 2 * sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemcpy(_train_image, train_image, ROW * COL * sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemcpy(_conv_w, conv_w, CONV_W_NUM*CONV_W_SIZE*CONV_W_SIZE*sizeof(float), cudaMemcpyHostToDevice);
	dim3 grid2(ROW, COL);
	
	//___________________________________________CPU________________________________________________
	cout << "First Image" << endl;
	set_input(0, train_image);
	input_fc1();
	fc1_fc2();
	set_answer(0, train_label);
	check_answer(correct_cnt);
	get_error(avg_error);
	update_fc2_b();
	update_fc2_w();
	update_fc1_b();
	update_fc1_w(0);
	/*for (int i = 0;i < FC2_SIZE;i++) {
		cout << fc2_delta[i] << " ";
	}
	cout << endl;
	for (int i = 0;i < FC1_SIZE;i++) {
		cout << fc1_delta[i] << " ";
	}*/
	cout << endl;
	cout << "Second Image" << endl;
	set_input(1, train_image);
	input_fc1();
	fc1_fc2();
	set_answer(1, train_label);
	check_answer(correct_cnt);
	get_error(avg_error);
	update_fc2_b();
	update_fc2_w();
	update_fc1_b();
	update_fc1_w(1);
	cout << "fc2_db" << endl;
	for (int i = 0;i < FC2_SIZE;i++) {
		cout << fc2_db[i] << " ";
	}
	cout << endl;
	cout << "fc2_dw" << endl;
	for (int i = 0;i < FC2_SIZE;i++) {
		for (int j = 0;j < FC1_SIZE;j++) {
			cout << fc2_dw[i][j] << " ";
		}
		cout << endl;
	}
	cout << endl;
	cout << "fc1_db" << endl;
	for (int i = 0;i < FC1_SIZE;i++) {
		cout << fc1_db[i] << " ";
	}
	cout << endl;
	cout << "fc1_dw" << endl;
	for (int r = 0;r < FC1_SIZE;r++) {
		for (int i = 0;i < ROW;i++) {
			for (int j = 0;j < COL;j++) {
				cout << fc1_dw[r][i][j] << " ";
			}
			cout << endl;
		}
		cout << endl;
	}
	cout << "___________________________________________GPU________________________________________________" << endl;
	//_input_conv << <1, grid2>> > ((float (*)[4])_train_image, (float (*)[3][3])_conv_w, (float (*)[2][2])_conv_z);
	_input_fc1 << <dim3(2, FC1_SIZE), grid2 >> > (0);
	_fc1_fc2 << <dim3(BATCH_SIZE, FC2_SIZE), FC1_SIZE>> > ();
	_set_answer_train << < BATCH_SIZE, FC2_SIZE >> > (0);
	_check_answer_get_error <<<1, BATCH_SIZE >>> ();
	_calcDelta << <BATCH_SIZE, FC2_SIZE>> > ();
	//float testOut[BATCH_SIZE][FC2_SIZE];
	//_conv_pool << <1, grid2 >> > ();
	//cudaMemcpyFromSymbol(&conv_z, _pool, CONV_W_NUM * CONV_SIZE * CONV_SIZE * sizeof(float));
	//cudaMemcpyFromSymbol(&conv_z, _pool, 8 * sizeof(float));
	//cudaMemcpyFromSymbol(&testOut, _fc2_del, BATCH_SIZE*FC2_SIZE * sizeof(float));
	float testOut[FC2_SIZE];
	cudaMemcpyFromSymbol(&testOut, _fc2_db, FC2_SIZE* sizeof(float));
	cout << "First Image" << endl;
	cout << "fc2_db" << endl;
	for (int i = 0;i < FC2_SIZE;i++) {
			cout << testOut[i] << " ";
	}
	cout << endl;
	float testOut1[FC2_SIZE][FC1_SIZE];
	cudaMemcpyFromSymbol(&testOut1, _fc2_dw, FC2_SIZE*FC1_SIZE * sizeof(float));
	cout << "fc2_dw" << endl;
	for (int i = 0;i < FC2_SIZE;i++) {
		for (int j = 0;j < FC1_SIZE;j++) {
			cout << testOut1[i][j] << " ";
		}
		cout << endl;
	}
	float testOut2[FC1_SIZE];
	cudaMemcpyFromSymbol(&testOut2, _fc1_db, FC1_SIZE * sizeof(float));
	cout << "fc1_db" << endl;
	for (int i = 0;i < FC1_SIZE;i++) {
		cout << testOut2[i] << " ";
	}
	cout << endl;
	float testOut3[FC1_SIZE][ROW][COL];
	cudaMemcpyFromSymbol(&testOut3, _fc1_dw, FC1_SIZE*ROW*COL* sizeof(float));
	cout << "fc1_dw" << endl;
	for (int r = 0;r < FC1_SIZE;r++) {
		for (int i = 0;i < ROW;i++) {
			for (int j = 0;j < COL;j++) {
				cout << testOut3[r][i][j] << " ";
			}
			cout << endl;
		}
		cout << endl;
	}
	/*cout << "Second Image" << endl;
	for (int i = 0;i < FC1_SIZE;i++) {
		cout << testOut[1][i] << endl;
	}*/
	/*for (int i = 0;i < 2;i++) {
		for (int j = 0;j <2;j++) {
			cout << conv_z[0][i][j] << " ";
		}
		cout << endl;
	}
	for (int i = 0;i < 2;i++) {
		for (int j = 0;j < 2;j++) {
			cout << conv_z[1][i][j] << " ";
		}
		cout << endl;
	}*/
	return 0;
}