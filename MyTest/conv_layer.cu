//#pragma comment (lib, "cublas.lib")
//#include "stdio.h"
//#include <cuda.h>
//using namespace std;
//#include <ctime>
//#include "cuda_runtime.h"
//#include "curand_kernel.h"
//#include "device_launch_parameters.h"
//#include <stdio.h>
//#include <stdlib.h>
//
//#include <string>
//#include <iomanip>
//#include <time.h>
//#include <iostream>
//#include <cmath>
//#include <math.h>
//
//#define TRAIN_NUM 30000
//#define TEST_NUM 10000
//#define ROW 28
//#define COL 28
//#define CONV_SIZE 24
//#define POOL_SIZE 12
//#define FC1_SIZE 5
//#define FC2_SIZE 10
//#define CONV_W_SIZE 5
//#define CONV_W_NUM 6
//
//float conv_w[CONV_W_NUM][CONV_W_SIZE][CONV_W_SIZE];
//float conv_b[CONV_W_NUM];
//float input[ROW][COL];
//float conv_z[CONV_W_NUM][CONV_SIZE][CONV_SIZE];
//float conv_a[CONV_W_NUM][CONV_SIZE][CONV_SIZE];
//
//__device__ float _conv_w[CONV_W_NUM][CONV_W_SIZE][CONV_W_SIZE];
//__device__ float _conv_b[CONV_W_NUM];
//__device__ float _input[ROW][COL];
//__device__ float _conv_z[CONV_W_NUM][CONV_SIZE][CONV_SIZE];
//__device__ float _conv_a[CONV_W_NUM][CONV_SIZE][CONV_SIZE];
//
//float get_rand(float fan_in)
//{
//	float sum = 0;
//	for (int i = 0;i < 12;i++)
//		sum += (float)rand() / RAND_MAX;
//	sum -= 6;
//	sum *= 1 / sqrt(fan_in);
//	return sum;
//}
//void init_params()
//{
//	for (int i = 0;i < CONV_W_NUM;i++)
//	{
//		for (int j = 0;j < CONV_W_SIZE;j++)
//			for (int k = 0;k < CONV_W_SIZE;k++)
//				conv_w[i][j][k] = get_rand(CONV_W_SIZE * CONV_W_SIZE);
//		conv_b[i] = get_rand(CONV_W_SIZE * CONV_W_SIZE);
//	}
//
//	/*for (int i = 0;i < FC1_SIZE;i++)
//	{
//		for (int j = 0;j < CONV_W_NUM;j++)
//			for (int k = 0;k < POOL_SIZE;k++)
//				for (int l = 0;l < POOL_SIZE;l++)
//					fc1_w[i][j][k][l] = get_rand(POOL_SIZE * POOL_SIZE * CONV_W_NUM);
//		fc1_b[i] = get_rand(POOL_SIZE * POOL_SIZE * CONV_W_NUM);
//	}
//
//	for (int i = 0;i < FC2_SIZE;i++)
//	{
//		for (int j = 0;j < FC1_SIZE;j++)
//			fc2_w[i][j] = get_rand(FC1_SIZE);
//		fc2_b[i] = get_rand(FC1_SIZE);
//	}*/
//}
//
//float sigmoid(float x)
//{
//	return (1 / (1 + exp(-1 * x)));
//}
//
//void input_conv()
//{
//	for (int i = 0;i < CONV_W_NUM;i++)
//		for (int j = 0;j < CONV_SIZE;j++)
//			for (int k = 0;k < CONV_SIZE;k++)
//			{
//				conv_z[i][j][k] = 0;
//				for (int l = 0;l < CONV_W_SIZE;l++)
//					for (int m = 0;m < CONV_W_SIZE;m++)
//						conv_z[i][j][k] += input[j + l][k + m] * conv_w[i][l][m];
//				conv_z[i][j][k] += conv_b[i];
//				conv_a[i][j][k] = sigmoid(conv_z[i][j][k]);
//			}
//}
//
//__device__ float _sigmoid(float x)
//{
//	return (1 / (1 + exp(-1 * x)));
//}
//
//__global__ void _input_conv()
//{
//	int ix = threadIdx.x + blockDim.x * blockIdx.x;
//	int iy = threadIdx.y + blockDim.y * blockIdx.y;
//	int iz = threadIdx.z + blockDim.z * blockIdx.z;
//	if (ix < CONV_W_NUM && iy < CONV_SIZE && iz < CONV_SIZE)
//	{
//		_conv_z[ix][iy][iz] = 0;
//		// #pragma unroll
//		for (int l = 0;l < CONV_W_SIZE;l++)
//			for (int m = 0;m < CONV_W_SIZE;m++)
//				_conv_z[ix][iy][iz] += _input[iy + l][iz + m] * _conv_w[ix][l][m];
//		_conv_z[ix][iy][iz] += _conv_b[ix];
//		_conv_a[ix][iy][iz] = _sigmoid(_conv_z[ix][iy][iz]);
//	}
//}
//
//__global__ void _input_conv_reduce()
//{
//	int ix = threadIdx.x + blockDim.x * blockIdx.x;
//	int iy = threadIdx.y + blockDim.y * blockIdx.y;
//	int iz = threadIdx.z + blockDim.z * blockIdx.z;
//
//	__shared__ int data[25];
//	int tid = threadIdx.x;
//	//int i = 2*blockIdx.x * blockDim.x + threadIdx.x;
//	data[tid] = _input[blockIdx.x+iy][blockIdx.y+iz] * _conv_w[blockIdx.x][iy][iz];
//	__syncthreads();
//	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
//		if (tid < s)
//			data[tid] += data[tid + s];
//		__syncthreads();
//	}
//	if (tid == 0) {
//		_conv_z[ix][iy][iz] = data[0];
//		data[0] += _conv_b[ix];
//		_conv_a[ix][iy][iz] = _sigmoid(data[0]);
//	}
//}
//int main() {
//	clock_t t = clock();
//
//
//	cout << "-----------------------CPU------------------" << endl;
//	for (int j = 0;j < TRAIN_NUM;j++)
//	{
//		//input_conv();
//		if (j && j % 100 == 0)
//			printf("Training  Time spent : %.0fs Image count : %d \r", floor(((float)(clock() - t)) / CLOCKS_PER_SEC), j);
//	}
//	printf("Training  Time spent : %.0fs Image count : %d \n", floor(((float)(clock() - t)) / CLOCKS_PER_SEC), TRAIN_NUM);
//
//	cout << "-----------------------GPU------------------" << endl;
//	t = clock();
//	//dim3 grid_input(1, 24, 24);
//	dim3 block_input(6, 24, 24);
//	dim3 grid_input(1, 5, 5);
//	for (int j = 0;j < TRAIN_NUM;j++)
//	{
//		_input_conv_reduce << <block_input, grid_input>> > ();
//		if (j && j % 100 == 0)
//			printf("Training  Time spent : %.0fs Image count : %d \r", floor(((float)(clock() - t)) / CLOCKS_PER_SEC), j);
//	}
//	printf("Training  Time spent : %.0fs Image count : %d \n", floor(((float)(clock() - t)) / CLOCKS_PER_SEC), TRAIN_NUM);
//
//	
//
//	return 0;
//}