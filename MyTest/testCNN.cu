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
//#define TRAIN_NUM 60000
//#define TEST_NUM 10000
//#define ROW 4
//#define COL 4
//#define CONV_SIZE 2
//#define POOL_SIZE 1
//#define FC1_SIZE 45
//#define FC2_SIZE 10
//#define CONV_W_SIZE 3
//#define CONV_W_NUM 1
//
//__device__ float _alpha = 20;
//__device__ int _train_label[TRAIN_NUM];
//__device__ float _train_image[ROW][COL];
//__device__ float _conv_w[CONV_W_NUM][CONV_W_SIZE][CONV_W_SIZE];
//__device__ float _conv_z[CONV_W_NUM][CONV_SIZE][CONV_SIZE];
//
////__global__ void _input_conv(float _train_image[4][4], float _conv_w[1][3][3], float _conv_z[1][2][2])
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
//				_conv_z[ix][iy][iz] += _train_image[iy + l][iz + m] * _conv_w[ix][l][m];
//		_conv_z[ix][iy][iz] += _alpha;
//		//_conv_a[ix][iy][iz] = _sigmoid(_conv_z[ix][iy][iz]);
//	}
//}
//int main() {
//	float train_image[ROW][COL] = { { 3, 1, 2, 4 },
//	{ 2, 4, 3, 1 },
//	{ 1, 5, 2, 3 },
//	{ 2, 3, 4, 1 } };
//	float conv_w[CONV_W_NUM][CONV_W_SIZE][CONV_W_SIZE] = { {
//	{1, 2, 3},
//	{4, 3, 1},
//	{1, 2, 4}}};
//
//	float** h_weight = new float* [3];
//			for (int i = 0;i < 3;i++) {
//				h_weight[i] = new float[3];
//			}
//	float conv_z[CONV_W_NUM][CONV_SIZE][CONV_SIZE];
//
//	float train_label[2] = { 3,2 };
//	//float* _train_image;
//	//float* _train_label;
//	//float* _conv_w;
//	//float* _conv_z;
//
//	//cudaMalloc(&_conv_w, CONV_W_NUM * CONV_W_SIZE * CONV_W_SIZE * sizeof(float));
//	//cudaMalloc(&_conv_z, CONV_W_NUM * CONV_SIZE * CONV_SIZE * sizeof(float));
//	//cudaMalloc(&_train_image, ROW*COL*sizeof(float));
//	//cudaMalloc(&_train_label, 2*sizeof(float));
//
//	float alpha = 10.0;
//	cudaMemcpyToSymbol(_alpha, &alpha, sizeof(float));
//	cudaMemcpyToSymbol(_train_image, train_image, ROW * COL * sizeof(float));
//	cudaMemcpyToSymbol(_conv_w, conv_w, CONV_W_NUM * CONV_W_SIZE * CONV_W_SIZE * sizeof(float));
//	//cudaMemcpy(_train_label, train_label, 2 * sizeof(float), cudaMemcpyHostToDevice);
//	//cudaMemcpy(_train_image, train_image, ROW * COL * sizeof(float), cudaMemcpyHostToDevice);
//	//cudaMemcpy(_conv_w, conv_w, CONV_W_NUM*CONV_W_SIZE*CONV_W_SIZE*sizeof(float), cudaMemcpyHostToDevice);
//	dim3 grid2(1, 2, 2);
//
//	//_input_conv << <1, grid2>> > ((float (*)[4])_train_image, (float (*)[3][3])_conv_w, (float (*)[2][2])_conv_z);
//	_input_conv << <1, grid2>> > ();
//	cudaMemcpyFromSymbol(&conv_z, _conv_z, CONV_W_NUM * CONV_SIZE * CONV_SIZE*sizeof(float));
//	for (int i = 0;i < CONV_SIZE;i++) {
//		for (int j = 0;j < CONV_SIZE;j++) {
//			cout << conv_z[0][i][j]<<" ";
//		}
//		cout << endl;
//	}
//	return 0;
//}