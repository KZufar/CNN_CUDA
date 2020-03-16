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
//__device__ int _correct_cnt;
//
//__device__ float _res;
//__device__ float _arr[10][10];
//float arr[10][10];
//
//__global__ void test() {
//	int ib = blockIdx.x;
//	int ix = threadIdx.x;
//	int iy = threadIdx.y;
//	//for (int l = 0;l < 10;l++)
//	//	for (int m = 0;m < 10;m++)
//	//		_res += _arr[l][m];
//		__shared__ float data[1024];
//		int tid = threadIdx.y*blockDim.x+threadIdx.x;
//		data[tid] = 0;
//		if(ix<10&&iy<10)
//			data[tid] = _arr[ix][iy];
//		__syncthreads();
//		for (int s = 1024 / 2; s > 0; s >>= 1) {
//			if (tid < s)
//				data[tid] += data[tid + s];
//			__syncthreads();
//		}
//		if (tid == 0) {
//			_res= data[0];
//		}
//}
//int main() {
//	for (int i = 0;i < 10; i++) {
//		for (int j = 0;j < 10;j++) {
//			arr[i][j] = rand() % 5;
//		}
//	}
//	cudaMemcpyToSymbol(_arr, &arr, 10 * 10 * sizeof(float));
//	float sum = 0;
//	for (int i = 0;i < 10; i++) {
//		for (int j = 0;j < 10;j++) {
//			sum += arr[i][j];
//		}
//	}
//	cout << "CPU sum: " <<sum << endl;
//	test << <1, dim3(32,32)>> > ();
//	float res=0;
//	cudaMemcpyFromSymbol(&res, _res, sizeof(float));
//	cout << "GPU sum: " << res << endl;
//	return 0;
//}