////подключение библиотек
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
//using namespace std;
//
//#define N 10000
//#define M 32
//#define BASE_TYPE int
//
//__global__ void scalMult(const BASE_TYPE* A, const BASE_TYPE* B, BASE_TYPE* C) {
//	BASE_TYPE sum = 0;
//	//__shared__ BASE_TYPE ash[M];
//	//__shared__ BASE_TYPE bsh[M];
//
//	//ash[threadIdx.x] = A[blockIdx.x * blockDim.x + threadIdx.x];
//	//bsh[threadIdx.x] = B[blockIdx.x * blockDim.x + threadIdx.x];
//	//__syncthreads();
//	//if (threadIdx.x == 0) {
//	//	sum = 0;
//	//	for (int j = 0;j < blockDim.x;j++) {
//	//		sum += ash[j] * bsh[j];
//	//	}
//	//	atomicAdd(C, sum);
//	//	//C[blockIdx.x] = sum;
//	//}
//
//	int idx = blockIdx.x * blockDim.x + threadIdx.x;
//	sum = A[idx] * B[idx];
//	atomicAdd(C, sum);
//
//
//
//}
//
////KernelTme: 0.11 millseconds
////	Result : 203394
//
////KernelTme: 0.34 millseconds
////	Result : 203394
//
//int main() {
//
//	cudaEvent_t start, stop;
//	cudaEventCreate(&start);
//	cudaEventCreate(&stop);
//
//	int host_a[N], host_b[N];
//	int* host_c = (int*)malloc(sizeof(int));
//	int* dev_a, * dev_b, * dev_c, * dev_res;
//	cout << "a" << "  " << "b" << endl;
//	for (int i = 0; i < N; i++)
//	{
//		host_a[i] = rand() % 10;
//		host_b[i] = rand() % 10;
//		//cout << host_a[i] << " " << host_b[i] << endl;
//	}
//	cudaMalloc((void**)&dev_a, N * sizeof(int));
//	cudaMalloc((void**)&dev_b, N * sizeof(int));
//	cudaMalloc((void**)&dev_c, sizeof(int));
//	cudaMemcpy(dev_a, host_a, N * sizeof(int), cudaMemcpyHostToDevice);
//	cudaMemcpy(dev_b, host_b, N * sizeof(int), cudaMemcpyHostToDevice);
//	cudaMemset(dev_c, 0, sizeof(int));
//	//dim3 threadsPerBlock = dim3(BS, BS);
//	dim3 blocksPerGrid = dim3(N / M);
//	cudaEventRecord(start, 0);
//	scalMult << <blocksPerGrid, M>> > (dev_a, dev_b, dev_c);
//	
//	cudaEventRecord(stop, 0);
//	cudaEventSynchronize(stop);
//	float KernelTime;
//	cudaEventElapsedTime(&KernelTime, start, stop);
//	printf("KernelTme: %.2f millseconds\n", KernelTime);
//	cudaMemcpy(host_c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
//	printf("Result: %d", host_c[0]);
//	cudaFree(dev_a);
//	cudaFree(dev_b);
//	cudaFree(dev_c);
//	cudaEventDestroy(start);
//	cudaEventDestroy(stop);
//	return 0;
//}
