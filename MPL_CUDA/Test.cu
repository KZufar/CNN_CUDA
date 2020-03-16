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
//__device__ float _avg_error;
//
//
//__global__ void test() {
//	_correct_cnt += _correct_cnt + 1;
//	_correct_cnt += _correct_cnt - 1;
//}
//int main() {
//	test << <dim3(100000,1000,1000), 10 >> > ();
//	int value = 0;
//	cudaMemcpyFromSymbol(&value, _correct_cnt, sizeof(int));
//	cout << value << endl;
//	return 0;
//}