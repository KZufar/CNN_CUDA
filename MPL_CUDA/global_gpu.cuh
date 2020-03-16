//#ifndef _GLOBAL_H_
//#define _GLOBAL_H_
//
//#include "cuda_runtime.h"
//#define TROWAIN_NUM 60000
//#define TEST_NUM 10000
//#define ROWOW 28
//#define COL 28
//#define FC1_SIZE 500
//#define FC2_SIZE 10
//#define BATCH_SIZE 10
//
//extern float fc1_b[FC1_SIZE];
//extern float fc1_w[FC1_SIZE][ROWOW][COL];
//extern float fc2_b[FC2_SIZE];
//extern float fc2_w[FC2_SIZE][FC1_SIZE];
//
//extern __constant__ float _alpha;
//extern __constant__ int _minibatch;
//extern __constant__ int _epochs;
//
//extern __device__ int _correct_cnt;
//extern __device__ float _avg_error;
//
//int correct_cnt = 3;
//float avg_error = 2;
//float max_acc;
//
//float alpha = 0.2;
//int epochs = 5;
//int minibatch = 1;
//
//extern float train_image[TROWAIN_NUM][ROWOW][COL];
//extern int train_label[TROWAIN_NUM];
//extern float test_image[TEST_NUM][ROWOW][COL];
//extern int test_label[TEST_NUM];
//
//extern float input[ROWOW][COL];
//extern float fc1_z[FC1_SIZE];
//extern float fc1_a[FC1_SIZE];
//extern float fc2_z[FC2_SIZE];
//extern float fc2_a[FC2_SIZE];
//extern float output[FC2_SIZE];
//extern int answer[FC2_SIZE];
//
//extern __device__ float _train_image[TROWAIN_NUM][ROWOW][COL];
//extern __device__ int _train_label[TROWAIN_NUM];
//extern __device__ float _test_image[TEST_NUM][ROWOW][COL];
//extern __device__ int _test_label[TEST_NUM];
//
//extern __device__ float _fc1_b[FC1_SIZE];
//extern __device__ float _fc1_w[FC1_SIZE][ROWOW][COL];
//extern __device__ float _fc2_b[FC2_SIZE];
//extern __device__ float _fc2_w[FC2_SIZE][FC1_SIZE];
//
////__device__ float _input[ROWOW][COL];
//extern __device__ float _fc1_z[BATCH_SIZE][FC1_SIZE];
//extern __device__ float _fc1_a[BATCH_SIZE][FC1_SIZE];
//extern __device__ float _fc2_z[BATCH_SIZE][FC2_SIZE];
//extern __device__ float _fc2_a[BATCH_SIZE][FC2_SIZE];
//extern __device__ float _output[BATCH_SIZE][FC2_SIZE];
//extern __device__ int _answer[BATCH_SIZE][FC2_SIZE];
//
//extern __device__ float _fc1_db[BATCH_SIZE][FC1_SIZE];
//extern __device__ float _fc1_dw[BATCH_SIZE][FC1_SIZE][ROWOW][COL];
//extern __device__ float _fc2_db[BATCH_SIZE][FC2_SIZE];
//extern __device__ float _fc2_dw[BATCH_SIZE][FC2_SIZE][FC1_SIZE];
//extern __device__ float _C[BATCH_SIZE][FC2_SIZE];
//extern __device__ float _fc2_delta[BATCH_SIZE][FC2_SIZE];
//extern __device__ float _fc1_delta[BATCH_SIZE][FC1_SIZE];
//
//extern __device__ int tmp;
//
//#endif