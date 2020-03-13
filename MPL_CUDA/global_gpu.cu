#include "global_gpu.cuh"

float fc1_b[FC1_SIZE];
float fc1_w[FC1_SIZE][ROW][COL];
float fc2_b[FC2_SIZE];
float fc2_w[FC2_SIZE][FC1_SIZE];

__constant__ float _alpha;
__constant__ int _minibatch;
__constant__ int _epochs;

__device__ int _correct_cnt;
__device__ float _avg_error;

//int correct_cnt = 3;
//float avg_error = 2;
//float max_acc;
//
//float alpha = 0.2;
//int epochs = 5;
//int minibatch = 1;

float train_image[TRAIN_NUM][ROW][COL];
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

__device__ float _train_image[TRAIN_NUM][ROW][COL];
__device__ int _train_label[TRAIN_NUM];
__device__ float _test_image[TEST_NUM][ROW][COL];
__device__ int _test_label[TEST_NUM];

__device__ float _fc1_b[FC1_SIZE];
__device__ float _fc1_w[FC1_SIZE][ROW][COL];
__device__ float _fc2_b[FC2_SIZE];
__device__ float _fc2_w[FC2_SIZE][FC1_SIZE];

//__device__ float _input[ROW][COL];
__device__ float _fc1_z[BATCH_SIZE][FC1_SIZE];
__device__ float _fc1_a[BATCH_SIZE][FC1_SIZE];
__device__ float _fc2_z[BATCH_SIZE][FC2_SIZE];
__device__ float _fc2_a[BATCH_SIZE][FC2_SIZE];
__device__ float _output[BATCH_SIZE][FC2_SIZE];
__device__ int _answer[BATCH_SIZE][FC2_SIZE];

__device__ float _fc1_db[BATCH_SIZE][FC1_SIZE];
__device__ float _fc1_dw[BATCH_SIZE][FC1_SIZE][ROW][COL];
__device__ float _fc2_db[BATCH_SIZE][FC2_SIZE];
__device__ float _fc2_dw[BATCH_SIZE][FC2_SIZE][FC1_SIZE];
__device__ float _C[BATCH_SIZE][FC2_SIZE];
__device__ float _fc2_delta[BATCH_SIZE][FC2_SIZE];
__device__ float _fc1_delta[BATCH_SIZE][FC1_SIZE];

__device__ int tmp;
