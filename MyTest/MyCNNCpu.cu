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
//#define ROW 28
//#define COL 28
//#define CONV_SIZE 24
//#define POOL_SIZE 12
//#define FC1_SIZE 5
//#define FC2_SIZE 10
//#define CONV_W_SIZE 5
//#define CONV_W_NUM 6
//
//int correct_cnt;
//float avg_error;
//float max_acc;
//
//float alpha = 0.2;
//int epochs = 5;
//int minibatch = 1;
//
//float train_image[TRAIN_NUM][ROW][COL];
//int train_label[TRAIN_NUM];
//float test_image[TEST_NUM][ROW][COL];
//int test_label[TEST_NUM];
//
//float conv_w[CONV_W_NUM][CONV_W_SIZE][CONV_W_SIZE];
//float conv_b[CONV_W_NUM];
//float fc1_b[FC1_SIZE];
//float fc1_w[FC1_SIZE][CONV_W_NUM][POOL_SIZE][POOL_SIZE];
//float fc2_b[FC2_SIZE];
//float fc2_w[FC2_SIZE][FC1_SIZE];
//
//float input[ROW][COL];
//float conv_z[CONV_W_NUM][CONV_SIZE][CONV_SIZE];
//float conv_a[CONV_W_NUM][CONV_SIZE][CONV_SIZE];
//int pool_pos[CONV_W_NUM][POOL_SIZE][POOL_SIZE];
//float pool[CONV_W_NUM][POOL_SIZE][POOL_SIZE];
//float fc1_z[FC1_SIZE];
//float fc1_a[FC1_SIZE];
//float fc2_z[FC2_SIZE];
//float fc2_a[FC2_SIZE];
//float output[FC2_SIZE];
//int answer[FC2_SIZE];
//
//float conv_dw[CONV_W_NUM][CONV_W_SIZE][CONV_W_SIZE];
//float conv_db[CONV_W_NUM];
//float fc1_db[FC1_SIZE];
//float fc1_dw[FC1_SIZE][CONV_W_NUM][POOL_SIZE][POOL_SIZE];
//float fc2_db[FC2_SIZE];
//float fc2_dw[FC2_SIZE][FC1_SIZE];
//float C[FC2_SIZE];
//float fc2_delta[FC2_SIZE];
//float fc1_delta[FC1_SIZE];
//float conv_sigma_delta[CONV_W_NUM];
//float conv_delta[CONV_W_NUM][POOL_SIZE][POOL_SIZE];
//
//int swap_endian(int val)
//{
//	unsigned char c1, c2, c3, c4;
//	c1 = val & 255;
//	c2 = (val >> 8) & 255;
//	c3 = (val >> 16) & 255;
//	c4 = (val >> 24) & 255;
//	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
//}
//void load_data()
//{
//	FILE* f_images = fopen("D:\\\\Zufar\\\\CUDA-CNN\\\\CudaCNN2\\\\CudaCNN2\\\\data\\\\train-images.idx3-ubyte", "rb");
//	FILE* f_labels = fopen("D:\\\\Zufar\\\\CUDA-CNN\\\\CudaCNN2\\\\CudaCNN2\\\\data\\\\train-labels.idx1-ubyte", "rb");
//
//	int tmp;
//
//	int magic_num;
//	fread(&magic_num, sizeof(int), 1, f_images);
//	fread(&magic_num, sizeof(int), 1, f_labels);
//
//	// printf("debug:%d\n",swap_endian(magic_num));
//
//	int train_size;
//	fread(&train_size, sizeof(int), 1, f_images);
//	fread(&train_size, sizeof(int), 1, f_labels);
//	train_size = swap_endian(train_size);
//
//	// printf("debug:%d\n",swap_endian(train_size));
//
//	int rows, cols;
//	fread(&rows, sizeof(int), 1, f_images);
//	fread(&cols, sizeof(int), 1, f_images);
//	rows = swap_endian(rows);
//	cols = swap_endian(cols);
//
//	// printf("debug:%d\n",swap_endian(rows));
//	// printf("debug:%d\n",swap_endian(cols));
//
//	for (int i = 0;i < train_size;i++)
//	{
//		fread(&train_label[i], 1, 1, f_labels);
//		if (i % 1000 == 0)
//			printf("Training labels : Already read %5d labels\r", i);
//		// printf("%d:debug:%d\r",i,train_label[i]);
//		// system("pause");
//	}
//	printf("Training labels : Already read %5d labels\n", train_size);
//
//	for (int i = 0;i < train_size;i++)
//	{
//		for (int j = 0;j < rows;j++)
//			for (int k = 0;k < cols;k++)
//			{
//				tmp = 0;
//				fread(&tmp, 1, 1, f_images);
//				train_image[i][j][k] = tmp;
//				train_image[i][j][k] /= 255;
//				// printf("%d %d %d debug: %f\n",i,j,k,train_image[i][j][k]);
//				// system("pause");
//			}
//		if (i % 1000 == 0)
//			printf("Training images : Already read %5d images\r", i);
//	}
//	printf("Training images : Already read %5d images\n", train_size);
//
//	fclose(f_images);
//	fclose(f_labels);
//
//	f_images = fopen("D:\\\\Zufar\\\\CUDA-CNN\\\\CudaCNN2\\\\CudaCNN2\\\\data\\\\t10k-images.idx3-ubyte", "rb");
//	f_labels = fopen("D:\\\\Zufar\\\\CUDA-CNN\\\\CudaCNN2\\\\CudaCNN2\\\\data\\\\t10k-labels.idx1-ubyte", "rb");
//
//	fread(&magic_num, sizeof(int), 1, f_images);
//	fread(&magic_num, sizeof(int), 1, f_labels);
//
//	int test_size;
//	fread(&test_size, sizeof(int), 1, f_images);
//	fread(&test_size, sizeof(int), 1, f_labels);
//	test_size = swap_endian(test_size);
//
//	fread(&rows, sizeof(int), 1, f_images);
//	fread(&cols, sizeof(int), 1, f_images);
//	rows = swap_endian(rows);
//	cols = swap_endian(cols);
//
//	for (int i = 0;i < test_size;i++)
//	{
//		fread(&test_label[i], 1, 1, f_labels);
//		if (i % 1000 == 0)
//			printf("Testing labels : Already read %5d labels\r", i);
//	}
//	printf("Testing labels : Already read %5d labels\n", test_size);
//
//	for (int i = 0;i < test_size;i++)
//	{
//		for (int j = 0;j < rows;j++)
//			for (int k = 0;k < cols;k++)
//			{
//				tmp = 0;
//				fread(&tmp, 1, 1, f_images);
//				test_image[i][j][k] = tmp;
//				test_image[i][j][k] /= 255;
//			}
//		if (i % 1000 == 0)
//			printf("Testing images : Already read %5d images\r", i);
//	}
//	printf("Testing images : Already read %5d images\n\n", test_size);
//
//	fclose(f_images);
//	fclose(f_labels);
//}
//
//float sigmoid(float x)
//{
//	return (1 / (1 + exp(-1 * x)));
//}
//
//void set_input(int idx, float image[TRAIN_NUM][ROW][COL])
//{
//	for (int i = 0;i < ROW;i++)
//		for (int j = 0;j < COL;j++)
//			input[i][j] = image[idx][i][j];
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
//void conv_pool()
//{
//	for (int i = 0;i < CONV_W_NUM;i++)
//		for (int j = 0;j < POOL_SIZE;j++)
//			for (int k = 0;k < POOL_SIZE;k++)
//			{
//				float _max = conv_a[i][j * 2][k * 2];
//				pool_pos[i][j][k] = 0;
//				if (conv_a[i][j * 2][k * 2 + 1] > _max)
//				{
//					_max = conv_a[i][j * 2][k * 2 + 1];
//					pool_pos[i][j][k] = 1;
//				}
//				if (conv_a[i][j * 2 + 1][k * 2] > _max)
//				{
//					_max = conv_a[i][j * 2 + 1][k * 2];
//					pool_pos[i][j][k] = 2;
//				}
//				if (conv_a[i][j * 2 + 1][k * 2 + 1] > _max)
//				{
//					_max = conv_a[i][j * 2 + 1][k * 2 + 1];
//					pool_pos[i][j][k] = 3;
//				}
//				pool[i][j][k] = _max;
//			}
//}
//
//void pool_fc1()
//{
//	for (int i = 0;i < FC1_SIZE;i++)
//	{
//		fc1_z[i] = 0;
//		for (int j = 0;j < CONV_W_NUM;j++)
//			for (int k = 0;k < POOL_SIZE;k++)
//				for (int l = 0;l < POOL_SIZE;l++)
//					fc1_z[i] += pool[j][k][l] * fc1_w[i][j][k][l];
//		fc1_z[i] += fc1_b[i];
//		fc1_a[i] = sigmoid(fc1_z[i]);
//	}
//}
//
//void fc1_fc2()
//{
//	for (int i = 0;i < FC2_SIZE;i++)
//	{
//		fc2_z[i] = 0;
//		for (int j = 0;j < FC1_SIZE;j++)
//			fc2_z[i] += fc1_a[j] * fc2_w[i][j];
//		fc2_z[i] += fc2_b[i];
//		fc2_a[i] = sigmoid(fc2_z[i]);
//	}
//}
//
//void set_answer(int idx, int label[TRAIN_NUM])
//{
//	for (int i = 0;i < FC2_SIZE;i++)
//	{
//		output[i] = fc2_a[i];
//		answer[i] = (label[idx] == i) ? 1 : 0;
//	}
//}
//
//void check_answer(int& correct_cnt)
//{
//	float _max = output[0];
//	int max_pos = 0;
//	for (int i = 0;i < FC2_SIZE;i++)
//	{
//		if (_max < output[i])
//		{
//			_max = output[i];
//			max_pos = i;
//		}
//	}
//	if (answer[max_pos])
//		correct_cnt++;
//}
//
//void get_error(float& avg_error)
//{
//	for (int i = 0;i < FC2_SIZE;i++)
//	{
//		C[i] = output[i] - answer[i];
//		avg_error += C[i] * C[i] * 0.5;
//	}
//}
//
//
//void update_fc2_b()
//{
//	for (int i = 0;i < FC2_SIZE;i++)
//	{
//		fc2_delta[i] = alpha * C[i] * (fc2_a[i] * (1.0 - fc2_a[i]));
//		fc2_db[i] += fc2_delta[i];
//	}
//}
//
//void update_fc2_w()
//{
//	for (int i = 0;i < FC2_SIZE;i++)
//		for (int j = 0;j < FC1_SIZE;j++)
//			fc2_dw[i][j] += fc2_delta[i] * fc1_a[j];
//}
//
//void update_fc1_b()
//{
//	for (int i = 0;i < FC1_SIZE;i++)
//	{
//		float error = 0;
//		for (int j = 0;j < FC2_SIZE;j++)
//			error += fc2_delta[j] * fc2_w[j][i];
//		fc1_delta[i] = error * (fc1_a[i] * (1.0 - fc1_a[i]));
//		fc1_db[i] += fc1_delta[i];
//	}
//}
//
//void update_fc1_w()
//{
//	for (int i = 0;i < FC1_SIZE;i++)
//		for (int j = 0;j < CONV_W_NUM;j++)
//			for (int k = 0;k < POOL_SIZE;k++)
//				for (int l = 0;l < POOL_SIZE;l++)
//					fc1_dw[i][j][k][l] += fc1_delta[i] * pool[j][k][l];
//}
//
//void update_conv_b()
//{
//	for (int i = 0;i < CONV_W_NUM;i++)
//	{
//		conv_sigma_delta[i] = 0;
//		for (int j = 0;j < POOL_SIZE;j++)
//			for (int k = 0;k < POOL_SIZE;k++)
//			{
//				float error = 0;
//				conv_delta[i][j][k] = 0;
//				for (int l = 0;l < FC1_SIZE;l++)
//					error += fc1_delta[l] * fc1_w[l][i][j][k];
//				conv_delta[i][j][k] = error * (pool[i][j][k] * (1.0 - pool[i][j][k]));
//				conv_sigma_delta[i] += error * (pool[i][j][k] * (1.0 - pool[i][j][k]));
//			}
//		conv_db[i] += conv_sigma_delta[i];
//	}
//}
//
//void update_conv_w()
//{
//	for (int i = 0;i < CONV_W_NUM;i++)
//		for (int j = 0;j < CONV_W_SIZE;j++)
//			for (int k = 0;k < CONV_W_SIZE;k++)
//			{
//				float error = 0;
//				for (int m = 0;m < POOL_SIZE;m++)
//					for (int n = 0;n < POOL_SIZE;n++)
//					{
//						int x = pool_pos[i][m][n] / 2;
//						int y = pool_pos[i][m][n] % 2;
//						error += conv_delta[i][m][n] * input[2 * m + j + x][2 * n + k + y];
//					}
//				conv_dw[i][j][k] += error;
//			}
//}
//
//void assign_grads()
//{
//	for (int i = 0;i < FC2_SIZE;i++)
//	{
//		fc2_b[i] -= (fc2_db[i] / minibatch);
//		fc2_db[i] = 0;
//	}
//
//	for (int i = 0;i < FC2_SIZE;i++)
//		for (int j = 0;j < FC1_SIZE;j++)
//		{
//			fc2_w[i][j] -= (fc2_dw[i][j] / minibatch);
//			fc2_dw[i][j] = 0;
//		}
//
//	for (int i = 0;i < FC1_SIZE;i++)
//	{
//		fc1_b[i] -= (fc1_db[i] / minibatch);
//		fc1_db[i] = 0;
//	}
//
//	for (int i = 0;i < FC1_SIZE;i++)
//		for (int j = 0;j < CONV_W_NUM;j++)
//			for (int k = 0;k < POOL_SIZE;k++)
//				for (int l = 0;l < POOL_SIZE;l++)
//				{
//					fc1_w[i][j][k][l] -= (fc1_dw[i][j][k][l] / minibatch);
//					fc1_dw[i][j][k][l] = 0;
//				}
//
//	for (int i = 0;i < CONV_W_NUM;i++)
//	{
//		conv_b[i] -= (conv_db[i] / minibatch);
//		conv_db[i] = 0;
//	}
//
//	for (int i = 0;i < CONV_W_NUM;i++)
//		for (int l = 0;l < CONV_W_SIZE;l++)
//			for (int m = 0;m < CONV_W_SIZE;m++)
//			{
//				conv_w[i][l][m] -= (conv_dw[i][l][m] / minibatch);
//				conv_dw[i][l][m] = 0;
//			}
//}
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
//	for (int i = 0;i < FC1_SIZE;i++)
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
//	}
//}
//int main() {
//
//	load_data();
//	clock_t t = clock();
//	init_params();
//
//	for (int i = 1;i <= epochs;i++)
//	{
//		correct_cnt = 0;
//		avg_error = 0;
//
//		for (int j = 0;j < TRAIN_NUM;j++)
//		{
//			set_input(j, train_image);
//			input_conv();
//			conv_pool();
//			pool_fc1();
//			fc1_fc2();
//			set_answer(j, train_label);
//			check_answer(correct_cnt);
//			get_error(avg_error);
//
//			update_fc2_b();
//			update_fc2_w();
//			update_fc1_b();
//			update_fc1_w();
//			update_conv_b();
//			update_conv_w();
//			if ((j + 1) % minibatch == 0)
//				assign_grads();
//
//			if (j && j % 100 == 0)
//				printf("Training  Time spent : %.0fs Image count : %d Accuracy : %0.4f%% Error : %0.4f%% Epoch : %d \r", floor(((float)(clock() - t)) / CLOCKS_PER_SEC), j, ((float)correct_cnt / j) * 100, (avg_error / j) * 100, i);
//		}
//		printf("Training  Time spent : %.0fs Image count : %d Accuracy : %0.4f%% Error : %0.4f%% Epoch : %d \n", floor(((float)(clock() - t)) / CLOCKS_PER_SEC), TRAIN_NUM, ((float)correct_cnt / TRAIN_NUM) * 100, (avg_error / TRAIN_NUM) * 100, i);
//
//		correct_cnt = 0;
//		avg_error = 0;
//
//		for (int j = 0;j < TEST_NUM;j++)
//		{
//			set_input(j, test_image);
//			input_conv();
//			conv_pool();
//			pool_fc1();
//			fc1_fc2();
//			set_answer(j, test_label);
//			check_answer(correct_cnt);
//			get_error(avg_error);
//
//			if (j && j % 100 == 0)
//				printf("Testing   Time spent : %.0fs Image count : %d Accuracy : %0.4f%% Error : %0.4f%% \r", floor(((float)(clock() - t)) / CLOCKS_PER_SEC), j, ((float)correct_cnt / j) * 100, (avg_error / j) * 100);
//		}
//		printf("Testing   Time spent : %.0fs Image count : %d Accuracy : %0.4f%% Error : %0.4f%% \n", floor(((float)(clock() - t)) / CLOCKS_PER_SEC), TEST_NUM, ((float)correct_cnt / TEST_NUM) * 100, (avg_error / TEST_NUM) * 100);
//
//		if ((float)correct_cnt / TEST_NUM * 100 > max_acc)
//		{
//			max_acc = (float)correct_cnt / TEST_NUM * 100;
//			//export_params();
//			printf("The new model has been exported.Accuracy has reached to %0.5f%%\n\n", max_acc);
//		}
//		else
//		{
//			alpha = alpha - (alpha / 3);
//			printf("Learning rate has been reduced to %f\n\n", alpha);
//		}
//	}
//
//
//
//	//float train_image[ROW][COL] = {
//	//{ 3, 1, 2, 4, 3, 3 },
//	//{ 2, 4, 3, 1, 1, 4 },
//	//{ 1, 5, 2, 3, 2, 5 },
//	//{ 2, 3, 4, 1, 4, 1 },
//	//{ 1, 4, 2, 1, 2, 3 },
//	//{ 2, 3, 6, 5, 4, 1 }, };
//	//float conv_w[CONV_W_NUM][CONV_W_SIZE][CONV_W_SIZE] = { {
//	//{1, 2, 3},
//	//{4, 3, 1},
//	//{1, 2, 4}},
//	//{{4, 2, 5},
//	//{2, 3, 1},
//	//{1, 2, 3}} };
//
//	////float conv_z[CONV_W_NUM][CONV_SIZE][CONV_SIZE];
//	//float conv_z[2][2][2];
//	//float train_label[2] = { 3,2 };
//
//	//cudaMemcpyToSymbol(_train_image, train_image, ROW * COL * sizeof(float));
//	//cudaMemcpyToSymbol(_conv_w, conv_w, CONV_W_NUM * CONV_W_SIZE * CONV_W_SIZE * sizeof(float));
//	////cudaMemcpy(_train_label, train_label, 2 * sizeof(float), cudaMemcpyHostToDevice);
//	////cudaMemcpy(_train_image, train_image, ROW * COL * sizeof(float), cudaMemcpyHostToDevice);
//	////cudaMemcpy(_conv_w, conv_w, CONV_W_NUM*CONV_W_SIZE*CONV_W_SIZE*sizeof(float), cudaMemcpyHostToDevice);
//	//dim3 grid2(2, 4, 4);
//
//	////_input_conv << <1, grid2>> > ((float (*)[4])_train_image, (float (*)[3][3])_conv_w, (float (*)[2][2])_conv_z);
//	//_input_conv << <1, grid2 >> > ();
//	//_conv_pool << <1, grid2 >> > ();
//	////cudaMemcpyFromSymbol(&conv_z, _pool, CONV_W_NUM * CONV_SIZE * CONV_SIZE * sizeof(float));
//	//cudaMemcpyFromSymbol(&conv_z, _pool, 8 * sizeof(float));
//	//for (int i = 0;i < 2;i++) {
//	//	for (int j = 0;j <2;j++) {
//	//		cout << conv_z[0][i][j] << " ";
//	//	}
//	//	cout << endl;
//	//}
//	//for (int i = 0;i < 2;i++) {
//	//	for (int j = 0;j < 2;j++) {
//	//		cout << conv_z[1][i][j] << " ";
//	//	}
//	//	cout << endl;
//	//}
//	return 0;
//}