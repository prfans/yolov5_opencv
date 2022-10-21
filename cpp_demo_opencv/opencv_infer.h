#pragma once

#ifndef __H_OPENCV_INFER__
#define __H_OPENCV_INFER__

#include <fstream>
#include <vector>
#include <chrono>
#include <memory>
#include <string>
#include <mutex>

#include "opencv2/opencv.hpp"
#include "net_params.h"
#include "post_process.h"

#if USE_OPENCV_DNN

class OpencvDNNEngine
{
public:
	std::string input_name;
	std::string output_name;
	std::vector<std::string>output_names;

private:
	const float anchors[3][6] = { {10.0, 13.0, 16.0, 30.0, 33.0, 23.0}, {30.0, 61.0, 62.0, 45.0, 59.0, 119.0},{116.0, 90.0, 156.0, 198.0, 373.0, 326.0} };
	const float stride[3] = { 8.0, 16.0, 32.0 };
  float* box_cache;

private:
	std::mutex __mutex_net;
	cv::dnn::Net net;

public:
	OpencvDNNEngine();
	OpencvDNNEngine(std::string onnx_file);
	~OpencvDNNEngine();
	int initfromONNX(std::string onnx_file);
	cv::Mat compute(unsigned char* data, int height, int width, int width_step);
	int process(unsigned char* data, int height, int width, int step, float* results, int* num_results, int* results_dim);
};
#endif

#endif //__H_OPENCV_INFER__
