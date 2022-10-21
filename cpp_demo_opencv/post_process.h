#pragma once

#ifndef __HEAD_POST_PROCESS__
#define __HEAD_POST_PROCESS__

#include <fstream>
#include <vector>
#include <chrono>
#include <memory>
#include <string>
#include "opencv2/opencv.hpp"

typedef struct BoxInfo
{
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	int label;
} BoxInfo;

int non_max_suppression(float* bboxes, int size_bboxs, int dim_bbox, float conf_thresh, float nms_thresh, float* results, int* num, bool class_agnostic);

#endif //__HEAD_POST_PROCESS__
