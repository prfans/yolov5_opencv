#pragma once

#ifndef __H_IMAGE_PROCESS__
#define __H_IMAGE_PROCESS__

#include "opencv2/opencv.hpp"

void resize(cv::Mat& image_orig, cv::Mat& image_border, int* newh, int* neww, int* top, int* left);

#endif //__H_IMAGE_PROCESS__
