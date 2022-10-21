#pragma once

#ifndef __NETPARAMS_HEADER__
#define __NETPARAMS_HEADER__

#include <fstream>
#include <vector>
#include <chrono>
#include <memory>
#include <string>

#define USE_NNIE 0
#define USE_OPENVINO 0
#define USE_AMBA 0
#define USE_OPENCV_DNN 1
#define USE_RKNN 0
#define USE_MNN 0
#define USE_NCNN 0
#define USE_XXXXXXX 0

#define USE_CLASSIFY 0
#define USE_DET_HBB 1
#define USE_DET_OBB 0

#define NORMAL_SIZE 1024
#define IMAGE_CHANNEL 1
#define IMAGE_TYPE "gray"

constexpr bool log_open = false;

constexpr float PI = 3.14159265f;

constexpr int dim_coords_ = 4;
constexpr int dim_obj_conf_ = 1;
constexpr int dim_classes_ = 30;
constexpr int dim_c_ = dim_coords_ + dim_obj_conf_ + dim_classes_;

constexpr int num_anchors_ = 3;
constexpr int num_block_output_ = 3;
constexpr int max_objs_ = num_anchors_ * (128*128 + 64*64 + 32*32);

// # bboxŒ¨∂»–≈œ¢ [xmin, ymin, xmax, ymax, cls_id, conf]
constexpr int INDEX_CONF = 5;
constexpr int INDEX_CLS = 4;
constexpr int DIM_OUTPUT_ = 6;

#endif //__NETPARAMS_HEADER__
