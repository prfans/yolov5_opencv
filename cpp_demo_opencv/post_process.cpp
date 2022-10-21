
#include <math.h>
#include <stdio.h>
#include "post_process.h"
#include "ImgDetect.h"
#include <opencv2/opencv.hpp>

//class_agnostic
int non_max_suppression(float* bboxes, int size_bboxs, int dim_bbox, float conf_thresh, float nms_thresh, float* results, int* num, bool class_agnostic)
{
	int cache_size = size_bboxs * dim_bbox
 		+ size_bboxs * size_bboxs
		+ size_bboxs
		+ size_bboxs + 1024;
	float* cache = new float[cache_size]; //需要保证申请足够，如果担心内存碎片，可以放在全局区申请，但需要注意线程安全

	//申请内存
	float* bbox_c = cache;
	float* ious = cache + size_bboxs * dim_bbox; //计算iou矩阵，未使用
	float* scores = cache + size_bboxs * dim_bbox + size_bboxs * size_bboxs;
	bool* results_nms = (bool*)(cache + size_bboxs * dim_bbox + size_bboxs * size_bboxs + size_bboxs);
	int num_results = 0;

	if (class_agnostic)
	{
		bbox_c = bboxes;

		std::vector<int> classIds;
		std::vector<float> confidences;
		std::vector<cv::Rect> boxes;
		for (int n = 0; n < size_bboxs; n++)
		{
			float x1 = bbox_c[n * dim_bbox];
			float y1 = bbox_c[n * dim_bbox + 1];
			float x2 = bbox_c[n * dim_bbox + 2];
			float y2 = bbox_c[n * dim_bbox + 3];

			int centerX = (int)((x1 + x2) / 2.0 + 0.5f);
			int centerY = (int)((y1 + y2) / 2.0 + 0.5f);
			int width = (int)(x2 - x1 + 0.5f);
			int height = (int)(y2 - y1 + 0.5f);
			int left = centerX - width / 2;
			int top = centerY - height / 2;

			classIds.push_back(bbox_c[n * dim_bbox + INDEX_CLS]);
			confidences.push_back((float)bbox_c[n * dim_bbox + INDEX_CONF]);
			boxes.push_back(cv::Rect(left, top, width, height));
		}

		// nms
		std::vector<int> indices;
		cv::dnn::NMSBoxes(boxes, confidences, 0.01f, nms_thresh, indices);

		for (int i = 0; i < indices.size(); i++)
		{
			int n = indices[i];

			memcpy(&results[num_results * dim_bbox], &bbox_c[n * dim_bbox], dim_bbox * sizeof(float));
			num_results++;
		}
	}

	delete[] cache;

	*num = num_results;

	return 0;
}
