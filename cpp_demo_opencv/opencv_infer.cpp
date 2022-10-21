
#include "opencv_infer.h"
#include "opencv2/core/utils/logger.hpp"

#if USE_OPENCV_DNN

OpencvDNNEngine::OpencvDNNEngine()
{
	box_cache = nullptr;
}

OpencvDNNEngine::OpencvDNNEngine(std::string onnx_file)
{
	box_cache = new float[dim_c_*max_objs_];
	initfromONNX(onnx_file);
}

OpencvDNNEngine::~OpencvDNNEngine()
{
	delete[] box_cache;
}

int OpencvDNNEngine::initfromONNX(std::string onnx_file)
{
	if (configs.log_config.open)
	{
		printf_d("loading ONNX files: %s ", onnx_file.c_str());
	}

	//去除日志信息输出
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

	//load onnx model
	net = cv::dnn::readNetFromONNX(onnx_file);
	
	//set inference engine backend
	//net.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
	
	if (configs.log_config.open)
		printf_d("set opencv inference device: %s", configs.net_config.device.c_str());

	//set compute device 
	//net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

	if (configs.log_config.open)
		printf_d("load onnx successfully.");

	//cache
	int type = (configs.img_config.channel == 3) ? CV_8UC3 : CV_8UC1;
	cv::Mat temp(configs.img_config.normal_size, configs.img_config.normal_size, type);
	cv::randn(temp, 128, 10);
	compute(temp.data, temp.rows, temp.cols, temp.step);

	return 1;
}

cv::Mat OpencvDNNEngine::compute(unsigned char* data, int height, int width, int width_step)
{
	int type = (configs.img_config.channel == 3) ? CV_8UC3 : CV_8UC1;
	cv::Mat image(height, width, type, (void*)data);

	cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0, { configs.img_config.normal_size,configs.img_config.normal_size });

	net.setInput(blob, "data");

	return net.forward();
}

int OpencvDNNEngine::process(unsigned char* data, int height, int width, int step, float* results, int* num_results, int* results_dim)
{
	std::unique_lock<std::mutex> __lock_net(__mutex_net);

	int padw = 0, padh = 0;

	auto detectionMat = compute(data, height, width, step);
	if (configs.log_config.open)
	{
		printf_d("detectionMat: %d %d\n", detectionMat.rows, detectionMat.cols);
	}

	//generate proposals
	std::vector<BoxInfo> generate_boxes;
	float ratioh = 1.0f, ratiow = 1.0f;
	int n = 0, q = 0, i = 0, j = 0, k = 0; ///xmin,ymin,xamx,ymax,box_score,class_score
	const int nout = dim_classes_ + 5;
	float* box_ = box_cache;
	int dim_bbox = nout;
	const float* preds = detectionMat.ptr<float>(0);

	int size_bbox = 0;
	for (n = 0; n < num_block_output_; n++)   //
	{
		int num_grid_x = (int)(width / this->stride[n]);
		int num_grid_y = (int)(height / this->stride[n]);
		for (q = 0; q < num_anchors_; q++)    ///anchor
		{
			const float anchor_w = this->anchors[n][q * 2];
			const float anchor_h = this->anchors[n][q * 2 + 1];
			for (i = 0; i < num_grid_y; i++)
			{
				for (j = 0; j < num_grid_x; j++)
				{
					float box_score = preds[4];
					if (box_score > configs.det_config.conf_thresh)
					{
						float class_score = 0;
						int class_ind = 0;
						for (k = 0; k < dim_classes_; k++)
						{
							if (preds[k + 5] > class_score)
							{
								class_score = preds[k + 5];
								class_ind = k;
							}
						}

						float cx = (preds[0] * 2.f - 0.5f + j) * this->stride[n];  ///cx
						float cy = (preds[1] * 2.f - 0.5f + i) * this->stride[n];   ///cy
						float w = powf(preds[2] * 2.f, 2.f) * anchor_w;   ///w
						float h = powf(preds[3] * 2.f, 2.f) * anchor_h;  ///h

						float xmin = (cx - padw - 0.5 * w) * ratiow;
						float ymin = (cy - padh - 0.5 * h) * ratioh;
						float xmax = (cx - padw + 0.5 * w) * ratiow;
						float ymax = (cy - padh + 0.5 * h) * ratioh;

						generate_boxes.push_back(BoxInfo{ xmin, ymin, xmax, ymax, class_score, class_ind });

						box_[0] = xmin;
						box_[1] = ymin;
						box_[2] = xmax;
						box_[3] = ymax;
						box_[4] = class_ind;
						box_[5] = class_score;

						size_bbox++;

						box_ += dim_bbox;
					}
					preds += nout;
				}
			}
		}
	}

	*results_dim = dim_bbox;
	non_max_suppression(box_cache, size_bbox, dim_bbox, configs.det_config.conf_thresh, configs.det_config.iou_thresh, results, num_results, true);

	if (configs.log_config.open)
		printf_d("post-process bbox num: %d", *num_results);

	return 0;
}

#endif
