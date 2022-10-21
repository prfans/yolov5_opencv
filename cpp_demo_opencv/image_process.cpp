

#include "image_process.h"
#include "net_params.h"

int resize_image(cv::Mat& srcimg, cv::Mat& dstimg, int* newh, int* neww, int* top, int* left)
{
	bool keep_ratio = configs.img_config.keep_ratio;
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = configs.img_config.normal_size;
	*neww = configs.img_config.normal_size;
	*left = *top = 0;

	if (keep_ratio && srch != srcw) {
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1) {
			*newh = configs.img_config.normal_size;
			*neww = int(configs.img_config.normal_size / hw_scale);
			cv::resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_LINEAR);
			*left = int((configs.img_config.normal_size - *neww) * 0.5);
			cv::copyMakeBorder(dstimg, dstimg, 0, 0, *left, configs.img_config.normal_size - *neww - *left, cv::BORDER_CONSTANT, 0);
		}
		else {
			*newh = (int)configs.img_config.normal_size * hw_scale;
			*neww = configs.img_config.normal_size;
			cv::resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_LINEAR);
			*top = (int)(configs.img_config.normal_size - *newh) * 0.5;
			copyMakeBorder(dstimg, dstimg, *top, configs.img_config.normal_size - *newh - *top, 0, 0, cv::BORDER_CONSTANT, 0);
		}
	}
	else {
		cv::resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_LINEAR);
	}
	return 0;
}

void resize(cv::Mat& image_orig, cv::Mat& image_border, int* newh, int* neww, int* top, int* left)
{
	if (configs.log_config.open)
	{
		printf_d("image resized to {%d %d}", configs.img_config.normal_size, configs.img_config.normal_size);
	}
	resize_image(image_orig, image_border, newh, neww, top, left);
	//cv::resize(image_orig, image_border, cv::Size(configs.img_config.normal_size, configs.img_config.normal_size), 0.0, 0.0, cv::INTER_LINEAR);
}

