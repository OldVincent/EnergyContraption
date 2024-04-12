#pragma once

#include <opencv4/opencv2/opencv.hpp>

namespace Gaia::Modules
{
	class ImageDebugUtility
	{
	public:
		static void DrawAxis(cv::Mat& image, cv::Point start, cv::Point end,
							 const cv::Scalar& color, float scale = 0.2);

		static void DrawRectangle(cv::Mat& canvas,
								  const cv::Point& center, const cv::Size& size,
								  const cv::Scalar& color, int thickness = 3);

		static void DrawRotatedRectangle(cv::Mat& canvas,
										 const cv::Point& center, const cv::Size& size, float angle,
										 const cv::Scalar& color, int thickness = 3);

		static void DrawRotatedRectangle(cv::Mat& canvas,
										 const cv::RotatedRect& rotated_rect,
										 const cv::Scalar& color, int thickness = 3);

		static void DrawCross(cv::Mat& canvas, const cv::Point& center, const cv::Scalar& color, int length = 5, int thickness = 2);
	};
}