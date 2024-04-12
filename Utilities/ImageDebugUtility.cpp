#include "ImageDebugUtility.hpp"

#include <opencv4/opencv2/opencv.hpp>

#define PI 3.14159265f

namespace Gaia::Modules
{
	void ImageDebugUtility::DrawRectangle(cv::Mat& canvas,
										  const cv::Point& center, const cv::Size& size,
										  const cv::Scalar& color, int thickness)
	{
		cv::line(canvas,
				 cv::Point(center.x - size.width / 2, center.y - size.height / 2),
				 cv::Point(center.x - size.width / 2, center.y + size.height / 2),
				 color, thickness);
		cv::line(canvas,
				 cv::Point(center.x - size.width / 2, center.y + size.height / 2),
				 cv::Point(center.x + size.width / 2, center.y + size.height / 2),
				 color, thickness);
		cv::line(canvas,
				 cv::Point(center.x + size.width / 2, center.y + size.height / 2),
				 cv::Point(center.x + size.width / 2, center.y - size.height / 2),
				 color, thickness);
		cv::line(canvas,
				 cv::Point(center.x + size.width / 2, center.y - size.height / 2),
				 cv::Point(center.x - size.width / 2, center.y - size.height / 2),
				 color, thickness);
	}

	void ImageDebugUtility::DrawRotatedRectangle(cv::Mat& canvas,
												 const cv::Point &center, const cv::Size &size, float angle,
												 const cv::Scalar &color, int thickness)
	{
		ImageDebugUtility::DrawRotatedRectangle(canvas, cv::RotatedRect(center, size, angle),
												color, thickness);
	}

	void ImageDebugUtility::DrawRotatedRectangle(
			cv::Mat &canvas, const cv::RotatedRect& rotated_rect,
			const cv::Scalar &color, int thickness)
	{
		cv::Point2f vertices[4];

		rotated_rect.points(vertices);

		for(int i=0;i<4;i++)
		{
			line(canvas, vertices[i], vertices[(i + 1) % 4], color, thickness);
		}
	}

	void ImageDebugUtility::DrawAxis(cv::Mat &image, cv::Point start, cv::Point end, const cv::Scalar& color, const float scale)
	{
		using namespace cv;
		double angle = atan2((double) start.y - end.y, (double) start.x - end.x ); // angle in radians
		double hypotenuse = sqrt((double) (start.y - end.y) * (start.y - end.y) + (start.x - end.x) * (start.x - end.x));
		// lengthen the arrow by a factor of scale
		end.x = (int) (start.x - scale * hypotenuse * cos(angle));
		end.y = (int) (start.y - scale * hypotenuse * sin(angle));
		line(image, start, end, color, 1, cv::LINE_AA);
		// create the arrow hooks
		start.x = (int) (end.x + 9 * cos(angle + CV_PI / 4));
		start.y = (int) (end.y + 9 * sin(angle + CV_PI / 4));
		line(image, start, end, color, 1, cv::LINE_AA);
		start.x = (int) (end.x + 9 * cos(angle - CV_PI / 4));
		start.y = (int) (end.y + 9 * sin(angle - CV_PI / 4));
		line(image, start, end, color, 1, cv::LINE_AA);
	}

	void ImageDebugUtility::DrawCross(cv::Mat &canvas, const cv::Point &center,
									  const cv::Scalar& color, int length, int thickness)
	{
		auto point_up = center - cv::Point(thickness / 2, length / 2);
		point_up.x = point_up.x >= 0 ? point_up.x : 0;
		point_up.y = point_up.y >= 0 ? point_up.y : 0;
		auto point_left = center - cv::Point(length / 2, thickness / 2);
		point_left.x = point_left.x >= 0 ? point_left.x : 0;
		point_left.y = point_left.y >= 0 ? point_left.y : 0;
		cv::rectangle(canvas, cv::Rect(point_up.x, point_up.y, thickness, length), color, cv::FILLED);
		cv::rectangle(canvas, cv::Rect(point_left.x, point_left.y, length, thickness), color, cv::FILLED);
	}
}