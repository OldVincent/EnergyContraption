#include <optional>
#include <opencv2/opencv.hpp>

class CircleTool
{
public:
	/// 计算点到中心的距离的标准差除以其均值的比值
	static double GetCircleSimilarity(const std::vector<cv::Point>& contour);
};