#include "CircleTool.hpp"

double CircleTool::GetCircleSimilarity(const std::vector<cv::Point> &contour)
{
	auto moment = cv::moments(cv::Mat(contour), false);
	auto center = cv::Point(
			cvRound(moment.m10 / moment.m00),
			cvRound(moment.m01 / moment.m00));

	// 计算各个点到中心的距离
	std::vector<int> distances;
	distances.reserve(contour.size());

	for (const auto& point : contour)
	{
		distances.emplace_back(
				(point.x - center.x) * (point.x - center.x) +(point.y - center.y) * (point.y - center.y));
	}

	// 计算均值
	double distances_mean = 0;
	for (auto distance : distances)
		distances_mean += distance;
	distances_mean /= (double)distances.size();

	// 计算标准差
	double distances_error = 0;
	for (auto distance : distances)
		distances_error += (distance - distances_mean) * (distance - distances_mean);
	distances_error = std::sqrt(distances_error);

	return distances_error / distances_mean;
}
