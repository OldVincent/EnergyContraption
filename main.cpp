#include <iostream>
#include <vector>
#include <tuple>
#include <string>
#include <opencv2/opencv.hpp>

#include "Utilities/GeometryFeature.hpp"
#include "Utilities/RectangleTool.hpp"
#include "Utilities/CircleTool.hpp"
#include "Utilities/ImageDebugUtility.hpp"

struct DetectRequest
{
	cv::Mat Picture;
	bool DebugMode;
	int ThresholdColor;
	int ThresholdValue;
	int CloseKernelSize;
	int PanelMinArea;
	int PanelMaxArea;
	double CircleRatio;
	double BlankRatio;
	int CircleValue;
};

struct DetectResponse
{
	std::vector<std::tuple<std::string, cv::Mat>> DebugViews;
};

void Detect(const DetectRequest& request, DetectResponse& result)
{
	using namespace Gaia::Modules;

	// 找出蒙板区域
	std::vector<cv::Mat> channels;
	cv::split(request.Picture, channels);
	auto& channel_b = channels[0];
	auto& channel_g = channels[1];
	auto& channel_r = channels[2];
	for(auto& channel : channels)
		channel.convertTo(channel, CV_16S);
	cv::Mat mask_color;
	cv::Mat mask_value;
	cv::threshold(channel_r - channel_b, mask_color,
				  request.ThresholdColor, 255, cv::THRESH_BINARY);
	cv::threshold(channel_r, mask_value,
				  request.ThresholdValue, 255, cv::THRESH_BINARY);
	mask_color.convertTo(mask_color, CV_8UC1);
	mask_value.convertTo(mask_value, CV_8UC1);
	cv::Mat mask = mask_color & mask_value;
	if (request.DebugMode)
	{
		result.DebugViews.emplace_back("Color Mask", mask_color);
		result.DebugViews.emplace_back("Value Mask", mask_value);
	}
	// 进行闭运算
	if (request.CloseKernelSize > 0)
	{
		cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
						 cv::getStructuringElement(
								 cv::MORPH_CROSS,
								 cv::Size(request.CloseKernelSize, request.CloseKernelSize)));

	}
	if (request.DebugMode)
	{
		result.DebugViews.emplace_back("Closed Mask", mask);
	}


	// 查找轮廓
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(mask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);
	// 计算几何特征
	struct Element
	{
		const int Index;
		const GeometryFeature Feature;
		const RectangleTool::ContourRelationship Relationship;
		const cv::RotatedRect Rectangle;
		const double Area;
		const std::vector<cv::Point> Contour;
		const int SiblingsCount;
		const int ChildrenCount;
		const cv::Point Center;
	};
	std::vector<Element> elements;
	elements.reserve(contours.size());
	for (auto index = 0; index < contours.size(); ++index)
	{
		auto relationship = RectangleTool::TranslateContourRelationship(hierarchy[index]);
		auto rectangle = cv::minAreaRect(contours[index]);
		elements.emplace_back(Element{
			index,
			GeometryFeature::Standardize(rectangle),
			relationship,
			rectangle,
			cv::contourArea(contours[index]),
			contours[index],
			RectangleTool::CountSiblings(hierarchy, index),
			relationship.ChildrenIndex > 0 ?
			RectangleTool::CountSiblings(hierarchy, relationship.ChildrenIndex) : 0,
			RectangleTool::GetCenter(contours[index])
		});
	}

	std::vector<Element> candidate_panels;
	for (const auto& element : elements)
	{
		// 筛选面积
		if (element.Area > request.PanelMaxArea || element.Area < request.PanelMinArea)
			continue;
		bool satisfied = true;
		// 子轮廓面积不得大于该轮廓的一定比例，用以排除已经激活的面板
		for (const auto& children_index : RectangleTool::CollectChildren(hierarchy, element.Index))
		{
			const auto& children = elements[children_index];
			if (children.Area / element.Area > request.BlankRatio)
			{
				satisfied = false;
				break;
			}
		}
		if (satisfied)
			candidate_panels.push_back(element);
	}
	if (request.DebugMode)
	{
		cv::Mat view = request.Picture.clone();
		for (const auto& element : candidate_panels)
		{
			drawContours(view, contours, element.Index,
						 cv::Scalar(102, 255, 102),2, cv::LINE_8, hierarchy);
		}
		result.DebugViews.emplace_back("Candidate Panels", view);
	}

	// 靶心的查找将在灰度图上进行
	cv::Mat picture_gray;
	cv::cvtColor(request.Picture, picture_gray, cv::COLOR_BGR2GRAY);

	if (request.DebugMode)
	{

		auto gray = picture_gray.clone();
		cv::threshold(gray, gray,
					  request.CircleValue, 255, cv::THRESH_BINARY);
		std::vector<std::vector<cv::Point>> gray_contours;
		cv::findContours(gray, gray_contours, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);
		std::vector<int> candidate_circles;
		for(auto index = 0; index < gray_contours.size(); ++index)
		{
			const auto& contour = gray_contours[index];
			if (CircleTool::GetCircleSimilarity(contour) < request.CircleRatio)
				candidate_circles.push_back(index);
		}
		auto view = request.Picture.clone();
		for (auto index : candidate_circles)
		{
			cv::drawContours(view, gray_contours, index, cv::Scalar(240,255,250), 2);
		}
		result.DebugViews.emplace_back("Candidate Circles", view);
	}

	if (request.DebugMode)
	{
		// 从灰度图中选出该候选面板区域，并拷贝到专门的检测图片中
		auto view = picture_gray.clone();
		cv::threshold(view, view,
					  request.CircleValue, 255, cv::THRESH_BINARY);
		if (request.CloseKernelSize > 0)
		{
			cv::morphologyEx(view, view, cv::MORPH_CLOSE,
							 cv::getStructuringElement(
									 cv::MORPH_CROSS,
									 cv::Size(request.CloseKernelSize, request.CloseKernelSize)));

		}
		result.DebugViews.emplace_back("Circle Mask", view);
	}


	struct PossiblePanel
	{
		const Element Panel;
		const cv::Point Target;
	};
	std::vector<PossiblePanel> candidate_targets;

	for (const auto& element : candidate_panels)
	{
		auto candidate_region = element.Rectangle.boundingRect();
		cv::Mat candidate_area;
		// 从灰度图中选出该候选面板区域，并拷贝到专门的检测图片中
		picture_gray(candidate_region).copyTo(candidate_area, mask(candidate_region));
		cv::threshold(candidate_area, candidate_area,
					  request.CircleValue, 255, cv::THRESH_BINARY);
		if (request.CloseKernelSize > 0)
		{
			cv::morphologyEx(candidate_area, candidate_area, cv::MORPH_CLOSE,
							 cv::getStructuringElement(
									 cv::MORPH_CROSS,
									 cv::Size(request.CloseKernelSize, request.CloseKernelSize)));
		}
		std::vector<std::vector<cv::Point>> gray_contours;
		std::vector<cv::Vec4i> gray_hierarchy;
		cv::findContours(candidate_area, gray_contours, gray_hierarchy,
						 cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

		// Debug
		cv::Mat view;
		cv::cvtColor(candidate_area, view, cv::COLOR_GRAY2BGR);
//		cv::drawContours(view, gray_contours, -1,
//						 cv::Scalar(0, 215, 255), 1);
		cv::imshow("Area", view);

		// 查找内圆
		cv::Point selected_circle_center;
		int selected_circle_distance = -1;
		for (auto child_index = 0; child_index < gray_contours.size(); ++child_index)
		{
			const auto& contour = gray_contours[child_index];
			if (CircleTool::GetCircleSimilarity(contour) > request.CircleRatio)
				continue;
			// 从兴趣区转换到全局坐标
			auto child_center = element.Rectangle.boundingRect().tl() + RectangleTool::GetCenter(contour);
			int distance = (child_center.x - element.Center.x) * (child_center.x - element.Center.x) +
					(child_center.y - element.Center.y) * (child_center.y - element.Center.y);
			if (selected_circle_distance >= 0)
			{
				if (distance > selected_circle_distance)
					continue;
			}
			selected_circle_distance = distance;
			selected_circle_center = child_center;
		}
		if (selected_circle_distance >= 0)
		{
			candidate_targets.emplace_back(PossiblePanel{
					element,selected_circle_center
			});
		}
	}

	if (request.DebugMode)
	{
		cv::Mat view = request.Picture.clone();
		for (const auto& candidate : candidate_targets)
		{
			cv::drawContours(view, contours, candidate.Panel.Index,
							 cv::Scalar(0, 215, 255),2,
							 cv::LINE_8, hierarchy, 0);
//			ImageDebugUtility::DrawCross(view, candidate.Panel.Center,
//										 cv::Scalar(255,0,0), 15, 2);
			ImageDebugUtility::DrawCross(view, candidate.Target,
										 cv::Scalar(0,255,0), 25, 2);
		}
		result.DebugViews.emplace_back("Candidate Targets", view);
	}
}

int main()
{
	DetectRequest request
	{
		.DebugMode = true,
		.ThresholdColor = 80,
		.ThresholdValue = 200,
		.CloseKernelSize = 4,
		.PanelMinArea = 1100,
		.PanelMaxArea = 2670,
		.CircleValue = 170
	};
	int threshold_circle = 308;
	int threshold_blank = 10;

	int view_index = 0;
	int view_count = 1;

	cv::namedWindow("Main");
	cv::createTrackbar("Color", "Main", &request.ThresholdColor, 255);
	cv::createTrackbar("Value", "Main", &request.ThresholdValue, 255);
	cv::createTrackbar("Close", "Main", &request.CloseKernelSize, 5);
	cv::createTrackbar("Circle Sim.", "Main", &threshold_circle, 500);
	cv::createTrackbar("Circle Val.", "Main", &request.CircleValue, 255);
	cv::createTrackbar("Blank", "Main", &threshold_blank, 100);
	cv::createTrackbar("Panel Min", "Main", &request.PanelMinArea, 9000);
	cv::createTrackbar("Panel Max", "Main", &request.PanelMaxArea, 9000);
	cv::createTrackbar("Views", "Main", &view_index, view_count);
	auto video = cv::VideoCapture("RealWorldVideo.mp4");

	int key = 0;
	bool paused = false;

	while(true)
	{
		key = cv::waitKey(1);
		if (key == 27) break;
		if (key == 32)
			paused = !paused;

		if (!paused)
		{
			if(!video.read(request.Picture))
			{
				video.set(cv::CAP_PROP_POS_FRAMES, 0);
				continue;
			}
		}

		request.CircleRatio = (double)threshold_circle / 100.0;
		request.BlankRatio = (double)threshold_blank / 100.0;

		DetectResponse response;
		Detect(request, response);

		if (!response.DebugViews.empty() && response.DebugViews.size() != view_count)
		{
			cv::setTrackbarMax("Views", "Main", (int)response.DebugViews.size());
			view_count = (int)response.DebugViews.size();
		}

		if (view_index == 0)
			cv::imshow("Main", request.Picture);
		else
		{
			auto view = std::get<1>(response.DebugViews[view_index - 1]);
			if (view.channels() == 1)
				cv::cvtColor(view, view, cv::COLOR_GRAY2BGR);
			cv::putText(view, std::get<0>(response.DebugViews[view_index - 1]),
			        cv::Point(5, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0,
					cv::Scalar(201, 207, 142));
			cv::imshow("Main", view);
		}
	}
	return 0;
}
