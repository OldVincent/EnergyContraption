#pragma once

#include <opencv2/opencv.hpp>

namespace Gaia::Modules
{
	class RectangleTool
	{
	public:
		/// Get safe rectangle insides the max size.
		static cv::Rect GetSafeRectangle(const cv::Rect& rectangle, const cv::Size& max_size);
		/// Get safe scaled rectangle.
		static cv::Rect GetScaledRectangle(const cv::Rect& rectangle, double width_scale, double height_scale);

		struct ContourRelationship
		{
			int ParentIndex {-1};
			int ChildrenIndex {-1};
			int NextIndex {-1};
			int PreviousIndex {-1};
		};
		/// Translate the relationship vector into human readable relationship structure.
		static ContourRelationship TranslateContourRelationship(cv::Vec4i relationship);

		/// Count all siblings of the target contour, the count includes the target itself.
		static int CountSiblings(std::vector<cv::Vec4i> hierarchy, int index);

		static std::vector<int> CollectChildren(const std::vector<cv::Vec4i>& hierarchy, int index, int max_depth = -1);

		/// Get the position of the weight center of the specified contour.
		static cv::Point GetCenter(const std::vector<cv::Point>& contour);
	};
}