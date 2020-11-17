#include <opencv2/highgui.hpp>
#include <nlohmann/json.hpp>

namespace cv {
	void to_json(nlohmann::json& j, const cv::Point3d& p);
	void to_json(nlohmann::json& j, const cv::Point2d& p);
}
