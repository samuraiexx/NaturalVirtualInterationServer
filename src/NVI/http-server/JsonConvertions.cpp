#include <NVI/http-server/JsonConvertions.h>

using json = nlohmann::json;

namespace cv {
	void to_json(json& j, const cv::Point3d& p) {
		j = json{ { "x", p.x },{ "y", p.y },{ "z", p.z } };
	}

	void to_json(json& j, const cv::Point2d& p) {
		j = json{ { "x", p.x },{ "y", p.y } };
	}
}
