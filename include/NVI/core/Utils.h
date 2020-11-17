#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <utility>

namespace Utils {
  int findStringInArgv(int argc, char** argv, std::string s, std::string alt = "");
  cv::Rect createBoundBox(
    const std::vector<std::pair<cv::Point2d, double >> &rnnPose,
    const std::vector<cv::Point2d> &ikPose,
    cv::Mat &img,
    bool &lostTrack
  );
  void drawBox(cv::Mat &img, cv::Rect roi);
}
