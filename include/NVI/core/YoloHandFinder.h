#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

class YoloHandFinder {
public:
  static YoloHandFinder* getYolo();

  YoloHandFinder(YoloHandFinder&) = delete;
  void operator=(const YoloHandFinder&) = delete;

  cv::Rect getHandRoi(const cv::Mat &frame);
private:
  YoloHandFinder::YoloHandFinder();
  std::vector<cv::Rect> postProcess(const cv::Mat &frame, const std::vector<cv::Mat>& outs);

  const int CNN_H = 256;
  const int CNN_W = 256;
  float confThreshold = 0.5; // Confidence threshold
  float nmsThreshold = 0.4;  // Non-maximum suppression threshold
  cv::dnn::Net net;

  const std::string modelPath = "models/cross-hands.cfg";
  const std::string weightsPath = "models/cross-hands.weights";

};