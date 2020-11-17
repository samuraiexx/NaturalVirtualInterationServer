#pragma once
#include <vector>
#include <opencv2/highgui.hpp>

#include "OpenPose.h"
#include "InverseKinematics.h"
#include "OneEuroFilter.h"

template<typename T>
struct EulerAngle;

class ImageProcessor {
public:
  ImageProcessor();
  void processImage(cv::Mat &img);

  std::vector<double> getPoseAngles();
  std::vector<cv::Point2d> ImageProcessor::getPose2d();
  std::vector<cv::Point3d> ImageProcessor::getPose3d();
  double getMeanError();
  bool getLostTrack();
  void resetPose();

private:
  void trackKeyPoints(cv::Mat &img, cv::Point2d offset);
  void filterKeyPoints();

  std::vector<std::pair<cv::Point2d, double>> keyPoints;
  std::vector<std::pair<cv::Point2d, double>> prevKeyPoints;
  cv::Mat grayFrame;
  cv::Mat prevGrayFrame;

  int imageWidth;
  int imageHeight;
  int currentFrame = -1;
  bool lostTrack = true;
  OpenPose pose2dProcessor;
  InverseKinematics inverseKinematics;
  std::vector<OneEuroFilter> filters;

  const bool USE_LK = false;
  const bool USE_ONE_EURO = true;
  const int RNN_FRAME_INTERVAL = 1;
};

