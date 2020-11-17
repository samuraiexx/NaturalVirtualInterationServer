#pragma once
#include <opencv2/dnn.hpp>
#include <time.h>

#include "HandPoseEstimator.h"

class OpenPose :
  public HandPoseEstimator
{
public:
  OpenPose();
  std::vector<std::pair<cv::Point2d, double>> ProcessImage(cv::Mat &img, cv::Point offset = cv::Point());

private:
  // Originally CNN_H = CNN_W = 368
  const int CNN_H = 256;
  const int CNN_W = 256;

  cv::dnn::Net net;

  const std::string modelProtoPath = "models/pose_deploy.prototxt";
  const std::string modelBinPath = "models/pose_iter_102000.caffemodel";
};

