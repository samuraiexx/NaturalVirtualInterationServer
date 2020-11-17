#pragma once

#include <opencv2/highgui.hpp>

class HandPoseEstimator
{
public:
  virtual std::vector<std::pair<cv::Point2d, double>> ProcessImage(cv::Mat &img, cv::Point offset = cv::Point()) = 0;

protected:
  std::vector<std::pair<cv::Point2d, double>> GetPointsFromHeatmap(cv::Mat &heatmap, cv::Mat &img);
};
