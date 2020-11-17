#pragma once
#include <opencv2/highgui.hpp>
#include <Eigen/Geometry>
#include <ceres/ceres.h>

const double CONFIDENCE_TRASHOLD = 0.1; // Points bellow this confidence trashold will be ignored

template<typename T>
struct EulerAngle;

class InverseKinematics {
public:
  InverseKinematics(int adjustmentFrames = 30);

  void run(
    const std::vector<std::pair<cv::Point2d, double>> &observedHand2dPose
  );

  std::vector<double> getAngles();
  std::vector<cv::Point3d> get3dPoints();
  std::vector<cv::Point2d> get2dPoints();
  void resetPose();
  int getFrame();

private:
  int adjustmentFrames;
  int currentFrame = -1;
  double pulsePosition[3] = {};
  Eigen::Quaterniond pulseRotation;
  double jointsAngles[20] = {};
  double bonesSize[21] = {};

  std::vector<double> IKWeights;
  std::vector<std::pair<cv::Point2d, double>> observedHand2dPose;
  ceres::Problem problem;
  ceres::CostFunction* costFunction;
  ceres::CostFunction* costFunction2;
  ceres::ResidualBlockId residualBlockId;
  ceres::Solver::Options options;
  ceres::Solver::Summary summary;
};

