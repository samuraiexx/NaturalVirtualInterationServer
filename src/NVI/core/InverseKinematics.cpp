#include <iostream>
#include <fstream>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <NVI/utilities/Debug.h>

#include <NVI/core/HandJoints.hpp>
#include <NVI/core/InverseKinematics.h>

using namespace std;
using namespace cv;
using namespace Eigen;

using ceres::Solver;
using ceres::LocalParameterization;
using ceres::EigenQuaternionParameterization;

const double PI = acos(-1);

template<typename T>
vector<Matrix<T, 2, 1>> projectPoints(vector<Matrix<T, 3, 1>> points);

class CostFunctor {
private:
  double *bonesSize;
  vector<double> weights;
  vector<pair<Point2d, double>> *observedHand2dPose;

public:
  template <typename T>
  bool operator()(
    const T* const pulsePosition,
    const T* const angles,
    const T* const rotationPtr,
    T* residuals
    ) const {
    Eigen::Map<const Eigen::Quaternion<T> > rotation(rotationPtr);
    HandJoints<T> handJoints(bonesSize);
    vector<pair<Point2d, double>> &observedHand2dPose = *this->observedHand2dPose;

    Matrix<T, 3, 1> offset;
    offset << pulsePosition[0], pulsePosition[1], pulsePosition[2];

    vector<EulerAngle<T>> eulerAngles = HandJoints<T>::GetEulerAnglesFromAnglesArray(angles);
    vector<Matrix<T, 2, 1>> predictedPoints = projectPoints(handJoints.forwardKinematics(offset, eulerAngles, rotation));

    for (int i = 0; i < predictedPoints.size(); i++) {
      // Ignores Points with low confidence
      const T confidence = T(observedHand2dPose[i].second);
      if (observedHand2dPose[i].second < CONFIDENCE_TRASHOLD) {
        residuals[2 * i] = T(0);
        residuals[2 * i + 1] = T(0);
        continue;
      }

      T errorX = predictedPoints[i](0, 0) - T(observedHand2dPose[i].first.x);
      T errorY = predictedPoints[i](1, 0) - T(observedHand2dPose[i].first.y);

      residuals[2 * i] = errorX * confidence*confidence*confidence * T(weights[i]);
      residuals[2 * i + 1] = errorY * confidence*confidence*confidence * T(weights[i]);
    }

    return true;
  }

  CostFunctor(
    double *bonesSize,
    vector<double> weights,
    vector<pair<Point2d, double>> *observedHand2dPose
  ) :
    bonesSize(bonesSize),
    weights(weights),
    observedHand2dPose(observedHand2dPose)
  {}
};

void InverseKinematics::resetPose() {
  pulseRotation = Quaterniond(AngleAxis<double>(PI / 2, Eigen::Vector3d(1, 0, 0)));
  pulsePosition[2] = -490;
  pulsePosition[0] = -50;
  pulsePosition[1] = 120;

  for (int i = 0; i < 20; i++) {
    jointsAngles[i] = 0;
  }
}

InverseKinematics::InverseKinematics(
  int adjustmentFrames
) : adjustmentFrames(adjustmentFrames) {
  pulseRotation = Quaterniond(AngleAxis<double>(PI / 2, Eigen::Vector3d(1, 0, 0)));
  pulsePosition[2] = -490;
  pulsePosition[0] = -50;
  pulsePosition[1] = 120;

  // Change it to your own path
  std::ifstream ifs("IKWeights.in", std::ifstream::in);

  IKWeights = vector<double>(25, 0.0);

  for (int i = 0; i < 21; i++) {
    if (ifs.is_open()) {
      double weight;
      ifs >> weight;
      IKWeights[i] = weight;
    }
  }

  ifs.close();

  costFunction = new ceres::AutoDiffCostFunction<CostFunctor, 42, 3, 20, 4>(
    new CostFunctor(
      bonesSize,
      IKWeights,
      &observedHand2dPose
    )
    );

  residualBlockId = problem.AddResidualBlock(
    costFunction,
    NULL,
    pulsePosition,
    jointsAngles,
    pulseRotation.coeffs().data()
  );

  LocalParameterization* quaternionLocalParameterization =
    new EigenQuaternionParameterization;
  problem.SetParameterization(pulseRotation.coeffs().data(), quaternionLocalParameterization);

  // Set Parameters Upper and Lower bounds
  problem.SetParameterUpperBound(pulsePosition, 2, 0);

  for (int i = 0; i < 20; i++) {
    if (i % 4 == 1) {
      problem.SetParameterLowerBound(jointsAngles, i, -PI / 6);
      problem.SetParameterUpperBound(jointsAngles, i, PI / 6);
      continue;
    }

    problem.SetParameterLowerBound(jointsAngles, i, 0);
    problem.SetParameterUpperBound(jointsAngles, i, PI / 2);
  }

  // Solve
  options.num_threads = 8;
  // options.linear_solver_type = ceres::DENSE_QR;
  options.logging_type = ceres::SILENT;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
}

void InverseKinematics::run(
  const vector<pair<Point2d, double>> &observedHand2dPose
) {
  currentFrame++;

  if (currentFrame < adjustmentFrames) {
    double frameBonesSize[21];
    HandJoints<double>::GetBonesSizeFromHandPoints(observedHand2dPose, frameBonesSize);

    for (int i = 0; i < 21; i++) {
      bonesSize[i] = (currentFrame*bonesSize[i] + frameBonesSize[i])/(currentFrame + 1);
    }
  }

  this->observedHand2dPose = observedHand2dPose;
  ceres::Solve(options, &problem, &summary);
}

vector<double> InverseKinematics::getAngles() {
  vector<double> angles({pulseRotation.x(), pulseRotation.y(), pulseRotation.z(), pulseRotation.w()});
  for (int i = 0; i < 20; i++)
    angles.push_back(jointsAngles[i]);

  return angles;
}

std::vector<cv::Point3d> InverseKinematics::get3dPoints() {
  Matrix<double, 3, 1> offset;
  offset << pulsePosition[0], pulsePosition[1], pulsePosition[2];

  auto eulerAngles = HandJoints<double>::GetEulerAnglesFromAnglesArray(jointsAngles);
  HandJoints<double> handJoints(bonesSize);

  auto jointsPosition = handJoints.forwardKinematics(offset, eulerAngles, pulseRotation);
  vector<Point3d> cvJointsPosition(21);

  for (int i = 0; i < jointsPosition.size(); i++) {
    cvJointsPosition[i].x = jointsPosition[i](0);
    cvJointsPosition[i].y = jointsPosition[i](1);
    cvJointsPosition[i].z = jointsPosition[i](2);
  }

  return cvJointsPosition;
}

std::vector<cv::Point2d> InverseKinematics::get2dPoints() {
  auto points = get3dPoints();
  vector<Vector3d> eigen3dPoints;

  for (const auto &point : points) {
    Vector3d eigen3dPoint;
    eigen3dPoint << point.x, point.y, point.z;
    eigen3dPoints.push_back(eigen3dPoint);
  }

  auto projectedPoints = projectPoints(eigen3dPoints);
  vector<Point2d> cvProjectedPoints;

  for (const auto &projectedPoint : projectedPoints) {
    Point2d cvProjectedPoint;
    cvProjectedPoint.x = projectedPoint(0);
    cvProjectedPoint.y = projectedPoint(1);

    cvProjectedPoints.push_back(cvProjectedPoint);
  }

  return cvProjectedPoints;
}

template <typename T>
vector<Matrix<T, 2, 1>> projectPoints(vector<Matrix<T, 3, 1>> points) {
  vector<Matrix<T, 2, 1>> projectedPoints;

  for (const auto &point : points) {
    /* Based on https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html */
    Matrix<T, 2, 1> v1;
    Matrix<T, 2, 1> v2;

    // Tangencial Distortion
    const T p1 = T(-2.8880193013935932e-03);
    const T p2 = T(-1.2948094406435170e-03);
    // Radial Distortion
    const T k1 = T(1.8773590962971398e-01);
    const T k2 = T(-1.4102294060582985e+00);
    const T k3 = T(3.0596587859370281e+00);

    v1 << -point(0) / point(2), -point(1) / point(2);

    const T r2 = v1(0)*v1(0) + v1(1)*v1(1);
    const T k = T(1) + k1*r2 + k2*r2*r2 + k3*r2*r2*r2;

    v2 << v1(0)*k + T(2) * p1*v1(0)*v1(1) + p2*(r2 + T(2) * v1(0)*v1(0)),
      v1(1)*k + T(2) * p2*v1(0)*v1(1) + p1*(r2 + T(2) * v1(1)*v1(1));

    // cx == cy == 0
    const T fx = T(4.8779979884985454e+02);
    const T fy = T(4.9043008807801573e+02);

    Matrix<T, 2, 1> projectedPoint;
    projectedPoint << fx*v2(0), fy*v2(1);
    projectedPoints.push_back(projectedPoint);
  }

  return projectedPoints;
}

int InverseKinematics::getFrame() {
  return currentFrame;
}
