#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <iostream>

#include <NVI/utilities/Debug.h>

#include <NVI/core/OpenPose.h>
#include <NVI/core/InverseKinematics.h>
#include <NVI/core/Utils.h>

#include <NVI/core/ImageProcessor.h>

using namespace std;
using namespace cv;

ImageProcessor::ImageProcessor() {
  for (int i = 0; i < 42; i++) {
    filters.push_back(OneEuroFilter(10, 0.5, 0.007));
  }
}

void ImageProcessor::resetPose() {
  inverseKinematics.resetPose();
}

vector<double> ImageProcessor::getPoseAngles() {
  if (lostTrack) return vector<double>();
  return inverseKinematics.getAngles();
}

vector<Point2d> ImageProcessor::getPose2d() {
  if (lostTrack) return vector<Point2d>();

  auto points = inverseKinematics.get2dPoints();
  for (auto &point : points) {
    point = Point(point.x + imageWidth / 2, point.y + imageHeight / 2);
  }

  return points;
}

vector<Point3d> ImageProcessor::getPose3d() {
  if (lostTrack) return vector<Point3d>();
  return inverseKinematics.get3dPoints();
}

double ImageProcessor::getMeanError() {
  if (lostTrack && inverseKinematics.getFrame() == -1) return 0;
  auto observedPoints = keyPoints;
  auto estimatedPoints = inverseKinematics.get2dPoints();

  double meanSquaredError = 0;
  for (int i = 0; i < observedPoints.size(); i++) {
    Point2d error = Point2d(observedPoints[i].first) - estimatedPoints[i];
    meanSquaredError += norm(error)/21;
  }

  return meanSquaredError;
}

bool ImageProcessor::getLostTrack() {
  return lostTrack;
}

void ImageProcessor::trackKeyPoints(Mat &img, Point2d offset) {
  if (prevKeyPoints.empty()) {
    return;
  }

  vector<Point2f> prevPoints, nextPoints;
  vector<uchar> status;
  vector<float> err;

  for (int i = 0; i < prevKeyPoints.size(); i++) {
    const auto &keyPoint = prevKeyPoints[i];

    if (keyPoint.second < CONFIDENCE_TRASHOLD) {
      prevPoints.push_back(keyPoints[i].first + offset);
      continue;
    }
    prevPoints.push_back(keyPoint.first + offset);
  }

  for (const auto &keyPoint : keyPoints)
    nextPoints.push_back(keyPoint.first + offset);

  calcOpticalFlowPyrLK(
    prevGrayFrame,
    grayFrame,
    prevPoints,
    nextPoints,
    status,
    err,
    Size(50, 50),
    2,
    TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01),
    OPTFLOW_USE_INITIAL_FLOW
  );

  for (int i = 0; i < keyPoints.size(); i++) {
    const double dist = norm(keyPoints[i].first - (Point2d(nextPoints[i]) - offset));
    if (status[i] == 0 || err[i] > 5 || dist > 15) {
      continue;
    }
    keyPoints[i].first = Point2d(nextPoints[i]) - offset;
  }
}

void ImageProcessor::filterKeyPoints() {
  for (int i = 0; i < 21; i++) {
    keyPoints[i].first.x = filters[2*i](keyPoints[i].first.x);
    keyPoints[i].first.y = filters[2*i + 1](keyPoints[i].first.y);
  }
}

void ImageProcessor::processImage(Mat &img) {
  currentFrame++;
  imageWidth = img.cols;
  imageHeight = img.rows;

  if (USE_LK) {
    cvtColor(img, grayFrame, COLOR_BGR2GRAY);
  }

  Rect roi = Utils::createBoundBox(keyPoints, getPose2d(), img, lostTrack);

  if (lostTrack) {
    if (debug_mode) {
      imshow("Display window", img);
    }
    return;
  }

  Mat croppedImg = img(roi);

  Point offset(roi.x - imageWidth / 2, roi.y - imageHeight / 2);

  if (currentFrame % RNN_FRAME_INTERVAL == 0) {
    keyPoints = pose2dProcessor.ProcessImage(croppedImg, offset);
  } else {
    keyPoints = prevKeyPoints;
  }

  int confidentPoints = 0;
  for (auto keyPoint : keyPoints) {
    confidentPoints += keyPoint.second > CONFIDENCE_TRASHOLD;
  }

  lostTrack = lostTrack || confidentPoints == 0;

  if (lostTrack) {
    if (debug_mode) {
      imshow("Display window", img);
    }
    return;
  }

  if (USE_LK) {
    trackKeyPoints(img, Point(roi.x + roi.width / 2, roi.y + roi.width / 2));
    prevGrayFrame = grayFrame;
  }

  if (USE_ONE_EURO) {
    filterKeyPoints();
  }

  inverseKinematics.run(keyPoints);

  prevKeyPoints = keyPoints;

  if (debug_mode) {
    Debug::addPointToGraph("average", getMeanError());
    auto keyPoints2d = getPose2d();

    for (int i = 0; i < keyPoints.size(); i++) {
      auto keyPoint = keyPoints[i];

      if (keyPoint.second > CONFIDENCE_TRASHOLD)
        circle(img, Point(keyPoint.first.x + imageWidth / 2, keyPoint.first.y + imageHeight / 2), 2, Scalar(255, 0, 0), FILLED);
      else
        circle(img, Point(keyPoint.first.x + imageWidth / 2, keyPoint.first.y + imageHeight / 2), 2, Scalar(0, 0, 255), FILLED);
    }

    Utils::drawBox(img, roi);

    for (int i = 0; i < keyPoints2d.size(); i++) {
      auto keyPoint = keyPoints2d[i];

      Debug::addPointToGraph("jointX" + to_string(i), keyPoint.x);
      Debug::addPointToGraph("jointY" + to_string(i), keyPoint.y);
      circle(img, keyPoint, 2, Scalar(0, 255, 0), FILLED);
    }

    imshow("Display window", img);
  }
}
