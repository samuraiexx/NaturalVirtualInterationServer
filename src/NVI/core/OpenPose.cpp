#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/dnn/layer.details.hpp>  // CV_DNN_REGISTER_LAYER_CLASS

#include <NVI/utilities/Debug.h>

#include <NVI/core/OpenPose.h>

using namespace cv;
using namespace cv::dnn;
using namespace std;

OpenPose::OpenPose()
{
  String modelTxt = samples::findFile(modelProtoPath);
  String modelBin = samples::findFile(modelBinPath);

  net = readNet(modelBin, modelTxt);
  net.setPreferableBackend(DNN_BACKEND_CUDA);
  net.setPreferableTarget(DNN_TARGET_CUDA);
}

vector<pair<Point2d, double>> OpenPose::ProcessImage(Mat &img, Point offset) {
  flip(img, img, 1);
  Mat inputBlob = blobFromImage(img, 0.003922, Size(CNN_W, CNN_H), Scalar(0, 0, 0), false, false);
  flip(img, img, 1);
  net.setInput(inputBlob);
  Mat result = net.forward();

  auto points = GetPointsFromHeatmap(result, img);
  for (auto &point : points) {
    point.first.x = img.cols - point.first.x + offset.x;
    point.first.y = point.first.y + offset.y;
  }

  return points;
}
