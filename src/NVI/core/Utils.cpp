#include <iostream>

#include <NVI/utilities/debug.h>
#include <NVI/core/Utils.h>
#include <NVI/core/YoloHandFinder.h>

using namespace std;
using namespace cv;

namespace Utils {
  int findStringInArgv(int argc, char** argv, string s, string alt) {
    for (int i = 1; i < argc; i++) {
      if (argv[i] == s || argv[i] == alt) return i;
    }

    return -1;
  }

  void drawBox(Mat &img, Rect roi) {
    line(img, Point(roi.x, roi.y), Point(roi.x + roi.width, roi.y), Scalar(0, 0, 255));
    line(img, Point(roi.x, roi.y), Point(roi.x, roi.y + roi.height), Scalar(0, 0, 255));
    line(img, Point(roi.x + roi.width, roi.y + roi.height), Point(roi.x + roi.width, roi.y), Scalar(0, 0, 255));
    line(img, Point(roi.x + roi.width, roi.y + roi.height), Point(roi.x, roi.y + roi.height), Scalar(0, 0, 255));
  }

  Rect createBoundBox(const vector<pair<Point2d, double>> &rnnPose, const vector<Point2d> &ikPose, Mat &img, bool &lostTrack) {
    const int MIN_BOX_WIDTH = max(int(img.cols*0.375), int(img.rows*0.5));
    const int INF = 0x3f3f3f3f;
    Rect roi(INF, INF, 0, 0); // Region Of Interest

    for (int i = 0; i < rnnPose.size(); i++) {
      const auto &rnnPoint = rnnPose[i].first;

      roi.x = min({ (double)roi.x, rnnPoint.x + img.cols / 2});
      roi.y = min({ (double)roi.y, rnnPoint.y + img.rows / 2});
    }

    for (int i = 0; i < ikPose.size(); i++) {
      const auto &ikPoint = ikPose[i];

      roi.x = min({ roi.x, int(ikPoint.x) });
      roi.y = min({ roi.y, int(ikPoint.y) });
    }

    for (int i = 0; i < rnnPose.size(); i++) {
      const auto &rnnPoint = rnnPose[i].first;

      roi.width = max({ (double)roi.width, rnnPoint.x + img.cols / 2 - roi.x });
      roi.height = max({ (double)roi.height, rnnPoint.y + img.rows / 2 - roi.y });
    }

    for (int i = 0; i < ikPose.size(); i++) {
      const auto &ikPoint = ikPose[i];

      roi.width = max({ roi.width, int(ikPoint.x) - roi.x });
      roi.height = max({ roi.height, int(ikPoint.y) - roi.y });
    }

    if (roi.x == INF || roi.y == INF || lostTrack) { // No points were read so try to detect hand
      dbs("Using Yolo to get hand position...");
      YoloHandFinder &yolo = *YoloHandFinder::getYolo();
      roi = yolo.getHandRoi(img);

      if (roi.width == 0 || roi.height == 0) return Rect();
      lostTrack = false;
    }

    int boxWidth = max((int)(max(roi.width, roi.height)*1.25), MIN_BOX_WIDTH);
    boxWidth = min({ boxWidth, img.rows, img.cols });

    roi.x -= (boxWidth - roi.width) / 2;
    roi.y -= (boxWidth - roi.height) / 2;
    roi.width += boxWidth - roi.width;
    roi.height += boxWidth - roi.height;

    roi.x -= max(0, roi.x + roi.width - img.cols);
    roi.y -= max(0, roi.y + roi.height - img.rows);

    roi.x = max(roi.x, 0);
    roi.y = max(roi.y, 0);

    return roi;
  }
}
