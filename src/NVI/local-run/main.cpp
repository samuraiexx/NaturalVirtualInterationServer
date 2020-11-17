#include <opencv2/highgui.hpp>
#include <time.h>
#include <iostream>

#include <NVI/utilities/Debug.h>

#include <NVI/core/Utils.h>
#include <NVI/core/ImageProcessor.h>
#include <NVI/core/HandJoints.hpp>

using namespace std;
using namespace cv;

bool debug_mode = false;

int main(int argc, char *argv[]) {
  cout << " Flags: --debug/-d --cam/-c --file/-f --ip/-i --port/-p"
          " examples:\n"
          " NeuralVirtualInteractionServer --cam 0\n"
          " NeuralVirtualInteractionServer --file Bolt/img/%d.jpg\n"
          " NeuralVirtualInteractionServer --debug --file Bolt/img/%04d.jpg\n"
       << endl;

  int debugArgIdx = Utils::findStringInArgv(argc, argv, "--debug", "-d");
  int camArgIdx = Utils::findStringInArgv(argc, argv, "--cam", "-c");
  int fileArgIdx = Utils::findStringInArgv(argc, argv, "--file", "-f");

  if (debugArgIdx > 0) {
    debug_mode = true;
  }

  VideoCapture cap;
  Mat img;
  int currentFrame = 0;

  if (camArgIdx > 0) {
    cap.open(stoi(argv[camArgIdx + 1]));
    waitKey(100);
  }
  else if (fileArgIdx > 0) {
    cap.open(argv[fileArgIdx + 1]);
  } else {
    cout << "No input was specified." << endl;
    return 0;
  }

  ImageProcessor imageProcessor;
  clock_t tStart = clock();
  double totalMeanResidualValue = 0;

  while (cap >> img, !img.empty()) {
    clock_t tStart = clock();

    imageProcessor.processImage(img);
    totalMeanResidualValue += imageProcessor.getMeanError();

    int waitTime = 33 - 1000 * (clock() - tStart) / CLOCKS_PER_SEC;
    waitKey(max(waitTime, 1));

    currentFrame++;
  }
  totalMeanResidualValue /= currentFrame;

  Debug::plotGraph("jointX0");
  Debug::plotGraph("jointY0");
  Debug::plotGraph("jointX8");
  Debug::plotGraph("jointY8");
  waitKey(0);
  printf("Time taken: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
  printf("Mean Residual Value (average across images): %.6lf\n", totalMeanResidualValue);

  return 0;
}