#include <string>
#include <opencv2/imgproc.hpp>

#include <NVI/utilities/Debug.h>

#include <NVI/core/YoloHandFinder.h>
#include <NVI/core/Utils.h>

using namespace cv;
using namespace std;
using namespace dnn;

YoloHandFinder* yolo;

vector<String> getOutputsNames(const Net& net);

YoloHandFinder::YoloHandFinder() {
  string model = samples::findFile(modelPath);
  string weights = samples::findFile(weightsPath);

  net = readNetFromDarknet(model, weights);
  net.setPreferableBackend(DNN_BACKEND_CUDA);
  net.setPreferableTarget(DNN_TARGET_CUDA);
}

YoloHandFinder* YoloHandFinder::getYolo() {
  if (yolo == nullptr) yolo = new YoloHandFinder();

  return yolo;
}

Rect YoloHandFinder::getHandRoi(const Mat &frame) {
  Mat blob;
  // Create a 4D blob from a frame.
  blobFromImage(frame, blob, 1 / 255.0, Size(CNN_W, CNN_H), Scalar(0, 0, 0), true, false);

  //Sets the input to the network
  net.setInput(blob);

  // Runs the forward pass to get output of the output layers
  vector<Mat> outs;
  net.forward(outs, getOutputsNames(net));
  vector<Rect> boxes = postProcess(frame, outs);

  if (boxes.size() == 0) {
    return Rect();
  }

  Rect box = boxes[0];

  return box;
}

// Remove the bounding boxes with low confidence using non-maxima suppression
vector<Rect> YoloHandFinder::postProcess(const Mat &frame, const vector<Mat>& outs)
{
  vector<int> classIds;
  vector<float> confidences;
  vector<Rect> boxes;

  for (size_t i = 0; i < outs.size(); ++i)
  {
    // Scan through all the bounding boxes output from the network and keep only the
    // ones with high confidence scores. Assign the box's class label as the class
    // with the highest score for the box.
    float* data = (float*)outs[i].data;
    for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
    {
      Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
      Point classIdPoint;
      double confidence;
      // Get the value and location of the maximum score
      minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
      if (confidence > confThreshold)
      {
        int centerX = (int)(data[0] * frame.cols);
        int centerY = (int)(data[1] * frame.rows);
        int width = (int)(data[2] * frame.cols);
        int height = (int)(data[3] * frame.rows);
        int left = centerX - width / 2;
        int top = centerY - height / 2;

        classIds.push_back(classIdPoint.x);
        confidences.push_back((float)confidence);
        boxes.push_back(Rect(left, top, width, height));
      }
    }
  }

  // Perform non maximum suppression to eliminate redundant overlapping boxes with
  // lower confidences
  vector<int> indices;
  NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

  return boxes;
}

vector<String> getOutputsNames(const Net& net) {
  static vector<String> names;
  if (names.empty())
  {
    //Get the indices of the output layers, i.e. the layers with unconnected outputs
    vector<int> outLayers = net.getUnconnectedOutLayers();

    //get the names of all the layers in the network
    vector<String> layersNames = net.getLayerNames();

    // Get the names of the output layers in names
    names.resize(outLayers.size());
    for (size_t i = 0; i < outLayers.size(); ++i)
      names[i] = layersNames[outLayers[i] - 1];
  }
  return names;
}
