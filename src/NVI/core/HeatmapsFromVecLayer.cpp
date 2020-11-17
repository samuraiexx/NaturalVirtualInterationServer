#include <vector>
#include <opencv2/dnn.hpp>

#include <NVI/core/HeatmapsFromVecLayer.h>

using namespace std;
using namespace cv::dnn;
using namespace cv;

HeatmapsFromVecLayer::HeatmapsFromVecLayer(const LayerParams &params) : Layer(params) {
  range_ = 1.5f;
  heatmap_size_ = 32;
  kernel_size_ = 3;
  gaussian_.resize((kernel_size_ + 1)*(kernel_size_ + 1)); // bottom-right quarter Gaussian values

                                                           // un-normalized Gaussian (bottom-right quarter) in pixel space
  for (int k_r = 0; k_r <= kernel_size_; k_r++)
  {
    for (int k_c = 0; k_c <= kernel_size_; k_c++)
    {
      int linID = k_r * (kernel_size_ + 1) + k_c;
      gaussian_[linID] = exp(-0.5 * (k_r*k_r + k_c*k_c));
    }
  }
}

Ptr<Layer> HeatmapsFromVecLayer::create(LayerParams& params) {
  return Ptr<Layer>(new HeatmapsFromVecLayer(params));
}

bool HeatmapsFromVecLayer::getMemoryShapes(
  const vector<vector<int>> &inputs,
  const int requiredOutputs,
  vector<vector<int>> &outputs,
  vector<vector<int>> &internals
) const {
  CV_UNUSED(requiredOutputs); CV_UNUSED(internals);

  vector<int> outShape = inputs[0];
  outShape[2] = heatmap_size_;
  outShape[3] = heatmap_size_;

  outputs.assign(1, outShape);
  return false;
}

void HeatmapsFromVecLayer::forward(
  InputArrayOfArrays inputs_arr,
  OutputArrayOfArrays outputs_arr,
  OutputArrayOfArrays internals_arr
) {
  std::vector<cv::Mat> inputs, outputs;
  inputs_arr.getMatVector(inputs);
  outputs_arr.getMatVector(outputs);

  const float* bottom_data = (float*)inputs[0].data;
  float* top_data = (float*)outputs[0].data;

  const int num = inputs[0].size[0];
  const int num_vecs = inputs[0].size[1];
  const int inpHeight = inputs[0].size[2];
  const int inpWidth = inputs[0].size[3];

  vector<int> proj_vecs(2 * num_vecs);

  const int example_size = num_vecs * inpHeight * inpWidth;

  int u_id, v_id; // top-left corner is 0,0
                  // x -> left to right, y -> top to bottom, assuming ortographic camera for the moment
  int top_step = heatmap_size_ * heatmap_size_;

  for (int n = 0; n < num; n++)
  {
    for (int j = 0; j < num_vecs; j++)
    {
      u_id = (int)round((bottom_data[n * (3 * num_vecs) + j * 3] + range_) / (2 * range_) * (heatmap_size_ - 1));
      v_id = (int)round((bottom_data[n * (3 * num_vecs) + j * 3 + 1] + range_) / (2 * range_) * (heatmap_size_ - 1));
      proj_vecs[j * 2] = u_id;
      proj_vecs[j * 2 + 1] = v_id;

      for (int w = -kernel_size_; w <= kernel_size_; w++)
      {
        for (int h = -kernel_size_; h <= kernel_size_; h++)
        {
          if (u_id + w >= 0 && u_id + w < heatmap_size_ && v_id + h >= 0 && v_id + h < heatmap_size_)
          { // in this case, put Gaussian in top blob centered at u_id, v_id
            top_data[n * example_size + j * top_step + (v_id + h) * heatmap_size_ + (u_id + w)] = gaussian_[abs(h) * (kernel_size_ + 1) + abs(w)];
          }
        }
      }
    }
  }
}
