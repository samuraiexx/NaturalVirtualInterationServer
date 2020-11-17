#include <opencv2/plot.hpp>

#include <NVI/utilities/debug.h>

using namespace std;
using namespace cv;
using namespace plot;

map<string, Mat> graphsData;

void Debug::addPointToGraph(string id, double y) {
  if (!debug_mode) return;
  if (!graphsData.count(id)) {
    graphsData[id] = Mat(1, 0, CV_64F);
  }

  auto &graph = graphsData[id];
  graph.reshape(1, graph.cols);
  graph.push_back(y);
  graph.reshape(graph.rows);
}

void Debug::plotGraph(string id) {
  if (!debug_mode) return;
  const auto graphData = graphsData[id];
  double mn, mx;
  auto plot = Plot2d::create(graphData);

  minMaxLoc(graphData, &mn, &mx);
  Mat display;

  plot->setPlotAxisColor(Scalar(255, 255, 255));
  plot->setPlotLineColor(Scalar(227, 227, 255));
  plot->setMinY(mn - 0.2*(mx - mn));
  plot->setMaxY(mx + 0.2*(mx - mn));

  plot->render(display);
  imshow(id, display);
}
