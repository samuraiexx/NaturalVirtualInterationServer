#include <opencv2/core/utility.hpp>

struct LowPassFilter {
  LowPassFilter();
  double operator()(double x, double alpha);
  double hatxprev;
  double xprev;
  bool hadprev;
};


struct OneEuroFilter {
  OneEuroFilter(double _freq, double _mincutoff, double _beta, double _dcutoff = 1); // Time in seconds
  double operator()(double x);

  double freq;
  double mincutoff, beta, dcutoff;
private:
  double alpha(double cutoff);

  double last_time_;
  cv::TickMeter tickMeter;
  LowPassFilter xfilt_, dxfilt_;
};
