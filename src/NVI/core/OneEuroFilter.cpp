#include <NVI/core/OneEuroFilter.h>

#include <cmath>

const double PI = acos(-1);

LowPassFilter::LowPassFilter() : hatxprev(0), xprev(0), hadprev(false) {}
double LowPassFilter::operator()(double x, double alpha) {
  double hatx;
  if (hadprev) {
    hatx = alpha * x + (1 - alpha) * hatxprev;
  }
  else {
    hatx = x;
    hadprev = true;
  }
  hatxprev = hatx;
  xprev = x;
  return hatx;
}

OneEuroFilter::OneEuroFilter(double _freq, double _mincutoff, double _beta, double _dcutoff)
  : freq(_freq), mincutoff(_mincutoff), beta(_beta), dcutoff(_dcutoff), last_time_(-1) {}

double OneEuroFilter::operator()(double x) {
  double dx = 0;
  double t = tickMeter.getTimeSec();

  if (last_time_ != -1 && t != last_time_) {
    freq = 1.0 / (t - last_time_);
  }
  last_time_ = t;

  if (xfilt_.hadprev)
    dx = (x - xfilt_.xprev) * freq;

  double edx = dxfilt_(dx, alpha(dcutoff));
  double cutoff = mincutoff + beta * std::abs(static_cast<double>(edx));
  return xfilt_(x, alpha(cutoff));
}

double OneEuroFilter::alpha(double cutoff) {
  double tau = 1.0 / (2 * PI * cutoff);
  double te = 1.0 / freq;
  return 1.0 / (1.0 + tau / te);
}
