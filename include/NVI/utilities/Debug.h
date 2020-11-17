#pragma once
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <map>

extern bool debug_mode;

#define db(x) if(debug_mode) std::cout << #x << " = " << x << std::endl
#define dbs(x) if(debug_mode) std::cout << x << std::endl
#define _ << ", " <<

namespace Debug {
  void addPointToGraph(std::string id, double y);
  void plotGraph(std::string id);
}
