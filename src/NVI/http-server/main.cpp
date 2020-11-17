#define CPPHTTPLIB_RECV_BUFSIZ size_t(500000u)

#include <iostream>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <httplib.h>
#include <nlohmann/json.hpp>

#include <NVI/decoder.h>
#include <NVI/utilities.h>
#include <NVI/core/Utils.h>
#include <NVI/core/ImageProcessor.h>

#include <NVI/http-server/JsonConvertions.h>

using namespace httplib;
using namespace cv;
using json = nlohmann::json;

bool debug_mode = false;

char rcvBuffer[CPPHTTPLIB_RECV_BUFSIZ];
char frameBuffer[CPPHTTPLIB_RECV_BUFSIZ];

ImageProcessor* imageProcessor;

void processFrameRequest(char* rcvBuffer, int total_len, Response& res, bool encoded);

int main(int argc, char* argv[]) {
  std::cout <<
    " Flags: --debug/-d --ip/-i --port/-p"
    " Default:\n"
    " HttpServer --ip 127.0.0.1 --port 3030\n"
    << std::endl;

  int debugArgIdx = Utils::findStringInArgv(argc, argv, "--debug", "-d");
  int ipArgIdx = Utils::findStringInArgv(argc, argv, "--ip", "-i");
  int portArgIdx = Utils::findStringInArgv(argc, argv, "--port", "-p");

  if (debugArgIdx > 0) {
    debug_mode = true;
  }

  Server server;

  Decoder::vpx_init();
  imageProcessor = new ImageProcessor();

  server.Put("/processFrame", [&](const Request& req, Response& res, const ContentReader& content_reader) {
    int total_len = get_header_value_uint64(req.headers, "Content-Length", 0);
    char* rcvPtr = rcvBuffer;

    content_reader([&](const char* data, size_t data_length) {
      memcpy(rcvPtr, data, data_length);
      rcvPtr += data_length;

      if (rcvBuffer + total_len == rcvPtr) {
        bool encoded = req.get_header_value_count("Encoded") > 0;
        processFrameRequest(rcvBuffer, total_len, res, encoded);
      }
      return true;
    });
  });

  std::string ip = "127.0.0.1";
  int port = 3030;

  if (ipArgIdx > 0) ip = argv[ipArgIdx + 1];
  if (portArgIdx > 0) port = std::stoi(argv[portArgIdx + 1]);

  server.listen(ip.c_str(), port);
}

void processFrameRequest(char* rcvBuffer, int total_len, Response& res, bool encoded) {
  int width, height;
  Mat img;

  if (encoded) {
    bool successful = Decoder::vpx_decode(rcvBuffer, total_len, frameBuffer, width, height);

    if (!successful) {
      res.status = 400;
      return;
    }
    Mat imgYV12 = Mat(height * 3 / 2, width, CV_8UC1, frameBuffer);
    cvtColor(imgYV12, img, COLOR_YUV2BGR_YV12);
  } else {
    std::vector<uchar> v;
    v.assign(rcvBuffer, rcvBuffer + total_len);
    img = imdecode(v, IMREAD_COLOR);
  }

  imageProcessor->processImage(img);

  const bool lostTrack = imageProcessor->getLostTrack();

  std::string body = json({
    {"lostTrack", imageProcessor->getLostTrack()},
    {"points2d", json(imageProcessor->getPose2d())},
    {"points3d", json(imageProcessor->getPose3d())},
    {"angles", json(imageProcessor->getPoseAngles())}
    }).dump();

  res.set_content(body, "application/json");

  if (debug_mode) {
    waitKey(0);
  }

  if (lostTrack) {
    dbs("Lost track...");
    imageProcessor->resetPose();
  }
}

