
namespace Decoder {
  bool vpx_init();
  bool vpx_decode(const char* encoded, int frame_size, char* yv12_frame, int &width, int &height);
  void vpx_cleanup();
}
