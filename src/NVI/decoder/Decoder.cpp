#define VPX_CODEC_DISABLE_COMPAT 1
#include <vpx/vpx_decoder.h>
#include <vpx/vp8dx.h>
#include <stdio.h>
#include <string.h>

#include <NVI/decoder/Decoder.h>

static vpx_codec_ctx_t *codec = nullptr;

namespace Decoder {
  bool vpx_init() {
    if (codec != nullptr) vpx_cleanup();

    codec = new vpx_codec_ctx_t();
    vpx_codec_iface_t* iface = vpx_codec_vp8_dx();

    printf("Using %s\n", vpx_codec_iface_name(iface));

    // Initialize codec
    int flags = 0;
    if (vpx_codec_dec_init(codec, iface, NULL, flags)) {
      printf("Failed to initialize decoder\n");
      return false;
    }

    return true;
  }

  bool vpx_decode(const char* encoded, int frame_size, char* yv12_frame, int& width, int& height) {
    // Decode the frame
    if (vpx_codec_decode(codec, (const unsigned char*)encoded, frame_size, NULL, 0)) {
      printf("Failed to decode frame (maybe a key frame has not been reached)\n");
      return false;
    }

    // Write decoded data to yv12_frame
    vpx_codec_iter_t iter = NULL;
    vpx_image_t* img;
    int              total = 0;
    while ((img = vpx_codec_get_frame(codec, &iter))) {
      width = img->d_w;
      height = img->d_h;

      for (int plane = 0; plane < 3; plane++) {
        unsigned char* buf = img->planes[plane];
        for (int y = 0; y < (plane ? (img->d_h + 1) >> 1 : img->d_h); y++) {
          int len = (plane ? (img->d_w + 1) >> 1 : img->d_w);
          memcpy(yv12_frame + total, buf, len);
          buf += img->stride[plane];
          total += len;
        }
      }
    }

    return true;
  }

  void vpx_cleanup() {
    if (codec == nullptr) return;

    if (vpx_codec_destroy(codec)) printf("Failed to destroy codec\n");
    codec = nullptr;
  }
}