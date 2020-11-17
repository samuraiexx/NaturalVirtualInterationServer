// Code from https://github.com/ngocdaothanh/vpxcam

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

#define VPX_CODEC_DISABLE_COMPAT 1
#include <vpx/vpx_encoder.h>
#include <vpx/vp8cx.h>

#include <NVI/encoder/Encoder.h>

// RGB -> YUV
#define RGB2Y(R, G, B) ( (  66 * (R) + 129 * (G) +  25 * (B) + 128) >> 8) +  16
#define RGB2U(R, G, B) ( ( -38 * (R) -  74 * (G) + 112 * (B) + 128) >> 8) + 128
#define RGB2V(R, G, B) ( ( 112 * (R) -  94 * (G) -  18 * (B) + 128) >> 8) + 128

static char buffer[(int)3e6];

static int width, height;
static int             frame_size;
static vpx_codec_ctx_t codec;
static vpx_image_t     raw;
static int             frame_cnt = 0;

int vpx_init(int _width, int _height) {
  width = _width, height = _height;
  frame_size = (int)(width * height * 1.5);

  vpx_codec_iface_t* interface = vpx_codec_vp8_cx();
  printf("Using %s\n", vpx_codec_iface_name(interface));

  if (!vpx_img_alloc(&raw, VPX_IMG_FMT_I420, width, height, 1)) {
    printf("Failed to allocate image %d x %d\n", width, height);
    return false;
  }

  vpx_codec_enc_cfg_t cfg;

  // Populate encoder configuration
  vpx_codec_err_t res = vpx_codec_enc_config_default(interface, &cfg, 0);
  if (res) {
    printf("Failed to get config: %s\n", vpx_codec_err_to_string(res));
    return false;
  }

  // Update the default configuration with our settings
  cfg.rc_target_bitrate = width * height * cfg.rc_target_bitrate / cfg.g_w / cfg.g_h;
  cfg.g_w = width;
  cfg.g_h = height;

  // Initialize codec
  if (vpx_codec_enc_init(&codec, interface, &cfg, 0)) {
    printf("Failed to initialize encoder\n");
  }

  return true;
}

void rgb2yv12(const unsigned char* rgb_frame, unsigned char* yv12_frame) {
  int size = width * height;

  unsigned char* yPtr = yv12_frame;
  unsigned char* vPtr = yv12_frame + size;
  unsigned char* uPtr = yv12_frame + 5 * size / 4;

  unsigned const char *currLine = rgb_frame;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int R = *currLine++;
      int G = *currLine++;
      int B = *currLine++;

      *yPtr++ = RGB2Y(R, G, B);
    }
  }

  currLine = rgb_frame;
  const unsigned char *nextLine = rgb_frame + 3 * width;

  for (int i = 0; i < height; i += 2) {
    for (int j = 0; j < width; j += 2) {
      int R = 0;
      int G = 0;
      int B = 0;

      for (int k = 0; k < 2; k++) {
        R += *currLine++;
        G += *currLine++;
        B += *currLine++;
      }

      for (int k = 0; k < 2; k++) {
        R += *nextLine++;
        G += *nextLine++;
        B += *nextLine++;
      }

      R /= 4;
      G /= 4;
      B /= 4;

      *vPtr++ = RGB2V(R, G, B);
      *uPtr++ = RGB2U(R, G, B);
    }
    currLine += 3 * width;
    nextLine += 3 * width;
  }
}

int vpx_encode(const char* rgb_frame, char* encoded, bool force_key_frame) {
  rgb2yv12((unsigned char*)rgb_frame, (unsigned char*)buffer);
  char* yv12_frame = buffer;

  int size = 0;
  int flags = force_key_frame ? VPX_EFLAG_FORCE_KF : 0;

  // This does not work correctly (only the 1st plane is encoded), why?
  // raw.planes[0] = (unsigned char *) yv12_frame;
  memcpy(raw.planes[0], yv12_frame, frame_size);
  if (vpx_codec_encode(&codec, &raw, frame_cnt, 1, flags, VPX_DL_REALTIME)) {
    printf("Failed to encode frame\n");
    return size;
  }

  vpx_codec_iter_t iter = NULL;
  const vpx_codec_cx_pkt_t* pkt;
  while ((pkt = vpx_codec_get_cx_data(&codec, &iter))) {
    if (pkt->kind == VPX_CODEC_CX_FRAME_PKT) {
      if (!size) {
        size = pkt->data.frame.sz;
        memcpy(encoded, pkt->data.frame.buf, pkt->data.frame.sz);
      }

      bool key_frame = pkt->data.frame.flags & VPX_FRAME_IS_KEY;
      printf(key_frame ? "K" : ".");
      fflush(stdout);
    }
  }
  frame_cnt++;

  return size;
}

void vpx_cleanup() {
  printf("Processed %d frames\n", frame_cnt);

  vpx_img_free(&raw);

  if (vpx_codec_destroy(&codec)) {
    printf("Failed to destroy codec\n");
  }
}