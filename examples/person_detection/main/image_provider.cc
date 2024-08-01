
/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_spi_flash.h"
#include "esp_system.h"
#include "esp_timer.h"

#include "app_camera_esp.h"
#include "esp_camera.h"
#include "model_settings.h"
#include "image_provider.h"
#include "esp_main.h"

static const char *TAG = "app_camera";

static uint16_t *display_buf; // buffer to hold data to be sent to display

// Get the camera module ready
TfLiteStatus InitCamera()
{
#if CLI_ONLY_INFERENCE
  ESP_LOGI(TAG, "CLI_ONLY_INFERENCE enabled, skipping camera init");
  return kTfLiteOk;
#endif
// if display support is present, initialise display buf
#if DISPLAY_SUPPORT
  if (display_buf == NULL)
  {
    // Size of display_buf:
    // Frame 96x96 from camera is extrapolated to 192x192. RGB565 pixel format -> 2 bytes per pixel
    display_buf = (uint16_t *)heap_caps_malloc(96 * 2 * 96 * 2 * sizeof(uint16_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  }
  if (display_buf == NULL)
  {
    ESP_LOGE(TAG, "Couldn't allocate display buffer");
    return kTfLiteError;
  }
#endif // DISPLAY_SUPPORT

#if ESP_CAMERA_SUPPORTED
  int ret = app_camera_init();
  if (ret != 0)
  {
    MicroPrintf("Camera init failed\n");
    return kTfLiteError;
  }
  MicroPrintf("Camera Initialized\n");
#else
  ESP_LOGE(TAG, "Camera not supported for this device");
#endif
  return kTfLiteOk;
}

void *image_provider_get_display_buf()
{
  return (void *)display_buf;
}

// Get an image from the camera module
TfLiteStatus GetImage(int image_width, int image_height, int channels, float *image_data)
{
#if ESP_CAMERA_SUPPORTED
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb)
  {
    ESP_LOGE(TAG, "Camera capture failed");
    return kTfLiteError;
  }

  // We have initialised camera to grayscale
  // Just quantize to int8_t

  for (int i = 0; i < image_width * image_height; i++)
  {
    image_data[i] = ((float) fb->buf[i]) / 255;
  //  MicroPrintf("pixel %i = %f\n",i, image_data[i]);
  }

  esp_camera_fb_return(fb);
  /* here the esp camera can give you grayscale image directly */
  return kTfLiteOk;
#else
  return kTfLiteError;
#endif
}
