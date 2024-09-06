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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "esp_log.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "main_functions.h"

#include "esp_spiffs.h"
#include "spi_flash_mmap.h"
#include <nvs_flash.h>

#include "esp_http_server.h"
#include "esp_wifi.h"
#include "esp_event.h"

#include "micro_model_settings.h"

// Wi-Fi credentials
#define WIFI_SSID "Room"
#define WIFI_PASS "AspireE15"

// HTTP GET handler to serve the image file
esp_err_t file_get_handler(httpd_req_t *req)
{
  FILE *f = fopen("/storage/dv.pgm", "r");
  if (f == NULL)
  {
    httpd_resp_send_404(req);
    return ESP_FAIL;
  }

  // Set the content type to PGM
  httpd_resp_set_type(req, "image/x-portable-graymap");

  // Write the PGM header
  char header[32];
  int header_len = snprintf(header, sizeof(header), "P5\n%d %d\n255\n", kFeatureSize , kFeatureCount);
  httpd_resp_send_chunk(req, header, header_len);

  // Read and send the file content in chunks
  char buffer[1024];
  size_t read_bytes;
  while ((read_bytes = fread(buffer, 1, sizeof(buffer), f)) > 0)
  {
    httpd_resp_send_chunk(req, buffer, read_bytes);
  }
  fclose(f);

  // Signal the end of the response
  httpd_resp_send_chunk(req, NULL, 0);
  return ESP_OK;
}

// Function to start the HTTP server
httpd_handle_t start_webserver(void)
{
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  httpd_handle_t server = NULL;
  if (httpd_start(&server, &config) == ESP_OK)
  {
    httpd_uri_t file_uri = {
        .uri = "/dv.jpg",
        .method = HTTP_GET,
        .handler = file_get_handler,
        .user_ctx = NULL};
    httpd_register_uri_handler(server, &file_uri);
  }
  return server;
}

// Wi-Fi event handler
static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data)
{
  if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START)
  {
    esp_wifi_connect();
  }
  else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED)
  {
    esp_wifi_connect();
  }
  else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP)
  {
    ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
    printf("got ip:" IPSTR, IP2STR(&event->ip_info.ip));
    start_webserver();
  }
}

// Function to initialize Wi-Fi
void wifi_init_sta(void)
{
  esp_netif_init();
  esp_event_loop_create_default();
  esp_netif_create_default_wifi_sta();
  wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
  esp_wifi_init(&cfg);

  esp_event_handler_instance_t instance_any_id;
  esp_event_handler_instance_t instance_got_ip;
  esp_event_handler_instance_register(WIFI_EVENT,
                                      ESP_EVENT_ANY_ID,
                                      &wifi_event_handler,
                                      NULL,
                                      &instance_any_id);
  esp_event_handler_instance_register(IP_EVENT,
                                      IP_EVENT_STA_GOT_IP,
                                      &wifi_event_handler,
                                      NULL,
                                      &instance_got_ip);

  wifi_config_t wifi_config = {
      .sta = {
          .ssid = WIFI_SSID,
          .password = WIFI_PASS,
      },
  };
  esp_wifi_set_mode(WIFI_MODE_STA);
  esp_wifi_set_config(WIFI_IF_STA, &wifi_config);
  esp_wifi_start();
}

void tf_main(void)
{

  // Initialize NVS
  esp_err_t ret = nvs_flash_init();
  if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND)
  {
    ESP_ERROR_CHECK(nvs_flash_erase());
    ret = nvs_flash_init();
  }
  ESP_ERROR_CHECK(ret);

  // Initialize SPIFFS
  esp_vfs_spiffs_conf_t spiffs_conf = {
      .base_path = "/storage",
      .partition_label = NULL,
      .max_files = 5,
      .format_if_mount_failed = true};

  esp_err_t result = esp_vfs_spiffs_register(&spiffs_conf);
  if (result != ESP_OK)
  {
    printf("Failed to load file (%s)", esp_err_to_name(result));
    return;
  }

  // Open the image file to check if it exists
  FILE *f = fopen("/storage/dv.jpg", "r");
  if (f == NULL)
  {
    printf("Failed to load dv.jpg");
  }
  else
  {
    printf("load dv.jpg\n");
    fclose(f);
  }

  // Initialize Wi-Fi
  wifi_init_sta();

  setup();
  while (true)
  {
    loop();
  }
}

extern "C" void app_main()
{
  xTaskCreate((TaskFunction_t)&tf_main, "tensorflow", 8 * 1024, NULL, 8, NULL);
  vTaskDelete(NULL);
}
