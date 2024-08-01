#include "main_functions.h"
#include "detection_responder.h"
#include "image_provider.h"
#include "model_settings.h"
#include "drowsiness_detect_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include <esp_heap_caps.h>
#include <esp_timer.h>
#include <esp_log.h>
#include "esp_main.h"
#include "driver/gpio.h"

#define LED_BUILTIN GPIO_NUM_2

// Globals, used for compatibility with Arduino-style sketches.
namespace
{
  const tflite::Model *model = nullptr;
  tflite::MicroInterpreter *interpreter = nullptr;
  TfLiteTensor *input = nullptr;

  // In order to use optimized tensorflow lite kernels, a signed int8_t quantized
  // model is preferred over the legacy unsigned model format. This means that
  // throughout this project, input images must be converted from unisgned to
  // signed format. The easiest and quickest way to convert from unsigned to
  // signed 8-bit integers is to subtract 128 from the unsigned value to get a
  // signed value.

#ifdef CONFIG_IDF_TARGET_ESP32S3
  constexpr int scratchBufSize = 40 * 1024;
#else
  constexpr int scratchBufSize = 0;
#endif
  // An area of memory to use for input, output, and intermediate arrays.
  constexpr int kTensorArenaSize = 384 * 1024 + scratchBufSize;
  static uint8_t *tensor_arena; //[kTensorArenaSize]; // Maybe we should move this to external
} // namespace

// The name of this function is important for Arduino compatibility.
void setup()
{
  gpio_set_direction(LED_BUILTIN, GPIO_MODE_OUTPUT);

  size_t psram_size = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
  if (psram_size > 0)
  {
    ESP_LOGI("PSRAM", "PSRAM is available: %d bytes", psram_size);
  }
  else
  {
    ESP_LOGE("PSRAM", "No PSRAM available");
  }
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(drowsiness_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION)
  {
    MicroPrintf("Model provided is schema version %d not equal to supported "
                "version %d.",
                model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  if (tensor_arena == NULL)
  {
    tensor_arena = (uint8_t *)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  }
  if (tensor_arena == NULL)
  {
    printf("Couldn't allocate memory of %d bytes\n", kTensorArenaSize);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroMutableOpResolver<5> micro_op_resolver;
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddSoftmax();

  // Build an interpreter to run the model with.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk)
  {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0);

  TfLiteIntArray *dims = input->dims;
  MicroPrintf("Input Tensor Shape: [%d, %d, %d, %d]",
              dims->data[0],  // Batch size
              dims->data[1],  // Height
              dims->data[2],  // Width
              dims->data[3]); // Channels

#ifndef CLI_ONLY_INFERENCE
  // Initialize Camera
  TfLiteStatus init_status = InitCamera();
  if (init_status != kTfLiteOk)
  {
    MicroPrintf("InitCamera failed\n");
    return;
  }
#endif

  MicroPrintf("Number of dimension %d", input->dims->size);
  MicroPrintf("Dim 1 size %d", input->dims->data[0]);
  MicroPrintf("Dim 2 size %d", input->dims->data[1]);
  MicroPrintf("Input type %d", input->type);
}

#ifndef CLI_ONLY_INFERENCE
// The name of this function is important for Arduino compatibility.
void loop()
{
  // Get image from provider.
  if (kTfLiteOk != GetImage(kNumCols, kNumRows, kNumChannels, input->data.f))
  {
    MicroPrintf("Image capture failed.");
  }

  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != interpreter->Invoke())
  {
    MicroPrintf("Invoke failed.");
  }

  TfLiteTensor *output = interpreter->output(0);

  // Process the inference results.
  // int8_t person_score = output->data.uint8[kDrowsyIndex];
  // int8_t no_person_score = output->data.uint8[kNotDrowsyIndex];

  // MicroPrintf("Size: %i", sizeof(output->data.f));
  // MicroPrintf("1: %f", output->data.f[0]);
  // MicroPrintf("2: %f", output->data.f[1]);

  float noDrowsy = output->data.f[0];
  float drowsy = output->data.f[1];

  // float person_score_f =
  //     (person_score - output->params.zero_point) * output->params.scale;
  // float no_person_score_f =
  //     (no_person_score - output->params.zero_point) * output->params.scale;

  MicroPrintf("Non Drowsy:%f, Drowsy:%f", noDrowsy, drowsy);
  if (drowsy > 0.4)
  {
    gpio_set_level(LED_BUILTIN, 1); // Turn the LED on
  }
  else
  {
    gpio_set_level(LED_BUILTIN, 0); // Turn the LED on
  }
  // Respond to detection
  // RespondToDetection(person_score_f, no_person_score_f);
  vTaskDelay(1000 / portTICK_PERIOD_MS); // to avoid watchdog trigger
}
#endif

#if defined(COLLECT_CPU_STATS)
long long total_time = 0;
long long start_time = 0;
extern long long softmax_total_time;
extern long long dc_total_time;
extern long long conv_total_time;
extern long long fc_total_time;
extern long long pooling_total_time;
extern long long add_total_time;
extern long long mul_total_time;
#endif

void run_inference(void *ptr)
{
  /* Convert from uint8 picture data to int8 */
  for (int i = 0; i < kNumCols * kNumRows; i++)
  {
    input->data.int8[i] = ((uint8_t *)ptr)[i] ^ 0x80;
  }

#if defined(COLLECT_CPU_STATS)
  long long start_time = esp_timer_get_time();
#endif
  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != interpreter->Invoke())
  {
    MicroPrintf("Invoke failed.");
  }

#if defined(COLLECT_CPU_STATS)
  long long total_time = (esp_timer_get_time() - start_time);
  printf("Total time = %lld\n", total_time / 1000);
  // printf("Softmax time = %lld\n", softmax_total_time / 1000);
  printf("FC time = %lld\n", fc_total_time / 1000);
  printf("DC time = %lld\n", dc_total_time / 1000);
  printf("conv time = %lld\n", conv_total_time / 1000);
  printf("Pooling time = %lld\n", pooling_total_time / 1000);
  printf("add time = %lld\n", add_total_time / 1000);
  printf("mul time = %lld\n", mul_total_time / 1000);

  /* Reset times */
  total_time = 0;
  // softmax_total_time = 0;
  dc_total_time = 0;
  conv_total_time = 0;
  fc_total_time = 0;
  pooling_total_time = 0;
  add_total_time = 0;
  mul_total_time = 0;
#endif

  TfLiteTensor *output = interpreter->output(0);

  // Process the inference results.
  int8_t person_score = output->data.uint8[kDrowsyIndex];
  int8_t no_person_score = output->data.uint8[kNotDrowsyIndex];

  float person_score_f =
      (person_score - output->params.zero_point) * output->params.scale;
  float no_person_score_f =
      (no_person_score - output->params.zero_point) * output->params.scale;
  RespondToDetection(person_score_f, no_person_score_f);
}
