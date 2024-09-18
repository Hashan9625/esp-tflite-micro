#include <algorithm>
#include <cstdint>
#include <iterator>

#include "main_functions.h"

#include "audio_provider.h"
#include "command_responder.h"
#include "feature_provider.h"
#include "micro_model_settings.h"
#include "model.h"
#include "recognize_commands.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include <esp_heap_caps.h>
#include <esp_log.h>

// Globals, used for compatibility with Arduino-style sketches.
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
FeatureProvider* feature_provider = nullptr;
RecognizeCommands* recognizer = nullptr;
int32_t previous_time = 0;

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
constexpr int kTensorArenaSize = 30 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
int8_t feature_buffer[kFeatureElementCount];
float* model_input_buffer = nullptr;
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
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
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model provided is schema version %d not equal to supported "
                "version %d.", model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // if (tensor_arena == NULL)
  // {
  //   tensor_arena = (uint8_t *)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  // }
  // if (tensor_arena == NULL)
  // {
  //   ESP_LOGE( "PSRAM","Couldn't allocate memory of %d bytes", kTensorArenaSize);
  //   return;
  // }

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
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  model_input = interpreter->input(0);

  TfLiteIntArray *dims = model_input->dims;
  MicroPrintf("Input Tensor Shape: [%d, %d, %d, %d]",
              dims->data[0],  // Batch size
              dims->data[1],  // Height
              dims->data[2],  // Width
              dims->data[3]); // Channels

  MicroPrintf("Number of dimension %d", model_input->dims->size);
  MicroPrintf("Input type %d", model_input->type);

  // if ((model_input->dims->size != 2) || (model_input->dims->data[0] != 1) ||
  //     (model_input->dims->data[1] != (kFeatureCount * kFeatureSize)) ||
  //     (model_input->type != kTfLiteInt8)) {
  //   MicroPrintf("Bad input tensor parameters in model");
  //   return;
  // }
  model_input_buffer = model_input->data.f;

  // Prepare to access the audio spectrograms from a microphone or other source
  // that will provide the inputs to the neural network.
  // NOLINTNEXTLINE(runtime-global-variables)
  static FeatureProvider static_feature_provider(kFeatureElementCount,
                                                 feature_buffer);
  feature_provider = &static_feature_provider;

  static RecognizeCommands static_recognizer;
  recognizer = &static_recognizer;

  previous_time = 0;
}

// The name of this function is important for Arduino compatibility.
void loop() {
  // Fetch the spectrogram for the current time.
  const int32_t current_time = LatestAudioTimestamp();
  int how_many_new_slices = 0;
  TfLiteStatus feature_status = feature_provider->PopulateFeatureData(
      previous_time, current_time, &how_many_new_slices);
  if (feature_status != kTfLiteOk) {
    MicroPrintf( "Feature generation failed");
    return;
  }
  previous_time = current_time;
  // If no new audio samples have been received since last time, don't bother
  // running the network model.

  // MicroPrintf("# how_many_new_slices: %d", how_many_new_slices);

  if (how_many_new_slices == 0) {
    return;
  }

  // Copy feature buffer to input tensor
  for (int i = 0; i < kFeatureElementCount; i++)
  {
    model_input_buffer[i] = ((float)feature_buffer[i]) / 255;
  }

  FILE *file = fopen("/storage/dv.pgm", "wb");
  if (file)
  {
    // Write the PGM header
    fprintf(file, "P5\n%d %d\n255\n", kFeatureSize, kFeatureCount);
    fwrite(model_input_buffer, 1, kFeatureElementCount, file);
    fclose(file);
    printf("Image saved to /storage/dv.pgm");
  }
  else
  {
    printf( "Failed to open file for writing");
  }

  // Run the model on the spectrogram input and make sure it succeeds.
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    MicroPrintf( "Invoke failed");
    return;
  }

  // Obtain a pointer to the output tensor
  TfLiteTensor* output = interpreter->output(0);
#if 1 // using simple argmax instead of recognizer
  float output_scale = output->params.scale;
  int output_zero_point = output->params.zero_point;
  int max_idx = 0;
  float max_result = 0.0;
  // Dequantize output values and find the max
  for (int i = 0; i < kCategoryCount; i++) {
    float current_result =  (tflite::GetTensorData<float>(output)[i]);
    if (current_result > max_result) {
      max_result = current_result; // update max result
      max_idx = i; // update category
    }
  }

  MicroPrintf("output %f , %f , %f , %f", output->data.f[0], output->data.f[1], output->data.f[2], output->data.f[3]);

  //  MicroPrintf("output %f , %f , %f , %f", tflite::GetTensorData<float>(output)[0],  tflite::GetTensorData<float>(output)[1],  tflite::GetTensorData<float>(output)[2],  tflite::GetTensorData<float>(output)[3]);
  if (max_result > 100) {
  //  MicroPrintf("Detected %7s, score: %.2f", kCategoryLabels[max_idx],
    //    static_cast<double>(max_result));
 }
#else
  // Determine whether a command was recognized based on the output of inference
  const char* found_command = nullptr;
  float score = 0;
  bool is_new_command = false;
  TfLiteStatus process_status = recognizer->ProcessLatestResults(
      output, current_time, &found_command, &score, &is_new_command);
  if (process_status != kTfLiteOk) {
    MicroPrintf("RecognizeCommands::ProcessLatestResults() failed");
    return;
  }
  // Do something based on the recognized command. The default implementation
  // just prints to the error console, but you should replace this with your
  // own function for a real application.
  RespondToCommand(current_time, found_command, score, is_new_command);
#endif
}
