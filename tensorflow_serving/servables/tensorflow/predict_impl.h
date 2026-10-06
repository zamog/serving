/* Copyright 2016 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_PREDICT_IMPL_H_
#define TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_PREDICT_IMPL_H_

#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow_serving/apis/predict.pb.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/model_servers/server_core.h"
#include "tensorflow_serving/servables/tensorflow/thread_pool_factory.h"

namespace tensorflow {
namespace serving {

// What a Predict started with TensorflowPredictor::PredictAsync() keeps until
// it completes: the handle that pins the model version, and the tensors the
// batch reads and writes.
struct PredictAsyncState {
  ServableHandle<SavedModelBundle> bundle;
  ScopedThreadPools thread_pools;
  std::vector<std::pair<string, Tensor>> inputs;
  std::vector<string> output_tensor_names;
  std::vector<string> output_tensor_aliases;
  std::vector<Tensor> outputs;
  RunMetadata run_metadata;
  Status run_status;
  uint64_t run_start_micros = 0;
};

// Utility methods for implementation of PredictionService::Predict.
class TensorflowPredictor {
 public:
  TensorflowPredictor() {}

  explicit TensorflowPredictor(ThreadPoolFactory* thread_pool_factory)
      : thread_pool_factory_(thread_pool_factory) {}

  Status Predict(const RunOptions& run_options, ServerCore* core,
                 const PredictRequest& request, PredictResponse* response);

  // Like Predict(), but uses 'model_spec' instead of the one embedded in
  // 'request'.
  Status PredictWithModelSpec(const RunOptions& run_options, ServerCore* core,
                              const ModelSpec& model_spec,
                              const PredictRequest& request,
                              PredictResponse* response);

  // Like Predict(), but does not wait for the model run when the model's
  // session can run requests asynchronously (see AsyncRunSession, which the
  // batching session implements). 'done' is called exactly once with the
  // result, after 'response' is filled in: on one of the model's batch threads
  // when the session runs the request asynchronously, otherwise on the calling
  // thread, which then also runs the model if the request is valid (as for a
  // model without batching). 'done' must not block. 'request', 'response' and
  // 'state' must stay alive until 'done' has been called, and 'state' must be
  // released on a thread that is not one of the model's batch threads: its
  // handle can be the last reference to the model version, whose unloading
  // waits for those threads to finish their batches.
  void PredictAsync(const RunOptions& run_options, ServerCore* core,
                    const PredictRequest& request, PredictResponse* response,
                    PredictAsyncState* state,
                    std::function<void(const Status&)> done);

 private:
  ThreadPoolFactory* thread_pool_factory_ = nullptr;
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_PREDICT_IMPL_H_
