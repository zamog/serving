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

#include "tensorflow_serving/servables/tensorflow/predict_impl.h"

#include <string>
#include <utility>

#include "absl/strings/substitute.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/env_time.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/threadpool_options.h"
#include "tensorflow_serving/batching/async_run_session.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/servables/tensorflow/predict_response_tensor_serialization_option.h"
#include "tensorflow_serving/servables/tensorflow/predict_util.h"
#include "tensorflow_serving/servables/tensorflow/thread_pool_factory.h"
#include "tensorflow_serving/servables/tensorflow/util.h"

namespace tensorflow {
namespace serving {

namespace {

// The part of an asynchronous Predict that follows the session run; mirrors
// the end of internal::RunPredict().
Status FinishPredictAsync(
    const string& model_name,
    internal::PredictResponseTensorSerializationOption option,
    PredictAsyncState* state, PredictResponse* response) {
  TF_RETURN_IF_ERROR(state->run_status);
  RecordRuntimeLatency(model_name, /*api=*/"Predict", /*runtime=*/"TF1",
                       EnvTime::NowMicros() - state->run_start_micros);
  return internal::PostProcessPredictionResult(
      state->output_tensor_aliases, state->outputs, option, response);
}

}  // namespace

absl::Status TensorflowPredictor::Predict(const RunOptions& run_options,
                                          ServerCore* core,
                                          const PredictRequest& request,
                                          PredictResponse* response) {
  if (!request.has_model_spec()) {
    return absl::Status(
        static_cast<absl::StatusCode>(absl::StatusCode::kInvalidArgument),
        "Missing ModelSpec");
  }
  return PredictWithModelSpec(run_options, core, request.model_spec(), request,
                              response);
}

absl::Status TensorflowPredictor::PredictWithModelSpec(
    const RunOptions& run_options, ServerCore* core,
    const ModelSpec& model_spec, const PredictRequest& request,
    PredictResponse* response) {
  ServableHandle<SavedModelBundle> bundle;
  TF_RETURN_IF_ERROR(core->GetServableHandle(model_spec, &bundle));
  return internal::RunPredict(
      run_options, bundle->meta_graph_def, bundle.id().version,
      core->predict_response_tensor_serialization_option(),
      bundle->session.get(), request, response,
      thread_pool_factory_ == nullptr
          ? thread::ThreadPoolOptions()
          : thread_pool_factory_->GetThreadPools().get());
}

void TensorflowPredictor::PredictAsync(
    const RunOptions& run_options, ServerCore* core,
    const PredictRequest& request, PredictResponse* response,
    PredictAsyncState* state, std::function<void(const Status&)> done) {
  if (!request.has_model_spec()) {
    done(absl::Status(
        static_cast<absl::StatusCode>(absl::StatusCode::kInvalidArgument),
        "Missing ModelSpec"));
    return;
  }
  Status status = core->GetServableHandle(request.model_spec(), &state->bundle);
  if (status.ok()) {
    status = internal::PrepareRunPredict(
        state->bundle->meta_graph_def, state->bundle.id().version, request,
        response, &state->inputs, &state->output_tensor_names,
        &state->output_tensor_aliases);
  }
  if (!status.ok()) {
    done(status);
    return;
  }
  if (thread_pool_factory_ != nullptr) {
    state->thread_pools = thread_pool_factory_->GetThreadPools();
  }
  const internal::PredictResponseTensorSerializationOption option =
      core->predict_response_tensor_serialization_option();
  const string& model_name = request.model_spec().name();
  state->run_start_micros = EnvTime::NowMicros();

  auto* async_session =
      dynamic_cast<AsyncRunSession*>(state->bundle->session.get());
  if (async_session == nullptr) {
    LOG_EVERY_N_SEC(WARNING, 120)
        << "Model " << model_name
        << " has no batching session, so its Predicts run synchronously on "
           "the calling thread.";
  } else {
    status = async_session->RunAsync(
        run_options, &state->inputs, &state->output_tensor_names,
        &state->outputs, &state->run_metadata, state->thread_pools.get(),
        &state->run_status,
        [model_name, option, state, response, done]() {
          done(FinishPredictAsync(model_name, option, state, response));
        });
    if (!absl::IsUnimplemented(status)) {
      // Scheduled (the batch thread calls 'done'), or rejected.
      if (!status.ok()) {
        done(status);
      }
      return;
    }
  }
  // This session runs these inputs only synchronously.
  state->run_status = state->bundle->session->Run(
      run_options, state->inputs, state->output_tensor_names, {},
      &state->outputs, &state->run_metadata, state->thread_pools.get());
  done(FinishPredictAsync(model_name, option, state, response));
}

}  // namespace serving
}  // namespace tensorflow
