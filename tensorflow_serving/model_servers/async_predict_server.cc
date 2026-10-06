/* Copyright 2026 Google Inc. All Rights Reserved.

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

#include "tensorflow_serving/model_servers/async_predict_server.h"

#include <memory>
#include <utility>
#include <vector>

#include "grpcpp/completion_queue.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/async_unary_call.h"
#include "grpcpp/support/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow_serving/apis/predict.pb.h"
#include "tensorflow_serving/servables/tensorflow/predict_impl.h"

namespace tensorflow {
namespace serving {

namespace {

// Calls kept posted on each completion queue, waiting for requests. A request
// that arrives while none is posted waits inside gRPC until one is, so this
// only smooths bursts.
constexpr int kPostedCallsPerQueue = 32;

// One Predict call, from waiting for its request until its reply is sent. The
// call is its own completion-queue tag and deletes itself once gRPC no longer
// refers to it. Each queue is drained by a single thread, so all of a call's
// events are handled on that thread.
class PredictCall {
 public:
  // Posts a new call on 'queue' to receive the next Predict request.
  static void Post(AsyncPredictionService* service,
                   ::grpc::ServerCompletionQueue* queue) {
    new PredictCall(service, queue);
  }

  // Handles this call's next event: its request arrived ('ok'), or the server
  // shut down before one did (not 'ok'), or its reply was sent or abandoned.
  void Proceed(bool ok) {
    if (replying_ || !ok) {
      delete this;
      return;
    }
    Post(service_, queue_);
    replying_ = true;
    service_->impl()->PredictAsync(
        &context_, &request_, &response_, &state_,
        [this](const ::grpc::Status& status) {
          // The last use of this call here: once the reply is posted, the
          // queue's thread can handle its completion and delete the call.
          if (status.ok()) {
            responder_.Finish(response_, status, this);
          } else {
            responder_.FinishWithError(status, this);
          }
        });
  }

 private:
  PredictCall(AsyncPredictionService* service,
              ::grpc::ServerCompletionQueue* queue)
      : service_(service), queue_(queue), responder_(&context_) {
    service_->RequestPredict(&context_, &request_, &responder_, queue_, queue_,
                             this);
  }

  AsyncPredictionService* const service_;
  ::grpc::ServerCompletionQueue* const queue_;
  ::grpc::ServerContext context_;
  PredictRequest request_;
  PredictResponse response_;
  ::grpc::ServerAsyncResponseWriter<PredictResponse> responder_;
  // Released with the call, on the queue's thread. It pins the model version,
  // and the last reference to a model version must not be released on one of
  // that model's batch threads; see TensorflowPredictor::PredictAsync().
  PredictAsyncState state_;
  bool replying_ = false;
};

// Handles the events of 'queue' until it has been shut down and emptied.
void DrainQueue(::grpc::ServerCompletionQueue* queue) {
  void* tag;
  bool ok;
  while (queue->Next(&tag, &ok)) {
    static_cast<PredictCall*>(tag)->Proceed(ok);
  }
}

}  // namespace

::grpc::Status ForwardingPredictionService::Classify(
    ::grpc::ServerContext* context, const ClassificationRequest* request,
    ClassificationResponse* response) {
  return impl_->Classify(context, request, response);
}

::grpc::Status ForwardingPredictionService::Regress(
    ::grpc::ServerContext* context, const RegressionRequest* request,
    RegressionResponse* response) {
  return impl_->Regress(context, request, response);
}

::grpc::Status ForwardingPredictionService::MultiInference(
    ::grpc::ServerContext* context, const MultiInferenceRequest* request,
    MultiInferenceResponse* response) {
  return impl_->MultiInference(context, request, response);
}

::grpc::Status ForwardingPredictionService::GetModelMetadata(
    ::grpc::ServerContext* context, const GetModelMetadataRequest* request,
    GetModelMetadataResponse* response) {
  return impl_->GetModelMetadata(context, request, response);
}

AsyncPredictServer::AsyncPredictServer(
    AsyncPredictionService* service,
    std::vector<std::unique_ptr<::grpc::ServerCompletionQueue>> queues)
    : service_(service), queues_(std::move(queues)) {}

AsyncPredictServer::~AsyncPredictServer() { Stop(); }

void AsyncPredictServer::Start() {
  for (const auto& queue : queues_) {
    ::grpc::ServerCompletionQueue* const raw_queue = queue.get();
    for (int i = 0; i < kPostedCallsPerQueue; ++i) {
      PredictCall::Post(service_, raw_queue);
    }
    threads_.emplace_back(Env::Default()->StartThread(
        ThreadOptions(), "async_predict",
        [raw_queue] { DrainQueue(raw_queue); }));
  }
}

void AsyncPredictServer::Stop() {
  if (stopped_) {
    return;
  }
  stopped_ = true;
  for (const auto& queue : queues_) {
    queue->Shutdown();
  }
  if (threads_.empty()) {
    // Never started: no thread will drain the queues, and a queue must be
    // drained before it is destroyed.
    for (const auto& queue : queues_) {
      DrainQueue(queue.get());
    }
  }
  // Each thread returns once its queue is shut down and empty; destroying a
  // Thread joins it.
  threads_.clear();
}

}  // namespace serving
}  // namespace tensorflow
