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

#ifndef TENSORFLOW_SERVING_MODEL_SERVERS_ASYNC_PREDICT_SERVER_H_
#define TENSORFLOW_SERVING_MODEL_SERVERS_ASYNC_PREDICT_SERVER_H_

#include <memory>
#include <vector>

#include "grpcpp/completion_queue.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#include "tensorflow_serving/model_servers/prediction_service_impl.h"

namespace tensorflow {
namespace serving {

// Answers every PredictionService method except Predict by forwarding it to a
// PredictionServiceImpl. Exists because the base class gRPC generates for an
// asynchronous method (WithAsyncMethod_Predict, below) must be default
// constructible, which PredictionServiceImpl is not.
class ForwardingPredictionService : public PredictionService::Service {
 public:
  ::grpc::Status Classify(::grpc::ServerContext* context,
                          const ClassificationRequest* request,
                          ClassificationResponse* response) override;

  ::grpc::Status Regress(::grpc::ServerContext* context,
                         const RegressionRequest* request,
                         RegressionResponse* response) override;

  ::grpc::Status MultiInference(::grpc::ServerContext* context,
                                const MultiInferenceRequest* request,
                                MultiInferenceResponse* response) override;

  ::grpc::Status GetModelMetadata(::grpc::ServerContext* context,
                                  const GetModelMetadataRequest* request,
                                  GetModelMetadataResponse* response) override;

 protected:
  PredictionServiceImpl* impl_ = nullptr;
};

// The PredictionService to register with the gRPC server when Predict is
// served asynchronously: Predict through gRPC's completion-queue API (served by
// AsyncPredictServer), every other method through the synchronous server, as
// before.
class AsyncPredictionService
    : public PredictionService::WithAsyncMethod_Predict<
          ForwardingPredictionService> {
 public:
  // 'impl' must outlive this service.
  explicit AsyncPredictionService(PredictionServiceImpl* impl) {
    impl_ = impl;
  }

  PredictionServiceImpl* impl() const { return impl_; }
};

// Serves the Predict calls of an AsyncPredictionService from gRPC completion
// queues, with one thread per queue. A thread receives a request, starts it
// (PredictionServiceImpl::PredictAsync()) and moves on; the request's reply is
// sent from the batch thread that runs it, and the request is released on its
// queue's thread once the reply is out. A request waiting for or running in a
// batch therefore holds no thread, whereas the synchronous server needs one
// thread per request in flight, which multiplies with load.
//
// Once these queues are added, gRPC does all of the server's network I/O, for
// every service, on their threads only, so a thread that blocks stalls that
// I/O. A Predict on a model without batching runs synchronously on its queue's
// thread.
class AsyncPredictServer {
 public:
  // 'service' must have been registered with the gRPC server, and each of
  // 'queues' added to its ServerBuilder with AddCompletionQueue(), before the
  // server was built. 'service' must outlive this object.
  AsyncPredictServer(
      AsyncPredictionService* service,
      std::vector<std::unique_ptr<::grpc::ServerCompletionQueue>> queues);

  AsyncPredictServer(const AsyncPredictServer&) = delete;
  AsyncPredictServer& operator=(const AsyncPredictServer&) = delete;

  // Calls Stop() if it was not called.
  ~AsyncPredictServer();

  // Starts accepting Predict calls. Call once, after the gRPC server is built.
  void Start();

  // Shuts the queues down and waits for their threads to drain them. Must not
  // be called while a Predict is in flight: it replies through its queue when
  // its batch finishes, even after the gRPC server has shut down.
  void Stop();

 private:
  AsyncPredictionService* const service_;
  std::vector<std::unique_ptr<::grpc::ServerCompletionQueue>> queues_;
  std::vector<std::unique_ptr<Thread>> threads_;
  bool stopped_ = false;
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_MODEL_SERVERS_ASYNC_PREDICT_SERVER_H_
