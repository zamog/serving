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

#ifndef TENSORFLOW_SERVING_BATCHING_ASYNC_RUN_SESSION_H_
#define TENSORFLOW_SERVING_BATCHING_ASYNC_RUN_SESSION_H_

#include <functional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/threadpool_options.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {
namespace serving {

// A session that can run a request without blocking the calling thread while
// the request waits for, and runs in, a batch. Session::Run() parks its caller
// until the outputs are ready, so a server calling it holds one thread per
// request in flight.
class AsyncRunSession {
 public:
  virtual ~AsyncRunSession() = default;

  // Schedules a run of 'inputs' and returns without waiting for it.
  //
  // If the returned status is OK, 'done' is called exactly once, after
  // '*status' holds the outcome of the run and, if that is OK, '*outputs' holds
  // the tensors named by 'output_tensor_names'. 'done' runs on one of the
  // session's batch threads, so it must not block, and it must not release
  // anything whose destruction waits for those threads (such as the last
  // reference to this session). Calling 'done' is the session's last access to
  // 'inputs', 'output_tensor_names', 'outputs', 'run_metadata' and 'status',
  // which the caller keeps alive until then.
  //
  // If the returned status is not OK, the run was not scheduled and 'done' is
  // never called. kUnimplemented means this session cannot run 'inputs'
  // asynchronously; run them with Session::Run() instead.
  virtual absl::Status RunAsync(
      const RunOptions& run_options,
      const std::vector<std::pair<string, Tensor>>* inputs,
      const std::vector<string>* output_tensor_names,
      std::vector<Tensor>* outputs, RunMetadata* run_metadata,
      const thread::ThreadPoolOptions& thread_pool_options,
      absl::Status* status, std::function<void()> done) = 0;
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_BATCHING_ASYNC_RUN_SESSION_H_
