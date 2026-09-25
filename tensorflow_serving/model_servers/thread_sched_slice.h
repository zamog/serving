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

#ifndef TENSORFLOW_SERVING_MODEL_SERVERS_THREAD_SCHED_SLICE_H_
#define TENSORFLOW_SERVING_MODEL_SERVERS_THREAD_SCHED_SLICE_H_

#include <cstdint>
#include <functional>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"

namespace tensorflow {
namespace serving {

// Per-thread EEVDF CPU slices (Linux >= 6.12, sched_setattr() sched_runtime
// for SCHED_OTHER/SCHED_BATCH threads). A thread with a shorter slice gets an
// earlier virtual deadline and is scheduled sooner when the CPU is contended;
// fairness (weights) is unchanged. Threads inherit the slice of the thread
// that creates them, so setting it on the server's main thread before a
// subsystem creates its threads classifies all of that subsystem's threads.

// The kernel clamps slices to this range; values outside it are rejected so
// that the effective slice can be verified to equal the requested one.
inline constexpr int64_t kMinThreadSliceNs = 100 * 1000;         // 0.1 ms
inline constexpr int64_t kMaxThreadSliceNs = 100 * 1000 * 1000;  // 100 ms

// The kernel's struct sched_attr, first version (SCHED_ATTR_SIZE_VER0).
// glibc 2.31 (the serving base image) has neither the struct nor a wrapper.
struct KernelSchedAttr {
  uint32_t size = 0;
  uint32_t sched_policy = 0;
  uint64_t sched_flags = 0;
  int32_t sched_nice = 0;
  uint32_t sched_priority = 0;
  uint64_t sched_runtime = 0;
  uint64_t sched_deadline = 0;
  uint64_t sched_period = 0;
};
static_assert(sizeof(KernelSchedAttr) == 48,
              "KernelSchedAttr must match SCHED_ATTR_SIZE_VER0");

// sched_getattr/sched_setattr for the calling thread. Each returns 0 on
// success or an errno value. Replaceable so tests can model other kernels.
struct SchedAttrSyscalls {
  std::function<int(KernelSchedAttr*)> get;
  std::function<int(const KernelSchedAttr&)> set;
};

// The real syscalls; on non-Linux builds both return ENOSYS.
SchedAttrSyscalls LinuxSchedAttrSyscalls();

// Returns InvalidArgument if the flag pair is invalid. Only the values are
// checked, not the kernel: an invalid pair fails the same way on every host.
absl::Status ValidateThreadSliceFlags(int64_t tf_thread_slice_ns,
                                      int64_t grpc_thread_slice_ns);

enum class SliceLogSeverity { kInfo, kWarning, kError };
using SliceLogger =
    std::function<void(SliceLogSeverity severity, const std::string& message)>;

// Applies --tf_thread_slice_ns and --grpc_thread_slice_ns to the calling
// thread in two phases. Never fails the server: if the kernel or the runtime
// environment cannot honour the flags (kernel < 6.12, syscall blocked by
// seccomp, non-fair scheduling policy, SCHED_RESET_ON_FORK, unexpected
// read-back) it logs one warning, leaves or restores the kernel default slice
// and turns itself off. With both flags zero it issues no syscalls at all.
//
// Not thread-safe; both phases must run on the same thread, the one that
// creates the TensorFlow threads and then the gRPC threads.
class ThreadSliceSetter final {
 public:
  // Flags must have passed ValidateThreadSliceFlags().
  ThreadSliceSetter(int64_t tf_thread_slice_ns, int64_t grpc_thread_slice_ns,
                    SliceLogger logger,
                    SchedAttrSyscalls syscalls = LinuxSchedAttrSyscalls());

  ThreadSliceSetter(const ThreadSliceSetter&) = delete;
  ThreadSliceSetter& operator=(const ThreadSliceSetter&) = delete;

  // Call before any TensorFlow thread is created.
  void ApplyTensorFlowSlice();

  // Call after ApplyTensorFlowSlice() and before gRPC's threads are created.
  // Sets the gRPC slice, or, if only the TensorFlow slice was set, restores
  // the kernel default so gRPC's threads do not inherit the TensorFlow slice.
  void ApplyGrpcSlice();

  // False once the feature has been turned off for this process.
  bool enabled() const { return state_ != State::kDisabled; }

 private:
  enum class State { kUnprobed, kReady, kDisabled };

  // Reads the calling thread's attributes and decides whether the feature
  // can work here. Changes nothing.
  bool Probe();
  // Sets the calling thread's slice (0 = kernel default) keeping its policy
  // and nice value, then verifies it; *effective_ns receives the slice read
  // back.
  absl::Status SetSlice(int64_t slice_ns, uint64_t* effective_ns);
  void Disable(absl::string_view reason);

  const int64_t tf_thread_slice_ns_;
  const int64_t grpc_thread_slice_ns_;
  const SliceLogger logger_;
  const SchedAttrSyscalls syscalls_;
  State state_ = State::kUnprobed;
  bool tf_slice_applied_ = false;
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_MODEL_SERVERS_THREAD_SCHED_SLICE_H_
