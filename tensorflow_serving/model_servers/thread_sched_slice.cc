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

#include "tensorflow_serving/model_servers/thread_sched_slice.h"

#include <cerrno>
#include <cstring>
#include <string>
#include <utility>

#if defined(__linux__)
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace tensorflow {
namespace serving {

namespace {

// Linux UAPI values; glibc 2.31 does not define all of them.
constexpr uint32_t kSchedOther = 0;
constexpr uint32_t kSchedBatch = 3;
constexpr uint64_t kSchedFlagResetOnFork = 0x01;
constexpr uint32_t kSchedAttrSizeVer0 = 48;

std::string ErrnoText(int error) {
  return std::string(std::strerror(error)) + " (errno " +
         std::to_string(error) + ")";
}

std::string SliceFlagRangeError(const char* name, int64_t value) {
  return std::string("server_options.") + name + " (" + std::to_string(value) +
         ") must be 0 or between " + std::to_string(kMinThreadSliceNs) +
         " and " + std::to_string(kMaxThreadSliceNs) +
         " ns; the kernel clamps slices outside that range.";
}

}  // namespace

SchedAttrSyscalls LinuxSchedAttrSyscalls() {
#if defined(__linux__) && defined(SYS_sched_getattr) && \
    defined(SYS_sched_setattr)
  return {
      [](KernelSchedAttr* attr) {
        return syscall(SYS_sched_getattr, 0, attr, kSchedAttrSizeVer0, 0) == 0
                   ? 0
                   : errno;
      },
      [](const KernelSchedAttr& attr) {
        return syscall(SYS_sched_setattr, 0, &attr, 0) == 0 ? 0 : errno;
      },
  };
#else
  return {[](KernelSchedAttr*) { return ENOSYS; },
          [](const KernelSchedAttr&) { return ENOSYS; }};
#endif
}

std::string ValidateThreadSliceFlags(int64_t tf_thread_slice_ns,
                                     int64_t grpc_thread_slice_ns) {
  if (tf_thread_slice_ns < 0 || grpc_thread_slice_ns < 0) {
    return "server_options.tf_thread_slice_ns (" +
           std::to_string(tf_thread_slice_ns) +
           ") and server_options.grpc_thread_slice_ns (" +
           std::to_string(grpc_thread_slice_ns) + ") must not be negative.";
  }
  const auto out_of_range = [](int64_t value) {
    return value != 0 &&
           (value < kMinThreadSliceNs || value > kMaxThreadSliceNs);
  };
  if (out_of_range(tf_thread_slice_ns)) {
    return SliceFlagRangeError("tf_thread_slice_ns", tf_thread_slice_ns);
  }
  if (out_of_range(grpc_thread_slice_ns)) {
    return SliceFlagRangeError("grpc_thread_slice_ns", grpc_thread_slice_ns);
  }
  if (tf_thread_slice_ns != 0 && grpc_thread_slice_ns != 0 &&
      tf_thread_slice_ns > grpc_thread_slice_ns) {
    return "server_options.tf_thread_slice_ns (" +
           std::to_string(tf_thread_slice_ns) +
           ") must not be greater than server_options.grpc_thread_slice_ns (" +
           std::to_string(grpc_thread_slice_ns) +
           "): a shorter slice means earlier scheduling, and the TensorFlow "
           "threads are the ones that determine graph run time.";
  }
  return "";
}

ThreadSliceSetter::ThreadSliceSetter(int64_t tf_thread_slice_ns,
                                     int64_t grpc_thread_slice_ns,
                                     SliceLogger logger,
                                     SchedAttrSyscalls syscalls)
    : tf_thread_slice_ns_(tf_thread_slice_ns),
      grpc_thread_slice_ns_(grpc_thread_slice_ns),
      logger_(std::move(logger)),
      syscalls_(std::move(syscalls)) {
  if (tf_thread_slice_ns_ == 0 && grpc_thread_slice_ns_ == 0) {
    state_ = State::kDisabled;  // Nothing requested: never touch the kernel.
  }
}

void ThreadSliceSetter::Disable(const std::string& reason) {
  state_ = State::kDisabled;
  logger_(SliceLogSeverity::kWarning,
          "Per-thread EEVDF slices disabled: " + reason +
              ". Starting with the kernel's default scheduling; the "
              "--tf_thread_slice_ns/--grpc_thread_slice_ns flags have no "
              "effect.");
}

bool ThreadSliceSetter::Probe() {
  if (state_ != State::kUnprobed) return state_ == State::kReady;
  KernelSchedAttr attr;
  if (const int error = syscalls_.get(&attr); error != 0) {
    Disable("sched_getattr failed: " + ErrnoText(error));
    return false;
  }
  if (attr.sched_policy != kSchedOther && attr.sched_policy != kSchedBatch) {
    Disable("the server runs under scheduling policy " +
            std::to_string(attr.sched_policy) +
            ", and slices apply only to SCHED_OTHER/SCHED_BATCH; the policy "
            "is left unchanged");
    return false;
  }
  if ((attr.sched_flags & kSchedFlagResetOnFork) != 0) {
    Disable(
        "SCHED_RESET_ON_FORK is set, so new threads would not inherit the "
        "slice");
    return false;
  }
  if (attr.sched_runtime == 0) {
    // Kernels with per-task slices (6.12+) always report the effective slice
    // of a fair task here; older kernels report 0.
    Disable(
        "the kernel does not support per-task slices (needs Linux >= 6.12; "
        "sched_getattr reports no slice)");
    return false;
  }
  state_ = State::kReady;
  return true;
}

std::string ThreadSliceSetter::SetSlice(int64_t slice_ns,
                                        uint64_t* effective_ns) {
  // Re-read so that the policy and nice value passed back are current. Flags
  // must stay 0: SCHED_FLAG_KEEP_PARAMS would make the kernel skip the slice.
  KernelSchedAttr attr;
  if (const int error = syscalls_.get(&attr); error != 0) {
    return "sched_getattr failed: " + ErrnoText(error);
  }
  attr.size = kSchedAttrSizeVer0;
  attr.sched_flags = 0;
  attr.sched_priority = 0;
  attr.sched_runtime = static_cast<uint64_t>(slice_ns);
  attr.sched_deadline = 0;
  attr.sched_period = 0;
  if (const int error = syscalls_.set(attr); error != 0) {
    return "sched_setattr(slice " + std::to_string(slice_ns) +
           " ns) failed: " + ErrnoText(error);
  }
  KernelSchedAttr back;
  if (const int error = syscalls_.get(&back); error != 0) {
    return "sched_getattr failed: " + ErrnoText(error);
  }
  *effective_ns = back.sched_runtime;
  // slice_ns == 0 restores the kernel's base slice, whose value is not known
  // here; any non-zero read-back is fine then.
  const bool matches = slice_ns == 0
                           ? back.sched_runtime != 0
                           : back.sched_runtime ==
                                 static_cast<uint64_t>(slice_ns);
  if (!matches) {
    return "the kernel reports an effective slice of " +
           std::to_string(back.sched_runtime) + " ns after a request for " +
           std::to_string(slice_ns) + " ns";
  }
  return "";
}

void ThreadSliceSetter::ApplyTensorFlowSlice() {
  if (tf_thread_slice_ns_ == 0 || !Probe()) return;
  uint64_t effective_ns = 0;
  const std::string error = SetSlice(tf_thread_slice_ns_, &effective_ns);
  if (!error.empty()) {
    // A set that went through but read back wrong may have changed the
    // slice; put the default back before giving up. Failure here is moot:
    // the feature is off and there is nothing more to try.
    uint64_t ignored = 0;
    SetSlice(0, &ignored);
    Disable("TensorFlow thread slice not applied: " + error);
    return;
  }
  tf_slice_applied_ = true;
  logger_(SliceLogSeverity::kInfo,
          "TensorFlow threads: EEVDF slice requested " +
              std::to_string(tf_thread_slice_ns_) + " ns, effective " +
              std::to_string(effective_ns) + " ns");
}

void ThreadSliceSetter::ApplyGrpcSlice() {
  // With only the TensorFlow slice requested this phase still has to run:
  // it restores the default so gRPC's threads do not inherit that slice.
  if (grpc_thread_slice_ns_ == 0 && !tf_slice_applied_) return;
  if (!Probe()) return;
  uint64_t effective_ns = 0;
  std::string error = SetSlice(grpc_thread_slice_ns_, &effective_ns);
  if (error.empty()) {
    logger_(SliceLogSeverity::kInfo,
            grpc_thread_slice_ns_ == 0
                ? "gRPC threads: kernel default EEVDF slice restored "
                  "(effective " +
                      std::to_string(effective_ns) + " ns)"
                : "gRPC threads: EEVDF slice requested " +
                      std::to_string(grpc_thread_slice_ns_) +
                      " ns, effective " + std::to_string(effective_ns) +
                      " ns");
    return;
  }
  if (grpc_thread_slice_ns_ != 0) {
    uint64_t ignored = 0;
    const std::string restore_error = SetSlice(0, &ignored);
    if (!restore_error.empty()) {
      error += "; restoring the default slice also failed: " + restore_error;
    }
    if (!tf_slice_applied_) {
      Disable("gRPC thread slice not applied: " + error);
      return;
    }
    if (restore_error.empty()) {
      state_ = State::kDisabled;
      logger_(SliceLogSeverity::kWarning,
              "gRPC thread slice not applied: " + error +
                  ". gRPC threads keep the kernel default slice; the "
                  "TensorFlow slice stays in effect.");
      return;
    }
  }
  // Only reachable after the TensorFlow slice was applied: this thread keeps
  // it, and gRPC's threads will inherit it.
  state_ = State::kDisabled;
  logger_(SliceLogSeverity::kError,
          "gRPC threads could not be moved off the TensorFlow EEVDF slice (" +
              error + "). gRPC threads will share the " +
              std::to_string(tf_thread_slice_ns_) +
              " ns TensorFlow slice, so TensorFlow threads are no longer "
              "scheduled ahead of them. Serving continues; remove "
              "--tf_thread_slice_ns if this persists.");
}

}  // namespace serving
}  // namespace tensorflow
