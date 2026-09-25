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

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

#if defined(__linux__)
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
  return absl::StrCat(std::strerror(error), " (errno ", error, ")");
}

absl::Status SliceFlagRangeError(const char* name, int64_t value) {
  return absl::InvalidArgumentError(absl::StrCat(
      "server_options.", name, " (", value, ") must be 0 or between ",
      kMinThreadSliceNs, " and ", kMaxThreadSliceNs,
      " ns; the kernel clamps slices outside that range."));
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

absl::Status ValidateThreadSliceFlags(int64_t tf_thread_slice_ns,
                                      int64_t grpc_thread_slice_ns) {
  if (tf_thread_slice_ns < 0 || grpc_thread_slice_ns < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("server_options.tf_thread_slice_ns (", tf_thread_slice_ns,
                     ") and server_options.grpc_thread_slice_ns (",
                     grpc_thread_slice_ns, ") must not be negative."));
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
    return absl::InvalidArgumentError(absl::StrCat(
        "server_options.tf_thread_slice_ns (", tf_thread_slice_ns,
        ") must not be greater than server_options.grpc_thread_slice_ns (",
        grpc_thread_slice_ns,
        "): a shorter slice means earlier scheduling, and the TensorFlow "
        "threads are the ones that determine graph run time."));
  }
  return absl::OkStatus();
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

void ThreadSliceSetter::Disable(absl::string_view reason) {
  state_ = State::kDisabled;
  logger_(SliceLogSeverity::kWarning,
          absl::StrCat("Per-thread EEVDF slices disabled: ", reason,
                       ". Starting with the kernel's default scheduling; the "
                       "--tf_thread_slice_ns/--grpc_thread_slice_ns flags "
                       "have no effect."));
}

bool ThreadSliceSetter::Probe() {
  if (state_ != State::kUnprobed) return state_ == State::kReady;
  KernelSchedAttr attr;
  if (const int error = syscalls_.get(&attr); error != 0) {
    Disable(absl::StrCat("sched_getattr failed: ", ErrnoText(error)));
    return false;
  }
  if (attr.sched_policy != kSchedOther && attr.sched_policy != kSchedBatch) {
    Disable(absl::StrCat(
        "the server runs under scheduling policy ", attr.sched_policy,
        ", and slices apply only to SCHED_OTHER/SCHED_BATCH; the policy is "
        "left unchanged"));
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

absl::Status ThreadSliceSetter::SetSlice(int64_t slice_ns,
                                         uint64_t* effective_ns) {
  // Re-read so that the policy and nice value passed back are current. Flags
  // must stay 0: SCHED_FLAG_KEEP_PARAMS would make the kernel skip the slice.
  KernelSchedAttr attr;
  if (const int error = syscalls_.get(&attr); error != 0) {
    return absl::InternalError(
        absl::StrCat("sched_getattr failed: ", ErrnoText(error)));
  }
  attr.size = kSchedAttrSizeVer0;
  attr.sched_flags = 0;
  attr.sched_priority = 0;
  attr.sched_runtime = static_cast<uint64_t>(slice_ns);
  attr.sched_deadline = 0;
  attr.sched_period = 0;
  if (const int error = syscalls_.set(attr); error != 0) {
    return absl::InternalError(absl::StrCat("sched_setattr(slice ", slice_ns,
                                            " ns) failed: ", ErrnoText(error)));
  }
  KernelSchedAttr back;
  if (const int error = syscalls_.get(&back); error != 0) {
    return absl::InternalError(
        absl::StrCat("sched_getattr failed: ", ErrnoText(error)));
  }
  *effective_ns = back.sched_runtime;
  // slice_ns == 0 restores the kernel's base slice, whose value is not known
  // here; any non-zero read-back is fine then.
  const bool matches =
      slice_ns == 0 ? back.sched_runtime != 0
                    : back.sched_runtime == static_cast<uint64_t>(slice_ns);
  if (!matches) {
    return absl::InternalError(absl::StrCat(
        "the kernel reports an effective slice of ", back.sched_runtime,
        " ns after a request for ", slice_ns, " ns"));
  }
  return absl::OkStatus();
}

void ThreadSliceSetter::ApplyTensorFlowSlice() {
  if (tf_thread_slice_ns_ == 0 || !Probe()) return;
  uint64_t effective_ns = 0;
  const absl::Status status = SetSlice(tf_thread_slice_ns_, &effective_ns);
  if (!status.ok()) {
    // A set that went through but read back wrong may have changed the
    // slice; put the default back before giving up. Failure here is moot:
    // the feature is off and there is nothing more to try.
    uint64_t ignored = 0;
    SetSlice(0, &ignored).IgnoreError();
    Disable(absl::StrCat("TensorFlow thread slice not applied: ",
                         status.message()));
    return;
  }
  tf_slice_applied_ = true;
  logger_(SliceLogSeverity::kInfo,
          absl::StrCat("TensorFlow threads: EEVDF slice requested ",
                       tf_thread_slice_ns_, " ns, effective ", effective_ns,
                       " ns"));
}

void ThreadSliceSetter::ApplyGrpcSlice() {
  // With only the TensorFlow slice requested this phase still has to run:
  // it restores the default so gRPC's threads do not inherit that slice.
  if (grpc_thread_slice_ns_ == 0 && !tf_slice_applied_) return;
  if (!Probe()) return;
  uint64_t effective_ns = 0;
  const absl::Status status = SetSlice(grpc_thread_slice_ns_, &effective_ns);
  if (status.ok()) {
    logger_(
        SliceLogSeverity::kInfo,
        grpc_thread_slice_ns_ == 0
            ? absl::StrCat("gRPC threads: kernel default EEVDF slice restored "
                           "(effective ",
                           effective_ns, " ns)")
            : absl::StrCat("gRPC threads: EEVDF slice requested ",
                           grpc_thread_slice_ns_, " ns, effective ",
                           effective_ns, " ns"));
    return;
  }
  std::string error(status.message());
  if (grpc_thread_slice_ns_ != 0) {
    uint64_t ignored = 0;
    const absl::Status restore = SetSlice(0, &ignored);
    if (!restore.ok()) {
      absl::StrAppend(&error, "; restoring the default slice also failed: ",
                      restore.message());
    }
    if (!tf_slice_applied_) {
      Disable(absl::StrCat("gRPC thread slice not applied: ", error));
      return;
    }
    if (restore.ok()) {
      state_ = State::kDisabled;
      logger_(SliceLogSeverity::kWarning,
              absl::StrCat("gRPC thread slice not applied: ", error,
                           ". gRPC threads keep the kernel default slice; "
                           "the TensorFlow slice stays in effect."));
      return;
    }
  }
  // Only reachable after the TensorFlow slice was applied: this thread keeps
  // it, and gRPC's threads will inherit it.
  state_ = State::kDisabled;
  logger_(SliceLogSeverity::kError,
          absl::StrCat("gRPC threads could not be moved off the TensorFlow "
                       "EEVDF slice (",
                       error, "). gRPC threads will share the ",
                       tf_thread_slice_ns_,
                       " ns TensorFlow slice, so TensorFlow threads are no "
                       "longer scheduled ahead of them. Serving continues; "
                       "remove --tf_thread_slice_ns if this persists."));
}

}  // namespace serving
}  // namespace tensorflow
