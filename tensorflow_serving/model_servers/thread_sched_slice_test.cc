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
#include <cstdint>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"

namespace tensorflow {
namespace serving {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Pair;

constexpr uint32_t kSchedOther = 0;
constexpr uint32_t kSchedFifo = 1;
constexpr uint32_t kSchedBatch = 3;
constexpr uint32_t kSchedIdle = 5;
constexpr uint64_t kBaseSliceNs = 2800000;

std::string ValidationError(int64_t tf_ns, int64_t grpc_ns) {
  const absl::Status status = ValidateThreadSliceFlags(tf_ns, grpc_ns);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  return std::string(status.message());
}

// Models the scheduler state of one thread as sched_getattr/sched_setattr see
// it on Linux >= 6.12 (per_task_slices) or on older kernels.
class FakeKernel {
 public:
  bool per_task_slices = true;
  uint32_t policy = kSchedOther;
  int32_t nice = 0;
  uint64_t flags = 0;
  uint64_t custom_slice = 0;  // 0 = none, the base slice applies.
  int get_error = 0;
  int set_error = EPERM;
  std::set<int> failing_set_calls;  // 1-based indices into set_calls.
  bool sets_ignored = false;        // sched_setattr succeeds but does nothing.
  std::vector<KernelSchedAttr> set_calls;

  uint64_t EffectiveSlice() const {
    if (!per_task_slices) return 0;
    return custom_slice != 0 ? custom_slice : kBaseSliceNs;
  }

  SchedAttrSyscalls Syscalls() {
    return {
        [this](KernelSchedAttr* attr) {
          if (get_error != 0) return get_error;
          *attr = KernelSchedAttr{};
          attr->size = 48;
          attr->sched_policy = policy;
          attr->sched_flags = flags;
          attr->sched_nice = nice;
          attr->sched_runtime = EffectiveSlice();
          return 0;
        },
        [this](const KernelSchedAttr& attr) {
          set_calls.push_back(attr);
          if (failing_set_calls.count(static_cast<int>(set_calls.size())) > 0) {
            return set_error;
          }
          if (sets_ignored) return 0;
          policy = attr.sched_policy;
          nice = attr.sched_nice;
          if (per_task_slices) custom_slice = attr.sched_runtime;
          return 0;
        },
    };
  }
};

class ThreadSliceSetterTest : public ::testing::Test {
 protected:
  ThreadSliceSetter MakeSetter(int64_t tf_ns, int64_t grpc_ns) {
    return ThreadSliceSetter(
        tf_ns, grpc_ns,
        [this](SliceLogSeverity severity, const std::string& message) {
          logs_.emplace_back(severity, message);
        },
        kernel_.Syscalls());
  }

  // Runs both phases the way Server::BuildAndStart does and returns the
  // slice the TensorFlow threads and the gRPC threads would inherit.
  std::pair<uint64_t, uint64_t> RunBothPhases(ThreadSliceSetter& setter) {
    setter.ApplyTensorFlowSlice();
    const uint64_t tf_threads = kernel_.EffectiveSlice();
    setter.ApplyGrpcSlice();
    return {tf_threads, kernel_.EffectiveSlice()};
  }

  std::vector<SliceLogSeverity> Severities() const {
    std::vector<SliceLogSeverity> result;
    for (const auto& log : logs_) result.push_back(log.first);
    return result;
  }

  FakeKernel kernel_;
  std::vector<std::pair<SliceLogSeverity, std::string>> logs_;
};

TEST(ValidateThreadSliceFlagsTest, AcceptsValidPairs) {
  EXPECT_TRUE(ValidateThreadSliceFlags(0, 0).ok());
  EXPECT_TRUE(ValidateThreadSliceFlags(500000, 4000000).ok());
  EXPECT_TRUE(ValidateThreadSliceFlags(500000, 0).ok());
  EXPECT_TRUE(ValidateThreadSliceFlags(0, 4000000).ok());
  EXPECT_TRUE(ValidateThreadSliceFlags(1000000, 1000000).ok());
  EXPECT_TRUE(
      ValidateThreadSliceFlags(kMinThreadSliceNs, kMaxThreadSliceNs).ok());
}

TEST(ValidateThreadSliceFlagsTest, RejectsNegative) {
  EXPECT_THAT(ValidationError(-1, 0), HasSubstr("must not be negative"));
  EXPECT_THAT(ValidationError(0, -1), HasSubstr("must not be negative"));
}

TEST(ValidateThreadSliceFlagsTest, RejectsValuesTheKernelWouldClamp) {
  EXPECT_THAT(ValidationError(1000, 0),
              HasSubstr("server_options.tf_thread_slice_ns (1000) must be 0 "
                        "or between 100000 and 100000000 ns"));
  EXPECT_THAT(ValidationError(0, kMaxThreadSliceNs + 1),
              HasSubstr("server_options.grpc_thread_slice_ns (100000001) must "
                        "be 0 or between"));
}

TEST(ValidateThreadSliceFlagsTest, RejectsInvertedPair) {
  EXPECT_THAT(ValidationError(4000000, 500000),
              HasSubstr("server_options.tf_thread_slice_ns (4000000) must not "
                        "be greater than server_options.grpc_thread_slice_ns "
                        "(500000)"));
}

TEST_F(ThreadSliceSetterTest, BothFlagsZeroIssuesNoSyscalls) {
  kernel_.get_error = EFAULT;  // Any syscall would log.
  auto setter = MakeSetter(0, 0);
  RunBothPhases(setter);
  EXPECT_THAT(kernel_.set_calls, IsEmpty());
  EXPECT_THAT(logs_, IsEmpty());
  EXPECT_FALSE(setter.enabled());
}

TEST_F(ThreadSliceSetterTest, AppliesBothSlices) {
  auto setter = MakeSetter(500000, 4000000);
  EXPECT_EQ(RunBothPhases(setter),
            std::make_pair(uint64_t{500000}, uint64_t{4000000}));
  ASSERT_EQ(kernel_.set_calls.size(), 2);
  for (const KernelSchedAttr& attr : kernel_.set_calls) {
    EXPECT_EQ(attr.size, 48);
    EXPECT_EQ(attr.sched_policy, kSchedOther);
    EXPECT_EQ(attr.sched_flags, 0);  // No SCHED_FLAG_KEEP_PARAMS.
  }
  EXPECT_THAT(
      logs_,
      ElementsAre(
          Pair(SliceLogSeverity::kInfo,
               "TensorFlow threads: EEVDF slice requested 500000 ns, "
               "effective 500000 ns"),
          Pair(SliceLogSeverity::kInfo,
               "gRPC threads: EEVDF slice requested 4000000 ns, effective "
               "4000000 ns")));
  EXPECT_TRUE(setter.enabled());
}

TEST_F(ThreadSliceSetterTest, TensorFlowOnlyRestoresDefaultForGrpc) {
  auto setter = MakeSetter(500000, 0);
  EXPECT_EQ(RunBothPhases(setter),
            std::make_pair(uint64_t{500000}, kBaseSliceNs));
  ASSERT_EQ(kernel_.set_calls.size(), 2);
  EXPECT_EQ(kernel_.set_calls[1].sched_runtime, 0);
  EXPECT_THAT(logs_[1].second,
              HasSubstr("gRPC threads: kernel default EEVDF slice restored "
                        "(effective 2800000 ns)"));
}

TEST_F(ThreadSliceSetterTest, GrpcOnlyLeavesTensorFlowThreadsAlone) {
  auto setter = MakeSetter(0, 4000000);
  EXPECT_EQ(RunBothPhases(setter),
            std::make_pair(kBaseSliceNs, uint64_t{4000000}));
  ASSERT_EQ(kernel_.set_calls.size(), 1);
  EXPECT_THAT(Severities(), ElementsAre(SliceLogSeverity::kInfo));
}

TEST_F(ThreadSliceSetterTest, KeepsPolicyAndNice) {
  kernel_.policy = kSchedBatch;
  kernel_.nice = 5;
  auto setter = MakeSetter(500000, 4000000);
  RunBothPhases(setter);
  ASSERT_EQ(kernel_.set_calls.size(), 2);
  for (const KernelSchedAttr& attr : kernel_.set_calls) {
    EXPECT_EQ(attr.sched_policy, kSchedBatch);
    EXPECT_EQ(attr.sched_nice, 5);
  }
}

TEST_F(ThreadSliceSetterTest, OldKernelWarnsOnceAndChangesNothing) {
  kernel_.per_task_slices = false;  // e.g. 6.6: sched_runtime reads back 0.
  auto setter = MakeSetter(500000, 4000000);
  RunBothPhases(setter);
  EXPECT_THAT(kernel_.set_calls, IsEmpty());
  ASSERT_THAT(Severities(), ElementsAre(SliceLogSeverity::kWarning));
  EXPECT_THAT(logs_[0].second, HasSubstr("needs Linux >= 6.12"));
  EXPECT_THAT(logs_[0].second,
              HasSubstr("Starting with the kernel's default scheduling"));
  EXPECT_FALSE(setter.enabled());
}

TEST_F(ThreadSliceSetterTest, OldKernelGrpcOnlyWarnsOnce) {
  kernel_.per_task_slices = false;
  auto setter = MakeSetter(0, 4000000);
  RunBothPhases(setter);
  EXPECT_THAT(kernel_.set_calls, IsEmpty());
  EXPECT_THAT(Severities(), ElementsAre(SliceLogSeverity::kWarning));
}

TEST_F(ThreadSliceSetterTest, BlockedSyscallWarnsAndChangesNothing) {
  for (const int error : {ENOSYS, EPERM}) {
    kernel_ = FakeKernel{};
    kernel_.get_error = error;
    logs_.clear();
    auto setter = MakeSetter(500000, 4000000);
    RunBothPhases(setter);
    EXPECT_THAT(kernel_.set_calls, IsEmpty());
    ASSERT_THAT(Severities(), ElementsAre(SliceLogSeverity::kWarning));
    EXPECT_THAT(logs_[0].second, HasSubstr("sched_getattr failed"));
  }
}

TEST_F(ThreadSliceSetterTest, NonFairPolicyIsLeftUnchanged) {
  for (const uint32_t policy : {kSchedFifo, kSchedIdle}) {
    kernel_ = FakeKernel{};
    kernel_.policy = policy;
    logs_.clear();
    auto setter = MakeSetter(500000, 4000000);
    RunBothPhases(setter);
    EXPECT_THAT(kernel_.set_calls, IsEmpty());
    EXPECT_EQ(kernel_.policy, policy);
    ASSERT_THAT(Severities(), ElementsAre(SliceLogSeverity::kWarning));
    EXPECT_THAT(logs_[0].second, HasSubstr("scheduling policy"));
  }
}

TEST_F(ThreadSliceSetterTest, ResetOnForkDisables) {
  kernel_.flags = 0x01;
  auto setter = MakeSetter(500000, 4000000);
  RunBothPhases(setter);
  EXPECT_THAT(kernel_.set_calls, IsEmpty());
  ASSERT_THAT(Severities(), ElementsAre(SliceLogSeverity::kWarning));
  EXPECT_THAT(logs_[0].second, HasSubstr("SCHED_RESET_ON_FORK"));
}

TEST_F(ThreadSliceSetterTest, TensorFlowSetFailureDisablesBothPhases) {
  kernel_.failing_set_calls = {1};
  auto setter = MakeSetter(500000, 4000000);
  EXPECT_EQ(RunBothPhases(setter), std::make_pair(kBaseSliceNs, kBaseSliceNs));
  ASSERT_EQ(kernel_.set_calls.size(), 2);  // The failed set, then a restore.
  EXPECT_EQ(kernel_.set_calls[1].sched_runtime, 0);
  ASSERT_THAT(Severities(), ElementsAre(SliceLogSeverity::kWarning));
  EXPECT_THAT(logs_[0].second,
              HasSubstr("TensorFlow thread slice not applied: "
                        "sched_setattr(slice 500000 ns) failed"));
  EXPECT_FALSE(setter.enabled());
}

TEST_F(ThreadSliceSetterTest, ReadBackMismatchRestoresDefault) {
  kernel_.sets_ignored = true;  // Accepted but not applied.
  auto setter = MakeSetter(500000, 4000000);
  RunBothPhases(setter);
  ASSERT_EQ(kernel_.set_calls.size(), 2);
  EXPECT_EQ(kernel_.set_calls[1].sched_runtime, 0);
  ASSERT_THAT(Severities(), ElementsAre(SliceLogSeverity::kWarning));
  EXPECT_THAT(logs_[0].second,
              HasSubstr("effective slice of 2800000 ns after a request for "
                        "500000 ns"));
}

TEST_F(ThreadSliceSetterTest, GrpcSetFailureKeepsTensorFlowSlice) {
  kernel_.failing_set_calls = {2};
  auto setter = MakeSetter(500000, 4000000);
  EXPECT_EQ(RunBothPhases(setter),
            std::make_pair(uint64_t{500000}, kBaseSliceNs));
  ASSERT_THAT(Severities(),
              ElementsAre(SliceLogSeverity::kInfo, SliceLogSeverity::kWarning));
  EXPECT_THAT(logs_[1].second,
              HasSubstr("gRPC threads keep the kernel default slice; the "
                        "TensorFlow slice stays in effect"));
  EXPECT_FALSE(setter.enabled());
}

TEST_F(ThreadSliceSetterTest, GrpcSetAndRestoreFailureIsAnError) {
  kernel_.failing_set_calls = {2, 3};
  auto setter = MakeSetter(500000, 4000000);
  EXPECT_EQ(RunBothPhases(setter),
            std::make_pair(uint64_t{500000}, uint64_t{500000}));
  ASSERT_THAT(Severities(),
              ElementsAre(SliceLogSeverity::kInfo, SliceLogSeverity::kError));
  EXPECT_THAT(logs_[1].second, HasSubstr("restoring the default slice also "
                                         "failed"));
  EXPECT_THAT(logs_[1].second, HasSubstr("share the 500000 ns TensorFlow "
                                         "slice"));
}

TEST_F(ThreadSliceSetterTest, TensorFlowOnlyRestoreFailureIsAnError) {
  kernel_.failing_set_calls = {2};
  auto setter = MakeSetter(500000, 0);
  RunBothPhases(setter);
  ASSERT_EQ(kernel_.set_calls.size(), 2);  // No second restore attempt.
  ASSERT_THAT(Severities(),
              ElementsAre(SliceLogSeverity::kInfo, SliceLogSeverity::kError));
}

TEST_F(ThreadSliceSetterTest, GrpcOnlySetFailureWarns) {
  kernel_.failing_set_calls = {1};
  auto setter = MakeSetter(0, 4000000);
  RunBothPhases(setter);
  ASSERT_THAT(Severities(), ElementsAre(SliceLogSeverity::kWarning));
  EXPECT_THAT(logs_[0].second, HasSubstr("have no effect"));
}

#if defined(__linux__)
// Against the running kernel: on >= 6.12 threads created after each phase
// inherit that phase's slice; on older kernels nothing changes and one
// warning is logged. Either way the process keeps running.
TEST(ThreadSliceSetterRealKernelTest, NewThreadsInheritOrFeatureIsOff) {
  const SchedAttrSyscalls syscalls = LinuxSchedAttrSyscalls();
  const auto own_slice = [&syscalls] {
    KernelSchedAttr attr;
    return syscalls.get(&attr) == 0 ? attr.sched_runtime : ~uint64_t{0};
  };
  const auto slice_of_new_thread = [&own_slice] {
    uint64_t slice = 0;
    std::thread([&] { slice = own_slice(); }).join();
    return slice;
  };
  const uint64_t before = own_slice();
  ASSERT_NE(before, ~uint64_t{0});

  std::vector<SliceLogSeverity> severities;
  ThreadSliceSetter setter(
      500000, 4000000,
      [&severities](SliceLogSeverity severity, const std::string&) {
        severities.push_back(severity);
      });
  setter.ApplyTensorFlowSlice();
  const uint64_t tf_thread = slice_of_new_thread();
  setter.ApplyGrpcSlice();
  const uint64_t grpc_thread = slice_of_new_thread();

  if (before == 0) {
    EXPECT_THAT(severities, ElementsAre(SliceLogSeverity::kWarning));
    EXPECT_EQ(tf_thread, 0);
    EXPECT_EQ(grpc_thread, 0);
  } else {
    EXPECT_THAT(severities,
                ElementsAre(SliceLogSeverity::kInfo, SliceLogSeverity::kInfo));
    EXPECT_EQ(tf_thread, 500000);
    EXPECT_EQ(grpc_thread, 4000000);
    KernelSchedAttr attr;
    ASSERT_EQ(syscalls.get(&attr), 0);
    attr.sched_flags = 0;
    attr.sched_runtime = 0;
    EXPECT_EQ(syscalls.set(attr), 0);  // Leave the test thread as found.
  }
}
#endif

}  // namespace
}  // namespace serving
}  // namespace tensorflow
