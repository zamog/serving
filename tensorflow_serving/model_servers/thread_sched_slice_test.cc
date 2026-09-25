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
  int get_error = EFAULT;
  std::set<int> failing_get_calls;  // 1-based indices into get calls.
  int set_error = EPERM;
  std::set<int> failing_set_calls;  // 1-based indices into set_calls.
  std::set<int> ignored_set_calls;  // Succeed but change nothing.
  bool sets_ignored = false;        // Every sched_setattr is ignored.
  int get_calls = 0;
  std::vector<KernelSchedAttr> set_calls;

  uint64_t EffectiveSlice() const {
    if (!per_task_slices) return 0;
    return custom_slice != 0 ? custom_slice : kBaseSliceNs;
  }

  SchedAttrSyscalls Syscalls() {
    return {
        [this](KernelSchedAttr* attr) {
          if (failing_get_calls.count(++get_calls) > 0) return get_error;
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
          const int call = static_cast<int>(set_calls.size());
          if (failing_set_calls.count(call) > 0) return set_error;
          // Like the kernel: fair policies take no static priority.
          if (attr.sched_priority != 0) return EINVAL;
          if (sets_ignored || ignored_set_calls.count(call) > 0) return 0;
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

  std::vector<uint64_t> SetRuntimes() const {
    std::vector<uint64_t> result;
    for (const auto& attr : kernel_.set_calls) {
      result.push_back(attr.sched_runtime);
    }
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
  auto setter = MakeSetter(0, 0);
  RunBothPhases(setter);
  EXPECT_EQ(kernel_.get_calls, 0);
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
    EXPECT_EQ(attr.sched_priority, 0);
    EXPECT_EQ(attr.sched_deadline, 0);
    EXPECT_EQ(attr.sched_period, 0);
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

TEST_F(ThreadSliceSetterTest, TensorFlowOnlyRestoresStartingSliceForGrpc) {
  auto setter = MakeSetter(500000, 0);
  EXPECT_EQ(RunBothPhases(setter),
            std::make_pair(uint64_t{500000}, kBaseSliceNs));
  EXPECT_THAT(SetRuntimes(), ElementsAre(500000, 0));
  EXPECT_EQ(kernel_.custom_slice, 0);  // Follows base_slice_ns again.
  ASSERT_THAT(Severities(),
              ElementsAre(SliceLogSeverity::kInfo, SliceLogSeverity::kInfo));
  EXPECT_EQ(logs_[1].second,
            "gRPC threads: starting EEVDF slice restored (2800000 ns)");
}

TEST_F(ThreadSliceSetterTest, InheritedCustomSliceIsRestoredExactly) {
  kernel_.custom_slice = 1500000;  // Set by whatever launched the server.
  auto setter = MakeSetter(500000, 0);
  EXPECT_EQ(RunBothPhases(setter),
            std::make_pair(uint64_t{500000}, uint64_t{1500000}));
  EXPECT_THAT(SetRuntimes(), ElementsAre(500000, 0, 1500000));
  EXPECT_THAT(logs_[1].second, HasSubstr("restored (1500000 ns)"));
}

TEST_F(ThreadSliceSetterTest, GrpcOnlyLeavesTensorFlowThreadsAlone) {
  auto setter = MakeSetter(0, 4000000);
  EXPECT_EQ(RunBothPhases(setter),
            std::make_pair(kBaseSliceNs, uint64_t{4000000}));
  EXPECT_THAT(SetRuntimes(), ElementsAre(4000000));
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

TEST_F(ThreadSliceSetterTest, SingleFlagThatInvertsTheOrderDisables) {
  // Unset flags keep the 2.8 ms starting slice: a 10 ms TensorFlow slice
  // alone, or a 0.5 ms gRPC slice alone, would put gRPC first.
  for (const auto& flags : {std::make_pair(int64_t{10000000}, int64_t{0}),
                            std::make_pair(int64_t{0}, int64_t{500000})}) {
    kernel_ = FakeKernel{};
    logs_.clear();
    auto setter = MakeSetter(flags.first, flags.second);
    EXPECT_EQ(RunBothPhases(setter),
              std::make_pair(kBaseSliceNs, kBaseSliceNs));
    EXPECT_THAT(kernel_.set_calls, IsEmpty());
    ASSERT_THAT(Severities(), ElementsAre(SliceLogSeverity::kWarning));
    EXPECT_THAT(logs_[0].second, HasSubstr("starting slice of 2800000 ns"));
  }
}

TEST_F(ThreadSliceSetterTest, SingleFlagOnTheRightSideIsApplied) {
  auto setter = MakeSetter(kBaseSliceNs, 0);  // Equal is not inverted.
  RunBothPhases(setter);
  EXPECT_TRUE(setter.enabled());
  EXPECT_THAT(Severities(),
              ElementsAre(SliceLogSeverity::kInfo, SliceLogSeverity::kInfo));
}

TEST_F(ThreadSliceSetterTest, OldKernelWarnsOnceAndChangesNothing) {
  for (const auto& flags : {std::make_pair(int64_t{500000}, int64_t{4000000}),
                            std::make_pair(int64_t{500000}, int64_t{0}),
                            std::make_pair(int64_t{0}, int64_t{4000000})}) {
    kernel_ = FakeKernel{};
    kernel_.per_task_slices = false;  // e.g. 6.6: sched_runtime reads 0.
    logs_.clear();
    auto setter = MakeSetter(flags.first, flags.second);
    RunBothPhases(setter);
    EXPECT_THAT(kernel_.set_calls, IsEmpty());
    ASSERT_THAT(Severities(), ElementsAre(SliceLogSeverity::kWarning));
    EXPECT_THAT(logs_[0].second, HasSubstr("needs Linux >= 6.12"));
    EXPECT_THAT(logs_[0].second,
                HasSubstr("Starting with the kernel's default scheduling"));
    EXPECT_FALSE(setter.enabled());
  }
}

TEST_F(ThreadSliceSetterTest, BlockedSyscallWarnsAndChangesNothing) {
  for (const int error : {ENOSYS, EPERM}) {
    kernel_ = FakeKernel{};
    kernel_.get_error = error;
    kernel_.failing_get_calls = {1, 2, 3, 4};
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

TEST_F(ThreadSliceSetterTest, RejectedTensorFlowSetChangesNothingMore) {
  kernel_.custom_slice = 1500000;
  kernel_.failing_set_calls = {1};
  auto setter = MakeSetter(500000, 4000000);
  EXPECT_EQ(RunBothPhases(setter),
            std::make_pair(uint64_t{1500000}, uint64_t{1500000}));
  EXPECT_THAT(SetRuntimes(), ElementsAre(500000));  // No restore needed.
  ASSERT_THAT(Severities(), ElementsAre(SliceLogSeverity::kWarning));
  EXPECT_THAT(logs_[0].second,
              HasSubstr("TensorFlow thread slice not applied: "
                        "sched_setattr(slice 500000 ns) failed"));
  EXPECT_FALSE(setter.enabled());
}

TEST_F(ThreadSliceSetterTest, TensorFlowReadBackMismatchRestores) {
  kernel_.ignored_set_calls = {1};  // Accepted but not applied.
  auto setter = MakeSetter(500000, 4000000);
  RunBothPhases(setter);
  EXPECT_THAT(SetRuntimes(), ElementsAre(500000, 0));
  ASSERT_THAT(Severities(), ElementsAre(SliceLogSeverity::kWarning));
  EXPECT_THAT(logs_[0].second,
              HasSubstr("effective slice of 2800000 ns after a request for "
                        "500000 ns"));
}

TEST_F(ThreadSliceSetterTest, ReadBackFailureAfterAcceptedSetRestores) {
  kernel_.failing_get_calls = {3};  // Probe, pre-read, then the read-back.
  auto setter = MakeSetter(500000, 4000000);
  EXPECT_EQ(RunBothPhases(setter), std::make_pair(kBaseSliceNs, kBaseSliceNs));
  EXPECT_THAT(SetRuntimes(), ElementsAre(500000, 0));
  ASSERT_THAT(Severities(), ElementsAre(SliceLogSeverity::kWarning));
  EXPECT_THAT(logs_[0].second, HasSubstr("sched_getattr failed"));
}

TEST_F(ThreadSliceSetterTest, PreReadFailureIssuesNoSet) {
  kernel_.failing_get_calls = {2};  // The re-read before the first set.
  auto setter = MakeSetter(500000, 4000000);
  RunBothPhases(setter);
  EXPECT_THAT(kernel_.set_calls, IsEmpty());
  ASSERT_THAT(Severities(), ElementsAre(SliceLogSeverity::kWarning));
}

TEST_F(ThreadSliceSetterTest, GrpcSetFailureKeepsTensorFlowSlice) {
  kernel_.failing_set_calls = {2};
  auto setter = MakeSetter(500000, 4000000);
  EXPECT_EQ(RunBothPhases(setter),
            std::make_pair(uint64_t{500000}, kBaseSliceNs));
  EXPECT_THAT(SetRuntimes(), ElementsAre(500000, 4000000, 0));
  ASSERT_THAT(Severities(),
              ElementsAre(SliceLogSeverity::kInfo, SliceLogSeverity::kWarning));
  EXPECT_THAT(logs_[1].second,
              HasSubstr("gRPC threads keep the starting slice (2800000 ns); "
                        "the TensorFlow slice stays in effect"));
  EXPECT_FALSE(setter.enabled());
}

TEST_F(ThreadSliceSetterTest, GrpcReadBackMismatchRestores) {
  kernel_.ignored_set_calls = {2};
  auto setter = MakeSetter(500000, 4000000);
  EXPECT_EQ(RunBothPhases(setter),
            std::make_pair(uint64_t{500000}, kBaseSliceNs));
  ASSERT_THAT(Severities(),
              ElementsAre(SliceLogSeverity::kInfo, SliceLogSeverity::kWarning));
  EXPECT_THAT(logs_[1].second, HasSubstr("after a request for 4000000 ns"));
}

TEST_F(ThreadSliceSetterTest, RestoreThatLeavesTheTensorFlowSliceIsAnError) {
  // The gRPC set and every restore are accepted but change nothing, so the
  // thread stays on the TensorFlow slice; that must not be reported as a
  // successful restore.
  kernel_.ignored_set_calls = {2, 3, 4};
  auto setter = MakeSetter(500000, 4000000);
  EXPECT_EQ(RunBothPhases(setter),
            std::make_pair(uint64_t{500000}, uint64_t{500000}));
  ASSERT_THAT(Severities(),
              ElementsAre(SliceLogSeverity::kInfo, SliceLogSeverity::kError));
  EXPECT_THAT(logs_[1].second, HasSubstr("share the 500000 ns TensorFlow"));
}

TEST_F(ThreadSliceSetterTest, GrpcSetAndRestoreFailureIsAnError) {
  kernel_.failing_set_calls = {2, 3};
  auto setter = MakeSetter(500000, 4000000);
  EXPECT_EQ(RunBothPhases(setter),
            std::make_pair(uint64_t{500000}, uint64_t{500000}));
  ASSERT_THAT(Severities(),
              ElementsAre(SliceLogSeverity::kInfo, SliceLogSeverity::kError));
  EXPECT_THAT(logs_[1].second,
              HasSubstr("restoring the starting slice also failed"));
  EXPECT_THAT(logs_[1].second,
              HasSubstr("share the 500000 ns TensorFlow slice"));
}

TEST_F(ThreadSliceSetterTest, TensorFlowOnlyRestoreFailureIsAnError) {
  kernel_.failing_set_calls = {2};
  auto setter = MakeSetter(500000, 0);
  RunBothPhases(setter);
  EXPECT_THAT(SetRuntimes(), ElementsAre(500000, 0));
  ASSERT_THAT(Severities(),
              ElementsAre(SliceLogSeverity::kInfo, SliceLogSeverity::kError));
}

TEST_F(ThreadSliceSetterTest, GrpcOnlyRejectedSetWarns) {
  kernel_.failing_set_calls = {1};
  auto setter = MakeSetter(0, 4000000);
  RunBothPhases(setter);
  EXPECT_THAT(SetRuntimes(), ElementsAre(4000000));  // Nothing to restore.
  ASSERT_THAT(Severities(), ElementsAre(SliceLogSeverity::kWarning));
  EXPECT_THAT(logs_[0].second, HasSubstr("have no effect"));
}

TEST_F(ThreadSliceSetterTest, GrpcOnlyReadBackMismatchRestores) {
  kernel_.ignored_set_calls = {1};
  auto setter = MakeSetter(0, 4000000);
  RunBothPhases(setter);
  EXPECT_THAT(SetRuntimes(), ElementsAre(4000000, 0));
  ASSERT_THAT(Severities(), ElementsAre(SliceLogSeverity::kWarning));
}

#if defined(__linux__)
// Against the running kernel: on >= 6.12 threads created after each phase
// inherit that phase's slice; on older kernels nothing changes and one
// warning is logged. Skipped where this process cannot use slices at all.
TEST(ThreadSliceSetterRealKernelTest, NewThreadsInheritOrFeatureIsOff) {
  const SchedAttrSyscalls syscalls = LinuxSchedAttrSyscalls();
  KernelSchedAttr start;
  if (syscalls.get(&start) != 0) GTEST_SKIP() << "sched_getattr unavailable";
  const auto own_slice = [&syscalls] {
    KernelSchedAttr attr;
    return syscalls.get(&attr) == 0 ? attr.sched_runtime : ~uint64_t{0};
  };
  const auto slice_of_new_thread = [&own_slice] {
    uint64_t slice = 0;
    std::thread([&] { slice = own_slice(); }).join();
    return slice;
  };

  std::vector<std::pair<SliceLogSeverity, std::string>> logs;
  ThreadSliceSetter setter(
      500000, 4000000,
      [&logs](SliceLogSeverity severity, const std::string& message) {
        logs.emplace_back(severity, message);
      });
  setter.ApplyTensorFlowSlice();
  const uint64_t tf_thread = slice_of_new_thread();
  setter.ApplyGrpcSlice();
  const uint64_t grpc_thread = slice_of_new_thread();

  if (start.sched_runtime == 0) {
    // Kernel without per-task slices.
    ASSERT_EQ(logs.size(), 1);
    EXPECT_EQ(logs[0].first, SliceLogSeverity::kWarning);
    EXPECT_EQ(tf_thread, 0);
    EXPECT_EQ(grpc_thread, 0);
    return;
  }
  if (!setter.enabled()) {
    // Supported kernel, but this process may not use slices (policy,
    // SCHED_RESET_ON_FORK, LSM): the library must have said so and changed
    // nothing.
    ASSERT_EQ(logs.size(), 1);
    EXPECT_EQ(own_slice(), start.sched_runtime);
    GTEST_SKIP() << logs[0].second;
  }
  EXPECT_THAT(logs, ElementsAre(Pair(SliceLogSeverity::kInfo, ::testing::_),
                                Pair(SliceLogSeverity::kInfo, ::testing::_)));
  EXPECT_EQ(tf_thread, 500000);
  EXPECT_EQ(grpc_thread, 4000000);
  // Leave the test thread as found: the kernel default first, then the
  // starting value if it was a custom one.
  KernelSchedAttr attr;
  ASSERT_EQ(syscalls.get(&attr), 0);
  attr.sched_flags = 0;
  attr.sched_runtime = 0;
  EXPECT_EQ(syscalls.set(attr), 0);
  if (own_slice() != start.sched_runtime) {
    attr.sched_runtime = start.sched_runtime;
    EXPECT_EQ(syscalls.set(attr), 0);
  }
  EXPECT_EQ(own_slice(), start.sched_runtime);
}
#endif

}  // namespace
}  // namespace serving
}  // namespace tensorflow
