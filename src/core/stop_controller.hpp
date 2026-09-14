// SPDX-License-Identifier: Apache-2.0
// SANKHYA - unified stop condition checker.

#pragma once

#include <atomic>
#include <chrono>

#include "sankhya/model.hpp"
#include "sankhya/solve_control.hpp"
#include "sankhya/timer.hpp"

namespace sankhya {

/// Unified checker for time limits, user interruptions, and progress callbacks.
///
/// Designed to be called at safe points inside the solver engines. The callback's window
/// is kept by the SolveControl it checks (see there for why), so one is built per engine
/// call without the callback firing once per engine call.
class StopController {
 public:
  StopController(SolveControl* control, const Timer& timer, double time_limit)
      : control_(control), timer_(timer), time_limit_(time_limit) {}

  /// Check if the solver should stop.
  /// `get_progress` is a callable returning `Progress`, evaluated only when the callback is
  /// due.
  template <typename F>
  bool should_stop(F&& get_progress, SolveStatus* out_status) {
    if (control_ && control_->interruption_requested()) {
      *out_status = SolveStatus::kInterrupted;
      return true;
    }

    const double elapsed = timer_.elapsed_seconds();
    if (elapsed > time_limit_) {
      *out_status = SolveStatus::kTimeLimit;
      return true;
    }

    if (control_ && control_->progress_callback) {
      if (control_->callback_due()) {
        Progress p = get_progress();
        p.elapsed_seconds = elapsed;
        if (control_->progress_callback(p) != 0) {
          control_->interrupt();
          *out_status = SolveStatus::kInterrupted;
          return true;
        }
        if (control_->interruption_requested()) {
          *out_status = SolveStatus::kInterrupted;
          return true;
        }
      }
    }

    return false;
  }

 private:
  SolveControl* control_;
  const Timer& timer_;
  double time_limit_;
};

}  // namespace sankhya
