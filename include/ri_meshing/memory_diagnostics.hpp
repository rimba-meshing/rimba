#pragma once

#include <string_view>

namespace ri_meshing {

// Memory snapshots are emitted from the frame-execution thread.  The memory
// build is deliberately sequential, so a thread-local context is sufficient
// and cannot race with an asynchronous prefetch worker.
#if defined(RI_MESHING_ENABLE_MEMORY_INSTRUMENTATION)
inline thread_local std::string_view current_memory_pass_kind{"unknown"};
inline thread_local int current_memory_run_index{-1};

inline void set_memory_pass(std::string_view pass_kind, int run_index)
{
    current_memory_pass_kind = pass_kind;
    current_memory_run_index = run_index;
}
#else
inline void set_memory_pass(std::string_view, int)
{
}
#endif

}  // namespace ri_meshing
