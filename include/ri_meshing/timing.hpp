#pragma once

#include <chrono>
#include <string>
#include <utility>
#include <vector>

namespace ri_meshing {

struct StageTiming {
    std::string name;
    double milliseconds{};
};

struct PipelineTiming {
    double total_milliseconds{};
    bool synchronize_cuda{true};
    std::vector<StageTiming> stages;

    void add_stage(std::string name, double milliseconds)
    {
        stages.push_back(StageTiming{std::move(name), milliseconds});
    }
};

class Stopwatch {
public:
    Stopwatch() : start_(Clock::now()) {}

    [[nodiscard]] double elapsed_milliseconds() const
    {
        const auto elapsed = Clock::now() - start_;
        return std::chrono::duration<double, std::milli>(elapsed).count();
    }

private:
    using Clock = std::chrono::steady_clock;
    Clock::time_point start_;
};

}  // namespace ri_meshing
