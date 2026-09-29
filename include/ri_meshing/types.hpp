#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace ri_meshing {

using MeshScalar = float;

struct Vec3f {
    float x{};
    float y{};
    float z{};
};

using Mat4f = std::array<float, 16>;

struct Triangle {
    int v0{};
    int v1{};
    int v2{};
};

template <typename T>
struct Grid {
    std::size_t height{};
    std::size_t width{};
    std::vector<T> values{};

    Grid() = default;
    Grid(std::size_t rows, std::size_t cols, T fill)
        : height(rows), width(cols), values(rows * cols, fill)
    {
    }

    [[nodiscard]] bool empty() const noexcept { return values.empty(); }
    [[nodiscard]] std::size_t size() const noexcept { return values.size(); }

    T& operator()(std::size_t row, std::size_t col)
    {
        return values[row * width + col];
    }

    const T& operator()(std::size_t row, std::size_t col) const
    {
        return values[row * width + col];
    }
};

}  // namespace ri_meshing
