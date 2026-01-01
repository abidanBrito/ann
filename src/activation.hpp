#pragma once

#include <cmath>

namespace ann::activation
{

    [[nodiscard]] constexpr auto linear(double x) -> double
    {
        return x;
    }

    [[nodiscard]] constexpr auto linear_derivative([[maybe_unused]] double x) -> double
    {
        return 1.0;
    }

} // namespace ann::activation
