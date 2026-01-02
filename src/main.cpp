#include "neuron.hpp"
#include "network.hpp"
#include "activation.hpp"

#include <print>

using namespace ann;

auto main() -> int
{
    Activation nonlinear_activation{.function = activation::tanh,
                                    .derivative = activation::tanh_derivative};
    Network net({2, 4, 1}, nonlinear_activation);

    constexpr std::array xor_inputs = {std::array{0.0, 0.0}, std::array{0.0, 1.0},
                                       std::array{1.0, 0.0}, std::array{1.0, 1.0}};

    for (const auto& input : xor_inputs)
    {
        net.feed_forward({input[0], input[1]});
        std::println("[{}, {}] -> {:.4f}", input[0], input[1], net.get_output()[0]);
    }

    return 0;
}
