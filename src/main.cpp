#include "neuron.hpp"
#include "network.hpp"
#include "activation.hpp"

#include <print>

using namespace ann;

namespace
{

    auto create_input_neuron(double input, std::size_t index, std::size_t next_layer_size) -> Neuron
    {
        Neuron neuron(next_layer_size, index);
        neuron.set_output(input);

        return neuron;
    }

} // namespace

auto main() -> int
{
    std::println("--- Testing forward pass (2 input, 1 output) ---");

    std::vector<ann::Neuron> inputs;
    inputs.emplace_back(create_input_neuron(0.75, 0, 1));
    inputs.emplace_back(create_input_neuron(0.25, 1, 1));

    Activation linear_activation{.function = activation::linear,
                                 .derivative = activation::linear_derivative};
    Neuron output(0, 0, linear_activation);

    output.feed_forward(inputs);
    std::println("Output: {:.4f}", output.get_output());

    std::println("\n--- Testing Network forward pass (2 input, 4 hidden, 1 output) ---");

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
