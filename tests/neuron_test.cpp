#include "neuron.hpp"
#include "activation.hpp"

#include <gtest/gtest.h>

using namespace axon;

namespace
{

    auto create_input_neuron(double input, std::size_t index, std::size_t next_layer_size) -> Neuron
    {
        Neuron neuron(next_layer_size, index);
        neuron.set_output(input);
        return neuron;
    }

} // namespace

TEST(NeuronTest, InputNeuronStoresValueCorrectly)
{
    Neuron input(1, 0);
    input.set_output(0.5);

    EXPECT_DOUBLE_EQ(input.get_output(), 0.5);
}

TEST(NeuronTest, InputNeuronCannotFeedForward)
{
    Neuron input(1, 0);
    std::vector<Neuron> dummy_prev;
    dummy_prev.emplace_back(1, 0);

    EXPECT_THROW(input.feed_forward(dummy_prev), std::logic_error);
}

TEST(NeuronTest, BiasNeuronOutputIsOne)
{
    Neuron bias(1, 0);
    bias.set_output(1.0);

    EXPECT_DOUBLE_EQ(bias.get_output(), 1.0);
}

TEST(NeuronTest, GradientCanBeSetAndRetrieved)
{
    Neuron neuron(0, 0);
    neuron.set_gradient(0.15);

    EXPECT_DOUBLE_EQ(neuron.get_gradient(), 0.15);
}

TEST(NeuronTest, NeuronHasCorrectNumberOfConnections)
{
    Neuron neuron(10, 0);

    EXPECT_EQ(neuron.get_connections().size(), 10);
}

TEST(NeuronTest, ConnectionsAreRandomlyInitialized)
{
    Neuron neuron1(10, 0);
    Neuron neuron2(10, 0);

    bool has_different_weight{false};
    for (std::size_t i{0}; i < 10; ++i)
    {
        if (neuron1.get_connections()[i].weight != neuron2.get_connections()[i].weight)
        {
            has_different_weight = true;
            break;
        }
    }

    EXPECT_TRUE(has_different_weight);
}

TEST(NeuronTest, SingleNeuronForwardPassWithLinearActivation)
{
    std::vector<Neuron> inputs;
    inputs.emplace_back(create_input_neuron(0.75, 0, 1));
    inputs.emplace_back(create_input_neuron(0.25, 1, 1));

    const Activation linear_activation{.function = activation::linear,
                                       .derivative = activation::linear_derivative};

    Neuron output(0, 0, linear_activation);
    output.feed_forward(inputs);

    // NOTE(abi): output should be non-zero since weights are random.
    EXPECT_NE(output.get_output(), 0.0);
}
