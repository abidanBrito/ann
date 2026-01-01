#pragma once

#include "connection.hpp"

#include <vector>
#include <functional>

namespace ann
{

    class Neuron
    {
    public:
        struct Activation
        {
            using Function = std::function<double(double)>;

            Function function;
            Function derivative;
        };

        Neuron(std::size_t num_outputs, std::size_t index, Activation activation);

        auto set_output(double value) -> void
        {
            output_value_ = value;
        }

        [[nodiscard]] auto get_output() const -> double
        {
            return output_value_;
        }

        [[nodiscard]] auto get_weights() const -> const std::vector<Connection>&
        {
            return output_weights_;
        }

        auto feed_forward(const std::vector<Neuron>& prev_layer) -> void;

    private:
        Activation activation_;
        std::size_t index_;

        double output_value_{0.0};
        std::vector<Connection> output_weights_;
    };

} // namespace ann
