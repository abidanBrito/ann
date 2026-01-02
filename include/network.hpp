#pragma once

#include "neuron.hpp"

#include <vector>

namespace ann
{

    class Network
    {
    public:
        explicit Network(const std::vector<std::size_t>& layer_sizes, Activation);

        [[nodiscard]] auto get_output() const -> std::vector<double>;

        auto feed_forward(const std::vector<double>& inputs) -> void;

    private:
        std::vector<std::vector<Neuron>> layers_;
        Activation activation_;
    };

} // namespace ann
