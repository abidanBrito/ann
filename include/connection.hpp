#pragma once

namespace ann
{

    struct Connection
    {
        double weight{0.0};
        double delta_weight{0.0};

        Connection() = default;

        explicit Connection(double initial_weight)
            : weight{initial_weight}
        {
        }
    };

} // namespace ann
