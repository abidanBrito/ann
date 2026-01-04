#include "activation.hpp"

#include <gtest/gtest.h>
#include <cmath>

using namespace axon::activation;

TEST(ActivationTest, LinearReturnsIdenticalInput)
{
    EXPECT_DOUBLE_EQ(linear(0.0), 0.0);
    EXPECT_DOUBLE_EQ(linear(1.5), 1.5);
    EXPECT_DOUBLE_EQ(linear(-5.0), -5.0);
}

TEST(ActivationTest, LinearDerivativeIsOne)
{
    EXPECT_DOUBLE_EQ(linear_derivative(0.0), 1.0);
    EXPECT_DOUBLE_EQ(linear_derivative(1.5), 1.0);
    EXPECT_DOUBLE_EQ(linear_derivative(-5.0), 1.0);
}

TEST(ActivationTest, SigmoidRangeIsZeroToOne)
{
    EXPECT_NEAR(sigmoid(-100.0), 0.0, 1e-10);
    EXPECT_NEAR(sigmoid(100.0), 1.0, 1e-10);
    EXPECT_NEAR(sigmoid(0.0), 0.5, 1e-10);
}

TEST(ActivationTest, SigmoidDerivativeIsCorrect)
{
    double output = sigmoid(0.0);
    EXPECT_NEAR(sigmoid_derivative(output), 0.25, 1e-10);
}

TEST(ActivationTest, TanhRangeIsMinusOneToOne)
{
    EXPECT_NEAR(std::tanh(-100.0), -1.0, 1e-10);
    EXPECT_NEAR(std::tanh(100.0), 1.0, 1e-10);
    EXPECT_NEAR(std::tanh(0.0), 0.0, 1e-10);
}

TEST(ActivationTest, TanhDerivativeIsCorrect)
{
    double output = std::tanh(0.0);
    EXPECT_NEAR(tanh_derivative(output), 1.0, 1e-10);

    output = std::tanh(100.0);
    EXPECT_NEAR(tanh_derivative(output), 0.0, 1e-6);
}

TEST(ActivationTest, ReLUPositivePassthrough)
{
    EXPECT_DOUBLE_EQ(relu(0.0), 0.0);
    EXPECT_DOUBLE_EQ(relu(5.0), 5.0);
    EXPECT_DOUBLE_EQ(relu(10.0), 10.0);
}

TEST(ActivationTest, ReLUNegativeClipped)
{
    EXPECT_DOUBLE_EQ(relu(-0.5), 0.0);
    EXPECT_DOUBLE_EQ(relu(-10.0), 0.0);
}

TEST(ActivationTest, ReLUDerivativeIsCorrect)
{
    EXPECT_DOUBLE_EQ(relu_derivative(5.0), 1.0);
    EXPECT_DOUBLE_EQ(relu_derivative(0.0), 0.0);
    EXPECT_DOUBLE_EQ(relu_derivative(-5.0), 0.0);
}
