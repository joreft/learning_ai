#pragma once

#include <array>
#include <cstdint>
#include <random>
#include <cstdlib>
#include <random>
#include "matrix_math.h"
#include <Eigen/Dense>
#include <iostream>

#include <fmt/format.h>
#include <functional>
#include <utility>

namespace jeagle
{

template<std::size_t NUMBER_OF_NODES, std::size_t NEXT_LAYER_SIZE> // Next layer size is 0 for the final layer
struct Layer
{
    static constexpr std::size_t number_of_nodes = NUMBER_OF_NODES;
    static constexpr std::size_t weights_per_node = NEXT_LAYER_SIZE;

    Eigen::Matrix<double, weights_per_node, number_of_nodes> weights;
    Eigen::Matrix<double, number_of_nodes, 1> activation_values;
    Eigen::Matrix<double, NEXT_LAYER_SIZE, 1> biases;

//    std::vector<double> activation_values;
    //std::vector<double> weights;
   // std::vector<double> biases;

    auto calculate_activation_values_for_next_layer()
    {
        auto out = weights * activation_values + biases;

        auto new_val = out.unaryExpr([](double val) {return sigmoid(val);});

        return new_val;
    }
};

template <typename LayerType>
void assign_random_weights(LayerType& layer)
{
    std::random_device rd;
    std::mt19937 g(rd());
    std::normal_distribution<> d{0,1};
    std::normal_distribution<> for_bias{0,40};
    auto set_random = [&](auto)
    {
        // TODO replace with proper random generator
        return d(g);
    };

    auto set_random_1 = [&](auto)
    {
        // TODO replace with proper random generator
        return for_bias(g);
    };


    layer.weights = layer.weights.unaryExpr(set_random);
    layer.biases = layer.biases.unaryExpr(set_random_1);
}

struct WeightsAdjustment
{
    std::pair<Eigen::Matrix<double, 10, 1>, Eigen::Matrix<double, 10, 16>> layer_3;
    std::pair<Eigen::Matrix<double, 16, 1>, Eigen::Matrix<double, 16, 784>> layer_1;
};

struct SampleNetwork
{
    SampleNetwork()
    {
        assign_random_weights(first_layer);
        assign_random_weights(second_layer);
        //assign_random_weights(third_layer);
    }

    Layer<784, 16> first_layer;
    Layer<16, 10> second_layer;
//    Layer<16, 10> third_layer;
    Layer<10, 0> final_layer;

    void process_input(decltype(Layer<784, 16>::activation_values) first_layer_activation)
    {
        first_layer.activation_values = std::move(first_layer_activation);
        //std::cout <<  "First layer"  << first_layer.activation_values << "\n";
        second_layer.activation_values = first_layer.calculate_activation_values_for_next_layer();
        //std::cout <<  "Second layer"  << second_layer.activation_values << "\n";
        //third_layer.activation_values = second_layer.calculate_activation_values_for_next_layer();
        final_layer.activation_values = second_layer.calculate_activation_values_for_next_layer();
        //std::cout <<  "Final layer"  << final_layer.activation_values << "\n";
    }

    auto backpropagate(int target_value)
    {
        auto final_activations = final_layer.activation_values;

        Eigen::Matrix<double, 10, 1> target_vector;
        target_vector[target_value] = 1;

        auto constexpr sigmoid_differentiate = [](auto& val)
        {
            return sigmoid_derivative(val);
        };

        auto nabla_c = final_activations - target_vector;

        auto const sigmoid_derivative_output = final_activations.unaryExpr(sigmoid_differentiate);

        auto const output_error = nabla_c.cwiseProduct(sigmoid_derivative_output); // hadamard product

        std::cout << "Activation values: \n" << final_activations << "\n";

        std::cout << "Output error: \n" << output_error << "\n";

        auto const nabla_biases_1 = output_error;

        // dot the output error with activation values from the preceding layer (transposed)
        auto const nabla_weights_1 = output_error * second_layer.activation_values.transpose();

        //std::cout << "SIZESIZESIZEsize " << nabla_weights_1 << "\n";

       // second_layer.weights -= nabla_weights_1;
//        first_layer.biases -= nabla_biases_1;

        //auto const nabla_weights_2
        {

        }
        auto const layer_3 = std::pair<Eigen::Matrix<double, 10, 1>, Eigen::Matrix<double, 10, 16>>{ nabla_biases_1, nabla_weights_1 };

        auto const second_layer_activations_der = second_layer.activation_values.unaryExpr(sigmoid_differentiate);
        auto const delta = (second_layer.weights.transpose() * output_error).cwiseProduct(second_layer_activations_der);
        auto const biases_3 = delta;

        auto const weights_3 = delta * first_layer.activation_values.transpose();

        auto const layer_1 = std::pair{biases_3, weights_3};

        return WeightsAdjustment{layer_3, layer_1};
    }
};

} // namespace jeagle