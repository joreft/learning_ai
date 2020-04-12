#pragma once

#include <array>
#include <cstdint>
#include <random>
#include <cstdlib>
#include <random>
#include "matrix_math.h"

#include <fmt/format.h>

namespace jeagle
{

template<std::size_t NUMBER_OF_NODES, std::size_t NEXT_LAYER_SIZE> // Next layer size is 0 for the final layer
struct Layer
{
    static constexpr std::size_t number_of_nodes = NUMBER_OF_NODES;
    static constexpr std::size_t weights_per_node = NEXT_LAYER_SIZE;

    std::vector<float> activation_values;
    std::vector<float> weights;
    std::vector<float> biases;

    std::vector<float> calculate_activation_values_for_next_layer()
    {
        Matrix<NEXT_LAYER_SIZE, number_of_nodes> weight_matrix (weights);
        ColumnVector<NUMBER_OF_NODES> activation (activation_values);

//        biases = std::vector<float>(NEXT_LAYER_SIZE, 0);
        std::vector<float> out = ((weight_matrix * activation) + ColumnVector<NEXT_LAYER_SIZE> {biases}).values;

        for (auto& val : out)
        {
            val = sigmoid(val);
        }
        if (out.size() != NEXT_LAYER_SIZE)
        {
            fmt::print("Length is {}\n", out.size());
            throw 2; // TODO oh well
        }
        return out;
    }


};

template <typename LayerType>
void assign_random_weights(LayerType& layer)
{
    layer.weights.clear();
    layer.weights.reserve(LayerType::weights_per_node * LayerType::number_of_nodes);

    for (std::size_t i = 0; i < LayerType::weights_per_node * LayerType::number_of_nodes; ++i)
    {
        // TODO replace rand() with proper random generator (maybe)
        layer.weights.push_back(static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX));
        layer.biases.push_back(static_cast<float>(std::rand() % 100 - 50));
    }
}

struct SampleNetwork
{
    SampleNetwork()
    {
        assign_random_weights(first_layer);
        assign_random_weights(second_layer);
        assign_random_weights(third_layer);
    }

    Layer<784, 16> first_layer;
    Layer<16, 16> second_layer;
    Layer<16, 10> third_layer;
    Layer<10, 0> final_layer;

    void process_input(std::vector<float> first_layer_activation)
    {
        first_layer.activation_values = std::move(first_layer_activation);
        second_layer.activation_values = first_layer.calculate_activation_values_for_next_layer();
        third_layer.activation_values = second_layer.calculate_activation_values_for_next_layer();
        final_layer.activation_values = third_layer.calculate_activation_values_for_next_layer();
    }

    void backpropagate(int target_value)
    {
        auto results = final_layer.activation_values;

        auto target_vector = ColumnVector<10>();
        target_vector.values.at(target_value) = 1;

        auto constexpr sigmoid_differentiate = [](auto& val)
        {
            val = sigmoid_derivative(val);
        };

        auto const dc_da = ColumnVector<10> {results} - target_vector;

        std::for_each(results.begin(), results.end(), sigmoid_differentiate);

        auto const output_error = hadamard(dc_da, ColumnVector<10> {results} );

        auto const nabla_biases = output_error;

                                // dot the output error with activation values from the preceding layer (transposed)
        //auto const nabla_weights = output_error

    }
};

} // namespace jeagle