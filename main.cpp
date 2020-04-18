#include <fmt/format.h>
#include <thread>
#include <algorithm>
#include "sdl_abstraction.h"
#include "mnist_parser.h"
#include <Eigen/Dense>
#include <random>

constexpr double sigmoid(double in)
{
    return 1/(1+std::exp(-in));
}

constexpr double sigmoid_derivative(double in)
{
    return sigmoid(in) * (1 - sigmoid(in));
}

auto image_to_first_layer_activation_values(Image const& image)
{
    std::vector<double> out;
    out.reserve(28*28);

    for (auto const val : image)
    {
        out.emplace_back(static_cast<double>(val)/ static_cast<double>(255));
    }

    using InputVectorType = Eigen::Matrix<double, 28*28, 1>;
    return Eigen::Map<InputVectorType>(out.data());
}

int calculate_best_guess(Eigen::Matrix<double, 10, 1> const& results)
{
    auto const max_coeff = results.maxCoeff();

//    std::cout << "Evaluating results\n";
//    for (int i = 0; i < results.size(); ++i)
//    {
//        std::cout << i << " ---" << results(i) << "\n";
//    }

    for (int i = 0; i < results.size(); ++i)
    {
        if (results[i] == max_coeff) return i;
    }
    throw "oh no";
}

//void train_network(SampleNetwork& net, std::vector<Image> const& images, std::vector<int> const& labels, float learning_rate)
//{
//    auto zipped = zip(images, labels);
//
//    std::random_device rd;
//    std::mt19937 g(rd());
//    std::shuffle(zipped.begin(), zipped.end(), g);
//    const int batches = 600;
//    auto const  batch_size = zipped.size() / batches; // only take the first 1/40 of the randomly shuffled samples
//
//    for (int iter = 0; iter < 1; ++iter)
//    {
//        std::shuffle(zipped.begin(), zipped.end(), g);
//        for (std::size_t batch_counter = 0; batch_counter < batches; ++batch_counter)
//        {
//            std::vector<Eigen::Matrix<double, 10, 1>> biases_vec{};
//            std::vector<Eigen::Matrix<double, 10, 16>> weights_vec{};
//
//            std::vector<Eigen::Matrix<double, 16, 1>> biases_vec_2 {};
//            std::vector<Eigen::Matrix<double, 16, 784>> weights_vec_2 {};
//
//
//            for (std::size_t i = 0; i < batch_size; ++i)
//            {
//                auto const &image = zipped.at(i + batch_size * batch_counter).image;
//                auto const target = zipped.at(i + batch_size * batch_counter).label;
//
//                net.process_input(image_to_first_layer_activation_values(image));
//
//                auto res = net.backpropagate(target);
//
//                auto const res_1 = res.layer_3;
//                auto const res_2 = res.layer_1;
//                biases_vec.emplace_back(res_1.first);
//                weights_vec.emplace_back(res_1.second);
//
//                biases_vec_2.emplace_back(res_2.first);
//                weights_vec_2.emplace_back(res_2.second);
//            }
//
//            Eigen::Matrix<double, 10, 1> biases_sum{};
//            Eigen::Matrix<double, 10, 16> weights_sum{};
//            for (auto const &val : biases_vec)
//            {
//                biases_sum += val;
//            }
//
//            for (auto const &val : weights_vec)
//            {
//                weights_sum += val;
//            }
//
//            Eigen::Matrix<double, 16, 1> biases_sum_2{};
//            Eigen::Matrix<double, 16, 784> weights_sum_2{};
//            for (auto const &val : biases_vec_2)
//            {
//                biases_sum_2 += val;
//            }
//
//            for (auto const &val : weights_vec_2)
//            {
//                weights_sum_2 += val;
//            }
//
//            auto nudge_bias = biases_sum / (double) batch_size;
//            auto nudge_weights = weights_sum / (double) batch_size;
//
//            auto nudge_bias2 = biases_sum_2 / (double) batch_size;
//            auto nudge_weights2 = weights_sum_2 / (double) batch_size;
//
//            auto constexpr eta = -0.5;
//            net.second_layer.biases -= eta * nudge_bias;
//            net.second_layer.weights -= eta * nudge_weights;
//
//            net.first_layer.biases -= eta * nudge_bias2;
//            net.first_layer.weights -= eta * nudge_weights2;
//        }
//
//        std::cout << "Iteration no " << iter << std::endl;
//    }
//}

//struct Layer
//{
//
//};
namespace jeagle
{

struct InputLayer
{
    Eigen::MatrixXd activation_values;
    Eigen::MatrixXd outgoing_weights;

    Eigen::VectorXd stored_input;
};

struct HiddenLayer
{
    Eigen::MatrixXd activation_values;
    Eigen::MatrixXd outgoing_weights;
    Eigen::MatrixXd biases; // Coming into the layer for calculating activation values at this

    Eigen::VectorXd stored_input;
};

struct OutputLayer
{
    Eigen::MatrixXd activation_values;
    Eigen::MatrixXd biases;

    Eigen::VectorXd stored_input;
};

struct Network
{
    InputLayer input;
    std::vector<HiddenLayer> layers;
    OutputLayer output;
};

void load_input(InputLayer& input, Image const& image)
{
    input.stored_input = image_to_first_layer_activation_values(image);
    input.activation_values = input.stored_input.unaryExpr([](double val){return sigmoid(val);});
}

// TODO find better place for this
constexpr std::array nodes_per_layer {28*28, 16, 5,10};

void initialise_network(Network& net)
{
    std::random_device rd;
    std::mt19937 generator(rd());

    // TODO find out what clang tidy considers a safe seed
    generator.seed(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));

    std::normal_distribution<> weights {0,1};
    std::normal_distribution<> biases {0,1};

//    std::uniform_real_distribution<> biases {-2,2};
//    std::uniform_real_distribution<> weights {-2,2};

//    weights(g);

    auto init_matrix = [&generator](auto& mat, auto& dist)
    {
        for (long int i = 0; i < mat.size(); ++i)
        {
            mat(i) = dist(generator);
        }
    };

    net.input.activation_values = Eigen::VectorXd(28*28);
    net.input.outgoing_weights = Eigen::MatrixXd(nodes_per_layer.at(1), 28*28);
    init_matrix(net.input.outgoing_weights, weights);

    for (long int i = 1; i < static_cast<long int>(nodes_per_layer.size()) - 1; ++i)
    {
        HiddenLayer layer;
        layer.outgoing_weights = Eigen::MatrixXd(nodes_per_layer.at(i + 1), nodes_per_layer.at(i ));
        init_matrix(layer.outgoing_weights, weights);

        layer.biases = Eigen::VectorXd(nodes_per_layer.at(i));
        init_matrix(layer.biases, biases);

        layer.activation_values = {};
        init_matrix(layer.activation_values, weights);


        net.layers.emplace_back(layer);
    }

    fmt::print("Initialised {} layers\n", net.layers.size());
    net.output.biases = Eigen::VectorXd(nodes_per_layer.back());
    init_matrix(net.output.activation_values, weights);
}

// TODO probably helpful to move the activation function into the layer data structures
void feed_forward(Network& net,
                  std::function<double(double)> activation_function = [](double val) { return sigmoid(val); })
{
    auto propagate_forward = [&activation_function](auto& from_layer, auto& to_layer)
    {
        to_layer.stored_input =  (from_layer.outgoing_weights * from_layer.activation_values + to_layer.biases);
        to_layer.activation_values =  to_layer.stored_input.unaryExpr(activation_function);
    };

    propagate_forward(net.input, net.layers.front());
    net.input.activation_values = net.input.stored_input;

    long int n = 0;
    while (n < static_cast<long int>(net.layers.size()) - 1)
    {
        propagate_forward(net.layers.at(n), net.layers.at(n + 1));
        ++n;
    }

        propagate_forward(net.layers.back(), net.output);
}

double network_accuracy_percentage(Network& net, std::vector<LabelImageZip> const& test_dataset)
{
    std::size_t correct_counter = 0;
    for (auto const& data : test_dataset)
    {
        load_input(net.input, data.image);
        feed_forward(net);

        if (calculate_best_guess(net.output.activation_values) == data.label)
        {
            correct_counter += 1;
        }
    }

    return (100.0 * static_cast<double>(correct_counter))/ static_cast<double>(test_dataset.size());
}

struct BackpropagationResult
{
    Eigen::VectorXd biases_to_succeeding_layer;
    Eigen::MatrixXd weights;
};

Eigen::VectorXd vectorized_result(int desired_value)
{
    Eigen::VectorXd res(nodes_per_layer.back());
    res.setZero();
    res(desired_value) = 1;
    return res;
}

std::vector<BackpropagationResult> run_backpropagation(Network& net, LabelImageZip const& data_point)
{
    auto const activation_derivative = [](double val)
    {
        return sigmoid_derivative(val);
    };

    load_input(net.input, data_point.image);
    feed_forward(net);

    // TODO: change with the activation function for the output layer
    Eigen::VectorXd error;// = (net.output.activation_values - vectorized_result(3)).cwiseProduct(net.output.stored_input.unaryExpr(activation_derivative));

    auto output_to_last_hidden_layer = [&]()
    {
        error = (vectorized_result(data_point.label) - net.output.activation_values).cwiseProduct(net.output.stored_input.unaryExpr(activation_derivative));
        auto const nabla_weights = error * net.layers.back().activation_values.transpose();
        return BackpropagationResult{error, nabla_weights};
    };

    auto backward_pass = [&](auto const& from_layer, auto const& to_layer)
    {
        auto const& z = from_layer.stored_input;
        auto const ad = z.unaryExpr(activation_derivative);

        error = (from_layer.outgoing_weights.transpose() * error).cwiseProduct(ad);

        auto const nabla_biases = error;
        auto const nabla_weights = error * to_layer.activation_values.transpose();

        return BackpropagationResult{nabla_biases, nabla_weights};
    };

    std::vector<BackpropagationResult> results {};

    results.emplace_back(output_to_last_hidden_layer());

    for (auto it = net.layers.crbegin(); it != (net.layers.crend() - 1); ++it)
    {
        results.emplace_back(backward_pass(*it, *(it + 1)));
    }

    results.emplace_back(backward_pass(net.layers.at(0), net.input));

    // There is some confusing reverse iterator action going on here
    // The return value should be ordered from first layer to last
    return {results.rbegin(), results.rend()};
}

void process_mini_batch(Network& net, std::vector<LabelImageZip> const& training_dataset, double learning_rate)
{
    // Zero initialise accumulated nablas with correct dimensions
    std::vector<Eigen::VectorXd> biases;
    std::vector<Eigen::MatrixXd> weights;
    for (auto it = nodes_per_layer.cbegin(); it != (nodes_per_layer.cend() - 1); ++it)
    {
        Eigen::VectorXd bias(*(it + 1));
        bias.setZero();
        Eigen::MatrixXd weight(*(it + 1), *it);
        weight.setZero();

        biases.emplace_back(bias);
        weights.emplace_back(weight);
    }

    for (auto const& data : training_dataset)
    {
        auto const new_nablas = run_backpropagation(net, data);

        // accumulate nablas
        for (auto it = new_nablas.begin(); it != (new_nablas.end()); ++it)
        {
            auto const idx = it - new_nablas.cbegin();

            biases.at(idx) += it->biases_to_succeeding_layer;
            weights.at(idx) += it->weights;
        }
    }

    // average the summed nablas times learning rate and update the network
    if (biases.size() != weights.size()) throw std::runtime_error("Inconsistent sizes between biases and weights");

    net.input.outgoing_weights += learning_rate * (weights.at(0) / static_cast<double>(weights.size()));
    net.layers.at(0).biases += learning_rate * (biases.at(0) / static_cast<double>(weights.size()));

    for (std::size_t i = 1; i < (biases.size() - 1); ++i)
    {
        net.layers.at(i - 1).outgoing_weights += (learning_rate / static_cast<double>(weights.size())) * weights.at(i);
        net.layers.at(i).biases += learning_rate * (biases.at(i)) / static_cast<double>(weights.size());
    }

    net.layers.back().outgoing_weights += learning_rate * (weights.back() / static_cast<double>(weights.size()));
    net.output.biases += (learning_rate / static_cast<double>(weights.size())) * biases.back();
}

void stochastic_gradient_descent(Network& net, std::vector<LabelImageZip> training_dataset,
                                 int const epochs, int const batch_size, double const learning_rate)
{
    if (training_dataset.size() % batch_size != 0)
    {
        throw std::runtime_error("The batch size must be such that one can divide the set into mini batches of equal size");
    }

    std::random_device rd;
    std::mt19937 g(rd());
    // TODO these seeds...
    g.seed(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));

    for (int epoch_counter = 0; epoch_counter < epochs; ++epoch_counter)
    {
        std::shuffle(training_dataset.begin(), training_dataset.end(), g);

        for (std::size_t batch_counter = 0; batch_counter < (training_dataset.size() / batch_size); ++batch_counter)
        {
            process_mini_batch(net, std::vector( begin(training_dataset) + batch_counter * batch_size,
                                                 begin(training_dataset) + batch_counter * batch_size + batch_size ),
                               learning_rate);
        }
        fmt::print("Finished with epoch {}/{}\n", epoch_counter + 1, epochs);
    }

}

} // namespace jeagle

int main()
{
    using namespace jeagle;

    srand(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
    std::cout << "First random = " << rand() << "\n";

    auto const training_set = zip(parse_images_file("assets/train-images-idx3-ubyte"), read_label_file("assets/train-labels-idx1-ubyte"));

    Network neural_net {};
    initialise_network(neural_net);

    auto const test_set = zip(parse_images_file("assets/t10k-images-idx3-ubyte"),
                              read_label_file("assets/t10k-labels-idx1-ubyte"));

    fmt::print("Accuracy on test set before training was {}%\n", network_accuracy_percentage(neural_net, test_set));

    auto const start_time = std::chrono::system_clock::now();

    stochastic_gradient_descent(neural_net, training_set, 10, 60, 0.2);

    auto const train_time = std::chrono::system_clock::now() - start_time;
    auto const elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(train_time).count();
    fmt::print("Training finished in {}ms\n", elapsed_seconds);


    SDLAbstraction sdl_handle;
    InputState input {};

    fmt::print("Accuracy on test set was {}%\n", network_accuracy_percentage(neural_net, test_set));
//return 0;
    std::size_t image_index = 0;
    while (!input.should_quit)
    {
        if (input.advance_image || (image_index == 0))
        {
            if (image_index == test_set.size())
            {
                fmt::print("Finished at image index: {}", image_index);
                break;
            }
            ++image_index;
            auto const& image = test_set[image_index].image;

            load_input(neural_net.input, image);
            feed_forward(neural_net);

            sdl_handle.clear_and_draw_background();

            sdl_handle.draw_best_guess(calculate_best_guess(neural_net.output.activation_values));
            sdl_handle.draw_mnist_image(image);
            sdl_handle.draw_network(40, 40, neural_net);

            sdl_handle.render_to_target();
            input.advance_image = false;
        }

        sdl_handle.populate_inputs(input);

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    fmt::print("Closing gracefully");
    return 0;
}
