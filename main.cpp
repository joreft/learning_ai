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

    std::cout << "Evaluating results\n";
    for (int i = 0; i < results.size(); ++i)
    {
        std::cout << i << " ---" << results(i) << "\n";
    }

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
};

struct HiddenLayer
{
    Eigen::MatrixXd activation_values;
    Eigen::MatrixXd outgoing_weights;
    Eigen::MatrixXd biases; // Coming into the layer for calculating activation values at this
};

struct OutputLayer
{
    Eigen::MatrixXd activation_values;
    Eigen::MatrixXd biases;
};

struct Network
{
    InputLayer input;
    std::vector<HiddenLayer> layers;
    OutputLayer output;
};

void load_input(InputLayer& input, Image const& image)
{
    input.activation_values = Eigen::VectorXd(28*28);
    input.activation_values = image_to_first_layer_activation_values(image);
}

// TODO find better place for this
std::vector<int> const nodes_per_layer {28*28, 16, 16, 10};

void initialise_network(Network& net)
{
    std::random_device rd;
    std::mt19937 generator(rd());
    std::normal_distribution<> weights {0,1};
    std::normal_distribution<> biases {0,1};

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

        layer.activation_values = Eigen::VectorXd(nodes_per_layer.at(i));
        init_matrix(layer.activation_values, weights);


        net.layers.emplace_back(layer);
    }

    fmt::print("Initialised {} layers\n", net.layers.size());
    net.output.biases = Eigen::VectorXd(nodes_per_layer.back());
    //net.output.activation_values = Eigen::VectorXd(nodes_per_layer.back());

    init_matrix(net.output.activation_values, weights);
}

void run_network(Network& net)
{
    long int n = 0;
    // TODO apply sigmoid
    fmt::print("activation: x: {}, y {}\n", net.input.activation_values.cols(), net.input.activation_values.rows());
    fmt::print("weights: x: {}, y {}\n", net.input.outgoing_weights.cols(), net.input.outgoing_weights.rows());
    fmt::print("biases: x: {}, y {}\n", net.layers.at(n).biases.cols(), net.layers.at(n).biases.rows());
    net.layers.at(n).activation_values =  (net.input.outgoing_weights * net.input.activation_values + net.layers.at(n).biases)
                                         .unaryExpr([](double val) {return sigmoid(val);});
    ++n;

    while (n < static_cast<long int>(net.layers.size()))
    {
        auto const new_values = net.layers.at(n - 1).outgoing_weights * net.layers.at(n - 1).activation_values + net.layers.at(n).biases;

        net.layers.at(n).activation_values = new_values.unaryExpr([](double val) {return sigmoid(val);});
        ++n;
    }

    fmt::print("X: {}, Y: {}\n", net.layers.back().activation_values.cols(), net.layers.back().activation_values.rows());
    fmt::print("Weights, X: {}, Y: {}\n", net.layers.back().outgoing_weights.cols(), net.layers.back().outgoing_weights.rows());
    fmt::print("biases: {}\n", net.output.biases.size());
    net.output.activation_values =  (net.layers.back().outgoing_weights * net.layers.back().activation_values + net.output.biases)
                                   .unaryExpr([](double val) {return sigmoid(val);});

    fmt::print("Output, x: {}, y: {}", net.output.activation_values.cols(), net.output.activation_values.rows());
    fmt::print("Biases, x: {}, y: {}", net.output.biases.cols(), net.output.biases.rows());
    std::cout << std::endl;
}

} // namespace jeagle

using namespace jeagle;

int main()
{
    srand(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
    std::cout << "First random = " << rand() << "\n";

    auto const training_set = zip(parse_images_file("assets/train-images-idx3-ubyte"), read_label_file("assets/train-labels-idx1-ubyte"));

    Network neural_net {};
    initialise_network(neural_net);

    auto const start_time = std::chrono::system_clock::now();
    auto const train_time = std::chrono::system_clock::now() - start_time;
    auto const elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(train_time).count();
    fmt::print("Training finished in {}ms\n", elapsed_seconds);


    SDLAbstraction sdl_handle;
    InputState input {};

    auto const test_set = zip(parse_images_file("assets/t10k-images-idx3-ubyte"),
                               read_label_file("assets/t10k-labels-idx1-ubyte"));

    std::size_t image_index = 0;
    while (!input.should_quit)
    {
        if (input.advance_image)
        {
            if (image_index == test_set.size())
            {
                fmt::print("Finished at image index: {}", image_index);
                break;
            }
            ++image_index;
            input.advance_image = false;
        }

        auto const& image = test_set[image_index].image;

        load_input(neural_net.input, image);
        run_network(neural_net);

        sdl_handle.populate_inputs(input);

        sdl_handle.clear_and_draw_background();
        sdl_handle.draw_mnist_image(image);
        sdl_handle.draw_best_guess(calculate_best_guess(neural_net.output.activation_values));

        sdl_handle.draw_network(40, 40, neural_net);

        sdl_handle.render_to_target();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    fmt::print("Closing gracefully");
    return 0;
}
