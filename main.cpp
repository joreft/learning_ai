#include <fmt/format.h>
#include <thread>
#include "neuroticpp.h"
#include <algorithm>
#include "sdl_abstraction.h"
#include <future>
#include "mnist_parser.h"

using namespace jeagle;
std::vector<double> image_to_first_layer_activation_values(Image const& image)
{
    std::vector<double> out;
    out.reserve(28*28);
    for (auto const val : image)
    {
        out.emplace_back(static_cast<double>(val)/ static_cast<double>(255));
    }

    return out;
}

int calculate_best_guess(jeagle::Layer<10, 0> const& final_layer)
{
    auto const max_coeff = final_layer.activation_values.maxCoeff();

    for (int i = 0; i < final_layer.activation_values.size(); ++i)
    {
        if (final_layer.activation_values[i] == max_coeff) return i;
    }
    throw "oh no";
}

struct LabelImageZip
{
    Image image;
    int label;
};

std::vector<LabelImageZip> zip(std::vector<Image> const& images, std::vector<int> const& labels)
{
    std::vector<LabelImageZip> out;
    out.reserve(images.size());
    for (std::size_t i = 0; i < images.size(); ++i)
    {
        out.emplace_back(LabelImageZip{images.at(i), labels.at(i)});
    }

    return out;
}

void calculate_cost(jeagle::SampleNetwork& net, std::vector<Image> const& images, std::vector<int> const& labels);

void train_network(SampleNetwork& net, std::vector<Image> const& images, std::vector<int> const& labels, float learning_rate)
{
    auto zipped = zip(images, labels);

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(zipped.begin(), zipped.end(), g);
    const int batches = 600;
    auto const  batch_size = zipped.size() / batches; // only take the first 1/40 of the randomly shuffled samples

    for (int iter = 0; iter < 1000; ++iter)
    {
        std::shuffle(zipped.begin(), zipped.end(), g);
        for (std::size_t batch_counter = 0; batch_counter < batches; ++batch_counter)
        {
            std::vector<Eigen::Matrix<double, 10, 1>> biases_vec{};
            std::vector<Eigen::Matrix<double, 10, 16>> weights_vec{};

            std::vector<Eigen::Matrix<double, 16, 1>> biases_vec_2 {};
            std::vector<Eigen::Matrix<double, 16, 784>> weights_vec_2 {};


            for (std::size_t i = 0; i < batch_size; ++i)
            {
                auto const &image = zipped.at(i + batch_size * batch_counter).image;
                auto const target = zipped.at(i + batch_size * batch_counter).label;
                //if (target != 4) continue; //gives incentive for 4

                auto as_vec = image_to_first_layer_activation_values(image);
                decltype(Layer<784, 16>::activation_values)
                    as_mat = Eigen::Map<decltype(Layer<784, 16>::activation_values)>(as_vec.data());
                net.process_input(as_mat);

                auto res = net.backpropagate(target);

                auto const res_1 = res.layer_3;
                auto const res_2 = res.layer_1;
                biases_vec.emplace_back(res_1.first);
                weights_vec.emplace_back(res_1.second);

                biases_vec_2.emplace_back(res_2.first);
                weights_vec_2.emplace_back(res_2.second);
            }

            Eigen::Matrix<double, 10, 1> biases_sum{};
            Eigen::Matrix<double, 10, 16> weights_sum{};
            for (auto const &val : biases_vec)
            {
                biases_sum += val;
            }

            for (auto const &val : weights_vec)
            {
                weights_sum += val;
            }

            Eigen::Matrix<double, 16, 1> biases_sum_2{};
            Eigen::Matrix<double, 16, 784> weights_sum_2{};
            for (auto const &val : biases_vec_2)
            {
                biases_sum_2 += val;
            }

            for (auto const &val : weights_vec_2)
            {
                weights_sum_2 += val;
            }

            auto nudge_bias = biases_sum / (double) batch_size;
            auto nudge_weights = weights_sum / (double) batch_size;

            auto nudge_bias2 = biases_sum_2 / (double) batch_size;
            auto nudge_weights2 = weights_sum_2 / (double) batch_size;

            auto constexpr eta = -0.5;
            net.second_layer.biases -= eta * nudge_bias;
            net.second_layer.weights -= eta * nudge_weights;

            net.first_layer.biases -= eta * nudge_bias2;
            net.first_layer.weights -= eta * nudge_weights2;
        }

        std::cout << "Iteration no " << iter << std::endl;
        calculate_cost( net, images, labels);
    }
}

void calculate_cost(jeagle::SampleNetwork& net, std::vector<Image> const& images, std::vector<int> const& labels)
{
    if (labels.size() != images.size()) throw "panic at the disco";

    auto zipped = zip(images, labels);

    std::random_device rd;
    std::mt19937 g(rd());

    std::shuffle(zipped.begin(), zipped.end(), g);

    auto const  batch_size = zipped.size();// / 40; // only take the first 1/40 of the randomly shuffled samples

    int correct_counter = 0;
    int incorrect_counter = 0;

    std::vector<float> costs{};

    for (std::size_t i = 0; i < batch_size; ++i)
    {
        auto const& image = zipped.at(i).image;
        auto const target = zipped.at(i).label;

        auto as_vec = image_to_first_layer_activation_values(image);
        decltype(Layer<784, 16>::activation_values) as_mat = Eigen::Map<decltype(Layer<784, 16>::activation_values)> (as_vec.data());
        net.process_input(as_mat);
        auto const& results = net.final_layer.activation_values;


        float cost = 0;
        for (int u = 0; u < 10; ++u)
        {
            if (u == target)
            {
                cost += std::pow(results[u] - 1.0, 2.0);
            }
            else
            {
                cost += std::pow(results[u], 2.0);
            }
        }

        auto correct = calculate_best_guess(net.final_layer) == target;

        if (correct)
        {
            ++correct_counter;
        }
        else
        {
            ++incorrect_counter;
        }

        costs.emplace_back(cost);
    }

    auto const correct_proportion = 100*((double)(correct_counter)/ (double)(incorrect_counter + correct_counter));

    // Mean squared error (MSE) / Quadratic cost
    auto const average_cost = std::accumulate(costs.begin(), costs.end(), 0.f) / (2.f * batch_size);
    fmt::print("Final itercost was {} and accuracy was {}% \n", average_cost, correct_proportion);
}

int main()
{//1262459412
    srand(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
    std::cout << "First random = " << rand() << "\n";
    auto const training_labels = read_label_file("assets/train-labels-idx1-ubyte");
    auto const training_images = parse_images_file("assets/train-images-idx3-ubyte");

    fmt::print("Training labels size = {}\n", training_labels.size());
    fmt::print("Images file size = {}\n", training_images.size());

    jeagle::SampleNetwork neural_net {};

    calculate_cost(neural_net, training_images, training_labels);
    auto const start_time = std::chrono::system_clock::now();
    fmt::print("Before training\n");
    train_network(neural_net, training_images, training_labels, 30.f);
    auto const train_time = std::chrono::system_clock::now() - start_time;

    auto const elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(train_time).count();
    fmt::print("Training finished in {}ms\n", elapsed_seconds);
    calculate_cost(neural_net, training_images, training_labels);
    //return 0;
    SDLAbstraction sdl_handle;
    InputState input {};

    auto const test_labels = read_label_file("assets/t10k-labels-idx1-ubyte");
    auto const test_images = parse_images_file("assets/t10k-images-idx3-ubyte");

    std::size_t image_index = 0;
    while (!input.should_quit)
    {
        if (input.advance_image)
        {
            if (image_index == test_images.size()) break;
            ++image_index;
            input.advance_image = false;
        }

        auto const& image = test_images[image_index];
        auto const supposed_number = test_labels[image_index];

        auto as_vec = image_to_first_layer_activation_values(image);
        decltype(Layer<784, 16>::activation_values) as_mat = Eigen::Map<decltype(Layer<784, 16>::activation_values)> (as_vec.data());
        neural_net.process_input(as_mat);

        sdl_handle.populate_inputs(input);

        sdl_handle.clear_and_draw_background();
        sdl_handle.draw_mnist_image(image);
        sdl_handle.draw_best_guess(calculate_best_guess(neural_net.final_layer));

        sdl_handle.draw_network(40, 40, neural_net);

        fmt::print("Showing supposed {}\n", supposed_number);
        sdl_handle.render_to_target();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    fmt::print("Closing gracefully");
    return 0;
}
