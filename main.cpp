#include <fmt/format.h>
#include <thread>
#include "neuroticpp.h"
#include <algorithm>
#include "sdl_abstraction.h"
#include <future>
#include "mnist_parser.h"

using namespace jeagle;
std::vector<float> image_to_first_layer_activation_values(Image const& image)
{
    std::vector<float> out;
    out.reserve(28*28);
    for (auto const val : image)
    {
        out.emplace_back(static_cast<float>(val)/ static_cast<float>(255));
    }

    return out;
}

int calculate_best_guess(jeagle::Layer<10, 0> const& final_layer)
{
    auto max_element_at = std::max_element(final_layer.activation_values.begin(), final_layer.activation_values.end());
    if (max_element_at == final_layer.activation_values.end()) throw "oh no";
    return max_element_at - final_layer.activation_values.begin() + 1;
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

void train_network(SampleNetwork& net, std::vector<Image> const& images, std::vector<int> const& labels, float learning_rate)
{
    auto zipped = zip(images, labels);

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(zipped.begin(), zipped.end(), g);
    auto const  batch_size = zipped.size() / 40; // only take the first 1/40 of the randomly shuffled samples

    auto const sigmoid_differentiate = [](auto& val)
    {
        val = sigmoid_derivative(val);
    };
    for (std::size_t i = 0; i < batch_size; ++i)
    {
        auto const& image = zipped.at(i).image;
        auto const target = zipped.at(i).label;

        net.process_input(image_to_first_layer_activation_values(image));
        auto const& results = net.final_layer.activation_values;

        std::for_each(results.begin())

        auto const output_error = hadamard();

    }

}

void calculate_cost(jeagle::SampleNetwork& net, std::vector<Image> const& images, std::vector<int> const& labels)
{
    if (labels.size() != images.size()) throw "panic at the disco";

    auto zipped = zip(images, labels);

    std::random_device rd;
    std::mt19937 g(rd());

    std::shuffle(zipped.begin(), zipped.end(), g);

    auto const  batch_size = zipped.size() / 40; // only take the first 1/40 of the randomly shuffled samples

    std::vector<float> costs{};

    for (std::size_t i = 0; i < batch_size; ++i)
    {
        auto const& image = zipped.at(i).image;
        auto const target = zipped.at(i).label;

        net.process_input(image_to_first_layer_activation_values(image));
        auto const& results = net.final_layer.activation_values;


        float cost = 0;
        for (int u = 0; u < 10; ++u)
        {
            if (u == target)
            {
                cost += std::pow(results.at(u) - 1.0f, 2.f);
            }
            else
            {
                cost += std::pow(results.at(u), 2.f);
            }
        }

        costs.emplace_back(cost);
    }
    // Mean squared error (MSE) / Quadratic cost
    auto const average_cost = std::accumulate(costs.begin(), costs.end(), 0.f) / (2.f * batch_size);
    fmt::print("Final cost was {}\n", average_cost);

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

    auto const start_time = std::chrono::system_clock::now();
    train_network(neural_net, training_images, training_labels, 30.f);
    auto const train_time = std::chrono::system_clock::now() - start_time;

    auto const elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(train_time).count();
    fmt::print("Training finished in {}ms\n", elapsed_seconds);

    return 0;
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

        neural_net.process_input(image_to_first_layer_activation_values(image));

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
