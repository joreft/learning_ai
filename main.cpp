#include <fmt/format.h>
#include <thread>
#include "neuroticpp.h"

#include "sdl_abstraction.h"

#include "mnist_parser.h"


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

int main()
{//1262459412
    srand(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
    std::cout << "First random = " << rand() << "\n";
    auto const training_labels = read_label_file("assets/train-labels-idx1-ubyte");
    auto const images = parse_images_file("assets/train-images-idx3-ubyte");

    fmt::print("Training labels size = {}\n", training_labels.size());
    fmt::print("Images file size = {}\n", images.size());

    jeagle::SampleNetwork neural_net {};

    SDLAbstraction sdl_handle;
    InputState input {};

    std::size_t image_index = 0;
    while (!input.should_quit)
    {
        if (input.advance_image)
        {
            if (image_index == images.size()) break;
            ++image_index;
            input.advance_image = false;
        }

        auto const& image = images[image_index];
        auto const supposed_number = training_labels[image_index];

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
