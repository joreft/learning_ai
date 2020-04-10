#pragma once

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>

#include "neuroticpp.h"
#include "mnist_parser.h"

#include <chrono>
#include <thread>
#include <iostream>

struct Color
{
    std::uint8_t r;
    std::uint8_t g;
    std::uint8_t b;
    std::uint8_t a;
};

inline constexpr auto green = Color{0, 255, 0, 255};
inline constexpr auto red  = Color{255, 0, 0, 255};
inline constexpr auto blue = Color{0, 0, 255, 255};
inline constexpr auto white = Color{255, 255, 255, 255};
inline constexpr auto black = Color{0, 0, 0, 255};
inline constexpr auto purple = Color{255, 0, 255, 255};

void set_sdl_color(SDL_Renderer& renderer, Color const& color)
{
    SDL_SetRenderDrawColor(&renderer, color.r, color.g, color.b, color.a);
}

enum class KeyEvent
{
    pressed, released
};

struct InputState
{
    bool should_quit {};
    bool advance_image {};
};

/*
constexpr void process_key_event(InputState& input_state, KeyEvent const key_event, std::int32_t const keysym)
{
    bool const pressed = key_event == KeyEvent::pressed;
    if (keysym == SDLK_UP)
    {
        input_state.rotate = pressed;
    }
    else if (keysym == SDLK_ESCAPE)
    {
        input_state.escape = pressed;
    }
    else if (keysym == SDLK_RIGHT)
    {
        input_state.go_right = pressed;
    }
    else if (keysym == SDLK_LEFT)
    {
        input_state.go_left = pressed;
    }
    else if (keysym == SDLK_DOWN)
    {
        input_state.speedup_pressed = pressed;
    }
    else if (keysym == SDLK_SPACE)
    {
        // a bit of special handling here, letting it be reset by other state
        input_state.force_settle = input_state.force_settle ? true : pressed;
    }
}*/

constexpr void process_sdl_event(InputState& input_state, SDL_Event const& event)
{
    switch (event.type)
    {
        case SDL_KEYDOWN:
        {
            auto const sym = event.key.keysym.sym;
            if (sym == SDLK_ESCAPE)
            {
                input_state.should_quit = true;
            }
            else if (sym == SDLK_SPACE)
            {
                input_state.advance_image = true;
            }

            break;
        }
        case SDL_KEYUP: 
        {
           // auto const sym = event.key.keysym.sym;
            break;
        }
    }
}

inline constexpr int rect_size = 9;

constexpr SDL_Rect getRectangle(int x, int y)
{
    return {x, y, rect_size, rect_size};
}

struct SDLContextOwner
{
    SDLContextOwner()
    {
        std::cout << "Initialising SDL2\n";
        auto const init_ret = SDL_Init(SDL_INIT_VIDEO);

        if (init_ret != 0)
        {
            std::cout << "SDL_Init failed with err: " << SDL_GetError() << "\n";
        }

        auto const create_window_ret = SDL_CreateWindowAndRenderer(500, 400, 0, &window, &renderer);
        if (create_window_ret != 0)
        {
            std::cout << "SDL_CreateWindowAndRenderer failed with err: " << SDL_GetError() << "\n";
        }

        if (TTF_Init() != 0)
        {
            std::cout << "TTF_Init failed with " << TTF_GetError() << "\n";
        }

        sans_font = TTF_OpenFont("assets/FreeSans.ttf", 24);
        if (!sans_font)
        {
            std::cout << "Opening sans font failed with " << TTF_GetError() << "\n";
        }

    }
    ~SDLContextOwner()
    {
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
    }

    SDLContextOwner& operator=(SDLContextOwner const&) = delete;
    SDLContextOwner& operator=(SDLContextOwner&&) = delete;
    SDLContextOwner(SDLContextOwner const&) = delete;
    SDLContextOwner(SDLContextOwner&&) = delete;

    SDL_Window* window {};
    SDL_Renderer* renderer {};

    TTF_Font* sans_font;
};

struct SDLSurfaceOwner
{
    void operator()(SDL_Surface* surface)
    {
        SDL_FreeSurface(surface);
    }
};

using SafeSDLSurface = std::unique_ptr<SDL_Surface, SDLSurfaceOwner>;

// Multiline textboxes
struct TextBox
{

};

struct SDLAbstraction
{
    SDLAbstraction() : renderer(*context.renderer)
    {}

    SDLContextOwner context {};
    SDL_Renderer& renderer;

    void sleep_ms(int time_ms)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(time_ms));
    }

    void clear_and_draw_background()
    {
        set_sdl_color(renderer, black);
        SDL_RenderClear(&renderer);
    }

    void draw_network(std::size_t offset_x, std::size_t offset_y, jeagle::SampleNetwork const& net)
    {
        auto const set_color = [&](float activation_value)
        {
            auto const value = static_cast<uint8_t>(activation_value * 255);
            Color col {value, value, value, 255};
            set_sdl_color(renderer, col);
        };

        static constexpr std::size_t neuron_size = 15; // 4 x 4 rectangles
        static constexpr std::size_t spacing = 2;

        auto const draw_layer = [&](std::size_t layer_offset_x, auto const& layer)
        {
            for (std::size_t i = 0; i < layer.activation_values.size(); ++i)
            {
                SDL_Rect rect {static_cast<int>(layer_offset_x + offset_x),
                               static_cast<int>(offset_y + i * neuron_size + i*spacing),
                               neuron_size,
                               neuron_size};


                set_color(layer.activation_values.at(i));
                SDL_RenderDrawRect(&renderer, &rect);
                SDL_RenderFillRect(&renderer, &rect);
            }
        };
        std::cout << "Second layer size: " << net.second_layer.activation_values.size() << "\n";

        draw_layer(10, net.second_layer);
        draw_layer(20 + neuron_size , net.third_layer);
        draw_layer(30 + neuron_size*2, net.final_layer);

        fmt::print("Output values:\n");
        for (std::size_t i = 0; i < net.final_layer.activation_values.size(); ++i)
        {
            fmt::print("----{}\n", net.final_layer.activation_values[i]);
        }
    }

/*
    void draw_tile(Tile const& tile, Vec2d const& pos)
    {
        constexpr Vec2d offset_position {130, 60};
        SDLTileDrawer drawer{renderer, Vec2d{pos.x*rect_size + offset_position.x, pos.y*rect_size + offset_position.y}};
        std::visit(drawer, tile);
    }
*/
    void render_to_target()
    {
        SDL_RenderPresent(&renderer);
    }

    bool populate_inputs(InputState& input_state)
    {
        SDL_Event event;
        if( SDL_PollEvent(&event) ) 
        {
            process_sdl_event(input_state, event);
            return true;
        }
        return false;
    }

    void draw_mnist_image(Image const& image)
    {
        // We assume 28x28
        for (int i = 0; i < 28 * 28; ++i)
        {
            Color color{image.at(i), image.at(i), image.at(i), 255};

            set_sdl_color(renderer, color);
            auto const y = i/28;
            auto const x = i - y*28;

            SDL_RenderDrawPoint(&renderer, x, y);
        }
    }

    void draw_best_guess(int guess)
    {
        if (!context.sans_font)
        {
            std::cerr << "Missing needed font\n";
            return;
        }

        SDL_Color white = {255, 255, 255, {}};

        SafeSDLSurface surface
        (
            TTF_RenderText_Solid(context.sans_font, (std::string("Best guess: ") + std::to_string(guess)).c_str(), white)
        );

        auto sdl_texture_deleter = [](auto&& text)
        {
            SDL_DestroyTexture(text);
        };

        std::unique_ptr<SDL_Texture, decltype(sdl_texture_deleter)> message
        (
            SDL_CreateTextureFromSurface(&renderer, surface.get()), sdl_texture_deleter
        );


        auto const text_width = surface->w;
        auto const text_height = surface->h;
        auto const Message_rect = SDL_Rect{40, 0, text_width, text_height};

        SDL_RenderCopy(&renderer, message.get(), nullptr, &Message_rect); //you put the renderer's name first, the Message, the crop size(you can ignore this if you don't want to dabble with cropping), and the rect which is the size and coordinate of your texture
    }

/*
    MenuAction handle_menu()
    {
        SDL_Event event;
        if( SDL_PollEvent(&event) )
        {
            if (event.type == SDL_KEYDOWN)
            {
                auto const symbol = event.key.keysym.sym;
                if (symbol == SDLK_p)
                {
                    std::cout << "play\n";
                    return MenuAction::play_game;
                }
                else if (symbol == SDLK_ESCAPE)
                {
                    std::cout << "exit\n";
                    return MenuAction::quit;
                }
            }
        }

        return MenuAction::none;
    }*/

    void log(std::string const& m)
    {
        std::cout << m << "\n";
    }
};