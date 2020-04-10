#pragma once

#include <cstdint>
#include <stdexcept>
#include <fstream>

#include <boost/endian/conversion.hpp>

struct MnistParseError : public std::runtime_error
{
    MnistParseError(std::string what) : std::runtime_error(std::move(what)) {}
};

// Based on information from http://yann.lecun.com/exdb/mnist/

int parse_32bit_number_from_stream(std::ifstream& in_file)
{
    char buf[4] {};
    in_file.read(&buf[0], 4);
    int number {};

    std::memcpy(&number, &buf[0], 4);
    return boost::endian::big_to_native(number);
}

inline std::vector<int> read_label_file(std::string const& path)
{
    std::ifstream in_file(path.c_str());

    auto const magic_number = parse_32bit_number_from_stream(in_file);
    auto const number_of_items = parse_32bit_number_from_stream(in_file);

    fmt::print("Magic number labels file: {}\n", magic_number);
    fmt::print("Contains {} labels\n", number_of_items);

    std::vector<int> out;
    out.reserve(number_of_items);
    char c {};
    while (in_file.read(&c, 1))
    {
        out.emplace_back(c);
    }

    return out;
}

using Image = std::vector<uint8_t>;

Image read_image_from_file(int const rows, int const columns, std::ifstream& istream)
{
    Image out;

    char c {};

    for (int i = 0; i < rows*columns; ++i)
    {
        if (!istream.read(&c, 1))
        {
            throw MnistParseError("File ended unexpectedly");
        }

        out.push_back(c);
    }

    return out;
}

std::vector<Image> parse_images_file(std::string const& path)
{
    std::ifstream in_file(path.c_str());

    auto const magic_number = parse_32bit_number_from_stream(in_file);
    auto const number_of_images = parse_32bit_number_from_stream(in_file);
    auto const number_of_rows = parse_32bit_number_from_stream(in_file);
    auto const number_of_columns = parse_32bit_number_from_stream(in_file);

    fmt::print("Parsing input file with:\n--magic_number: {}\n--number_of_images: {}\n--number_of_rows: {}\n--number_of_columns: {}\n",
                magic_number, number_of_images, number_of_rows, number_of_columns);

    std::vector<Image> out;

    std::vector<std::uint8_t> image;

    for (int i = 0; i < number_of_images; ++i)
    {
        out.emplace_back(read_image_from_file(number_of_rows, number_of_columns, in_file));
    }

    return out;
}