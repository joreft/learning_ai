#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>


namespace jeagle
{

template<std::size_t HEIGHT, std::size_t WIDTH>
struct Matrix
{
    static constexpr std::size_t height = HEIGHT;
    static constexpr std::size_t width = WIDTH;

    Matrix() : values(height*width, 0)
    {}

    Matrix(std::array<float, height*width> const& init)
    {
        static_assert(init.size() == height*width);

        *this = Matrix(); // zero initialise
        std::copy(init.begin(), init.end(), values.data());
    }

    Matrix(std::vector<float> init) : values(std::move(init))
    {}

    auto& at(std::size_t x, std::size_t y)
    {
        return values.at(x + y * width);
    }

    std::vector<float> values; // indexing follows x+width*y
};

template<std::size_t VECTOR_SIZE>
struct ColumnVector
{
    static constexpr std::size_t vector_size = VECTOR_SIZE;

    std::vector<float> values;

    ColumnVector() : values(vector_size, 0)
    {}

    ColumnVector(std::array<float, vector_size> const& init)
    {
        *this = ColumnVector();
        std::copy(init.begin(), init.end(), values.data());
    }

    ColumnVector(std::vector<float> init) : values(std::move(init))
    {}

    template <typename ColumnVectorType>
    auto operator+(ColumnVectorType const& other)
    {
        auto out = ColumnVectorType();
        for (std::size_t i = 0; i < ColumnVectorType::vector_size; ++i)
        {
            out.values.at(i) = this->values.at(i) + other.values.at(i);
        }

        return out;
    }

    template <typename ColumnVectorType>
    auto operator-(ColumnVectorType const& other)
    {
        auto out = ColumnVectorType();
        for (std::size_t i = 0; i < ColumnVectorType::vector_size; ++i)
        {
            out.values.at(i) = this->values.at(i) - other.values.at(i);
        }

        return out;
    }

    template <typename MatrixType>
    auto dot_product(MatrixType const& mat)
    {
        auto out = Matrix<MatrixType::width, MatrixType::height>();
        for (std::size_t column_counter = 0; column_counter < MatrixType::width; ++ column_counter)
        {
            for (std::size_t i = 0; i < vector_size; ++i)
            {
               // out.at(column_counter, )
            }
        }
    }
};

template<typename MatrixType, typename ColumnVectorType>
auto multiply(MatrixType const &matrix, ColumnVectorType const &column_vector)
{
    static_assert(MatrixType::width == ColumnVectorType::vector_size);

    auto out = ColumnVector<MatrixType::height>();
    for (std::size_t y = 0; y < MatrixType::height; ++y)
    {
        for (std::size_t x = 0; x < MatrixType::width; ++x)
        {
            out.values.at(y) += matrix.values.at(x + MatrixType::height * y) * column_vector.values.at(x);
        }
    }

    return out;
}
/*

template <typename ColumnVectorType>
auto hadamard(ColumnVectorType const& lhs, ColumnVectorType const& rhs)
{
    auto out = ColumnVectorType();
    for (std::size_t y = 0; y < ColumnVectorType::vector_size; ++y)
    {
        out.values.emplace_back(lhs.values.at(y) * rhs.values.at(y));
    }

    return out;
}

template<typename MatrixType, typename ColumnVectorType>
auto operator*(MatrixType const &matrix, ColumnVectorType const &column_vector)
{
    return multiply(matrix, column_vector);
}


*/


} // namespace jeagle

void test();