#include "neuroticpp.h"
#include "fmt/format.h"

using namespace jeagle;

void test()
{
    fmt::print("Test");

    Matrix<2, 2> mat (std::array<float, 4>{1.f, 2.f,
                            1.f, 3.f});

    auto col = ColumnVector<2>(std::array{1.f, 3.f});

    auto const res = mat*col;

    fmt::print("Result is: {}, {}", res.values.at(0), res.values.at(1));
}

