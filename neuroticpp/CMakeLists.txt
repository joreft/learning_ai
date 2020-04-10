add_library(neuroticpp STATIC src/neuroticpp.cpp)
target_include_directories(neuroticpp PUBLIC inc/)
target_link_libraries(neuroticpp fmt::fmt)
