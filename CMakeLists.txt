cmake_minimum_required(VERSION 3.12)

project(neuroexp)

# Exporting the assets
file(COPY assets DESTINATION ${CMAKE_BINARY_DIR})

find_program(CCACHE ccache)
if(CCACHE)
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE})
endif()

#set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})

set(COMMON_COMPILE_OPTIONS -std=c++17 -Wall -Wextra -pedantic -Wnon-virtual-dtor -DNO_CONCEPTS)

add_compile_options(${COMMON_COMPILE_OPTIONS})

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR} ${CMAKE_MODULE_PATH})

set (SOURCE_PROGRAM ${SOURCE} main.cpp)
set (SOURCE_TESTS  ${SOURCE_TESTS} ${SOURCE})

add_executable(leikeplass ${SOURCE_PROGRAM})
target_include_directories(leikeplass PUBLIC .)
#add_executable(testbuild ${SOURCE_TESTS})

if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
  message(
    STATUS
      "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
  file(
    DOWNLOAD
      "https://raw.githubusercontent.com/conan-io/cmake-conan/v0.12/conan.cmake"
      "${CMAKE_BINARY_DIR}/conan.cmake")
endif()

include(${CMAKE_BINARY_DIR}/conan.cmake)
conan_check(REQUIRED)
conan_cmake_run(CONANFILE
                conanfile.txt
                BASIC_SETUP
                CONAN_COMMAND
                ${CONAN_CMD}
                CMAKE_TARGETS
                BUILD
                missing)


set(CMAKE_MODULE_PATH /usr/share/SFML/cmake/Modules ${CMAKE_BINARY_DIR})
find_package(sdl2 REQUIRED)
find_package(sdl2_ttf REQUIRED)
find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED)

add_subdirectory(neuroticpp)

#set(OpenGL_GL_PREFERENCE GLVND)
#find_package(OpenGL)

target_link_libraries(leikeplass PRIVATE 
                                        sdl2::sdl2
                                        sdl2_ttf::sdl2_ttf
                                        fmt::fmt
                                        Eigen3::Eigen3)
                                       # neuroticpp)
