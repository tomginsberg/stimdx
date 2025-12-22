cmake_minimum_required(VERSION 3.18)
project(stimdx_cpp LANGUAGES CXX)

# Stim requires C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Python and pybind11
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(Protobuf REQUIRED)

# Fetch Stim from GitHub - use latest version that requires C++20
include(FetchContent)
FetchContent_Declare(
    stim
    GIT_REPOSITORY https://github.com/quantumlib/Stim.git
    GIT_TAG v1.15.0
    GIT_SHALLOW TRUE
)

# Configure stim build options
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# Make stim available
FetchContent_MakeAvailable(stim)

# Generate protobuf sources
set(PROTO_FILES ${CMAKE_SOURCE_DIR}/proto/stimdx.proto)
set(PROTO_OUTPUT_DIR ${CMAKE_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${PROTO_OUTPUT_DIR})

# Generate C++ protobuf files
add_custom_command(
    OUTPUT ${PROTO_OUTPUT_DIR}/stimdx.pb.cc ${PROTO_OUTPUT_DIR}/stimdx.pb.h
    COMMAND ${Protobuf_PROTOC_EXECUTABLE}
            --cpp_out=${PROTO_OUTPUT_DIR}
            --proto_path=${CMAKE_SOURCE_DIR}/proto
            ${PROTO_FILES}
    DEPENDS ${PROTO_FILES}
    COMMENT "Generating protobuf C++ sources"
)

add_custom_target(proto_gen DEPENDS ${PROTO_OUTPUT_DIR}/stimdx.pb.cc)

# Create the pybind11 module
pybind11_add_module(_stimdx_cpp
    cpp/src/executor.cpp
    cpp/src/bindings.cpp
    ${PROTO_OUTPUT_DIR}/stimdx.pb.cc
)

add_dependencies(_stimdx_cpp proto_gen)

target_include_directories(_stimdx_cpp PRIVATE
    ${CMAKE_SOURCE_DIR}/cpp/include
    ${PROTO_OUTPUT_DIR}
    ${stim_SOURCE_DIR}/src
)

target_link_libraries(_stimdx_cpp PRIVATE
    libstim
    protobuf::libprotobuf
)

# Install the extension module to src/stimdx for development
install(TARGETS _stimdx_cpp
    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/src/stimdx
)
