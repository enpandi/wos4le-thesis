cmake_minimum_required(VERSION 3.14)
project(wos4le)

# dependencies
include(FetchContent)
FetchContent_Declare(
  libigl
  GIT_REPOSITORY https://github.com/libigl/libigl.git
  GIT_TAG        v2.5.0
)
FetchContent_Declare(
  polyscope
  GIT_REPOSITORY https://github.com/nmwsharp/polyscope.git
  GIT_TAG        v2.4.0
)
FetchContent_MakeAvailable(libigl polyscope)

# compiler options
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wfatal-errors)
endif()

# define wos4le
add_executable(${PROJECT_NAME} wos4le.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE igl::core polyscope)
