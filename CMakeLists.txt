cmake_minimum_required(VERSION 3.10)
project(wos4le)

add_subdirectory(libigl)
add_subdirectory(polyscope)
add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE igl::core polyscope)
target_compile_options(${PROJECT_NAME} PRIVATE -Wfatal-errors)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
