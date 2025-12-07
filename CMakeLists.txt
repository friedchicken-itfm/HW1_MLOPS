cmake_minimum_required(VERSION 3.15)
project(hw1_mac)

# питон
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# pybind11 
find_package(pybind11 CONFIG REQUIRED)

#создаем модуль
python_add_library(hw1_mac MODULE src/main.cpp WITH_SOABI)

#линкуем
target_link_libraries(hw1_mac PRIVATE pybind11::headers)

# целевой путь 
install(TARGETS hw1_mac DESTINATION .)