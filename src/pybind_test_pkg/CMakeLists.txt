cmake_minimum_required(VERSION 3.8)
project(pybind_test_pkg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# --------------------------
# Find dependencies
# --------------------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# --------------------------
# Add executable
# --------------------------
add_executable(pybind_test_node src/pybind_test_node.cpp)

# --------------------------
# Include directories
# --------------------------
target_include_directories(pybind_test_node
  PRIVATE
    ${Python3_INCLUDE_DIRS}
)

# --------------------------
# ROS2 dependencies
# --------------------------
ament_target_dependencies(pybind_test_node
  rclcpp
)

# --------------------------
# Link pybind11 and Python3
# --------------------------
# Only one plain target_link_libraries call to avoid errors
target_link_libraries(pybind_test_node
  pybind11::embed
  Python3::Python
)

# --------------------------
# C++ standard
# --------------------------
set_target_properties(pybind_test_node PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
)

# --------------------------
# Install executable
# --------------------------
install(TARGETS
  pybind_test_node
  DESTINATION lib/${PROJECT_NAME}
)

# --------------------------
# Optional testing
# --------------------------
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
