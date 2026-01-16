#include <rclcpp/rclcpp.hpp>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // <-- Needed to convert std::vector to Python list

namespace py = pybind11;

class PybindTestNode : public rclcpp::Node
{
public: 
    PybindTestNode() : Node("pybind_test_node")
    {
        RCLCPP_INFO(this->get_logger(), "Initializing Python interpreter...");

        // Start Python interpreter
        py_guard_ = std::make_unique<py::scoped_interpreter>();

        try
        {
            // Import numpy
            py::module_ numpy = py::module_::import("piper_sdk");
            std::string numpy_version = numpy.attr("__version__").cast<std::string>();
            RCLCPP_INFO(this->get_logger(), "Successfully imported numpy, version %s", numpy_version.c_str());

            // // Example: create a numpy array from std::vector
            // std::vector<int> vec{1, 2, 3, 4, 5};
            // py::object arr = numpy.attr("array")(vec);  // Now works because of <pybind11/stl.h>

            // RCLCPP_INFO(this->get_logger(), "Created numpy array: %s", py::str(arr).cast<std::string>().c_str());

            // // Example: sum the array using numpy
            // int sum_result = numpy.attr("sum")(arr).cast<int>();
            // RCLCPP_INFO(this->get_logger(), "Sum of array: %d", sum_result);
        }
        catch (const py::error_already_set &e)
        {
            RCLCPP_ERROR(this->get_logger(), "Python error: %s", e.what());
        }
    }

private:
    std::unique_ptr<py::scoped_interpreter> py_guard_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PybindTestNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
