#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "transforms.h"

namespace py = pybind11;

PYBIND11_MODULE(cpp_transforms, m) {
    m.doc() = "C++ implementations of common torchvision v2 transforms";

    py::class_<Transform, std::shared_ptr<Transform>>(m, "Transform")
        .def("apply", &Transform::apply);

    m.def("to_dtype", [](const std::string& dtype_str, bool scale){
        torch::Dtype dt = torch::kFloat32;
        if (dtype_str == "float32") dt = torch::kFloat32;
        else if (dtype_str == "float16") dt = torch::kFloat16;
        else if (dtype_str == "uint8") dt = torch::kUInt8;
        else if (dtype_str == "int64") dt = torch::kInt64;
        return make_to_dtype(dt, scale);
    }, py::arg("dtype")="float32", py::arg("scale")=true);

    m.def("normalize", [](const std::vector<double>& mean, const std::vector<double>& std, bool inplace){
        return make_normalize(mean, std, inplace);
    }, py::arg("mean"), py::arg("std"), py::arg("inplace")=false);

    m.def("pad", [](int64_t padding, double fill){
        return make_pad(padding, fill);
    }, py::arg("padding"), py::arg("fill")=0.0);

    m.def("random_crop", [](int64_t size){
        return make_random_crop(size);
    });

    m.def("random_horizontal_flip", [](double p){
        return make_random_horizontal_flip(p);
    }, py::arg("p")=0.5);

    m.def("random_erasing", [](std::pair<double,double> scale, double value, bool inplace){
        return make_random_erasing(scale, value, inplace);
    }, py::arg("scale")=std::make_pair(0.02,0.33), py::arg("value")=0.0, py::arg("inplace")=false);

    m.def("identity", [](){
        return make_identity();
    });

    m.def("compose", [](std::vector<TransformPtr> list){
        return make_compose(list);
    });

    m.def("random_choice", [](std::vector<TransformPtr> choices, std::vector<double> probs){
        return make_random_choice(choices, probs);
    }, py::arg("choices"), py::arg("probs")=std::vector<double>());

    m.def("cutmix_batch", [](torch::Tensor x, torch::Tensor y, int num_classes, double alpha){
        return cutmix_batch(x, y, num_classes, alpha);
    }, py::arg("x"), py::arg("y"), py::arg("num_classes"), py::arg("alpha")=1.0);

    m.def("mixup_batch", [](torch::Tensor x, torch::Tensor y, int num_classes, double alpha){
        return mixup_batch(x, y, num_classes, alpha);
    }, py::arg("x"), py::arg("y"), py::arg("num_classes"), py::arg("alpha")=0.2);

    m.def("apply_transform", [](TransformPtr tr, torch::Tensor t) {
        return tr->apply(t);
    });
}
