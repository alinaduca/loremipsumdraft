#pragma once
#include <torch/extension.h>
#include <vector>
#include <memory>

struct Transform {
    virtual ~Transform() = default;
    virtual torch::Tensor apply(const torch::Tensor& img) = 0;
};

using TransformPtr = std::shared_ptr<Transform>;

TransformPtr make_to_dtype(torch::Dtype dtype, bool scale);
TransformPtr make_normalize(std::vector<double> mean, std::vector<double> std, bool inplace=false);
TransformPtr make_pad(int64_t padding, double fill=0.0);
TransformPtr make_random_crop(int64_t size);
TransformPtr make_random_horizontal_flip(double p=0.5);
TransformPtr make_random_erasing(std::pair<double,double> scale = {0.02,0.33}, double value = 0.0, bool inplace=false);

TransformPtr make_identity();

TransformPtr make_compose(const std::vector<TransformPtr>& transforms);
TransformPtr make_random_choice(const std::vector<TransformPtr>& choices, const std::vector<double>& probs = {});

std::pair<torch::Tensor, torch::Tensor> cutmix_batch(const torch::Tensor& x, const torch::Tensor& y, int num_classes, double alpha=1.0);
std::pair<torch::Tensor, torch::Tensor> mixup_batch(const torch::Tensor& x, const torch::Tensor& y, int num_classes, double alpha=0.2);

torch::Tensor ensure_CHW_float(const torch::Tensor& t);
