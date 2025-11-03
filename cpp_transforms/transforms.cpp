#include "transforms.h"
#include <random>
#include <algorithm>

torch::Tensor ensure_CHW_float(const torch::Tensor& t) {
    auto tensor = t;
    TORCH_CHECK(tensor.device().is_cpu() || tensor.is_cuda(), "Tensor must be on CPU or CUDA");

    if (tensor.dtype() == torch::kUInt8) {
        tensor = tensor.to(torch::kFloat32).div_(255.0);
    } else if (tensor.dtype() != torch::kFloat32 && tensor.dtype() != torch::kFloat16 && tensor.dtype() != torch::kDouble) {
        tensor = tensor.to(torch::kFloat32);
    }

    if (tensor.dim() == 2) {
        tensor = tensor.unsqueeze(0); // 1 x H x W
    } else if (tensor.dim() == 3) {
        if (tensor.size(2) == 3 || tensor.size(2) == 1) {
            tensor = tensor.permute({2,0,1}).contiguous();
        } else {
            // C x H x W
        }
    } else {
        TORCH_CHECK(false, "Unsupported tensor shape for image");
    }

    return tensor;
}

struct ToDtype : Transform {
    torch::Dtype dtype;
    bool scale;
    ToDtype(torch::Dtype d, bool s): dtype(d), scale(s) {}
    torch::Tensor apply(const torch::Tensor& img) override {
        auto t = ensure_CHW_float(img);
        if (scale && dtype == torch::kFloat32 && img.dtype() == torch::kUInt8) {
            return t;
        }
        return t.to(dtype);
    }
};

TransformPtr make_to_dtype(torch::Dtype dtype, bool scale) {
    return std::make_shared<ToDtype>(dtype, scale);
}

struct Normalize : Transform {
    std::vector<double> mean, std;
    bool inplace;
    Normalize(std::vector<double> m, std::vector<double> s, bool ip=false): mean(std::move(m)), std(std::move(s)), inplace(ip) {}
    torch::Tensor apply(const torch::Tensor& img) override {
        auto t = ensure_CHW_float(img);
        auto out = inplace ? t : t.clone();
        TORCH_CHECK((int)mean.size() == out.size(0) && (int)std.size() == out.size(0), "mean/std must match channels");
        for (int c=0;c<out.size(0);++c) {
            out[c].sub_(mean[c]).div_(std[c]);
        }
        return out;
    }
};

TransformPtr make_normalize(std::vector<double> mean, std::vector<double> std, bool inplace) {
    return std::make_shared<Normalize>(mean, std, inplace);
}

struct Pad : Transform {
    int64_t padding;
    double fill;
    Pad(int64_t p, double f): padding(p), fill(f) {}
    torch::Tensor apply(const torch::Tensor& img) override {
        auto t = ensure_CHW_float(img);
        std::vector<int64_t> pad = {padding, padding, padding, padding};
        torch::Tensor out = torch::constant_pad_nd(t, pad, fill);
        return out;
    }
};

TransformPtr make_pad(int64_t padding, double fill) {
    return std::make_shared<Pad>(padding, fill);
}

struct RandomCrop : Transform {
    int64_t size;
    std::mt19937 gen;
    RandomCrop(int64_t s): size(s), gen(std::random_device{}()) {}
    torch::Tensor apply(const torch::Tensor& img) override {
        auto t = ensure_CHW_float(img);
        int64_t C = t.size(0), H = t.size(1), W = t.size(2);
        TORCH_CHECK(H >= size && W >= size, "Image smaller than crop size");
        std::uniform_int_distribution<int64_t> dist_h(0, H - size);
        std::uniform_int_distribution<int64_t> dist_w(0, W - size);
        int64_t top = dist_h(gen);
        int64_t left = dist_w(gen);
        return t.index({torch::indexing::Slice(), torch::indexing::Slice(top, top+size), torch::indexing::Slice(left, left+size)}).contiguous();
    }
};

TransformPtr make_random_crop(int64_t size) {
    return std::make_shared<RandomCrop>(size);
}

struct RandomHorizontalFlip : Transform {
    double p;
    std::mt19937 gen;
    RandomHorizontalFlip(double prob=0.5): p(prob), gen(std::random_device{}()) {}
    torch::Tensor apply(const torch::Tensor& img) override {
        auto t = ensure_CHW_float(img);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        if (dist(gen) < p) {
            return t.flip({2}).contiguous();
        }
        return t;
    }
};

TransformPtr make_random_horizontal_flip(double p) {
    return std::make_shared<RandomHorizontalFlip>(p);
}

struct RandomErasing : Transform {
    double scale_min, scale_max;
    double value;
    bool inplace;
    std::mt19937 gen;
    RandomErasing(std::pair<double,double> sc, double v, bool ip=false)
      : scale_min(sc.first), scale_max(sc.second), value(v), inplace(ip), gen(std::random_device{}()) {}

    torch::Tensor apply(const torch::Tensor& img) override {
        auto t = ensure_CHW_float(img);
        auto out = inplace ? t : t.clone();
        int64_t C = out.size(0), H = out.size(1), W = out.size(2);
        std::uniform_real_distribution<double> dist_scale(scale_min, scale_max);
        std::uniform_real_distribution<double> dist_ur(0.0, 1.0);
        for (int attempt = 0; attempt < 10; ++attempt) {
            double target_area = dist_scale(gen) * (double)(H * W);
            double aspect_ratio = std::uniform_real_distribution<double>(0.3, 3.3)(gen);
            int64_t h = (int64_t)std::round(std::sqrt(target_area * aspect_ratio));
            int64_t w = (int64_t)std::round(std::sqrt(target_area / aspect_ratio));
            if (h < H && w < W && h > 0 && w > 0) {
                int64_t top = std::uniform_int_distribution<int64_t>(0, H - h)(gen);
                int64_t left = std::uniform_int_distribution<int64_t>(0, W - w)(gen);
                out.index_put_({torch::indexing::Slice(), torch::indexing::Slice(top, top+h), torch::indexing::Slice(left, left+w)}, value);
                return out;
            }
        }
        return out;
    }
};

TransformPtr make_random_erasing(std::pair<double,double> scale, double value, bool inplace) {
    return std::make_shared<RandomErasing>(scale, value, inplace);
}

struct Identity : Transform {
    torch::Tensor apply(const torch::Tensor& img) override {
        return ensure_CHW_float(img);
    }
};

TransformPtr make_identity() {
    return std::make_shared<Identity>();
}

struct Compose : Transform {
    std::vector<TransformPtr> transforms;
    Compose(const std::vector<TransformPtr>& t): transforms(t) {}
    torch::Tensor apply(const torch::Tensor& img) override {
        torch::Tensor x = img;
        for (auto &tr : transforms) {
            x = tr->apply(x);
        }
        return x;
    }
};

TransformPtr make_compose(const std::vector<TransformPtr>& transforms) {
    return std::make_shared<Compose>(transforms);
}

struct RandomChoice : Transform {
    std::vector<TransformPtr> choices;
    std::vector<double> probs;
    std::mt19937 gen;
    RandomChoice(const std::vector<TransformPtr>& c, const std::vector<double>& p)
        : choices(c), probs(p), gen(std::random_device{}()) {
        if (probs.empty()) {
            probs = std::vector<double>(choices.size(), 1.0 / choices.size());
        } else {
            double s = 0.0;
            for (double v : probs) s += v;
            for (double &v : const_cast<std::vector<double>&>(probs)) v /= s;
        }
    }
    torch::Tensor apply(const torch::Tensor& img) override {
        std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
        size_t idx = dist(gen);
        return choices[idx]->apply(img);
    }
};

TransformPtr make_random_choice(const std::vector<TransformPtr>& choices, const std::vector<double>& probs) {
    return std::make_shared<RandomChoice>(choices, probs);
}

static torch::Tensor to_one_hot(const torch::Tensor& y, int64_t C, torch::Device device=torch::kCPU) {
    auto y_long = y.to(torch::kLong).to(device);
    auto oh = torch::nn::functional::one_hot(y_long, C).to(torch::kFloat32);
    return oh;
}

std::pair<torch::Tensor, torch::Tensor> cutmix_batch(const torch::Tensor& x, const torch::Tensor& y, int num_classes, double alpha) {
    TORCH_CHECK(x.dim() == 4, "cutmix expects batched images NCHW");
    int64_t N = x.size(0);
    torch::Tensor x_perm = x;
    if (alpha <= 0.0) {
        auto y_oh = to_one_hot(y, num_classes, x.device());
        return {x, y_oh};
    }
    auto gamma = [](double a, std::mt19937& g) {
        std::gamma_distribution<double> dist(a, 1.0);
        return dist(g);
    };
    std::mt19937 gen(std::random_device{}());
    double a1 = gamma(alpha, gen);
    double a2 = gamma(alpha, gen);
    double lam = a1 / (a1 + a2);

    std::uniform_int_distribution<int64_t> perm_idx(0, N - 1);
    int64_t index = perm_idx(gen);
    int64_t H = x.size(2), W = x.size(3);
    double cut_rat = std::sqrt(1.0 - lam);
    int64_t cut_w = (int64_t)(W * cut_rat);
    int64_t cut_h = (int64_t)(H * cut_rat);
    std::uniform_int_distribution<int64_t> cx(0, W-1);
    std::uniform_int_distribution<int64_t> cy(0, H-1);
    int64_t cx_i = cx(gen);
    int64_t cy_i = cy(gen);
    int64_t bbx1 = std::max<int64_t>(0, cx_i - cut_w/2);
    int64_t bby1 = std::max<int64_t>(0, cy_i - cut_h/2);
    int64_t bbx2 = std::min<int64_t>(W, cx_i + cut_w/2);
    int64_t bby2 = std::min<int64_t>(H, cy_i + cut_h/2);
    auto out = x.clone();
    auto perm = torch::randperm(N, x.device());
    for (int64_t i = 0; i < N; ++i) {
        int64_t j = perm[i].item<int64_t>();
        out[i].index_put_({torch::indexing::Slice(), torch::indexing::Slice(bby1, bby2), torch::indexing::Slice(bbx1, bbx2)}, x[j].index({torch::indexing::Slice(), torch::indexing::Slice(bby1, bby2), torch::indexing::Slice(bbx1, bbx2)}));
    }
    double area = (double)((bbx2 - bbx1) * (bby2 - bby1));
    double lam_adjust = 1.0 - area / (double)(H * W);

    auto y1 = to_one_hot(y, num_classes, x.device());
    auto y_shuffled = to_one_hot(y.index({perm}), num_classes, x.device());
    auto y_out = y1 * lam_adjust + y_shuffled * (1.0 - lam_adjust);
    return {out, y_out};
}

std::pair<torch::Tensor, torch::Tensor> mixup_batch(const torch::Tensor& x, const torch::Tensor& y, int num_classes, double alpha) {
    TORCH_CHECK(x.dim() == 4, "mixup expects batched images NCHW");
    int64_t N = x.size(0);
    if (alpha <= 0.0) {
        return {x, to_one_hot(y, num_classes, x.device())};
    }
    std::mt19937 gen(std::random_device{}());
    std::gamma_distribution<double> ga(alpha, 1.0);
    double a1 = ga(gen);
    double a2 = ga(gen);
    double lam = a1 / (a1 + a2);

    auto perm = torch::randperm(N, x.device());
    auto x_shuf = x.index({perm});
    auto y1 = to_one_hot(y, num_classes, x.device());
    auto y_shuf = to_one_hot(y.index({perm}), num_classes, x.device());

    auto out = x * lam + x_shuf * (1.0 - lam);
    auto y_out = y1 * lam + y_shuf * (1.0 - lam);
    return {out, y_out};
}
