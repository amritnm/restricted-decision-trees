#pragma once

#include <stdexcept>
#include <string>

namespace rdt {

enum class Task { Classification, Regression };

enum class Criterion { Gini, Entropy, MSE, MAE };

inline Criterion default_criterion(Task task) {
    return task == Task::Classification ? Criterion::Gini : Criterion::MSE;
}

inline Criterion criterion_from_string(const std::string& name) {
    if (name == "gini") return Criterion::Gini;
    if (name == "entropy") return Criterion::Entropy;
    if (name == "mse") return Criterion::MSE;
    if (name == "mae") return Criterion::MAE;
    throw std::invalid_argument("Unknown criterion: " + name);
}

inline std::string to_string(Criterion c) {
    switch (c) {
        case Criterion::Gini: return "gini";
        case Criterion::Entropy: return "entropy";
        case Criterion::MSE: return "mse";
        case Criterion::MAE: return "mae";
    }
    return "";
}

struct TreeParams {
    Task task = Task::Classification;
    Criterion criterion = Criterion::Gini;
    int max_depth = 5;
    int min_samples_split = 2;
    int min_samples_leaf = 1;
};

} // namespace rdt
