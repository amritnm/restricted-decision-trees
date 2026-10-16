#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <map>
#include <memory>

#include <mutex>
#include <optional>
#include <stdexcept>
#include <unordered_map>

#include "rdt/forest.hpp"
#include "rdt/gradient_boosting.hpp"
#include "rdt/tree.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace rdt;

namespace {

using InputMatrix = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
// tree_method='hist' fits: float32 or float64, any strides (C/F order, views,
// memory-mapped files), read in place.
using AnyMatrix = nb::ndarray<nb::ro, nb::ndim<2>, nb::device::cpu>;
using InputVector = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

MatrixView to_matrix_view(const InputMatrix& X) {
    return MatrixView(X.data(), X.shape(0), X.shape(1), /*row_major=*/true);
}

VectorView to_vector_view(const InputVector& y) {
    return VectorView(y.data(), y.shape(0));
}

// The core indexes y/sample_weight by row and X by the column numbers seen at
// fit time without bounds checks, so mismatched shapes must never reach it.
// (The Python wrappers validate first; these guard direct use of the module.)
void check_fit_shapes(const InputMatrix& X, const InputVector& y, const std::optional<InputVector>& w) {
    if (y.shape(0) != X.shape(0)) throw std::invalid_argument("X and y have different numbers of rows");
    if (w && w->shape(0) != X.shape(0)) throw std::invalid_argument("X and sample_weight have different numbers of rows");
}

void check_predict_shape(const InputMatrix& X, std::size_t n_features) {
    if (X.shape(1) != n_features) throw std::invalid_argument("X has a different number of features than during fit");
}

// fit()/predict() run with the GIL released: the views above are plain
// pointers into the input arrays, which the bound function's arguments keep
// alive for the duration of the call, and the core never touches a Python
// object. Results are copied into NumPy arrays only after the GIL is
// re-acquired.
using OutputArray = nb::ndarray<nb::numpy, double>;

OutputArray to_numpy(std::vector<double>&& values, std::size_t n_rows, std::size_t n_cols, bool two_d) {
    auto* heap = new std::vector<double>(std::move(values));
    nb::capsule owner(heap, [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
    if (!two_d) return OutputArray(heap->data(), {n_rows}, owner);
    // v0.2.0 returned a list of per-row lists; for zero rows np.array() of
    // that is shape (0,), so keep that shape.
    if (n_rows == 0) return OutputArray(heap->data(), {std::size_t{0}}, owner);
    return OutputArray(heap->data(), {n_rows, n_cols}, owner);
}

// The native models are not internally synchronized, and their fit/predict
// run with the GIL released, so concurrent use of one native object is
// checked here: fit() needs the object to itself; predict(), predict_proba()
// and attribute reads may share it. A conflicting call raises RuntimeError
// instead of racing. (The rdt Python estimators fit a fresh native object and
// publish it only once fitted, so they never conflict.)
class UseRegistry {
public:
    static UseRegistry& instance() {
        static UseRegistry registry;
        return registry;
    }

    void acquire(const void* obj, bool exclusive) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = users_.find(obj);
        if (it != users_.end()) {
            if (it->second < 0) throw std::runtime_error("this model is being fitted by another thread");
            if (exclusive) throw std::runtime_error("fit() called while this model is in use by another thread");
        }
        if (exclusive) {
            users_[obj] = -1;
        } else {
            ++users_[obj];
        }
    }

    void release(const void* obj, bool exclusive) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = users_.find(obj);
        if (exclusive || --it->second == 0) users_.erase(it);
    }

private:
    std::mutex mutex_;
    std::unordered_map<const void*, long> users_; // -1: being fitted; > 0: number of readers
};

class UseGuard {
public:
    UseGuard(const void* obj, bool exclusive) : obj_(obj), exclusive_(exclusive) {
        UseRegistry::instance().acquire(obj_, exclusive_);
    }
    ~UseGuard() { UseRegistry::instance().release(obj_, exclusive_); }
    UseGuard(const UseGuard&) = delete;
    UseGuard& operator=(const UseGuard&) = delete;

private:
    const void* obj_;
    bool exclusive_;
};

// Attribute read under a shared UseGuard.
template <class Model, class F>
auto read(const Model& self, F&& f) {
    UseGuard guard(&self, /*exclusive=*/false);
    return f(self);
}

template <class Model>
void fit_released(Model& self, const InputMatrix& X, const InputVector& y,
                  const std::optional<InputVector>& sample_weight) {
    check_fit_shapes(X, y, sample_weight);
    const MatrixView xv = to_matrix_view(X);
    const VectorView yv = to_vector_view(y);
    std::optional<VectorView> wv;
    if (sample_weight) wv = to_vector_view(*sample_weight);
    UseGuard guard(&self, /*exclusive=*/true);
    nb::gil_scoped_release release;
    self.fit(xv, yv, wv ? &*wv : nullptr);
}

// Calls f(StridedMatrix<T>) for X's element type (float32 or float64).
template <class F>
void with_strided(const AnyMatrix& X, F&& f) {
    const std::size_t n = X.shape(0), p = X.shape(1);
    const auto rs = static_cast<std::ptrdiff_t>(X.stride(0)), cs = static_cast<std::ptrdiff_t>(X.stride(1));
    if (X.dtype() == nb::dtype<double>()) {
        f(StridedMatrix<double>(static_cast<const double*>(X.data()), n, p, rs, cs));
    } else if (X.dtype() == nb::dtype<float>()) {
        f(StridedMatrix<float>(static_cast<const float*>(X.data()), n, p, rs, cs));
    } else {
        throw std::invalid_argument("X must be a float32 or float64 array");
    }
}

void check_fit_shapes_any(const AnyMatrix& X, const InputVector& y, const std::optional<InputVector>& w) {
    if (y.shape(0) != X.shape(0)) throw std::invalid_argument("X and y have different numbers of rows");
    if (w && w->shape(0) != X.shape(0)) throw std::invalid_argument("X and sample_weight have different numbers of rows");
}

// fit_hist of a model: X read in place (no copy) with the GIL released.
template <class Model>
void fit_hist_released(Model& self, const AnyMatrix& X, const InputVector& y,
                       const std::optional<InputVector>& sample_weight) {
    check_fit_shapes_any(X, y, sample_weight);
    const VectorView yv = to_vector_view(y);
    std::optional<VectorView> wv;
    if (sample_weight) wv = to_vector_view(*sample_weight);
    UseGuard guard(&self, /*exclusive=*/true);
    nb::gil_scoped_release release;
    with_strided(X, [&](const auto& xs) { self.fit_hist(xs, yv, wv ? &*wv : nullptr); });
}

// ---- GradientBoostingModel fits with an optional eval set (X_val, y_val,
// sample_weight_val): the eval features are read in place (exact: C-contiguous
// float64 like X; hist: float32 / float64, any strides) and type-erased, so
// the boosting loop is instantiated once per training matrix type only.
void check_eval_args(std::size_t n_features, bool has_X_val, std::size_t val_rows, std::size_t val_cols,
                     const std::optional<InputVector>& y_val, const std::optional<InputVector>& w_val) {
    if (has_X_val != y_val.has_value()) throw std::invalid_argument("X_val and y_val must be given together");
    if (w_val && !has_X_val) throw std::invalid_argument("sample_weight_val needs X_val and y_val");
    if (!has_X_val) return;
    if (val_cols != n_features) throw std::invalid_argument("X_val has a different number of features than X");
    if (y_val->shape(0) != val_rows) throw std::invalid_argument("X_val and y_val have different numbers of rows");
    if (w_val && w_val->shape(0) != val_rows) {
        throw std::invalid_argument("X_val and sample_weight_val have different numbers of rows");
    }
}

std::unique_ptr<EvalFeatures> eval_features_any(const AnyMatrix& X) {
    const std::size_t n = X.shape(0), p = X.shape(1);
    const auto rs = static_cast<std::ptrdiff_t>(X.stride(0)), cs = static_cast<std::ptrdiff_t>(X.stride(1));
    if (X.dtype() == nb::dtype<double>()) {
        return std::make_unique<EvalFeaturesOf<StridedMatrix<double>>>(
            StridedMatrix<double>(static_cast<const double*>(X.data()), n, p, rs, cs));
    }
    if (X.dtype() == nb::dtype<float>()) {
        return std::make_unique<EvalFeaturesOf<StridedMatrix<float>>>(
            StridedMatrix<float>(static_cast<const float*>(X.data()), n, p, rs, cs));
    }
    throw std::invalid_argument("X_val must be a float32 or float64 array");
}

// Builds the EvalSet (pointing into `features` / the arrays) when an eval set is given.
struct EvalArgs {
    std::unique_ptr<EvalFeatures> features;
    std::optional<VectorView> weight;
    EvalSet set;

    const EvalSet* get() const { return features ? &set : nullptr; }

    EvalArgs(std::unique_ptr<EvalFeatures> f, const std::optional<InputVector>& y_val,
             const std::optional<InputVector>& w_val)
        : features(std::move(f)) {
        if (!features) return;
        set.X = features.get();
        set.y = to_vector_view(*y_val);
        if (w_val) {
            weight = to_vector_view(*w_val);
            set.sample_weight = &*weight;
        }
    }
    EvalArgs(const EvalArgs&) = delete;
    EvalArgs& operator=(const EvalArgs&) = delete;
};

void gb_fit_released(GradientBoostingModel& self, const InputMatrix& X, const InputVector& y,
                     const std::optional<InputVector>& sample_weight, const std::optional<InputMatrix>& X_val,
                     const std::optional<InputVector>& y_val, const std::optional<InputVector>& w_val) {
    check_fit_shapes(X, y, sample_weight);
    check_eval_args(X.shape(1), X_val.has_value(), X_val ? X_val->shape(0) : 0, X_val ? X_val->shape(1) : 0, y_val,
                    w_val);
    const MatrixView xv = to_matrix_view(X);
    const VectorView yv = to_vector_view(y);
    std::optional<VectorView> wv;
    if (sample_weight) wv = to_vector_view(*sample_weight);
    std::unique_ptr<EvalFeatures> features;
    if (X_val) features = std::make_unique<EvalFeaturesOf<MatrixView>>(to_matrix_view(*X_val));
    const EvalArgs eval(std::move(features), y_val, w_val);
    UseGuard guard(&self, /*exclusive=*/true);
    nb::gil_scoped_release release;
    self.fit(xv, yv, wv ? &*wv : nullptr, eval.get());
}

void gb_fit_hist_released(GradientBoostingModel& self, const AnyMatrix& X, const InputVector& y,
                          const std::optional<InputVector>& sample_weight, const std::optional<AnyMatrix>& X_val,
                          const std::optional<InputVector>& y_val, const std::optional<InputVector>& w_val) {
    check_fit_shapes_any(X, y, sample_weight);
    check_eval_args(X.shape(1), X_val.has_value(), X_val ? X_val->shape(0) : 0, X_val ? X_val->shape(1) : 0, y_val,
                    w_val);
    const VectorView yv = to_vector_view(y);
    std::optional<VectorView> wv;
    if (sample_weight) wv = to_vector_view(*sample_weight);
    const EvalArgs eval(X_val ? eval_features_any(*X_val) : nullptr, y_val, w_val);
    UseGuard guard(&self, /*exclusive=*/true);
    nb::gil_scoped_release release;
    with_strided(X, [&](const auto& xs) { self.fit_hist(xs, yv, wv ? &*wv : nullptr, eval.get()); });
}

EvalMetric eval_metric_from_string(const std::string& name) {
    if (name.empty()) return EvalMetric::Default;
    if (name == "logloss") return EvalMetric::LogLoss;
    if (name == "auc") return EvalMetric::AUC;
    if (name == "error") return EvalMetric::Error;
    if (name == "rmse") return EvalMetric::RMSE;
    if (name == "mae") return EvalMetric::MAE;
    throw std::invalid_argument("Unknown eval_metric: " + name);
}

std::string eval_metric_name(EvalMetric metric) {
    switch (metric) {
        case EvalMetric::LogLoss: return "logloss";
        case EvalMetric::AUC: return "auc";
        case EvalMetric::Error: return "error";
        case EvalMetric::RMSE: return "rmse";
        case EvalMetric::MAE: return "mae";
        case EvalMetric::Default: break;
    }
    return "";
}

// RDTModel::fit, with the per-row hessians a criterion='newton' tree needs.
void fit_tree_released(RDTModel& self, const InputMatrix& X, const InputVector& y,
                       const std::optional<InputVector>& sample_weight, const std::optional<InputVector>& hessian) {
    check_fit_shapes(X, y, sample_weight);
    if (hessian && hessian->shape(0) != X.shape(0)) throw std::invalid_argument("X and hessian have different numbers of rows");
    const MatrixView xv = to_matrix_view(X);
    const VectorView yv = to_vector_view(y);
    std::optional<VectorView> wv, hv;
    if (sample_weight) wv = to_vector_view(*sample_weight);
    if (hessian) hv = to_vector_view(*hessian);
    UseGuard guard(&self, /*exclusive=*/true);
    nb::gil_scoped_release release;
    self.fit(xv, yv, wv ? &*wv : nullptr, nullptr, hv ? &*hv : nullptr);
}

void fit_tree_hist_released(RDTModel& self, const AnyMatrix& X, const InputVector& y,
                            const std::optional<InputVector>& sample_weight, const std::optional<InputVector>& hessian) {
    check_fit_shapes_any(X, y, sample_weight);
    if (hessian && hessian->shape(0) != X.shape(0)) throw std::invalid_argument("X and hessian have different numbers of rows");
    const VectorView yv = to_vector_view(y);
    std::optional<VectorView> wv, hv;
    if (sample_weight) wv = to_vector_view(*sample_weight);
    if (hessian) hv = to_vector_view(*hessian);
    UseGuard guard(&self, /*exclusive=*/true);
    nb::gil_scoped_release release;
    with_strided(X, [&](const auto& xs) { self.fit_hist(xs, yv, wv ? &*wv : nullptr, nullptr, hv ? &*hv : nullptr); });
}

void check_max_bin(int max_bin) {
    if (max_bin < kMinMaxBin || max_bin > kMaxMaxBin) throw std::invalid_argument("max_bin must be in [2, 65536]");
}

SplitCriterion split_criterion_from_string(const std::string& name) {
    if (name == "friedman_mse") return SplitCriterion::FriedmanMSE;
    if (name == "newton") return SplitCriterion::Newton;
    throw std::invalid_argument("Unknown split_criterion: " + name);
}

template <class Model>
OutputArray predict_released(const Model& self, const InputMatrix& X) {
    const MatrixView xv = to_matrix_view(X);
    std::vector<double> out;
    {
        UseGuard guard(&self, /*exclusive=*/false);
        check_predict_shape(X, self.n_features());
        nb::gil_scoped_release release;
        out = self.predict(xv);
    }
    const std::size_t n = out.size();
    return to_numpy(std::move(out), n, 1, /*two_d=*/false);
}

template <class Model>
OutputArray predict_proba_released(const Model& self, const InputMatrix& X) {
    const MatrixView xv = to_matrix_view(X);
    std::vector<double> out;
    std::size_t k = 0;
    {
        UseGuard guard(&self, /*exclusive=*/false);
        check_predict_shape(X, self.n_features());
        k = self.n_classes();
        out.resize(xv.n_rows() * k);
        nb::gil_scoped_release release;
        self.predict_proba_into(xv, out.data());
    }
    return to_numpy(std::move(out), xv.n_rows(), k, /*two_d=*/true);
}

using U32Array = nb::ndarray<nb::numpy, std::uint32_t>;

U32Array to_numpy_u32(std::vector<std::uint32_t>&& values, std::size_t n_rows, std::size_t n_cols) {
    auto* heap = new std::vector<std::uint32_t>(std::move(values));
    nb::capsule owner(heap, [](void* p) noexcept { delete static_cast<std::vector<std::uint32_t>*>(p); });
    return U32Array(heap->data(), {n_rows, n_cols}, owner);
}

nb::dict int_map_to_dict(const std::map<int, int>& m) {
    nb::dict d;
    for (const auto& kv : m) d[nb::int_(kv.first)] = kv.second;
    return d;
}

nb::list int_maps_to_list(const std::vector<std::map<int, int>>& maps) {
    nb::list out;
    for (const auto& m : maps) out.append(int_map_to_dict(m));
    return out;
}

nb::list level_maps_to_list(const std::vector<std::map<int, std::vector<std::uint32_t>>>& maps) {
    nb::list out;
    for (const auto& m : maps) {
        nb::dict d;
        for (const auto& kv : m) {
            nb::list cols;
            for (auto c : kv.second) cols.append(c);
            d[nb::int_(kv.first)] = cols;
        }
        out.append(d);
    }
    return out;
}

void check_colsample_bylevel(double v) {
    if (!(v > 0.0 && v <= 1.0)) throw std::invalid_argument("colsample_bylevel must be in (0, 1]");
}

nb::dict double_map_to_dict(const std::map<int, double>& m) {
    nb::dict d;
    for (const auto& kv : m) d[nb::int_(kv.first)] = kv.second;
    return d;
}

GBLoss loss_from_string(const std::string& loss) {
    if (loss == "log_loss") return GBLoss::LogLoss;
    if (loss == "exponential") return GBLoss::Exponential;
    if (loss == "squared_error") return GBLoss::SquaredError;
    if (loss == "absolute_error") return GBLoss::AbsoluteError;
    if (loss == "huber") return GBLoss::Huber;
    if (loss == "quantile") return GBLoss::Quantile;
    throw std::invalid_argument("Unknown loss: " + loss);
}

// Staged predictions of a fitted GradientBoostingModel, one boosting round
// per __next__ (scikit-learn's staged_decision_function / staged_predict_proba
// / staged_predict). The raw scores are accumulated natively, one round at a
// time, with the GIL released; every stage is returned as a new array (no
// buffer is shared between stages). The iterator keeps X and the model alive;
// a refit of the model while iterating raises RuntimeError.
class StagedScores {
public:
    enum class Kind { Decision, Proba, Predict };

    StagedScores(const GradientBoostingModel& model, InputMatrix X, Kind kind)
        : model_(&model), X_(std::move(X)), kind_(kind) {
        UseGuard guard(model_, /*exclusive=*/false);
        check_predict_shape(X_, model_->n_features());
        generation_ = model_->fit_generation();
        F_ = model_->init_score_matrix(X_.shape(0));
    }

    OutputArray next() {
        std::lock_guard<std::mutex> lock(mutex_);
        const MatrixView xv = to_matrix_view(X_);
        const std::size_t n = xv.n_rows();
        std::vector<double> out;
        std::size_t width = 1;
        {
            UseGuard guard(model_, /*exclusive=*/false);
            if (model_->fit_generation() != generation_) {
                throw std::runtime_error("the model was refitted during staged prediction");
            }
            if (round_ >= model_->n_estimators_fitted()) throw nb::stop_iteration();
            width = kind_ == Kind::Proba ? model_->n_classes() : kind_ == Kind::Decision ? model_->n_outputs() : 1;
            out.resize(n * width);
            nb::gil_scoped_release release;
            model_->add_round(xv, round_, F_);
            ++round_;
            switch (kind_) {
                case Kind::Decision: model_->scores_to_decision(F_, n, out.data()); break;
                case Kind::Proba: model_->scores_to_proba(F_, n, out.data()); break;
                case Kind::Predict: model_->scores_to_labels(F_, n, out.data()); break;
            }
        }
        return to_numpy(std::move(out), n, width, /*two_d=*/width > 1 || kind_ == Kind::Proba);
    }

private:
    const GradientBoostingModel* model_;
    InputMatrix X_;
    Kind kind_;
    std::uint64_t generation_ = 0;
    std::vector<std::vector<double>> F_;
    std::size_t round_ = 0;
    std::mutex mutex_;
};

StagedScores::Kind staged_kind_from_string(const std::string& kind) {
    if (kind == "decision") return StagedScores::Kind::Decision;
    if (kind == "proba") return StagedScores::Kind::Proba;
    if (kind == "predict") return StagedScores::Kind::Predict;
    throw std::invalid_argument("Unknown staged prediction kind: " + kind);
}

} // namespace

NB_MODULE(_rdt_native, m) {
    nb::class_<Node>(m, "Node")
        .def_prop_ro("is_leaf", &Node::is_leaf)
        .def_prop_ro("indices",
                      [](const Node& n) {
                          if (!n.has_indices) {
                              throw nb::attribute_error(
                                  "Node.indices is not retained for this tree: gradient boosting trees and "
                                  "bootstrapped random forest trees drop their per-node training indices after "
                                  "fitting (a standalone RDT and RandomForestRDT(bootstrap=False) keep them). "
                                  "Node.n_samples holds the count.");
                          }
                          return n.indices;
                      })
        .def_prop_ro("n_samples", [](const Node& n) { return n.n_samples; })
        .def_ro("split_feature", &Node::split_feature)
        .def_ro("split_threshold", &Node::split_threshold)
        .def_ro("split_bin", &Node::split_bin)
        .def_ro("value", &Node::value)
        .def_ro("class_counts", &Node::class_counts)
        .def_prop_ro("left", [](const Node& n) -> const Node* { return n.left.get(); },
                      nb::rv_policy::reference_internal)
        .def_prop_ro("right", [](const Node& n) -> const Node* { return n.right.get(); },
                      nb::rv_policy::reference_internal);

    // criterion='newton' (task='regression'): fit(X, gradient, hessian=...)
    // grows an XGBoost-style tree on the given per-row gradients / hessians
    // (as GradientBoostingRDT(split_criterion='newton') does each round); the
    // reg_* / gamma / min_child_weight / max_delta_step arguments only apply then.
    nb::class_<RDTModel>(m, "RDTModel")
        .def(nb::new_([](std::string task, std::string criterion, int max_depth,
                          int min_samples_split, int min_samples_leaf, double min_impurity_decrease,
                          bool presort, double reg_lambda, double reg_alpha, double gamma,
                          double min_child_weight, double max_delta_step, std::string tree_method, int max_bin,
                          int n_jobs) {
                 TreeParams p;
                 p.tree_method = tree_method_from_string(tree_method);
                 check_max_bin(max_bin);
                 p.max_bin = max_bin;
                 p.n_jobs = n_jobs;
                 p.task = (task == "classification") ? Task::Classification : Task::Regression;
                 p.criterion = criterion_from_string(criterion);
                 p.max_depth = max_depth;
                 p.min_samples_split = min_samples_split;
                 p.min_samples_leaf = min_samples_leaf;
                 p.min_impurity_decrease = min_impurity_decrease;
                 p.presort = presort;
                 p.newton.reg_lambda = reg_lambda;
                 p.newton.reg_alpha = reg_alpha;
                 p.newton.gamma = gamma;
                 p.newton.min_child_weight = min_child_weight;
                 p.newton.max_delta_step = max_delta_step;
                 return new RDTModel(p);
             }),
             "task"_a, "criterion"_a, "max_depth"_a, "min_samples_split"_a, "min_samples_leaf"_a,
             "min_impurity_decrease"_a = 0.0, "presort"_a = true, "reg_lambda"_a = 1.0, "reg_alpha"_a = 0.0,
             "gamma"_a = 0.0, "min_child_weight"_a = 1.0, "max_delta_step"_a = 0.0, "tree_method"_a = "exact",
             "max_bin"_a = 256, "n_jobs"_a = 1)
        .def("fit",
             [](RDTModel& self, InputMatrix X, InputVector y, std::optional<InputVector> sample_weight,
                std::optional<InputVector> hessian) { fit_tree_released(self, X, y, sample_weight, hessian); },
             "X"_a, "y"_a, "sample_weight"_a = nb::none(), "hessian"_a = nb::none())
        .def("fit_hist",
             [](RDTModel& self, AnyMatrix X, InputVector y, std::optional<InputVector> sample_weight,
                std::optional<InputVector> hessian) { fit_tree_hist_released(self, X, y, sample_weight, hessian); },
             "X"_a, "y"_a, "sample_weight"_a = nb::none(), "hessian"_a = nb::none())
        .def("predict", [](const RDTModel& self, InputMatrix X) { return predict_released(self, X); }, "X"_a)
        .def("predict_proba", [](const RDTModel& self, InputMatrix X) { return predict_proba_released(self, X); },
             "X"_a)
        .def_prop_ro("classes_", [](const RDTModel& self) { return read(self, [](auto& m) { return m.classes(); }); })
        .def_prop_ro("n_classes_",
                      [](const RDTModel& self) { return read(self, [](auto& m) { return m.n_classes(); }); })
        .def_prop_ro("depth_features_",
                      [](const RDTModel& self) {
                          return read(self, [](auto& m) { return int_map_to_dict(m.depth_features()); });
                      })
        .def_prop_ro("depth_improvements_",
                      [](const RDTModel& self) {
                          return read(self, [](auto& m) { return double_map_to_dict(m.depth_improvements()); });
                      })
        .def_prop_ro("root",
                      [](const RDTModel& self) -> const Node* {
                          return read(self, [](auto& m) { return m.root(); });
                      },
                      nb::rv_policy::reference_internal);

    // Test hooks for the presort encoding (tests/test_split_search.py): the
    // presorted columns of X's `rows` (shape (n_cols, len(rows)); see
    // presort.hpp for the entry encoding), and the per-round derivation used by
    // gradient boosting (rows with new_id < 0 dropped, the rest renumbered).
    using U32Input = nb::ndarray<const std::uint32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
    using U32Matrix = nb::ndarray<const std::uint32_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
    using I32Input = nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
    m.def("_presort", [](InputMatrix X, U32Input rows) {
        const MatrixView xv = to_matrix_view(X);
        std::vector<std::uint32_t> r(rows.data(), rows.data() + rows.shape(0));
        for (auto v : r) {
            if (v >= xv.n_rows()) throw std::out_of_range("row out of range");
        }
        SortedColumns sorted(IndexedMatrixView(xv), r);
        std::vector<std::uint32_t> out(sorted.n_cols() * sorted.n_entries());
        for (std::size_t c = 0; c < sorted.n_cols(); ++c) {
            std::copy(sorted.column(c), sorted.column(c) + sorted.n_entries(), out.begin() + c * sorted.n_entries());
        }
        return to_numpy_u32(std::move(out), sorted.n_cols(), sorted.n_entries());
    }, "X"_a, "rows"_a);
    m.def("_presort_restrict", [](U32Matrix sorted, I32Input new_id, std::size_t n_kept) {
        PresortedView src;
        src.n_entries = sorted.shape(1);
        for (std::size_t c = 0; c < sorted.shape(0); ++c) {
            const std::uint32_t* col = sorted.data() + c * src.n_entries;
            for (std::size_t i = 0; i < src.n_entries; ++i) {
                if ((col[i] & kPresortRowMask) >= new_id.shape(0)) throw std::out_of_range("row out of range");
            }
            std::size_t kept = 0;
            for (std::size_t i = 0; i < src.n_entries; ++i) kept += new_id.data()[col[i] & kPresortRowMask] >= 0;
            if (kept != n_kept) throw std::invalid_argument("n_kept does not match new_id");
            src.columns.push_back(col);
        }
        SortedColumns derived(src, new_id.data(), n_kept);
        std::vector<std::uint32_t> out(derived.n_cols() * n_kept);
        for (std::size_t c = 0; c < derived.n_cols(); ++c) {
            std::copy(derived.column(c), derived.column(c) + n_kept, out.begin() + c * n_kept);
        }
        return to_numpy_u32(std::move(out), derived.n_cols(), n_kept);
    }, "sorted"_a, "new_id"_a, "n_kept"_a);

    // Test hook for hist binning (tests/test_hist.py): bins of every row of X
    // (shape (n_rows, n_cols), uint16) with cuts from `rows`, and each
    // column's cuts (x <= cuts[b] exactly when bin(x) <= b).
    m.def("_hist_bins", [](AnyMatrix X, U32Input rows, int max_bin) {
        check_max_bin(max_bin);
        std::vector<std::uint32_t> r(rows.data(), rows.data() + rows.shape(0));
        for (auto v : r) {
            if (v >= X.shape(0)) throw std::out_of_range("row out of range");
        }
        BinnedMatrix bins;
        {
            nb::gil_scoped_release release;
            with_strided(X, [&](const auto& xs) { bins = BinnedMatrix(xs, r, max_bin); });
        }
        auto* heap = new std::vector<std::uint16_t>(bins.n_rows() * bins.n_cols());
        for (std::size_t i = 0; i < bins.n_rows(); ++i) {
            for (std::size_t c = 0; c < bins.n_cols(); ++c) (*heap)[i * bins.n_cols() + c] = static_cast<std::uint16_t>(bins.bin(i, c));
        }
        nb::capsule owner(heap, [](void* p) noexcept { delete static_cast<std::vector<std::uint16_t>*>(p); });
        nb::ndarray<nb::numpy, std::uint16_t> out(heap->data(), {bins.n_rows(), bins.n_cols()}, owner);
        nb::list cuts;
        for (std::size_t c = 0; c < bins.n_cols(); ++c) {
            std::vector<double> cc(bins.cuts(c), bins.cuts(c) + (bins.n_bins(c) - 1));
            cuts.append(to_numpy(std::move(cc), bins.n_bins(c) - 1, 1, false));
        }
        return nb::make_tuple(out, cuts, bins.wide());
    }, "X"_a, "rows"_a, "max_bin"_a);

    // MaxFeaturesSpec is opaque from Python; constructed only via these
    // factories, which the rdt.ensemble wrapper calls after normalizing the
    // user's max_features/colsample_bytree argument ('sqrt'/'log2'/None/int/float).
    nb::class_<MaxFeaturesSpec>(m, "MaxFeaturesSpec");
    m.def("_max_features_sqrt", [] { return MaxFeaturesSpec::Sqrt(); });
    m.def("_max_features_log2", [] { return MaxFeaturesSpec::Log2(); });
    m.def("_max_features_all", [] { return MaxFeaturesSpec::All(); });
    m.def("_max_features_int", [](int v) { return MaxFeaturesSpec::FromInt(v); }, "v"_a);
    m.def("_max_features_float", [](double v) { return MaxFeaturesSpec::FromFloat(v); }, "v"_a);

    nb::class_<RandomForestModel>(m, "RandomForestModel")
        .def(nb::new_([](std::string task, std::string criterion, int max_depth,
                          int min_samples_split, int min_samples_leaf, double min_impurity_decrease,
                          int n_estimators, MaxFeaturesSpec max_features, bool bootstrap,
                          double max_samples, bool oob_score, std::uint64_t random_state, int n_jobs,
                          std::uint32_t max_samples_count, bool presort, std::string tree_method, int max_bin,
                          double colsample_bylevel) {
                 ForestParams p;
                 check_colsample_bylevel(colsample_bylevel);
                 p.tree_params.colsample_bylevel = colsample_bylevel;
                 p.tree_params.tree_method = tree_method_from_string(tree_method);
                 check_max_bin(max_bin);
                 p.tree_params.max_bin = max_bin;
                 p.tree_params.task = (task == "classification") ? Task::Classification : Task::Regression;
                 p.tree_params.criterion = criterion_from_string(criterion);
                 p.tree_params.max_depth = max_depth;
                 p.tree_params.min_samples_split = min_samples_split;
                 p.tree_params.min_samples_leaf = min_samples_leaf;
                 p.tree_params.min_impurity_decrease = min_impurity_decrease;
                 p.n_estimators = n_estimators;
                 p.max_features = max_features;
                 p.bootstrap = bootstrap;
                 p.max_samples = max_samples;
                 p.max_samples_count = max_samples_count;
                 p.oob_score = oob_score;
                 p.random_state = random_state;
                 p.n_jobs = n_jobs;
                 p.tree_params.presort = presort;
                 return new RandomForestModel(p);
             }),
             "task"_a, "criterion"_a, "max_depth"_a, "min_samples_split"_a, "min_samples_leaf"_a,
             "min_impurity_decrease"_a, "n_estimators"_a, "max_features"_a, "bootstrap"_a,
             "max_samples"_a, "oob_score"_a, "random_state"_a, "n_jobs"_a, "max_samples_count"_a = 0,
             "presort"_a = true, "tree_method"_a = "exact", "max_bin"_a = 256, "colsample_bylevel"_a = 1.0)
        .def("fit",
             [](RandomForestModel& self, InputMatrix X, InputVector y, std::optional<InputVector> sample_weight) {
                 fit_released(self, X, y, sample_weight);
             },
             "X"_a, "y"_a, "sample_weight"_a = nb::none())
        .def("fit_hist",
             [](RandomForestModel& self, AnyMatrix X, InputVector y, std::optional<InputVector> sample_weight) {
                 fit_hist_released(self, X, y, sample_weight);
             },
             "X"_a, "y"_a, "sample_weight"_a = nb::none())
        .def("predict", [](const RandomForestModel& self, InputMatrix X) { return predict_released(self, X); },
             "X"_a)
        .def("predict_proba",
             [](const RandomForestModel& self, InputMatrix X) { return predict_proba_released(self, X); }, "X"_a)
        .def_prop_ro("classes_",
                      [](const RandomForestModel& self) { return read(self, [](auto& m) { return m.classes(); }); })
        .def_prop_ro("n_classes_",
                      [](const RandomForestModel& self) { return read(self, [](auto& m) { return m.n_classes(); }); })
        .def_prop_ro("feature_importances_",
                      [](const RandomForestModel& self) {
                          return read(self, [](auto& m) { return m.feature_importances(); });
                      })
        .def_prop_ro("oob_score_",
                      [](const RandomForestModel& self) { return read(self, [](auto& m) { return m.oob_score(); }); })
        .def_prop_ro("has_oob_score_", [](const RandomForestModel& self) { return self.has_oob_score(); })
        .def_prop_ro("feature_subsets_",
                      [](const RandomForestModel& self) {
                          return read(self, [](auto& m) { return m.feature_subsets(); });
                      })
        .def_prop_ro("level_features_",
                      [](const RandomForestModel& self) {
                          return read(self, [](auto& m) { return level_maps_to_list(m.level_features()); });
                      })
        .def_prop_ro("depth_features_",
                      [](const RandomForestModel& self) {
                          return read(self, [](auto& m) { return int_maps_to_list(m.depth_features()); });
                      })
        .def_prop_ro(
            "estimators_",
            [](const RandomForestModel& self) {
                return read(self, [](auto& m) {
                    std::vector<const RDTModel*> out;
                    out.reserve(m.estimators().size());
                    for (const auto& t : m.estimators()) out.push_back(&t);
                    return out;
                });
            },
            nb::rv_policy::reference_internal);

    nb::class_<StagedScores>(m, "StagedScores")
        .def("__iter__", [](nb::handle self) { return nb::borrow(self); })
        .def("__next__", &StagedScores::next);

    nb::class_<GradientBoostingModel>(m, "GradientBoostingModel")
        .def(nb::new_([](std::string task, std::string loss, int max_depth,
                          int min_samples_split, int min_samples_leaf, double min_impurity_decrease,
                          int n_estimators, double learning_rate, double subsample,
                          MaxFeaturesSpec max_features, double validation_fraction, int n_iter_no_change,
                          double tol, double reg_alpha, double reg_lambda, double huber_delta,
                          double quantile_alpha, std::uint64_t random_state, int n_jobs, bool presort,
                          std::string split_criterion, double min_child_weight, double max_delta_step,
                          std::string tree_method, int max_bin, double colsample_bylevel,
                          std::string eval_metric) {
                 GBParams p;
                 p.eval_metric = eval_metric_from_string(eval_metric);
                 check_colsample_bylevel(colsample_bylevel);
                 p.tree_params.colsample_bylevel = colsample_bylevel;
                 p.tree_params.tree_method = tree_method_from_string(tree_method);
                 check_max_bin(max_bin);
                 p.tree_params.max_bin = max_bin;
                 p.split_criterion = split_criterion_from_string(split_criterion);
                 p.min_child_weight = min_child_weight;
                 p.max_delta_step = max_delta_step;
                 p.tree_params.task = (task == "classification") ? Task::Classification : Task::Regression;
                 p.tree_params.max_depth = max_depth;
                 p.tree_params.min_samples_split = min_samples_split;
                 p.tree_params.min_samples_leaf = min_samples_leaf;
                 p.tree_params.min_impurity_decrease = min_impurity_decrease;
                 p.n_estimators = n_estimators;
                 p.learning_rate = learning_rate;
                 p.subsample = subsample;
                 p.max_features = max_features;
                 p.loss = loss_from_string(loss);
                 p.validation_fraction = validation_fraction;
                 p.n_iter_no_change = n_iter_no_change;
                 p.tol = tol;
                 p.reg_alpha = reg_alpha;
                 p.reg_lambda = reg_lambda;
                 p.huber_delta = huber_delta;
                 p.quantile_alpha = quantile_alpha;
                 p.random_state = random_state;
                 p.n_jobs = n_jobs;
                 p.tree_params.presort = presort;
                 return new GradientBoostingModel(p);
             }),
             "task"_a, "loss"_a, "max_depth"_a, "min_samples_split"_a, "min_samples_leaf"_a,
             "min_impurity_decrease"_a, "n_estimators"_a, "learning_rate"_a, "subsample"_a,
             "max_features"_a, "validation_fraction"_a, "n_iter_no_change"_a, "tol"_a,
             "reg_alpha"_a, "reg_lambda"_a, "huber_delta"_a, "quantile_alpha"_a, "random_state"_a, "n_jobs"_a,
             "presort"_a = true, "split_criterion"_a = "friedman_mse", "min_child_weight"_a = 1.0,
             "max_delta_step"_a = 0.0, "tree_method"_a = "exact", "max_bin"_a = 256, "colsample_bylevel"_a = 1.0,
             "eval_metric"_a = "")
        // X_val / y_val / sample_weight_val: an eval set scored after every
        // round (evals_result_; early stopping on it when n_iter_no_change > 0).
        .def("fit",
             [](GradientBoostingModel& self, InputMatrix X, InputVector y, std::optional<InputVector> sample_weight,
                std::optional<InputMatrix> X_val, std::optional<InputVector> y_val,
                std::optional<InputVector> sample_weight_val) {
                 gb_fit_released(self, X, y, sample_weight, X_val, y_val, sample_weight_val);
             },
             "X"_a, "y"_a, "sample_weight"_a = nb::none(), "X_val"_a = nb::none(), "y_val"_a = nb::none(),
             "sample_weight_val"_a = nb::none())
        .def("fit_hist",
             [](GradientBoostingModel& self, AnyMatrix X, InputVector y, std::optional<InputVector> sample_weight,
                std::optional<AnyMatrix> X_val, std::optional<InputVector> y_val,
                std::optional<InputVector> sample_weight_val) {
                 gb_fit_hist_released(self, X, y, sample_weight, X_val, y_val, sample_weight_val);
             },
             "X"_a, "y"_a, "sample_weight"_a = nb::none(), "X_val"_a = nb::none(), "y_val"_a = nb::none(),
             "sample_weight_val"_a = nb::none())
        .def_prop_ro("evals_result_",
                      [](const GradientBoostingModel& self) {
                          return read(self, [](auto& m) { return m.evals_result(); });
                      })
        .def_prop_ro("best_iteration_",
                      [](const GradientBoostingModel& self) {
                          return read(self, [](auto& m) { return m.best_iteration(); });
                      })
        .def_prop_ro("best_score_",
                      [](const GradientBoostingModel& self) {
                          return read(self, [](auto& m) { return m.best_score(); });
                      })
        .def_prop_ro("eval_metric_",
                      [](const GradientBoostingModel& self) {
                          return read(self, [](auto& m) { return eval_metric_name(m.eval_metric()); });
                      })
        .def("predict",
             [](const GradientBoostingModel& self, InputMatrix X) { return predict_released(self, X); }, "X"_a)
        .def("predict_proba",
             [](const GradientBoostingModel& self, InputMatrix X) { return predict_proba_released(self, X); },
             "X"_a)
        .def("decision_function",
             [](const GradientBoostingModel& self, InputMatrix X) {
                 const MatrixView xv = to_matrix_view(X);
                 std::vector<double> out;
                 std::size_t k = 0;
                 {
                     UseGuard guard(&self, /*exclusive=*/false);
                     check_predict_shape(X, self.n_features());
                     k = self.n_outputs();
                     out.resize(xv.n_rows() * k);
                     nb::gil_scoped_release release;
                     self.decision_function_into(xv, out.data());
                 }
                 return to_numpy(std::move(out), xv.n_rows(), k, /*two_d=*/k > 1);
             },
             "X"_a)
        .def("_staged",
             [](const GradientBoostingModel& self, InputMatrix X, const std::string& kind) {
                 return new StagedScores(self, std::move(X), staged_kind_from_string(kind));
             },
             "X"_a, "kind"_a, nb::keep_alive<0, 1>())
        .def_prop_ro("init_scores_",
                      [](const GradientBoostingModel& self) {
                          return read(self, [](auto& m) { return m.init_scores(); });
                      })
        .def_prop_ro("n_outputs_",
                      [](const GradientBoostingModel& self) {
                          return read(self, [](auto& m) { return m.n_outputs(); });
                      })
        .def_prop_ro("level_features_",
                      [](const GradientBoostingModel& self) {
                          return read(self, [](auto& m) { return level_maps_to_list(m.level_features()); });
                      })
        .def_prop_ro("depth_features_",
                      [](const GradientBoostingModel& self) {
                          return read(self, [](auto& m) { return int_maps_to_list(m.depth_features()); });
                      })
        .def_prop_ro("classes_",
                      [](const GradientBoostingModel& self) {
                          return read(self, [](auto& m) { return m.classes(); });
                      })
        .def_prop_ro("n_classes_",
                      [](const GradientBoostingModel& self) {
                          return read(self, [](auto& m) { return m.n_classes(); });
                      })
        .def_prop_ro("feature_importances_",
                      [](const GradientBoostingModel& self) {
                          return read(self, [](auto& m) { return m.feature_importances(); });
                      })
        .def_prop_ro("train_score_",
                      [](const GradientBoostingModel& self) {
                          return read(self, [](auto& m) { return m.train_score(); });
                      })
        .def_prop_ro("n_estimators_fitted_",
                      [](const GradientBoostingModel& self) {
                          return read(self, [](auto& m) { return m.n_estimators_fitted(); });
                      });
}
