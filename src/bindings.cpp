#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <optional>
#include <stdexcept>

#include "rdt/forest.hpp"
#include "rdt/gradient_boosting.hpp"
#include "rdt/tree.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace rdt;

namespace {

using InputMatrix = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using InputVector = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

MatrixView to_matrix_view(const InputMatrix& X) {
    return MatrixView(X.data(), X.shape(0), X.shape(1), /*row_major=*/true);
}

VectorView to_vector_view(const InputVector& y) {
    return VectorView(y.data(), y.shape(0));
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

} // namespace

NB_MODULE(_rdt_native, m) {
    nb::class_<Node>(m, "Node")
        .def_prop_ro("is_leaf", &Node::is_leaf)
        .def_prop_ro("indices", [](const Node& n) { return n.indices; })
        .def_ro("split_feature", &Node::split_feature)
        .def_ro("split_threshold", &Node::split_threshold)
        .def_ro("value", &Node::value)
        .def_ro("class_counts", &Node::class_counts)
        .def_prop_ro("left", [](const Node& n) -> const Node* { return n.left.get(); },
                      nb::rv_policy::reference_internal)
        .def_prop_ro("right", [](const Node& n) -> const Node* { return n.right.get(); },
                      nb::rv_policy::reference_internal);

    nb::class_<RDTModel>(m, "RDTModel")
        .def(nb::new_([](std::string task, std::string criterion, int max_depth,
                          int min_samples_split, int min_samples_leaf, double min_impurity_decrease) {
                 TreeParams p;
                 p.task = (task == "classification") ? Task::Classification : Task::Regression;
                 p.criterion = criterion_from_string(criterion);
                 p.max_depth = max_depth;
                 p.min_samples_split = min_samples_split;
                 p.min_samples_leaf = min_samples_leaf;
                 p.min_impurity_decrease = min_impurity_decrease;
                 return new RDTModel(p);
             }),
             "task"_a, "criterion"_a, "max_depth"_a, "min_samples_split"_a, "min_samples_leaf"_a,
             "min_impurity_decrease"_a = 0.0)
        .def("fit",
             [](RDTModel& self, InputMatrix X, InputVector y, std::optional<InputVector> sample_weight) {
                 if (sample_weight) {
                     VectorView wv = to_vector_view(*sample_weight);
                     self.fit(to_matrix_view(X), to_vector_view(y), &wv);
                 } else {
                     self.fit(to_matrix_view(X), to_vector_view(y));
                 }
             },
             "X"_a, "y"_a, "sample_weight"_a = nb::none())
        .def("predict",
             [](const RDTModel& self, InputMatrix X) { return self.predict(to_matrix_view(X)); },
             "X"_a)
        .def("predict_proba",
             [](const RDTModel& self, InputMatrix X) { return self.predict_proba(to_matrix_view(X)); },
             "X"_a)
        .def_prop_ro("classes_", [](const RDTModel& self) { return self.classes(); })
        .def_prop_ro("n_classes_", [](const RDTModel& self) { return self.n_classes(); })
        .def_prop_ro("depth_features_",
                      [](const RDTModel& self) {
                          nb::dict d;
                          for (const auto& kv : self.depth_features()) {
                              d[nb::int_(kv.first)] = kv.second;
                          }
                          return d;
                      })
        .def_prop_ro("depth_improvements_",
                      [](const RDTModel& self) {
                          nb::dict d;
                          for (const auto& kv : self.depth_improvements()) {
                              d[nb::int_(kv.first)] = kv.second;
                          }
                          return d;
                      })
        .def_prop_ro("root", [](const RDTModel& self) -> const Node* { return self.root(); },
                      nb::rv_policy::reference_internal);

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
                          double max_samples, bool oob_score, std::uint64_t random_state, int n_jobs) {
                 ForestParams p;
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
                 p.oob_score = oob_score;
                 p.random_state = random_state;
                 p.n_jobs = n_jobs;
                 return new RandomForestModel(p);
             }),
             "task"_a, "criterion"_a, "max_depth"_a, "min_samples_split"_a, "min_samples_leaf"_a,
             "min_impurity_decrease"_a, "n_estimators"_a, "max_features"_a, "bootstrap"_a,
             "max_samples"_a, "oob_score"_a, "random_state"_a, "n_jobs"_a)
        .def("fit",
             [](RandomForestModel& self, InputMatrix X, InputVector y, std::optional<InputVector> sample_weight) {
                 if (sample_weight) {
                     VectorView wv = to_vector_view(*sample_weight);
                     self.fit(to_matrix_view(X), to_vector_view(y), &wv);
                 } else {
                     self.fit(to_matrix_view(X), to_vector_view(y));
                 }
             },
             "X"_a, "y"_a, "sample_weight"_a = nb::none())
        .def("predict",
             [](const RandomForestModel& self, InputMatrix X) { return self.predict(to_matrix_view(X)); }, "X"_a)
        .def("predict_proba",
             [](const RandomForestModel& self, InputMatrix X) { return self.predict_proba(to_matrix_view(X)); }, "X"_a)
        .def_prop_ro("classes_", [](const RandomForestModel& self) { return self.classes(); })
        .def_prop_ro("n_classes_", [](const RandomForestModel& self) { return self.n_classes(); })
        .def_prop_ro("feature_importances_", [](const RandomForestModel& self) { return self.feature_importances(); })
        .def_prop_ro("oob_score_", [](const RandomForestModel& self) { return self.oob_score(); })
        .def_prop_ro("has_oob_score_", [](const RandomForestModel& self) { return self.has_oob_score(); })
        .def_prop_ro("feature_subsets_", [](const RandomForestModel& self) { return self.feature_subsets(); })
        .def_prop_ro(
            "estimators_",
            [](const RandomForestModel& self) {
                std::vector<const RDTModel*> out;
                out.reserve(self.estimators().size());
                for (const auto& t : self.estimators()) out.push_back(&t);
                return out;
            },
            nb::rv_policy::reference_internal);

    nb::class_<GradientBoostingModel>(m, "GradientBoostingModel")
        .def(nb::new_([](std::string task, std::string loss, int max_depth,
                          int min_samples_split, int min_samples_leaf, double min_impurity_decrease,
                          int n_estimators, double learning_rate, double subsample,
                          MaxFeaturesSpec max_features, double validation_fraction, int n_iter_no_change,
                          double tol, double reg_alpha, double reg_lambda, double huber_delta,
                          double quantile_alpha, std::uint64_t random_state, int n_jobs) {
                 GBParams p;
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
                 return new GradientBoostingModel(p);
             }),
             "task"_a, "loss"_a, "max_depth"_a, "min_samples_split"_a, "min_samples_leaf"_a,
             "min_impurity_decrease"_a, "n_estimators"_a, "learning_rate"_a, "subsample"_a,
             "max_features"_a, "validation_fraction"_a, "n_iter_no_change"_a, "tol"_a,
             "reg_alpha"_a, "reg_lambda"_a, "huber_delta"_a, "quantile_alpha"_a, "random_state"_a, "n_jobs"_a)
        .def("fit",
             [](GradientBoostingModel& self, InputMatrix X, InputVector y, std::optional<InputVector> sample_weight) {
                 if (sample_weight) {
                     VectorView wv = to_vector_view(*sample_weight);
                     self.fit(to_matrix_view(X), to_vector_view(y), &wv);
                 } else {
                     self.fit(to_matrix_view(X), to_vector_view(y));
                 }
             },
             "X"_a, "y"_a, "sample_weight"_a = nb::none())
        .def("predict",
             [](const GradientBoostingModel& self, InputMatrix X) { return self.predict(to_matrix_view(X)); }, "X"_a)
        .def("predict_proba",
             [](const GradientBoostingModel& self, InputMatrix X) { return self.predict_proba(to_matrix_view(X)); }, "X"_a)
        .def_prop_ro("classes_", [](const GradientBoostingModel& self) { return self.classes(); })
        .def_prop_ro("n_classes_", [](const GradientBoostingModel& self) { return self.n_classes(); })
        .def_prop_ro("feature_importances_", [](const GradientBoostingModel& self) { return self.feature_importances(); })
        .def_prop_ro("train_score_", [](const GradientBoostingModel& self) { return self.train_score(); })
        .def_prop_ro("n_estimators_fitted_", [](const GradientBoostingModel& self) { return self.n_estimators_fitted(); });
}
