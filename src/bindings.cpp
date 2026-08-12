#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

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
                          int min_samples_split, int min_samples_leaf) {
                 TreeParams p;
                 p.task = (task == "classification") ? Task::Classification : Task::Regression;
                 p.criterion = criterion_from_string(criterion);
                 p.max_depth = max_depth;
                 p.min_samples_split = min_samples_split;
                 p.min_samples_leaf = min_samples_leaf;
                 return new RDTModel(p);
             }),
             "task"_a, "criterion"_a, "max_depth"_a, "min_samples_split"_a, "min_samples_leaf"_a)
        .def("fit",
             [](RDTModel& self, InputMatrix X, InputVector y) {
                 self.fit(to_matrix_view(X), to_vector_view(y));
             },
             "X"_a, "y"_a)
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
        .def_prop_ro("root", [](const RDTModel& self) -> const Node* { return self.root(); },
                      nb::rv_policy::reference_internal);
}
