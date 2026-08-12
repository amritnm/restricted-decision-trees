import numpy as np
import pytest

from rdt import RDT
from rdt.visualize import export_tree_dict, print_tree, visualize_d3_observable


def _fit_classification_model():
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    y = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    model = RDT(task="classification", max_depth=3)
    model.fit(X, y)
    return model, X, y


def test_export_tree_dict_before_fit_raises():
    model = RDT()
    with pytest.raises(ValueError):
        export_tree_dict(model, np.array([0.0, 1.0]))


def test_export_tree_dict_structure():
    model, X, y = _fit_classification_model()
    tree_dict = export_tree_dict(model, y, feature_names=["x"], class_names=["neg", "pos"])

    assert tree_dict["is_leaf"] is False
    assert tree_dict["feature"] == "x"
    assert tree_dict["samples"] == 6
    assert len(tree_dict["children"]) == 2
    for child in tree_dict["children"]:
        assert child["is_leaf"] is True
        assert child["purity"] == 100.0


def test_export_tree_dict_regression():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([10.0, 12.0, 50.0, 52.0])
    model = RDT(task="regression", max_depth=1)
    model.fit(X, y)

    tree_dict = export_tree_dict(model, y)
    assert tree_dict["criterion_name"] == "mse"
    left, right = tree_dict["children"]
    assert left["prediction"] == "11.0000"
    assert right["prediction"] == "51.0000"


def test_print_tree_runs_without_error(capsys):
    model, X, y = _fit_classification_model()
    print_tree(model, feature_names=["x"])
    captured = capsys.readouterr()
    assert "leaf" in captured.out
    assert "x <=" in captured.out


def test_visualize_d3_observable_writes_html(tmp_path):
    model, X, y = _fit_classification_model()
    out_file = tmp_path / "tree.html"

    html = visualize_d3_observable(model, y, filename=str(out_file), class_names=["neg", "pos"])

    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "d3.v7.min.js" in content
    assert "treeData" in content
    assert html == content
