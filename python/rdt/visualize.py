"""Tree introspection and D3.js visualization for fitted RDT models.

Ported from the project's original pure-Python `RestrictedDecisionTree`
implementation's `export_tree_dict`/`visualize_d3_observable` methods to work
against the native C++-backed `RDT` class: it recomputes per-node statistics
(impurity, target average, leaf purity) from `node.indices` and the training
labels, exactly as the original did, rather than assuming those stats are
cached on the node.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np


def _criterion_value(criterion: str, y_node: np.ndarray) -> float:
    if len(y_node) == 0:
        return 0.0
    if criterion == "gini":
        _, counts = np.unique(y_node, return_counts=True)
        probs = counts / len(y_node)
        return float(1.0 - np.sum(probs ** 2))
    if criterion == "entropy":
        _, counts = np.unique(y_node, return_counts=True)
        probs = counts / len(y_node)
        return float(-np.sum(probs * np.log2(probs + 1e-10)))
    if criterion == "mse":
        return float(np.var(y_node))
    if criterion == "mae":
        median = np.median(y_node)
        return float(np.mean(np.abs(y_node - median)))
    raise ValueError(f"Unknown criterion: {criterion}")


def _node_to_dict(model, node, y, depth, feature_names, class_names) -> Dict[str, Any]:
    indices = node.indices
    node_targets = y[indices]
    target_avg = float(np.mean(node_targets)) if len(node_targets) else 0.0
    criterion_value = _criterion_value(model.criterion, node_targets)

    if node.is_leaf:
        if model.task == "classification":
            classes = list(model.classes_) if model.classes_ is not None else []
            if class_names is not None and node.value in classes:
                pred_str = class_names[classes.index(node.value)]
            else:
                pred_str = f"Class {node.value:g}"
            counts = Counter(node_targets.tolist())
            purity = (counts.most_common(1)[0][1] / len(node_targets) * 100) if len(node_targets) else 0.0
        else:
            pred_str = f"{node.value:.4f}"
            purity = 100.0

        return {
            "name": f"leaf: {pred_str}",
            "samples": len(indices),
            "criterion_value": criterion_value,
            "criterion_name": model.criterion,
            "target_avg": target_avg,
            "depth": depth,
            "is_leaf": True,
            "prediction": pred_str,
            "purity": purity,
        }

    feature_idx = node.split_feature
    feature_name = feature_names[feature_idx] if feature_names else f"Feature {feature_idx}"
    node_dict: Dict[str, Any] = {
        "name": f"{feature_name} <= {node.split_threshold:.4f}",
        "samples": len(indices),
        "criterion_value": criterion_value,
        "criterion_name": model.criterion,
        "target_avg": target_avg,
        "depth": depth,
        "is_leaf": False,
        "feature": feature_name,
        "feature_idx": feature_idx,
        "threshold": float(node.split_threshold),
        "children": [],
    }
    if node.left is not None:
        node_dict["children"].append(_node_to_dict(model, node.left, y, depth + 1, feature_names, class_names))
    if node.right is not None:
        node_dict["children"].append(_node_to_dict(model, node.right, y, depth + 1, feature_names, class_names))
    return node_dict


def export_tree_dict(
    model,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Export a fitted RDT's tree structure as a nested dict with per-node statistics."""
    if model.root is None:
        raise ValueError("Tree has not been fitted yet. Call fit() first.")

    y = np.asarray(y)
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(model.n_features_)]

    return _node_to_dict(model, model.root, y, 0, feature_names, class_names)


def print_tree(model, feature_names: Optional[List[str]] = None) -> None:
    """Print a simple indented text representation of a fitted RDT's tree."""
    if model.root is None:
        raise ValueError("Tree has not been fitted yet. Call fit() first.")
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(model.n_features_)]

    def _print(node, depth):
        indent = "  " * depth
        if node.is_leaf:
            print(f"{indent}leaf: value={node.value:g}, samples={len(node.indices)}")
        else:
            fname = feature_names[node.split_feature]
            print(f"{indent}[{fname} <= {node.split_threshold:.4f}] samples={len(node.indices)}")
            _print(node.left, depth + 1)
            _print(node.right, depth + 1)

    _print(model.root, 0)


def visualize_d3_observable(
    model,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    filename: str = "rdt_tree_d3.html",
    width: int = 1200,
    height: int = 800,
) -> str:
    """Render a fitted RDT's tree as an interactive, collapsible D3.js visualization.

    Writes `filename` (open directly in a browser: click nodes to
    expand/collapse, scroll/drag to zoom and pan) and also returns the HTML
    as a string.
    """
    tree_data = export_tree_dict(model, y, feature_names, class_names)

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Restricted Decision Tree - D3 Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f5f5f5;
        }}

        #tree-container {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            padding: 20px;
            overflow: auto;
        }}

        .node {{
            cursor: pointer;
        }}

        .node rect {{
            stroke: #999;
            stroke-width: 2px;
            rx: 8;
        }}

        .node-internal rect {{
            fill: #95C0E8;
            stroke: #7AA8D6;
        }}

        .node-leaf rect {{
            fill: #C2B59B;
            stroke: #ADA088;
        }}

        .node text {{
            font-size: 12px;
            fill: #333;
            font-family: 'Courier New', monospace;
        }}

        .node-title {{
            font-weight: bold;
            font-size: 13px;
        }}

        .link {{
            fill: none;
            stroke: #999;
            stroke-width: 2px;
        }}

        .link-label {{
            font-size: 10px;
            fill: #666;
        }}

        #info-panel {{
            position: fixed;
            top: 20px;
            right: 20px;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            max-width: 250px;
        }}

        #info-panel h3 {{
            margin-top: 0;
            font-size: 16px;
            color: #333;
        }}

        #info-panel p {{
            margin: 5px 0;
            font-size: 13px;
            color: #666;
        }}

        .collapsed {{
            opacity: 0.6;
        }}
    </style>
</head>
<body>
    <div id="info-panel">
        <h3>Restricted Decision Tree</h3>
        <p><strong>Task:</strong> {model.task}</p>
        <p><strong>Criterion:</strong> {model.criterion}</p>
        <p><strong>Max Depth:</strong> {model.max_depth}</p>
        <p><br><em>Click nodes to expand/collapse</em></p>
        <p style="margin-top: 15px; font-size: 11px; color: #888;">
            Note: Same feature at each depth,<br>
            but different thresholds per node
        </p>
    </div>

    <div id="tree-container"></div>

    <script>
        const treeData = {json.dumps(tree_data, indent=2)};

        const width = {width};
        const height = {height};
        const nodeWidth = 220;
        const nodeHeight = 140;

        const svg = d3.select("#tree-container")
            .append("svg")
            .attr("width", width)
            .attr("height", height)
            .call(d3.zoom().on("zoom", (event) => {{
                g.attr("transform", event.transform);
            }}))
            .append("g")
            .attr("transform", "translate(100,50)");

        const g = svg;

        const tree = d3.tree()
            .nodeSize([nodeWidth + 100, nodeHeight + 60]);

        const root = d3.hierarchy(treeData, d => d.children);
        root.x0 = 0;
        root.y0 = 0;

        root.descendants().forEach(d => {{
            if (d.depth > 1) {{
                if (d.children) {{
                    d._children = d.children;
                    d.children = null;
                }}
            }}
        }});

        update(root);

        function update(source) {{
            const treeData = tree(root);
            const nodes = treeData.descendants();
            const links = treeData.links();

            const node = g.selectAll('.node')
                .data(nodes, d => d.id || (d.id = Math.random()));

            const nodeEnter = node.enter().append('g')
                .attr('class', d => `node ${{d.data.is_leaf ? 'node-leaf' : 'node-internal'}}`)
                .attr('transform', d => `translate(${{source.x0}},${{source.y0}})`)
                .on('click', click);

            nodeEnter.append('rect')
                .attr('width', nodeWidth)
                .attr('height', nodeHeight)
                .attr('x', -nodeWidth / 2)
                .attr('y', -nodeHeight / 2);

            nodeEnter.each(function(d) {{
                const g = d3.select(this);
                const data = d.data;

                g.append('text')
                    .attr('class', 'node-title')
                    .attr('dy', -nodeHeight/2 + 20)
                    .attr('text-anchor', 'middle')
                    .text(data.name.length > 30 ? data.name.substring(0, 27) + '...' : data.name);

                g.append('text')
                    .attr('dy', -nodeHeight/2 + 45)
                    .attr('text-anchor', 'middle')
                    .text(`Samples: ${{data.samples}}`);

                g.append('text')
                    .attr('dy', -nodeHeight/2 + 65)
                    .attr('text-anchor', 'middle')
                    .text(`${{data.criterion_name}}: ${{data.criterion_value.toFixed(4)}}`);

                g.append('text')
                    .attr('dy', -nodeHeight/2 + 85)
                    .attr('text-anchor', 'middle')
                    .text(`Target Avg: ${{data.target_avg.toFixed(4)}}`);

                g.append('text')
                    .attr('dy', -nodeHeight/2 + 105)
                    .attr('text-anchor', 'middle')
                    .attr('font-size', '10px')
                    .text(`Depth: ${{data.depth}}`);

                if (data.is_leaf) {{
                    g.append('text')
                        .attr('dy', -nodeHeight/2 + 125)
                        .attr('text-anchor', 'middle')
                        .attr('font-size', '10px')
                        .attr('fill', '#2ecc71')
                        .text(`Purity: ${{data.purity.toFixed(1)}}%`);
                }}
            }});

            const nodeUpdate = nodeEnter.merge(node)
                .transition()
                .duration(750)
                .attr('transform', d => `translate(${{d.x}},${{d.y}})`);

            const nodeExit = node.exit()
                .transition()
                .duration(750)
                .attr('transform', d => `translate(${{source.x}},${{source.y}})`)
                .remove();

            const link = g.selectAll('.link')
                .data(links, d => d.target.id);

            const linkEnter = link.enter().insert('path', 'g')
                .attr('class', 'link')
                .attr('d', d => {{
                    const o = {{x: source.x0, y: source.y0}};
                    return diagonal(o, o);
                }});

            const linkUpdate = linkEnter.merge(link)
                .transition()
                .duration(750)
                .attr('d', d => diagonal(d.source, d.target));

            const linkExit = link.exit()
                .transition()
                .duration(750)
                .attr('d', d => {{
                    const o = {{x: source.x, y: source.y}};
                    return diagonal(o, o);
                }})
                .remove();

            nodes.forEach(d => {{
                d.x0 = d.x;
                d.y0 = d.y;
            }});
        }}

        function diagonal(s, d) {{
            return `M ${{s.x}} ${{s.y}}
                    C ${{s.x}} ${{(s.y + d.y) / 2}},
                      ${{d.x}} ${{(s.y + d.y) / 2}},
                      ${{d.x}} ${{d.y}}`;
        }}

        function click(event, d) {{
            if (d.children) {{
                d._children = d.children;
                d.children = null;
            }} else {{
                d.children = d._children;
                d._children = null;
            }}
            update(d);
        }}
    </script>
</body>
</html>
"""

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"D3 visualization saved to: {filename}")
    print("Open the file in a browser to view the interactive tree.")
    return html
