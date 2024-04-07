from tensorgrad.tensor import Product, Zero, Copy, Variable, Sum
import json


def tensor_to_dict(tensor):
    """
    Recursively converts tensor objects into a nested dictionary format.
    """
    # Base case for Variable, Zero, and Identity types
    if isinstance(tensor, (Variable, Zero, Copy)):
        return {
            "name": str(tensor),
            "type": tensor.__class__.__name__,
            "children": [],  # No children for these types
        }

    # Recursive case for Contraction and LinearCombination types
    elif isinstance(tensor, Product):
        return {
            "name": "Contraction",
            "type": "Contraction",
            "children": [tensor_to_dict(t) for t in tensor.tensors],
        }

    elif isinstance(tensor, Sum):
        children = []
        for weight, t in zip(tensor.weights, tensor.tensors):
            child = tensor_to_dict(t)
            child["weight"] = weight  # Add weight information
            children.append(child)

        return {
            "name": "LinearCombination",
            "type": "LinearCombination",
            "children": children,
        }

    # Add additional cases here as necessary for other tensor types.
    else:
        raise ValueError(f"Unsupported tensor type: {type(tensor)}")


def to_d3(tensor):
    # Convert tensor to hierarchical data structure for D3.js
    hierarchical_data = tensor_to_dict(tensor)  # This needs to be implemented

    # HTML template with embedded JavaScript for D3.js visualization
    html_template = (
        """
<!DOCTYPE html>
<meta charset="utf-8">
<body>
<svg width="960" height="600"></svg>
<script src="https://d3js.org/d3.v6.min.js"></script>
<script>

var data = """
        + json.dumps(hierarchical_data)
        + """;

var treeLayout = d3.tree().size([560, 940]);

var rootNode = d3.hierarchy(data);

treeLayout(rootNode);

var svg = d3.select("svg")
    .style("width", "100%")
    .style("height", "100%")
    .append("g")
    .attr("transform", "translate(50,50)");

var links = svg.selectAll(".link")
    .data(rootNode.links())
    .enter()
    .append("path")
    .attr("class", "link")
    .attr("d", d3.linkHorizontal()
        .x(function(d) { return d.y; })
        .y(function(d) { return d.x; }));

var nodes = svg.selectAll(".node")
    .data(rootNode.descendants())
    .enter()
    .append("g")
    .attr("class", "node")
    .attr("transform", function(d) { return "translate(" + d.y + "," + d.x + ")"; });

nodes.append("circle")
    .attr("r", 5);

nodes.append("text")
    .attr("dy", 3)
    .attr("x", function(d) { return d.children ? -8 : 8; })
    .style("text-anchor", function(d) { return d.children ? "end" : "start"; })
    .text(function(d) { return d.data.name; });

</script>
"""
    )
    return html_template
