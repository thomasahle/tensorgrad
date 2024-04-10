from collections import defaultdict
from tensorgrad.tensor import Product, Zero, Copy, Variable, Sum, Function

# TODO: It could be cool if this function outputted a full pytorch module with a forward and backward method


def _to_pytorch_code(tensor, code_lines):
    if isinstance(tensor, Copy):
        code_lines.append(f"identity_{id(tensor)} = torch.eye(d).rename(*{tensor.edges})")
        return f"identity_{id(tensor)}"

    if isinstance(tensor, Variable):
        dims = [f"d_{e}" for e in tensor.original_edges]
        code_lines.append(f"# {tensor.name}.names = {tensor.edges}")
        return tensor.name

    if isinstance(tensor, Zero):
        dims = [f"d_{e}" for e in tensor.edges]
        code_lines.append(f"zero_{id(tensor)} = torch.zeros({', '.join(dims)}).rename(*{tensor.edges})")
        return f"zero_{id(tensor)}"

    if isinstance(tensor, Function):
        sub_ids = []
        for t in tensor.tensors:
            sub_id = _to_pytorch_code(t, code_lines)
            sub_ids.append(sub_id)

        einsum_str = ",".join(sub_ids) + "->" + ",".join(tensor.edges)
        code_lines.append(f"function_{id(tensor)} = torch.einsum('{einsum_str}', {', '.join(sub_ids)})")
        return f"function_{id(tensor)}"

    if isinstance(tensor, Product):
        sub_ids = []
        ein_edges = []
        for t in tensor.tensors:
            sub_id = _to_pytorch_code(t, code_lines)
            sub_ids.append(sub_id)
            ein_edges.append(" ".join(t.edges))

        contraction_indices = defaultdict(list)
        for e in tensor.contractions:
            t1, t2 = [t for t in tensor.tensors if e in t.edges]
            contraction_indices[id(t1)].append(e)
            contraction_indices[id(t2)].append(e)

        einsum_str = ",".join(ein_edges) + " -> " + " ".join(tensor.edges)
        einsum_str = einsum_str.replace("'", "_")

        code_lines.append(f"product_{id(tensor)} = torch.einsum('{einsum_str}', {', '.join(sub_ids)})")

        # einsum from einops is like ... einsum, generic and flexible dot-product
        # but 1) axes can be multi-lettered  2) pattern goes last 3) works with multiple frameworks
        # C = einsum(A, B, 'b t1 head c, b t2 head c -> b head t1 t2')
        return f"product_{id(tensor)}"

    if isinstance(tensor, Sum):
        sum_terms = []
        for w, t in zip(tensor.weights, tensor.tensors):
            sub_id = _to_pytorch_code(t, code_lines)
            sum_terms.append(f"{w} * {sub_id}")

        code_lines.append(f"sum_{id(tensor)} = {' + '.join(sum_terms)}")
        return f"sum_{id(tensor)}"


def to_pytorch(tensor):
    code_lines = [
        "import torch",
        "from einops import einsum",
    ]
    node_id = _to_pytorch_code(tensor, code_lines)
    code_lines.append(f"return {node_id}")
    return "\n".join(code_lines)
