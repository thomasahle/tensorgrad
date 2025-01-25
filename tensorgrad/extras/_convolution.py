import torch
from torch.nn import functional as F


def naive_tensor(X: int, Z: int, Y: int) -> torch.Tensor:
    assert X == Y + Z - 1, f"Expected X == Y + Z - 1, got {X=} {Y=} {Z=}"
    conv = torch.zeros(X, Z, Y)
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                if x == y + z:
                    conv[x, z, y] = 1
    return conv


def conv_einsum_dispatch(einsum_str: str, A: torch.Tensor | tuple, B: torch.Tensor | tuple) -> torch.Tensor:
    lhs, rhs = einsum_str.split("->")
    A_dims, B_dims = lhs.split(",")

    if isinstance(B, tuple):
        assert len(B_dims) == 3, f"Expected 3 dimensions for B tensor, got {einsum_str}"
    if isinstance(A, tuple):
        assert len(A_dims) == 3, f"Expected 3 dimensions for B tensor, got {einsum_str}"

        # Two convolutions should be easy to handle, but we don't support it yet
        # E.g. ijk,ilm->jklm corresponds to
        # Out[j,k,l,m] = sum_i [i=j+k][i=l+m] = [j+k=l+m] = [j=l+m-k]
        # which is some kind of generalized convolution...
        if isinstance(B, tuple):
            return torch.einsum(einsum_str, naive_tensor(*A), naive_tensor(*B))

        # Swap A and B so that A is the tensor and B is the convolution
        A_dims, B_dims = B_dims, A_dims
        einsum_str = f"{A_dims},{B_dims}->{rhs}"
        A, B = B, A

    # If we have to do a hadamard multiplication, we can't handle that right now
    if set(A_dims) & set(B_dims) & set(rhs):
        return torch.einsum(einsum_str, A, naive_tensor(*B))

    X, Z, Y = B
    assert X == Y + Z - 1, f"Expected X == Y + Z - 1, got {X=} {Y=} {Z=}"

    # Move the batch dimensions to the front
    batch_dims = [i for i, a in enumerate(A_dims) if a not in B_dims]
    contr_dims = [i for i, a in enumerate(A_dims) if a in B_dims]
    bd = "".join(A_dims[i] for i in batch_dims)
    # print(f"{batch_dims=}, {contr_dims=} {einsum_str=}")
    # Flatten to a 2D tensor
    A_flat = A.permute(*(batch_dims + contr_dims)).flatten(len(batch_dims))
    if len(batch_dims) > 0:
        A_flat = A_flat.flatten(0, len(batch_dims) - 1)
    elif len(batch_dims) == 0:
        A_flat = A_flat.unsqueeze(0)
    B = A_flat.shape[0]
    # Figure out what dimensions we need from the convolution
    b_contract = [i for i, b in enumerate(B_dims) if b in A_dims]

    res = None  # type: torch.Tensor
    # bx,xzy->bzy
    if b_contract == [0]:
        res = A_flat.unfold(dimension=1, size=Z, step=1)  # shape [B,Y,Z]
        res = res.permute(0, 2, 1)
        assert res.shape == (B, Z, Y), f"{res.shape=}"

    # by,xzy->bxz
    elif b_contract == [2]:
        # out[x,z] = A[x-z] if valid
        #   => typical "toeplitz" with negative shift
        #   => pad left/right by (Z-1), unfold(size=Z, step=1),
        #      slice first X, flip dimension 1
        assert A_flat.shape == (B, Y), f"{A_flat.shape=}"
        padded = F.pad(A_flat, (Z - 1, Z - 1))  # shape [Y + 2*(Z-1)]
        assert padded.shape == (B, Y + 2 * (Z - 1)), f"{padded.shape=}"
        tmp = padded.unfold(dimension=1, size=Z, step=1)  # => [B,Y+2*(Z-1),Z]
        tmp = tmp[:, :X]  # => [B,X,Z]
        res = tmp.flip(dims=(2,))  # => [B,X,Z]
        assert res.shape == (B, X, Z), f"{res.shape=}"

    # bz,xzy->bxy, using symmetry
    elif b_contract == [1]:
        eq = f"{A_dims},{B_dims[0]}{B_dims[2]}{B_dims[1]}->{rhs}"
        res = conv_einsum_dispatch(eq, A, (X, Y, Z))
        assert res.shape == (B, X, Y)

    # bzy,xzy->bx, and byz,xzy->bx
    elif b_contract == [1, 2]:
        # if bzy
        if A_dims[contr_dims[0]] == B_dims[1]:
            expanded = A_flat.reshape(B, Z, Y)
        else:
            expanded = A_flat.reshape(B, Y, Z).permute(0, 2, 1)
        # Fold assumes (B, C*K, L) => (B, C, H, W)
        res = F.fold(
            expanded,  # shape [B, Z, Y]
            output_size=(1, X),  # treat as (H=1, W=X)
            kernel_size=(1, Z),
            stride=(1, 1),
        ).squeeze(1, 2)  # shape [B, X]
        assert res.shape == (B, X)

    if res is not None:
        # Handle extra summations and permutations
        mid = bd + "".join(b for i, b in enumerate(B_dims) if i not in b_contract)
        res = res.reshape([A.shape[i] for i in batch_dims] + list(res.shape)[1:])
        return torch.einsum(f"{mid}->{rhs}", res)

    # Fallback to the naive implementation
    return torch.einsum(einsum_str, A, naive_tensor(X, Z, Y))
