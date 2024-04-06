import torch


def rand_values(variables, **shape):
    return {v: torch.randn([shape[e] for e in v.edges], names=v.edges) for v in variables}


def assert_close(a, b):
    assert set(a.names) == set(b.names)
    a = a.align_to(*b.names)
    torch.testing.assert_close(a.rename(None), b.rename(None))
