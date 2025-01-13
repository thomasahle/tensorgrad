from typing import TypeVar, Generic

K = TypeVar("K")  # Key type
V = TypeVar("V")  # Value type


class DisjointSets(Generic[K, V]):
    def __init__(self):
        self.parent: dict[K, K] = {}
        self.values: dict[K, V] = {}

    def find(self, x: K) -> K:
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: K, y: K) -> None:
        x_root = self.find(x)
        y_root = self.find(y)

        # Check value compatibility if both roots have values
        x_value = self.values.get(x_root)
        y_value = self.values.get(y_root)

        if x_value is not None and y_value is not None and x_value != y_value:
            raise ValueError(f"Cannot merge nodes with incompatible values: {x_value} and {y_value}")

        self.parent[x_root] = y_root

        # Keep the non-None value if one exists
        if y_value is None and x_value is not None:
            self.values[y_root] = x_value

    def __setitem__(self, key: K, value: V) -> None:
        root = self.find(key)
        if root in self.values and self.values[root] != value:
            raise ValueError(f"Node already had incompatible value: {self.values[root]} and {value}")
        self.values[root] = value

    def __getitem__(self, key: K) -> V:
        return self.values[self.find(key)]

    def get(self, key: K, default: V = None) -> V:
        return self.values.get(self.find(key), default)

    def items(self) -> list[tuple[list[K], V]]:
        groups = {}
        for key in self.parent:
            root = self.find(key)
            if root not in groups:
                groups[root] = []
            groups[root].append(key)
        return [(keys, self.values.get(root)) for root, keys in groups.items()]

    def __repr__(self) -> str:
        """{[k1, k2]: v, [k3]: v}"""
        return "{" + ", ".join(f"{keys}: {value}" for keys, value in self.items()) + "}"
