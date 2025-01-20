from typing import Any, TypeVar, Generic


class _MatchEdgesKey:
    """Normally Tensors use isomorphism as their test for equality, but they don't include
    the edge names in the comparison. This class is used to compare tensors based on their
    edge names. It is used in the Sum tensor to combine tensors with the same edge names."""

    def __init__(self, value: Any, **edge_names: str):
        self.value = value
        self.hash = hash(value)
        self.edge_names = edge_names

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, _MatchEdgesKey):
            return self.value.is_isomorphic(other.value, edge_names=self.edge_names, match_edges=True)
        return False

    def __hash__(self) -> int:
        return self.hash


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


class KeyStoringDict:
    """
    A dictionary-like class that:
      - Internally stores { key: (key, value) } in self._store.
      - Normal lookups return only 'value', but we can also retrieve
        the actual stored key object via get_with_key().
    """

    def __init__(self, *args, **kwargs):
        """
        Similar to dict's constructor, it can accept:
          - KeyStoringDict(mapping)
          - KeyStoringDict(iterable_of_pairs)
          - KeyStoringDict(key1=value1, key2=value2, ...)
        in any combination.
        """
        self._store = {}
        self.update(*args, **kwargs)

    def __setitem__(self, key, value):
        # Internally store (key, value).
        self._store[key] = (key, value)

    def __getitem__(self, key):
        # Return only the user_value portion
        _, user_value = self._store[key]
        return user_value

    def __delitem__(self, key):
        del self._store[key]

    def __contains__(self, key):
        return key in self._store

    def __len__(self):
        return len(self._store)

    def __iter__(self):
        """
        Iterating over this dict should iterate over its keys.
        """
        return iter(self._store)

    def __repr__(self):
        """
        Show a nice representation: KeyStoringDict({k: v, ...})
        """
        class_name = type(self).__name__
        items_str = ", ".join(f"{k!r}: {v!r}" for k, v in self.items())
        return f"{class_name}({{{items_str}}})"

    def get_with_key(self, key, default=None):
        """
        Return (stored_key, user_value) if key is found,
        else return 'default'.
        """
        return self._store.get(key, default)

    def get(self, key, default=None):
        """
        Normal dict .get(), returning just the user_value (or default).
        """
        _, value = self._store.get(key, (None, default))
        return value

    def pop(self, key):
        """
        pop(key) → user_value, removing the item if it exists.
        If key not found, raise KeyError.
        """
        _, user_value = self._store.pop(key)
        return user_value

    def update(self, other: dict):
        """
        update(...) adds/overwrites items from another dict/mapping.
        """
        for k, v in other.items():
            self[k] = v

    def keys(self):
        """
        Return a view (or iterable) of keys.
        """
        return self._store.keys()

    def values(self):
        """
        Return a view (or iterable) of user_values.
        """
        for _, user_value in self._store.values():
            yield user_value

    def items(self):
        """
        Return a view (or iterable) of (key, user_value).
        """
        return self._store.values()

    def copy(self):
        """
        Return a shallow copy of this KeyStoringDict.
        """
        return KeyStoringDict(self)
