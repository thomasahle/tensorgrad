import string
import random
from tensorgrad import Variable


def make_random_tree(nodes: int):
    components = [i for i in range(nodes)]
    def find(x):
        if components[x] != x:
            components[x] = find(components[x])
        return components[x]
    def union(x, y):
        components[find(x)] = find(y)

    edges = []
    adj = [[] for _ in range(nodes)]
    while len(edges) < nodes - 1:
        x, y = random.randint(0, nodes - 1), random.randint(0, nodes - 1)
        if len(adj[x]) < 3 and len(adj[y]) < 3 and find(x) != find(y):
            union(x, y)
            edges.append((x, y))
            adj[x].append(y)
            adj[y].append(x)

    # 3n edges, n-1 used, 2(n-1) used for connections, n+1 leaf nodes, 1 free edge left.
    names = string.ascii_uppercase
    vectors = []
    variables = []
    for i in range(nodes):
        ts = [f"{names[min(i,j)]}|{names[max(i,j)]}" for j in adj[i]]
        while len(ts) < 3 and len(vectors) < nodes + 1:
            vi = len(vectors)
            vectors.append(Variable(f"V{vi}", f"v{vi}"))
            ts.append(f"v{vi}")
        if len(ts) != 3:
            ts.append("free")
        variables.append(Variable(names[i], ts))

    return vectors, variables
