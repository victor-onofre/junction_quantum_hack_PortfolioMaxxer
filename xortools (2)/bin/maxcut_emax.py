#!/usr/bin/env python3

import networkx as nx
import random
import sys
from collections import Counter

def decode_T_join(G, T, weight='weight'):
    """
    G: networkx.Graph with edge attribute 'weight'
    T: iterable of nodes with syndrome=1, |T| even
    """
    # 1) All‐pairs shortest‐path distances between T
    #    also store the actual path for reconstruction
    dist = dict(nx.multi_source_dijkstra_path_length(G, T, weight=weight))
    # but we really need distances pairwise:    
    d = {u: nx.single_source_dijkstra_path_length(G, u, weight=weight) for u in T}

    # 2) Build complete graph on T
    K = nx.Graph()
    for u in T:
        for v in T:
            if u < v:
                K.add_edge(u, v, weight=d[u][v])

    # 3) Minimum‐weight perfect matching (negate to use max_weight_matching)
    #    NetworkX only has max_weight_matching, so we do:
    for u,v,data in K.edges(data=True):
        data['neg_weight'] = -data['weight']
    mate = nx.max_weight_matching(K, maxcardinality=True, weight='neg_weight')

    # 4) Reconstruct the T-join in G by concatenating shortest‐paths
    T_join_edges = []
    for u, v in mate:
        path = nx.shortest_path(G, u, v, weight=weight)
        # accumulate the edges on that path
        T_join_edges += list(zip(path, path[1:]))

    # Return the set of edges to flip
    return T_join_edges

def random_3_regular(n, weight=1, seed=None):
    """
    Build and return a random 3-regular undirected graph on n vertices.

    Each edge will have a 'weight' attribute set to `weight`.

    Parameters
    ----------
    n : int
        Number of vertices (must satisfy 3*n even, i.e. n even).
    weight : numeric, optional
        The value to assign to G[u][v]['weight'] for every edge.
    seed : int or None, optional
        Seed for the random number generator.

    Returns
    -------
    G : networkx.Graph
        A 3-regular graph with node labels 0..n-1 and a 'weight' attribute on each edge.
    """
    # 3-regular graph exists only if 3*n is even => n must be even
    if (3 * n) % 2 != 0:
        raise ValueError(f"Cannot build a 3-regular graph on {n} vertices (3*n must be even).")

    # networkx provides exactly this
    G = nx.random_regular_graph(3, n, seed=seed)

    # Annotate each edge with the given weight
    for u, v in G.edges():
        G[u][v]['weight'] = weight

    return G

def compute_syndrome(G, error_edges):
    """
    Given an undirected graph G and a subset of its edges (error_edges),
    return the list of vertices that have an odd number of those edges incident.

    Parameters
    ----------
    G : networkx.Graph
        The graph whose vertices are parity checks and edges are code bits.
    error_edges : iterable of (u, v) pairs
        The edges (bit flips) in G.  Each should match an edge in G,
        but orientation (u,v) vs (v,u) does not matter.

    Returns
    -------
    syndrome : list
        List of nodes in G for which the number of incident error_edges is odd.
    """
    # Count how many times each vertex appears in the error_edges list
    cnt = Counter()
    for u, v in error_edges:
        cnt[u] += 1
        cnt[v] += 1

    # Collect all vertices with odd count
    syndrome = [node for node, c in cnt.items() if (c % 2) == 1]
    return syndrome

def random_edge_subset(G, ell, seed=None):
    """
    Return a random subset of exactly `ell` edges from graph G.

    Parameters
    ----------
    G : networkx.Graph
        The input graph (can be directed or undirected).
    ell : int
        Number of edges to sample. Must satisfy 0 <= ell <= G.number_of_edges().
    seed : int or None
        Optional random seed for reproducibility.

    Returns
    -------
    subset : list of edge‐tuples
        A list of `ell` distinct edges chosen uniformly at random from G.
        Each edge is represented as the same tuple type that G.edges() yields.

    Raises
    ------
    ValueError
        If ell is out of bounds.
    """
    m = G.number_of_edges()
    if not (0 <= ell <= m):
        raise ValueError(f"ell must be between 0 and {m}, got {ell}")

    edges = list(G.edges())
    if seed is not None:
        random.seed(seed)
    subset = random.sample(edges, ell)
    return subset

def equal_edge_sets(edges1, edges2):
    """
    Return True iff edges1 and edges2 represent the same set of undirected edges.
    
    edges1, edges2 : iterables of 2-tuples (u, v)
      Each tuple is an edge; ordering of the tuples in the list and
      ordering within each tuple does not matter.
    """
    # Normalize each edge to an unordered frozenset of its endpoints
    set1 = { frozenset(e) for e in edges1 }
    set2 = { frozenset(e) for e in edges2 }
    return set1 == set2

def decode_test(n, verbose=True, trials=100):
    if n%2 != 0:
        print("n must be even")
        return
    G = random_3_regular(n)
    emax = 0
    for l in range(1,int(n/2)):
        successes = 0
        for trial in range(0,trials):
            err = random_edge_subset(G,l)
            S=compute_syndrome(G,err)
            dec = decode_T_join(G,S)
            if equal_edge_sets(err,dec):
                successes = successes + 1
        frac = successes/trials
        if frac > 0.9:
            emax = l
        if verbose:
            print("l = " + str(l) + " frac = " + str(frac))
        if frac < 0.8:
            break
    if verbose:
        print("emax = " + str(emax))
    return emax

if(len(sys.argv) != 2):
    print("usage: maxcut_emax n")
    sys.exit(0)

n = int(sys.argv[1])

if n < 4 or n > 100000:
    print("n out of range")
    sys.exit(0)

if n%2 != 0:
    print("n must be even")
    sys.exit(0)

print("k=2, D=3, n = " + str(n) + ", trials = 100")

emax = decode_test(n, False, 100)

print("emax = " + str(emax))
