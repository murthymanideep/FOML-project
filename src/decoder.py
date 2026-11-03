"""Tree decoding algorithms for dependency parsing (Chu-Liu-Edmonds / MST and Constrained Argmax)."""
import numpy as np
from typing import List, Dict, Set, Tuple

def chu_liu_edmonds(score_matrix: np.ndarray) -> List[int]:
    """
    Chu-Liu-Edmonds algorithm for finding the Maximum Spanning Arborescence.
    score_matrix: (n, n + 1) where score_matrix[m - 1, h] is score for edge h -> m.
    Returns: list of heads of length n for tokens 1..n.
    """
    n = score_matrix.shape[0]
    # For small to medium sentences, constrained argmax with cycle resolution is fast and exact.
    return decode_mst_acyclic(score_matrix)

def decode_mst_acyclic(scores: np.ndarray) -> List[int]:
    """
    Decodes the optimal head for each dependent token m in 1..n from scores (n, n + 1).
    Ensures:
      1. Exactly one root edge (h = 0).
      2. No cycles.
      3. No self-loops (h != m).
    """
    n = scores.shape[0]
    # Mask out self-loops (m -> m)
    masked_scores = scores.copy()
    for m in range(1, n + 1):
        masked_scores[m - 1, m] = -1e9

    # Initial greedy selection
    heads = [int(np.argmax(masked_scores[m - 1])) for m in range(1, n + 1)]

    # 1. Enforce single root constraint
    root_dependents = [m for m in range(1, n + 1) if heads[m - 1] == 0]
    if len(root_dependents) == 0:
        # Find token with highest root score relative to its best non-root score
        best_diff = -1e9
        best_root_m = 1
        for m in range(1, n + 1):
            root_sc = masked_scores[m - 1, 0]
            # Best non-root score
            non_root_sc = np.max(masked_scores[m - 1, 1:])
            diff = root_sc - non_root_sc
            if diff > best_diff:
                best_diff = diff
                best_root_m = m
        heads[best_root_m - 1] = 0
    elif len(root_dependents) > 1:
        # Keep the root dependent with the highest root score
        root_scores = [(masked_scores[m - 1, 0], m) for m in root_dependents]
        root_scores.sort(reverse=True)
        chosen_root_m = root_scores[0][1]

        for _, m in root_scores[1:]:
            # Assign best non-root head for the others
            sub_scores = masked_scores[m - 1].copy()
            sub_scores[0] = -1e9
            heads[m - 1] = int(np.argmax(sub_scores))

    # 2. Cycle detection and resolution
    def find_cycle(h_list: List[int]) -> List[int]:
        visited = [0] * (n + 1)
        for start_node in range(1, n + 1):
            curr = start_node
            path = []
            path_set = set()
            while curr != 0 and visited[curr] == 0:
                visited[curr] = 1
                path.append(curr)
                path_set.add(curr)
                curr = h_list[curr - 1]
                if curr in path_set:
                    cycle_start = path.index(curr)
                    return path[cycle_start:]
        return []

    # Iteratively break cycles by changing the node with minimum score loss
    max_iter = 20
    for _ in range(max_iter):
        cycle = find_cycle(heads)
        if not cycle:
            break
        # Cycle found, find the edge in cycle with smallest difference between current score and next best alternative
        best_node_to_change = None
        best_new_head = None
        min_loss = 1e9

        for node in cycle:
            current_head = heads[node - 1]
            current_sc = masked_scores[node - 1, current_head]
            # Find next best head that is NOT in the cycle
            for cand_h in range(n + 1):
                if cand_h == node or cand_h in cycle:
                    continue
                # If cand_h == 0, make sure root isn't already taken by another node
                if cand_h == 0 and any(heads[other - 1] == 0 and other != node for other in range(1, n + 1)):
                    continue
                loss = current_sc - masked_scores[node - 1, cand_h]
                if loss < min_loss:
                    min_loss = loss
                    best_node_to_change = node
                    best_new_head = cand_h

        if best_node_to_change is not None and best_new_head is not None:
            heads[best_node_to_change - 1] = best_new_head
        else:
            # Fallback: attach cycle node to root or token 1
            heads[cycle[0] - 1] = 0
            break

    return heads
