from math import inf
from itertools import permutations
import random

# ============================================================
# OPT-CET value only
# ============================================================
from math import inf

def _evaluate_sequence(seq_sorted_idx, p_sorted, d, start_time=0):
    t = start_time
    C = {}
    z = 0
    for j in seq_sorted_idx:
        t += p_sorted[j]
        C[j] = t
        z += abs(t - d)
    return z, C


def _stable_sort_jobs(p_list):
    jobs = list(enumerate(p_list, start=1))  # (original_id, p)
    jobs.sort(key=lambda x: (x[1], x[0]))
    sorted_ids = [jid for jid, _ in jobs]
    p_sorted = [p for _, p in jobs]
    return sorted_ids, p_sorted


def _prefix_sums(arr):
    pref = [0]
    for x in arr:
        pref.append(pref[-1] + x)
    return pref


def _map_back_sequence(seq_sorted_idx, sorted_ids):
    return [sorted_ids[j] for j in seq_sorted_idx]


def _build_completion_dict(seq_sorted_idx, p_sorted, d, sorted_ids, start_time=0):
    t = start_time
    C_sorted = {}
    C_orig = {}
    z = 0
    for j in seq_sorted_idx:
        t += p_sorted[j]
        C_sorted[j] = t
        C_orig[sorted_ids[j]] = t
        z += abs(t - d)
    return z, C_sorted, C_orig


def _unconstrained_vshape_indices(n):
    first = list(range(n - 1, -1, -2))
    second = list(range(n % 2, n, 2))
    return first + second, first


def _optcet_step1_unconstrained(sorted_ids, p_sorted, d):
    n = len(p_sorted)
    seq, early_block = _unconstrained_vshape_indices(n)
    early_sum = sum(p_sorted[j] for j in early_block)
    if early_sum <= d:
        start_time = d - early_sum
        z, _, C_orig = _build_completion_dict(seq, p_sorted, d, sorted_ids, start_time)
        return {
            "z": z,
            "seq": _map_back_sequence(seq, sorted_ids),
            "seq_sorted_idx": seq,
            "C": C_orig,
            "start_time": start_time,
            "method": "Step1-unconstrained-feasible"
        }
    return None


def _optcet_step1_spt(sorted_ids, p_sorted, d):
    if d < p_sorted[0]:
        seq = list(range(len(p_sorted)))
        z, _, C_orig = _build_completion_dict(seq, p_sorted, d, sorted_ids, start_time=0)
        return {
            "z": z,
            "seq": _map_back_sequence(seq, sorted_ids),
            "seq_sorted_idx": seq,
            "C": C_orig,
            "start_time": 0,
            "method": "Step1-SPT"
        }
    return None


def _evs_subroutine(sorted_ids, p_sorted, d):
    n = len(p_sorted)
    pref = _prefix_sums(p_sorted)

    prev = [inf] * (d + 1)
    prev[d] = 0

    parent = [[None] * (d + 1) for _ in range(n + 1)]

    for stage in range(n):
        j = n - 1 - stage
        pj = p_sorted[j]
        sum_1_to_j = pref[j + 1]

        curr = [inf] * (d + 1)

        for a in range(d + 1):
            best = inf
            best_parent = None

            # Early as possible:
            # f_{k+1}(a) <- a + f_k(a + p_j)
            if a + pj <= d and prev[a + pj] < inf:
                cand = a + prev[a + pj]
                if cand < best:
                    best = cand
                    best_parent = (a + pj, "E", j)

            # Late as possible:
            # f_{k+1}(a) <- sum_{i=1}^{j} p_i - a + f_k(a)
            # only if sum_{i=1}^{j} p_i > a
            if sum_1_to_j > a and prev[a] < inf:
                cand = sum_1_to_j - a + prev[a]
                if cand < best:
                    best = cand
                    best_parent = (a, "L", j)

            curr[a] = best
            parent[stage + 1][a] = best_parent

        prev = curr

    a_star = min(range(d + 1), key=lambda a: prev[a])
    z_star = prev[a_star]

    # Backtrack decisions
    decisions = []
    a = a_star
    for stage in range(n, 0, -1):
        par = parent[stage][a]
        if par is None:
            raise RuntimeError("EVS backtracking failed")
        prev_a, dec, j = par
        decisions.append((j, dec))
        a = prev_a
    decisions.reverse()

    early = []
    late = []
    for j, dec in decisions:
        if dec == "E":
            early.append(j)
        else:
            late.insert(0, j)

    seq = early + late
    z_check, _, C_orig = _build_completion_dict(seq, p_sorted, d, sorted_ids, start_time=0)

    return {
        "z": z_star,
        "z_check": z_check,
        "seq": _map_back_sequence(seq, sorted_ids),
        "seq_sorted_idx": seq,
        "C": C_orig,
        "start_time": 0,
        "state_star": a_star,
        "method": "EVS"
    }


def _tvs_subroutine(sorted_ids, p_sorted, d):
    n = len(p_sorted)
    P = sum(p_sorted)
    M = P - d
    pref = _prefix_sums(p_sorted)

    prev = [inf] * (M + 1)
    prev[M] = 0

    parent = [[None] * (M + 1) for _ in range(n + 1)]

    for stage in range(n):
        j = n - 1 - stage
        pj = p_sorted[j]
        sum_1_to_jm1 = pref[j]
        sum_1_to_j = pref[j + 1]

        curr = [inf] * (M + 1)

        for m in range(M + 1):
            best = inf
            best_parent = None

            # Early as possible:
            # g_{k+1}(m) <- |sum_{i=1}^{j-1} p_i - m| + g_k(m)
            # only if sum_{i=1}^{j} p_i > m
            if sum_1_to_j > m and prev[m] < inf:
                cand = abs(sum_1_to_jm1 - m) + prev[m]
                if cand < best:
                    best = cand
                    best_parent = (m, "E", j)

            # Late as possible:
            # g_{k+1}(m) <- m + p_j + g_k(m + p_j)
            if m + pj <= M and prev[m + pj] < inf:
                cand = m + pj + prev[m + pj]
                if cand < best:
                    best = cand
                    best_parent = (m + pj, "L", j)

            curr[m] = best
            parent[stage + 1][m] = best_parent

        prev = curr

    m_star = min(range(M + 1), key=lambda m: prev[m])
    z_star = prev[m_star]

    decisions = []
    m = m_star
    for stage in range(n, 0, -1):
        par = parent[stage][m]
        if par is None:
            raise RuntimeError("TVS backtracking failed")
        prev_m, dec, j = par
        decisions.append((j, dec))
        m = prev_m
    decisions.reverse()

    early = []
    late = []
    for j, dec in decisions:
        if dec == "E":
            early.append(j)
        else:
            late.insert(0, j)

    seq = early + late
    z_check, _, C_orig = _build_completion_dict(seq, p_sorted, d, sorted_ids, start_time=0)

    return {
        "z": z_star,
        "z_check": z_check,
        "seq": _map_back_sequence(seq, sorted_ids),
        "seq_sorted_idx": seq,
        "C": C_orig,
        "start_time": 0,
        "state_star": m_star,
        "method": "TVS"
    }


def _nosplit_subroutine(sorted_ids, p_sorted, d):
    n = len(p_sorted)
    pref = _prefix_sums(p_sorted)

    prev = [inf] * (d + 1)
    prev[0] = 0

    parent = [[None] * (d + 1) for _ in range(n + 1)]

    for stage in range(n):
        j = stage
        pj = p_sorted[j]
        total = pref[j + 1]

        curr = [inf] * (d + 1)

        for e in range(d + 1):
            best = inf
            best_parent = None

            # Late as possible:
            # h_{k+1}(e) <- sum_{i=1}^{k+1} p_i - e + h_k(e)
            if prev[e] < inf:
                cand = total - e + prev[e]
                if cand < best:
                    best = cand
                    best_parent = (e, "L", j)

            # Early as possible:
            # h_{k+1}(e) <- e - p_{k+1} + h_k(e - p_{k+1})
            if e >= pj and prev[e - pj] < inf:
                cand = e - pj + prev[e - pj]
                if cand < best:
                    best = cand
                    best_parent = (e - pj, "E", j)

            curr[e] = best
            parent[stage + 1][e] = best_parent

        prev = curr

    e_star = min(range(d + 1), key=lambda e: prev[e])
    z_star = prev[e_star]

    decisions = []
    e = e_star
    for stage in range(n, 0, -1):
        par = parent[stage][e]
        if par is None:
            raise RuntimeError("Nosplit backtracking failed")
        prev_e, dec, j = par
        decisions.append((j, dec))
        e = prev_e
    decisions.reverse()

    early = []
    late = []
    for j, dec in decisions:
        if dec == "E":
            early.insert(0, j)
        else:
            late.append(j)

    seq = early + late
    early_sum = sum(p_sorted[j] for j in early)
    start_time = d - early_sum

    z_check, _, C_orig = _build_completion_dict(seq, p_sorted, d, sorted_ids, start_time=start_time)

    return {
        "z": z_star,
        "z_check": z_check,
        "seq": _map_back_sequence(seq, sorted_ids),
        "seq_sorted_idx": seq,
        "C": C_orig,
        "start_time": start_time,
        "state_star": e_star,
        "method": "Nosplit"
    }


def optcet_hall(p_list, d, use_step1_shortcuts=True):
    """
    Literal Hall-Kubiak-Sethi style implementation of Optcet
    for 1||sum |C_j - d| with common due date d > 0.

    Returns:
        {
            'z': optimal value found by this implementation,
            'seq': sequence in original job labels,
            'C': completion times by original job label,
            'start_time': schedule start time,
            'method': one of Step1-unconstrained-feasible / Step1-SPT / EVS / TVS / Nosplit,
            'all_candidates': {...}
        }
    """
    if d <= 0:
        raise ValueError("This Optcet implementation assumes d > 0, as in Hall et al.")

    sorted_ids, p_sorted = _stable_sort_jobs(p_list)

    candidates = {}

    if use_step1_shortcuts:
        c1 = _optcet_step1_unconstrained(sorted_ids, p_sorted, d)
        if c1 is not None:
            candidates[c1["method"]] = c1

        c2 = _optcet_step1_spt(sorted_ids, p_sorted, d)
        if c2 is not None:
            candidates[c2["method"]] = c2

        # If one of Step 1 shortcuts applies, Hall stops immediately.
        if c1 is not None:
            out = c1.copy()
            out["all_candidates"] = candidates
            return out
        if c2 is not None:
            out = c2.copy()
            out["all_candidates"] = candidates
            return out

    evs = _evs_subroutine(sorted_ids, p_sorted, d)
    tvs = _tvs_subroutine(sorted_ids, p_sorted, d)
    nos = _nosplit_subroutine(sorted_ids, p_sorted, d)

    candidates["EVS"] = evs
    candidates["TVS"] = tvs
    candidates["Nosplit"] = nos

    best_key = min(candidates, key=lambda k: candidates[k]["z"])
    out = candidates[best_key].copy()
    out["all_candidates"] = candidates
    return out


# ------------------------------------------------------------
# Example
# ------------------------------------------------------------
if __name__ == "__main__":
    p = [3, 1, 8, 4, 7]
    d = 22

    res = optcet_hall(p, d)

    print("Method     :", res["method"])
    print("Value z    :", res["z"])
    print("Sequence   :", res["seq"])
    print("Start time :", res["start_time"])
    print("C          :", dict(sorted(res["C"].items())))

    print("\n--- Candidates ---")
    for name, cand in res["all_candidates"].items():
        print(
            name,
            "| z =", cand["z"],
            "| seq =", cand["seq"],
            "| start =", cand["start_time"]
        )


