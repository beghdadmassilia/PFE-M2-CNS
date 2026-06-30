from math import inf
import random
import csv
from time import perf_counter


# ============================================================
# OPT-CET (Hall-Kubiak-Sethi) for 1||sum |Cj - d|
# ============================================================

def _stable_sort_jobs(p_list):
    jobs = list(enumerate(p_list, start=1))
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

            if a + pj <= d and prev[a + pj] < inf:
                cand = a + prev[a + pj]
                if cand < best:
                    best = cand
                    best_parent = (a + pj, "E", j)

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

            if sum_1_to_j > m and prev[m] < inf:
                cand = abs(sum_1_to_jm1 - m) + prev[m]
                if cand < best:
                    best = cand
                    best_parent = (m, "E", j)

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

            if prev[e] < inf:
                cand = total - e + prev[e]
                if cand < best:
                    best = cand
                    best_parent = (e, "L", j)

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
    if d < 0:
        raise ValueError("This Optcet implementation assumes d >= 0.")

    sorted_ids, p_sorted = _stable_sort_jobs(p_list)

    if d == 0:
        seq = list(range(len(p_sorted)))
        z, _, C_orig = _build_completion_dict(seq, p_sorted, d, sorted_ids, start_time=0)
        return {
            "z": z,
            "seq": _map_back_sequence(seq, sorted_ids),
            "seq_sorted_idx": seq,
            "C": C_orig,
            "start_time": 0,
            "method": "SPT-d=0-start0",
            "all_candidates": {}
        }

    candidates = {}

    if use_step1_shortcuts:
        c1 = _optcet_step1_unconstrained(sorted_ids, p_sorted, d)
        if c1 is not None and c1["start_time"] == 0:
            c1 = c1.copy()
            c1["method"] = "Step1-unconstrained-feasible-start0"
            candidates[c1["method"]] = c1

        c2 = _optcet_step1_spt(sorted_ids, p_sorted, d)
        if c2 is not None and c2["start_time"] == 0:
            c2 = c2.copy()
            c2["method"] = "Step1-SPT-start0"
            candidates[c2["method"]] = c2

    evs = _evs_subroutine(sorted_ids, p_sorted, d)
    tvs = _tvs_subroutine(sorted_ids, p_sorted, d)

    candidates["EVS-start0"] = evs
    candidates["TVS-start0"] = tvs

    nos = _nosplit_subroutine(sorted_ids, p_sorted, d)
    if nos["start_time"] == 0:
        candidates["Nosplit-start0"] = nos

    if not candidates:
        raise RuntimeError("No start-at-zero candidate found in optcet_hall_start0")

    best_key = min(candidates, key=lambda k: candidates[k]["z"])
    out = candidates[best_key].copy()
    out["all_candidates"] = candidates
    return out


# ============================================================
# Cas dA < dB : outils côté A
# CORRECTION LOCALE : A1 = préfixe de seq, A2 = suffixe de seq
# ============================================================


# ============================================================
# DP sous la structure B1 - A1 - B2 - A2 - B3 pour dA < dB
# CORRECTION LOCALE : on choisit une coupure h dans seq(A)
# ============================================================

def exact_da_lt_db(pA, pB, dA, dB, q, debug=False):
    nA = len(pA)
    nB = len(pB)

    if not (0 <= q <= nB):
        raise ValueError("q doit vérifier 0 <= q <= |J_B|")

    k = nB - q

    A_cache = {}
    JA = list(range(1, nA + 1))
    for t1 in range(dB + 1):
        local_d = dA - t1

        if local_d <= 0:
            seq = sorted(JA, key=lambda j: (pA[j - 1], j))
            raw_A = {"method": "SPT-inline-if-local_d<=0"}
        else:
            out = optcet_hall([pA[j - 1] for j in JA], local_d)
            seq = [JA[j - 1] for j in out["seq"]]
            raw_A = out

        time = t1
        C = {}
        z = 0
        prefix_completion = [t1]

        for j in seq:
            time += pA[j - 1]
            C[j] = time
            z += abs(time - dA)
            prefix_completion.append(time)

        A_cache[t1] = {
            "zbase": z,
            "seq": seq,
            "C": C,
            "prefix_completion": prefix_completion,
            "raw": raw_A,
        }

    B_sorted = sorted(range(1, nB + 1), key=lambda j: (pB[j - 1], j))

    f = [[[inf] * (dB + 1) for _ in range(k + 1)] for __ in range(nB + 1)]
    parent = [[[None] * (dB + 1) for _ in range(k + 1)] for __ in range(nB + 1)]

    f[0][0][0] = 0

    for i, b in enumerate(B_sorted):
        pb = pB[b - 1]

        for s in range(k + 1):
            for t1 in range(dB + 1):
                t2 = f[i][s][t1]
                if t2 == inf:
                    continue

                if t2 < f[i + 1][s][t1]:
                    f[i + 1][s][t1] = t2
                    parent[i + 1][s][t1] = (i, s, t1, "B3")

                if s < k:
                    if t1 + pb <= dB and t2 < f[i + 1][s + 1][t1 + pb]:
                        f[i + 1][s + 1][t1 + pb] = t2
                        parent[i + 1][s + 1][t1 + pb] = (i, s, t1, "B1")

                    if t2 + pb <= dB and t2 + pb < f[i + 1][s + 1][t1]:
                        f[i + 1][s + 1][t1] = t2 + pb
                        parent[i + 1][s + 1][t1] = (i, s, t1, "B2")

    best = inf
    best_t1 = None
    best_t2 = None
    best_h = None

    for t1 in range(dB + 1):
        t2 = f[nB][k][t1]
        if t2 == inf:
            continue

        A = A_cache[t1]
        nA_local = len(A["seq"])

        for h in range(nA_local + 1):
            e_h = A["prefix_completion"][h]
            feasible = True if t2 == 0 else (e_h + t2 <= dB)

            if feasible:
                ell_h = nA_local - h
                z = A["zbase"] + t2 * ell_h
                if z < best:
                    best = z
                    best_t1 = t1
                    best_t2 = t2
                    best_h = h

    if best_t1 is None:
        if debug:
            print("DEBUG - aucun état final trouvé")
        raise RuntimeError("Aucune solution faisable trouvée")

    B1, B2, B3 = [], [], []

    i, s, t1 = nB, k, best_t1
    while i > 0:
        par = parent[i][s][t1]
        if par is None:
            raise RuntimeError("Backtracking impossible")

        pi, ps, pt1, move = par
        b = B_sorted[i - 1]

        if move == "B1":
            B1.append(b)
        elif move == "B2":
            B2.append(b)
        else:
            B3.append(b)

        i, s, t1 = pi, ps, pt1

    B1.reverse()
    B2.reverse()
    B3.reverse()

    A = A_cache[best_t1]
    A1 = A["seq"][:best_h]
    A2 = A["seq"][best_h:]
    e_star = A["prefix_completion"][best_h]

    full_seq = (
        [("B", j) for j in B1]
        + [("A", j) for j in A1]
        + [("B", j) for j in B2]
        + [("A", j) for j in A2]
        + [("B", j) for j in B3]
    )

    time = 0
    C_A = {}
    C_B = {}

    for agent, j in full_seq:
        if agent == "A":
            time += pA[j - 1]
            C_A[j] = time
        else:
            time += pB[j - 1]
            C_B[j] = time

    z_final = sum(abs(C_A[j] - dA) for j in range(1, nA + 1))
    tardy_B = sum(1 for j in range(1, nB + 1) if C_B[j] > dB)

    return {
        "z": z_final,
        "B1": B1,
        "B2": B2,
        "B3": B3,
        "A1": A1,
        "A2": A2,
        "full_seq": full_seq,
        "t1_star": best_t1,
        "t2_star": best_t2,
        "h_star": best_h,
        "ell_star": len(A2),
        "e_star": e_star,
        "tardy_B": tardy_B,
        "C_A": C_A,
        "C_B": C_B,
    }

# ============================================================
# Heuristique pour dA < dB
# CORRECTION LOCALE : on choisit aussi une coupure suffixe A2
# ============================================================


from math import inf

def heuristic_da_lt_db(pA, pB, dA, dB, q, debug=False):
    nA = len(pA)
    nB = len(pB)

    if not (0 <= q <= nB):
        raise ValueError("q doit vérifier 0 <= q <= |J_B|")

    k = nB - q
    B_sorted = sorted(range(1, nB + 1), key=lambda j: (pB[j - 1], j))

    B1 = B_sorted[:k].copy()
    B2 = []
    B3 = B_sorted[k:].copy()

    cap = dB - dA

    while B1:
        j_star = max(B1, key=lambda j: (pB[j - 1], j))
        pj = pB[j_star - 1]
        if sum(pB[j - 1] for j in B2) + pj <= cap:
            B1.remove(j_star)
            B2.append(j_star)
        else:
            break

    t1 = sum(pB[j - 1] for j in B1)
    t2 = sum(pB[j - 1] for j in B2)

    JA = list(range(1, nA + 1))
    d_prime = dA - t1

    if d_prime <= 0:
        seqA = sorted(JA, key=lambda j: (pA[j - 1], j))
        raw_A = {"method": "SPT-if-d_prime<=0"}
    else:
        out = optcet_hall([pA[j - 1] for j in JA], d_prime)
        seqA = [JA[j - 1] for j in out["seq"]]
        raw_A = out

    time = t1
    C = {}
    zbase = 0
    prefix_completion = [t1]

    for j in seqA:
        time += pA[j - 1]
        C[j] = time
        zbase += abs(time - dA)
        prefix_completion.append(time)

    best_h = None
    best_value = inf
    for h in range(nA + 1):
        e_h = prefix_completion[h]
        if e_h + t2 <= dB:
            value = zbase + t2 * (nA - h)
            if value < best_value:
                best_value = value
                best_h = h

    if best_h is None:
        raise RuntimeError("Aucune coupure faisable trouvée dans l'heuristique")

    A1 = seqA[:best_h]
    A2 = seqA[best_h:]
    e = prefix_completion[best_h]

    if not B1 and not A1:
        full_seq = (
            [("B", j) for j in B2]
            + [("A", j) for j in A2]
            + [("B", j) for j in B3]
        )
    else:
        full_seq = (
            [("B", j) for j in B1]
            + [("A", j) for j in A1]
            + [("B", j) for j in B2]
            + [("A", j) for j in A2]
            + [("B", j) for j in B3]
        )

    time = 0
    C_A = {}
    C_B = {}

    for agent, j in full_seq:
        if agent == "A":
            time += pA[j - 1]
            C_A[j] = time
        else:
            time += pB[j - 1]
            C_B[j] = time

    z_final = sum(abs(C_A[j] - dA) for j in range(1, nA + 1))
    tardy_B = sum(1 for j in range(1, nB + 1) if C_B[j] > dB)

    if debug:
        print("B_sorted =", B_sorted)
        print("B1 =", B1)
        print("B2 =", B2)
        print("B3 =", B3)
        print("t1 =", t1)
        print("t2 =", t2)
        print("d_prime =", d_prime)
        print("seqA =", seqA)
        print("raw_A =", raw_A)
        print("prefix_completion =", prefix_completion)
        print("h =", best_h)
        print("A1 =", A1)
        print("A2 =", A2)

    return {
        "z": z_final,
        "B1": B1,
        "B2": B2,
        "B3": B3,
        "A1": A1,
        "A2": A2,
        "full_seq": full_seq,
        "t1": t1,
        "t2": t2,
        "h": best_h,
        "ell": len(A2),
        "e": e,
        "tardy_B": tardy_B,
        "C_A": C_A,
        "C_B": C_B,
        "raw_A": raw_A,
    }

# ============================================================
# Génération aléatoire faisable
# ============================================================

def generate_feasible_instance(
    nA, nB,
    pA_min=1, pA_max=10,
    pB_min=1, pB_max=10,
    seed=None
):
    rng = random.Random(seed)

    while True:
        pA = [rng.randint(pA_min, pA_max) for _ in range(nA)]
        pB = [rng.randint(pB_min, pB_max) for _ in range(nB)]

        sumA = sum(pA)
        sumB = sum(pB)

        k = rng.randint(0, nB)
        q = nB - k

        pB_sorted = sorted(pB)
        sum_k_smallest = sum(pB_sorted[:k]) if k > 0 else 0

        lower_dB = max(1, sum_k_smallest)
        upper_dB = sumB - 1
        if lower_dB > upper_dB:
            continue

        dB = rng.randint(lower_dB, upper_dB)

        lower_dA = 1
        upper_dA = dB - 1
        if lower_dA > upper_dA:
            continue

        dA = rng.randint(lower_dA, upper_dA)

        if sumA > dA and sumB > dB and dA < dB and sum_k_smallest <= dB:
            return pA, pB, dA, dB, q


# ============================================================
# Outils
# ============================================================

def pretty_sequence(full_seq):
    return " - ".join(f"{agent}{j}" for agent, j in full_seq)


def run_one_instance(pA, pB, dA, dB, q, verbose=True, debug=False):
    t0 = perf_counter()
    res_exact = exact_da_lt_db(pA, pB, dA, dB, q, debug=debug)
    t1 = perf_counter()

    res_heur = heuristic_da_lt_db(pA, pB, dA, dB, q)
    t2 = perf_counter()

    z_exact = res_exact["z"]
    z_heur = res_heur["z"]
    diff_abs = z_heur - z_exact

    if z_exact > 0:
        ratio = z_heur / z_exact
        gap = (z_heur - z_exact) / z_exact
        gap_percent = 100 * gap
    else:
        ratio = None
        gap = None
        gap_percent = None

    if verbose:
        print("=" * 70)
        print("Instance")
        print(f"pA = {pA}")
        print(f"pB = {pB}")
        print(f"dA = {dA}, dB = {dB}, q = {q}")
        print("-" * 70)

        print("DP exacte")
        print("Séquence :", pretty_sequence(res_exact["full_seq"]))
        print("Coût A   :", res_exact["z"])
        print("B1       :", res_exact["B1"])
        print("A1       :", res_exact["A1"])
        print("B2       :", res_exact["B2"])
        print("A2       :", res_exact["A2"])
        print("B3       :", res_exact["B3"])
        print("t1*      :", res_exact["t1_star"])
        print("t2*      :", res_exact["t2_star"])
        print("h*       :", res_exact["h_star"])
        print("ell*     :", res_exact["ell_star"])
        print("e*       :", res_exact["e_star"])
        print("Tardy B  :", res_exact["tardy_B"])
        print(f"Temps    : {t1 - t0:.6f} sec")
        print("-" * 70)

        print("Heuristique")
        print("Séquence :", pretty_sequence(res_heur["full_seq"]))
        print("Coût A   :", res_heur["z"])
        print("B1       :", res_heur["B1"])
        print("A1       :", res_heur["A1"])
        print("B2       :", res_heur["B2"])
        print("A2       :", res_heur["A2"])
        print("B3       :", res_heur["B3"])
        print("t1       :", res_heur["t1"])
        print("t2       :", res_heur["t2"])
        print("h        :", res_heur["h"])
        print("ell      :", res_heur["ell"])
        print("e        :", res_heur["e"])
        print("Tardy B  :", res_heur["tardy_B"])
        print(f"Temps    : {t2 - t1:.6f} sec")
        print("-" * 70)

        print("Comparaison")
        print("z_exact     =", z_exact)
        print("z_heur      =", z_heur)
        print("diff_abs    =", diff_abs)
        print("ratio       =", ratio)
        print("gap         =", gap)
        print("gap_percent =", gap_percent)

    return {
        "exact": res_exact,
        "heur": res_heur,
        "z_exact": z_exact,
        "z_heur": z_heur,
        "diff_abs": diff_abs,
        "ratio": ratio,
        "gap": gap,
        "gap_percent": gap_percent,
        "time_exact": t1 - t0,
        "time_heur": t2 - t1,
    }


def save_results_csv(results, filename="resul_1000_tests.csv"):
    fieldnames = [
        "id", "pA", "pB", "dA", "dB", "q",
        "z_exact", "z_heur", "diff_abs", "ratio", "gap_percent",
        "time_exact", "time_heur", "error"
    ]

    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {key: r.get(key, None) for key in fieldnames}
            writer.writerow(row)


def run_random_tests(nb_tests=1000, nA=5, nB=4, seed=100, csv_filename="resul_1000_tests.csv"):
    rng = random.Random(seed)
    results = []

    for test_id in range(1, nb_tests + 1):
        local_seed = rng.randint(0, 10**9)
        pA, pB, dA, dB, q = generate_feasible_instance(nA=nA, nB=nB, seed=local_seed)

        try:
            out = run_one_instance(pA, pB, dA, dB, q, verbose=False, debug=False)
            results.append({
                "id": test_id,
                "pA": pA,
                "pB": pB,
                "dA": dA,
                "dB": dB,
                "q": q,
                "z_exact": out["z_exact"],
                "z_heur": out["z_heur"],
                "diff_abs": out["diff_abs"],
                "ratio": out["ratio"],
                "gap_percent": out["gap_percent"],
                "time_exact": out["time_exact"],
                "time_heur": out["time_heur"],
                "error": None
            })
        except Exception as e:
            results.append({
                "id": test_id,
                "pA": pA,
                "pB": pB,
                "dA": dA,
                "dB": dB,
                "q": q,
                "z_exact": None,
                "z_heur": None,
                "diff_abs": None,
                "ratio": None,
                "gap_percent": None,
                "time_exact": None,
                "time_heur": None,
                "error": str(e)
            })

    print("=" * 70)
    print("Résumé des tests aléatoires")

    nb_ok = 0
    sum_ratio = 0
    nb_ratio = 0
    sum_gap = 0
    nb_gap = 0
    sum_diff = 0
    sum_time_exact = 0
    sum_time_heur = 0

    for r in results:
        if r["error"] is not None:
            print(f"Test {r['id']}: ERREUR -> {r['error']}")
        else:
            print(
                f"Test {r['id']}: "
                f"z_exact={r['z_exact']}, "
                f"z_heur={r['z_heur']}, "
                f"diff_abs={r['diff_abs']}, "
                f"ratio={r['ratio']}, "
                f"gap_percent={r['gap_percent']}, "
                f"time_exact={r['time_exact']:.6f}, "
                f"time_heur={r['time_heur']:.6f}"
            )
            nb_ok += 1
            sum_diff += r["diff_abs"]
            sum_time_exact += r["time_exact"]
            sum_time_heur += r["time_heur"]

            if r["ratio"] is not None:
                sum_ratio += r["ratio"]
                nb_ratio += 1

            if r["gap_percent"] is not None:
                sum_gap += r["gap_percent"]
                nb_gap += 1

    print("=" * 70)
    if nb_ok > 0:
        print("MOYENNES SUR LES INSTANCES RÉUSSIES")
        print("Nombre d'instances réussies :", nb_ok)
        print("Différence absolue moyenne  :", sum_diff / nb_ok)
        print("Ratio moyen                 :", (sum_ratio / nb_ratio) if nb_ratio > 0 else None)
        print("Gap moyen (%)               :", (sum_gap / nb_gap) if nb_gap > 0 else None)
        print("Temps exact moyen           :", sum_time_exact / nb_ok)
        print("Temps heuristique moyen     :", sum_time_heur / nb_ok)
    print("=" * 70)

    save_results_csv(results, filename=csv_filename)
    print(f"Résultats sauvegardés dans : {csv_filename}")

    return results

def run_all_configs_da_lt_db(configs, nb_tests=50, seed=100, summary_filename="benchmark_summary_da_lt_db1.csv"):
    summary = []

    for nA, nB in configs:
        print(f"\n{'='*70}")
        print(f"Configuration : nA={nA}, nB={nB} (cas dA < dB)")
        print(f"{'='*70}")

        rng = random.Random(seed)
        results = []

        for test_id in range(1, nb_tests + 1):
            while True:
                local_seed = rng.randint(0, 10**9)
                rr = random.Random(local_seed)

                pA = [rr.randint(1, 10) for _ in range(nA)]
                pB = [rr.randint(1, 10) for _ in range(nB)]

                sumA = sum(pA)
                sumB = sum(pB)

                k = rr.randint(0, nB)
                q = nB - k
                sum_k_smallest = sum(sorted(pB)[:k]) if k > 0 else 0

                if sumA <= 1 or sumB <= 1:
                    continue

                dA = rr.randint(1, sumA - 1)
                lower_dB = max(dA + 1, sum_k_smallest)
                upper_dB = sumB - 1
                if lower_dB > upper_dB:
                    continue

                dB = rr.randint(lower_dB, upper_dB)
                break

            try:
                t0 = perf_counter()
                exact = exact_da_lt_db(pA, pB, dA, dB, q, debug=False)
                t1 = perf_counter()
                heur = heuristic_da_lt_db(pA, pB, dA, dB, q, debug=False)
                t2 = perf_counter()

                z_exact = exact["z"]
                z_heur = heur["z"]

                results.append({
                    "id": test_id,
                    "pA": pA,
                    "pB": pB,
                    "dA": dA,
                    "dB": dB,
                    "q": q,
                    "z_exact": z_exact,
                    "z_heur": z_heur,
                    "diff_abs": z_heur - z_exact,
                    "ratio": (z_heur / z_exact) if z_exact else None,
                    "gap_percent": ((z_heur - z_exact) / z_exact * 100) if z_exact else None,
                    "time_exact": (t1 - t0) * 1000,
                    "time_heur": (t2 - t1) * 1000,
                    "error": None
                })

            except Exception as e:
                results.append({
                    "id": test_id,
                    "pA": pA,
                    "pB": pB,
                    "dA": dA,
                    "dB": dB,
                    "q": q,
                    "z_exact": None,
                    "z_heur": None,
                    "diff_abs": None,
                    "ratio": None,
                    "gap_percent": None,
                    "time_exact": None,
                    "time_heur": None,
                    "error": str(e)
                })

        valid = [r for r in results if r["error"] is None]
        nb_ok = len(valid)
        nb_optimal = sum(1 for r in valid if r["diff_abs"] == 0)

        ratio_vals = [r["ratio"] for r in valid if r["ratio"] is not None]
        gap_vals = [r["gap_percent"] for r in valid if r["gap_percent"] is not None]

        summary_row = {
            "nA": nA,
            "nB": nB,
            "nb_instances": nb_ok,
            "optimal_%": round(100 * nb_optimal / nb_ok, 2) if nb_ok else None,
            "diff_abs_moyenne": round(sum(r["diff_abs"] for r in valid) / nb_ok, 4) if nb_ok else None,
            "ratio_moyen": round(sum(ratio_vals) / len(ratio_vals), 6) if ratio_vals else None,
            "gap_moyen_%": round(sum(gap_vals) / len(gap_vals), 4) if gap_vals else None,
            "temps_exact_moyen_ms": round(sum(r["time_exact"] for r in valid) / nb_ok, 4) if nb_ok else None,
            "temps_heur_moyen_ms": round(sum(r["time_heur"] for r in valid) / nb_ok, 4) if nb_ok else None,
        }

        print(summary_row)
        summary.append(summary_row)

        with open(f"benchmark_da_lt_db_nA{nA}_nB{nB}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    with open(summary_filename, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "nA", "nB", "nb_instances", "optimal_%",
            "diff_abs_moyenne", "ratio_moyen", "gap_moyen_%",
            "temps_exact_moyen_ms", "temps_heur_moyen_ms"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)

    print(f"\nRésumé global sauvegardé dans : {summary_filename}")
    return summary
# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    configs = [
        (5, 5),
        (10, 10),
        (20, 20),
        (30, 30),
        (40, 40),
        (50, 50),
        (60, 60),
        (70, 70) 
        
    ]

    run_all_configs_da_lt_db(
        configs=configs,
        nb_tests=50,
        seed=100,
        summary_filename="benchmark_summary_da_lt_db1.csv"
    )