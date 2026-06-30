from math import inf
import random
import csv
from time import perf_counter


# ============================================================
# OPT-CET (Hall-Kubiak-Sethi) for 1||sum |Cj - d|
# ============================================================

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
    """
    Version de Optcet adaptée au cas où la séquence doit commencer
    immédiatement à t = 0 dans le repère local.
    Donc on interdit toute solution avec start_time > 0.
    """

    if d < 0:
        raise ValueError("This Optcet implementation assumes d >= 0.")

    sorted_ids, p_sorted = _stable_sort_jobs(p_list)

    # Cas trivial
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

    # Step 1 - on ne garde que les solutions qui démarrent à 0
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

    # EVS et TVS sont déjà des solutions avec start_time = 0
    evs = _evs_subroutine(sorted_ids, p_sorted, d)
    tvs = _tvs_subroutine(sorted_ids, p_sorted, d)

    candidates["EVS-start0"] = evs
    candidates["TVS-start0"] = tvs

    # Nosplit peut avoir start_time > 0, donc on le filtre
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
# DP exacte pour dA >= dB
# ============================================================

def exact_da_ge_db(pA, pB, dA, dB, q, debug=False):
    nA = len(pA)
    nB = len(pB)

    if not (0 <= q <= nB):
        raise ValueError("q doit vérifier 0 <= q <= |J_B|")

    k = nB - q
    PB = sum(pB)

    def eval_A(t):
        d_local = dA - t

        if d_local >= 0:
            out = optcet_hall(pA, d_local)
            return {
                "z": out["z"],
                "seq": out["seq"][:],
                "method": out["method"]
            }

        ids = sorted(range(nA), key=lambda i: (pA[i], i))
        seq = [i + 1 for i in ids]
        time = 0
        z = 0
        for j in seq:
            time += pA[j - 1]
            z += abs(time - d_local)
        return {
            "z": z,
            "seq": seq,
            "method": "SPT-negative-local-due-date"
        }

    A_cache = {}
    for t in range(PB + 1):
        A_cache[t] = eval_A(t)

    layers = [dict() for _ in range(nB + 1)]
    parent = [dict() for _ in range(nB + 1)]

    start = (0, 0, 0)
    layers[0][start] = True

    for i in range(1, nB + 1):
        p = pB[i - 1]

        for (s, x, t) in list(layers[i - 1].keys()):

            # Choice 1: put job i in B2
            st = (s, x, t)
            if st not in layers[i]:
                layers[i][st] = True
                parent[i][st] = ((s, x, t), "B2")

            # Choice 2: put job i in B1 but tardy w.r.t. dB
            # Important: s does NOT increase here
            if t + p <= PB:
                st = (s, x, t + p)
                if st not in layers[i]:
                    layers[i][st] = True
                    parent[i][st] = ((s, x, t), "B1-late")

            # Choice 3: put job i in B1 and guaranteed on time
            if s < k and x + p <= dB and t + p <= PB:
                st = (s + 1, x + p, t + p)
                if st not in layers[i]:
                    layers[i][st] = True
                    parent[i][st] = ((s, x, t), "B1-early")

    best_state = None
    best_value = inf

    for (s, x, t) in layers[nB].keys():
        if s == k:
            z = A_cache[t]["z"]
            if z < best_value:
                best_value = z
                best_state = (s, x, t)

    if best_state is None:
        if debug:
            print("DEBUG - aucun état final trouvé")
            print("k =", k)
            print("pB =", pB)
            print("dB =", dB)
            print("Valeurs de s atteintes =", sorted(set(s for (s, x, t) in layers[nB].keys())))
            print("Quelques états finaux =", list(layers[nB].keys())[:30])
        raise RuntimeError("Aucune solution faisable trouvée")

    B1_early = []
    B1_late = []
    B2 = []

    cur = best_state
    for i in range(nB, 0, -1):
        prev, decision = parent[i][cur]
        job = i

        if decision == "B1-early":
            B1_early.append(job)
        elif decision == "B1-late":
            B1_late.append(job)
        else:
            B2.append(job)

        cur = prev

    B1_early.reverse()
    B1_late.reverse()
    B2.reverse()

    B1 = B1_early + B1_late
    A_seq = A_cache[best_state[2]]["seq"][:]

    full_seq = [("B", j) for j in B1] + [("A", j) for j in A_seq] + [("B", j) for j in B2]

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
        "B1_early": B1_early,
        "B1_late": B1_late,
        "B1": B1,
        "A_seq": A_seq,
        "B2": B2,
        "full_seq": full_seq,
        "t_star": best_state[2],
        "x_star": best_state[1],
        "method_A": A_cache[best_state[2]]["method"],
        "tardy_B": tardy_B,
        "C_A": C_A,
        "C_B": C_B
    }


# ============================================================
# Heuristique : prendre les k plus petites tâches de B avant A
# ============================================================

def heuristic_smallestB_before_A(pA, pB, dA, dB, q):
    nA = len(pA)
    nB = len(pB)

    if not (0 <= q <= nB):
        raise ValueError("q doit vérifier 0 <= q <= |J_B|")

    k = nB - q

    order_B = sorted(range(nB), key=lambda i: (pB[i], i))
    B1 = [i + 1 for i in order_B[:k]]
    B2 = [i + 1 for i in order_B[k:]]

    t = sum(pB[j - 1] for j in B1)
    d_local = dA - t

    if d_local >= 0:
        outA = optcet_hall(pA, d_local)
        A_seq = outA["seq"][:]
        method_A = outA["method"]
    else:
        ids = sorted(range(nA), key=lambda i: (pA[i], i))
        A_seq = [i + 1 for i in ids]
        method_A = "SPT-negative-local-due-date"

    full_seq = [("B", j) for j in B1] + [("A", j) for j in A_seq] + [("B", j) for j in B2]

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

    ontime_B1 = [j for j in B1 if C_B[j] <= dB]
    tardy_B1 = [j for j in B1 if C_B[j] > dB]

    return {
        "z": z_final,
        "B1": B1,
        "A_seq": A_seq,
        "B2": B2,
        "full_seq": full_seq,
        "t": t,
        "d_local": d_local,
        "method_A": method_A,
        "tardy_B": tardy_B,
        "ontime_B1": ontime_B1,
        "tardy_B1": tardy_B1,
        "C_A": C_A,
        "C_B": C_B
    }


# ============================================================
# Génération aléatoire faisable
# ============================================================

def generate_feasible_instance(
    nA, nB,
    pA_min=1, pA_max=20,
    pB_min=1, pB_max=20,
    seed=None
):
    rng = random.Random(seed)

    while True:
        pA = [rng.randint(pA_min, pA_max) for _ in range(nA)]
        pB = [rng.randint(pB_min, pB_max) for _ in range(nB)]

        sumA = sum(pA)
        sumB = sum(pB)

        q = 1
        k = nB - q


        pB_sorted = sorted(pB)
        sum_k_smallest = sum(pB_sorted[:k]) if k > 0 else 0

        lower_dB = max(1, sum_k_smallest)
        upper_dB = sumB - 1
        if lower_dB > upper_dB:
            continue

        dB = rng.randint(lower_dB, upper_dB)

        lower_dA = dB
        upper_dA = sumA - 1
        if lower_dA > upper_dA:
            continue

        dA = rng.randint(lower_dA, upper_dA)

        if sumA > dA and sumB > dB and dA >= dB and sum_k_smallest <= dB:
            return pA, pB, dA, dB, q


# ============================================================
# Outils
# ============================================================

def pretty_sequence(full_seq):
    return " - ".join(f"{agent}{j}" for agent, j in full_seq)


def run_one_instance(pA, pB, dA, dB, q, verbose=True, debug=False):
    t0 = perf_counter()
    res_exact = exact_da_ge_db(pA, pB, dA, dB, q, debug=debug)
    t1 = perf_counter()

    res_heur = heuristic_smallestB_before_A(pA, pB, dA, dB, q)
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
        print("B1 early :", res_exact["B1_early"])
        print("B1 late  :", res_exact["B1_late"])
        print("B1       :", res_exact["B1"])
        print("A        :", res_exact["A_seq"])
        print("B2       :", res_exact["B2"])
        print("t*       :", res_exact["t_star"])
        print("x*       :", res_exact["x_star"])
        print("Méthode A:", res_exact["method_A"])
        print("Tardy B  :", res_exact["tardy_B"])
        print(f"Temps    : {t1 - t0:.6f} sec")
        print("-" * 70)

        print("Heuristique")
        print("Séquence :", pretty_sequence(res_heur["full_seq"]))
        print("Coût A   :", res_heur["z"])
        print("B1       :", res_heur["B1"])
        print("A        :", res_heur["A_seq"])
        print("B2       :", res_heur["B2"])
        print("t        :", res_heur["t"])
        print("d_local  :", res_heur["d_local"])
        print("Méthode A:", res_heur["method_A"])
        print("Tardy B  :", res_heur["tardy_B"])
        print("B1 à temps   :", res_heur["ontime_B1"])
        print("B1 en retard :", res_heur["tardy_B1"])
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


def save_results_csv(results, filename="resulKtests.csv"):
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


def run_random_tests(nb_tests=1000, nA=5, nB=4, seed=100, csv_filename="resulKtests.csv"):
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
                "time_exact": out["time_exact"] * 1000,   # conversion en ms
                "time_heur": out["time_heur"] * 1000,     # conversion en ms
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
    nb_optimal = 0       
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
                f"time_exact={r['time_exact']:.4f} ms, "
                f"time_heur={r['time_heur']:.4f} ms"
            )
            nb_ok += 1
            sum_diff += r["diff_abs"]
            sum_time_exact += r["time_exact"]
            sum_time_heur += r["time_heur"]

            if r["diff_abs"] == 0:
                nb_optimal += 1

            if r["ratio"] is not None:
                sum_ratio += r["ratio"]
                nb_ratio += 1

            if r["gap_percent"] is not None:
                sum_gap += r["gap_percent"]
                nb_gap += 1

    print("=" * 70)
    if nb_ok > 0:
        print("MOYENNES SUR LES INSTANCES RÉUSSIES")
        print(f"Nombre d'instances réussies      : {nb_ok}")
        print(f"Instances optimales (%)          : {100 * nb_optimal / nb_ok:.2f}%")
        print(f"Différence absolue moyenne       : {sum_diff / nb_ok:.4f}")
        print(f"Ratio moyen                      : {(sum_ratio / nb_ratio) if nb_ratio > 0 else None:.6f}")
        print(f"Gap moyen (%)                    : {(sum_gap / nb_gap) if nb_gap > 0 else None:.4f}%")
        print(f"Temps exact moyen (ms)           : {sum_time_exact / nb_ok:.4f}")
        print(f"Temps heuristique moyen (ms)     : {sum_time_heur / nb_ok:.4f}")
    print("=" * 70)

    save_results_csv(results, filename=csv_filename)
    print(f"Résultats sauvegardés dans : {csv_filename}")

    return results

def run_all_configs(configs, nb_tests=100, seed=100, summary_filename="benchmark_summaryq1.csv"):
    summary = []

    for nA, nB in configs:
        print(f"\n{'='*70}")
        print(f"Configuration : nA={nA}, nB={nB}")
        print(f"{'='*70}")

        results = run_random_tests(
            nb_tests=nb_tests,
            nA=nA,
            nB=nB,
            seed=seed,
            csv_filename=f"benchmark_nA{nA}_nB{nB}.csv"
        )

        nb_ok = 0
        nb_optimal = 0
        sum_ratio = 0
        nb_ratio = 0
        sum_gap = 0
        nb_gap = 0
        sum_diff = 0
        sum_time_exact = 0
        sum_time_heur = 0

        for r in results:
            if r["error"] is None:
                nb_ok += 1
                sum_diff += r["diff_abs"]
                sum_time_exact += r["time_exact"]
                sum_time_heur += r["time_heur"]

                if r["diff_abs"] == 0:
                    nb_optimal += 1

                if r["ratio"] is not None:
                    sum_ratio += r["ratio"]
                    nb_ratio += 1

                if r["gap_percent"] is not None:
                    sum_gap += r["gap_percent"]
                    nb_gap += 1

        if nb_ok > 0:
            summary.append({
                "nA": nA,
                "nB": nB,
                "nb_instances": nb_ok,
                "optimal_%": round(100 * nb_optimal / nb_ok, 2),
                "diff_abs_moyenne": round(sum_diff / nb_ok, 4),
                "ratio_moyen": round(sum_ratio / nb_ratio, 6) if nb_ratio > 0 else None,
                "gap_moyen_%": round(sum_gap / nb_gap, 4) if nb_gap > 0 else None,
                "temps_exact_moyen_ms": round(sum_time_exact / nb_ok, 4),
                "temps_heur_moyen_ms": round(sum_time_heur / nb_ok, 4),
            })

    # Sauvegarde du résumé
    fieldnames = [
        "nA", "nB", "nb_instances", "optimal_%",
        "diff_abs_moyenne", "ratio_moyen", "gap_moyen_%",
        "temps_exact_moyen_ms", "temps_heur_moyen_ms"
    ]

    with open(summary_filename, mode="w", newline="", encoding="utf-8") as f:
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

    run_all_configs(
        configs=configs,
        nb_tests=50,
        seed=100,
        summary_filename="benchmark_summary.csv"
    )

