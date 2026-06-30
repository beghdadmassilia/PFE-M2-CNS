import itertools
from math import inf

def brute_force_two_duedates(A, B, p, dA, dB, q):
    J = list(A) + list(B)
    Aset, Bset = set(A), set(B)

    best_cost = inf
    best_perm = None
    best_C = None

    for perm in itertools.permutations(J):
        t = 0
        late_B = 0
        cost_A = 0
        C = {}

        for j in perm:
            t += p[j]
            C[j] = t

            # Constraint on B
            if j in Bset and t > dB:
                late_B += 1
                if late_B > q:
                    break

            # Objective on A
            if j in Aset:
                cost_A += abs(t - dA)

        else:
            if cost_A < best_cost:
                best_cost = cost_A
                best_perm = perm
                best_C = C

    if best_perm is None:
        return None, None, None
    return best_cost, best_perm, best_C


if __name__ == "__main__":
    tests_piegeux = [
        {"id": "T1",  "pA": [9, 8, 7, 6],       "pB": [1, 2, 10, 11],   "dA": 3, "dB": 12, "q": 2},
        {"id": "T2",  "pA": [4, 4, 4, 4, 4],   "pB": [3, 3, 3, 3],     "dA": 5, "dB": 6,  "q": 2},
        {"id": "T3",  "pA": [2, 9, 2, 9, 2],   "pB": [4, 4, 4, 9],     "dA": 6, "dB": 12, "q": 2},
        {"id": "T4",  "pA": [6, 1, 6, 1, 6],   "pB": [2, 2, 7, 8],     "dA": 4, "dB": 9,  "q": 2},
        {"id": "T7",  "pA": [3, 10, 3, 10],    "pB": [2, 9, 2, 9, 2],  "dA": 5, "dB": 11, "q": 3},
        {"id": "T8",  "pA": [7, 7, 1, 1, 1],   "pB": [4, 5, 6, 7],     "dA": 5, "dB": 10, "q": 2},
        {"id": "T9",  "pA": [2, 2, 2, 11, 11], "pB": [3, 3, 8, 8],     "dA": 6, "dB": 11, "q": 2},
        {"id": "T10", "pA": [1, 4, 7, 10],     "pB": [2, 2, 2, 2, 9],  "dA": 4, "dB": 8,  "q": 3},
        {"id": "T11", "pA": [1, 6],            "pB": [6, 1],           "dA": 5, "dB": 8,  "q": 0},
    ]
    

    for inst in tests_piegeux:
        A = [f"a{i+1}" for i in range(len(inst["pA"]))]
        B = [f"b{i+1}" for i in range(len(inst["pB"]))]

        p_by_id = {}
        for i, p in enumerate(inst["pA"], start=1):
            p_by_id[f"a{i}"] = p
        for i, p in enumerate(inst["pB"], start=1):
            p_by_id[f"b{i}"] = p

        dA = inst["dA"]
        dB = inst["dB"]
        q = inst["q"]

        best_cost, best_perm, best_C = brute_force_two_duedates(A, B, p_by_id, dA, dB, q)

        print("\n" + "=" * 80)
        print("TEST", inst["id"])
        print("pA =", inst["pA"])
        print("pB =", inst["pB"])
        print(f"dA = {dA}, dB = {dB}, q = {q}")
        print("Best cost (A):", best_cost)
        print("Best sequence:", best_perm)
        print("Late B jobs:", None if best_C is None else [b for b in B if best_C[b] > dB])
        # print("Completion times:", best_C)