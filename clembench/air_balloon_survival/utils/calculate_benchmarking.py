from itertools import combinations
from .pseudo_poly_knapsack import dp_pseudo_poly_knapsack

def compute_instance_benchmarks(items, item_weights, preference_p01, preference_p02, max_weight):
    """
    Given item weights, player preferences, and the weight limit,
    computes:
        - max utility for player 1 and 2
        - max total utility
        - max raw harmonic mean
        - pareto optima list
    Returns a dict which will be an input to the instance file.
    """

    all_valid_subsets = []
    # Compute max possible individual utility scores using efficient knapsack
    max_u1, _ = dp_pseudo_poly_knapsack(item_weights, preference_p01, max_weight)
    max_u2, _ = dp_pseudo_poly_knapsack(item_weights, preference_p02, max_weight)
    max_total_utility = 0 #highest possible total utility score
    max_harmonic = 0 #highest possible fairness-efficiency score
    best_set = set()

    # Iterate over all possible subsets of items
    for i in range(1, len(items) + 1):
        for subset in combinations(items, i):
            subset_set = set(subset)
            total_weight = sum(item_weights[item] for item in subset_set)

            if total_weight <= max_weight: #make sure that only combinations respecting the weight constraint are considered valid
                #For each valid subset, calculate the following:

                #Calculate personal utility scores
                util_p01 = sum(preference_p01[item] for item in subset_set)
                util_p02 = sum(preference_p02[item] for item in subset_set)
                all_valid_subsets.append((subset_set, util_p01, util_p02)) #subset_set = item set

                # Best total utility score (combined of both players)
                total_utility = util_p01 + util_p02
                max_total_utility = max(max_total_utility, total_utility)

                #Calculate maximum harmonic mean of normalized scores
                normalized_u1 = (util_p01 / max_u1 * 100) if max_u1 > 0 else 0
                normalized_u2 = (util_p02 / max_u2 * 100) if max_u2 > 0 else 0
                if normalized_u1 + normalized_u2 > 0:
                    harmonic_mean = 2 * (normalized_u1 * normalized_u2) / (normalized_u1 + normalized_u2)
                    if harmonic_mean > max_harmonic:
                        max_harmonic = harmonic_mean
                        best_set = subset_set

    # Find Pareto optima
    def dominates(a, b):
        _, a1, a2 = a
        _, b1, b2 = b
        return (a1 >= b1 and a2 >= b2) and (a1 > b1 or a2 > b2)
    #1) solution a must be at least as good as solution b for BOTH PLAYERS
    #2) solution A must be strictly better for AT LEAST ONE PLAYER

    pareto_optima = []
    for a in all_valid_subsets:
        if not any(dominates(b, a) for b in all_valid_subsets):
            #no solution 'b' dominates 'a' -> so 'a' is pareto optimal
            pareto_optima.append(a) #subsets not dominated by any other subset

    return {
        "max_u1": max_u1,
        "max_u2": max_u2,
        "max_total_utility": max_total_utility,
        "max_harmonic": max_harmonic,
        "pareto_optima_count": len(pareto_optima),
        #for efficient later check, store sets alone
        "pareto_optima_sets": [list(subset_set) for subset_set, _, _ in pareto_optima],
        "best_deal": best_set
    }
