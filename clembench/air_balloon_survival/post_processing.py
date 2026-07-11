import os
import json
import ast
import re
from itertools import chain, combinations

def extract_summary(log_file: str, game_instance_file: str, output_file: str):
    with open(log_file, "r") as f:
        log_data = json.load(f)
    with open(game_instance_file, "r") as f:
        game_instance = json.load(f)

    item_weights = game_instance["item_weights"]
    p1_prefs = game_instance["player1_preferences"]
    p2_prefs = game_instance["player2_preferences"]

    # regex tags from instance.json
    proposal_tag = game_instance["proposal_tag"]
    agree_tag = game_instance["agree_tag"]
    refusal_tag = game_instance["refusal_tag"]

    # --- Compute Pareto sets ---
    n_items = len(item_weights)
    given_sets = game_instance.get("pareto_optima_sets", [])

    if given_sets:  # use provided Pareto sets
        pareto_sets = [frozenset(s) for s in given_sets]
    elif n_items <= 15:  # compute if feasible
        def all_subsets(iterable):
            s = list(iterable)
            return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

        def evaluate_set(items):
            total_weight = sum(item_weights.get(i, 0) for i in items)
            u1 = sum(p1_prefs.get(i, 0) for i in items)
            u2 = sum(p2_prefs.get(i, 0) for i in items)
            return total_weight, u1, u2

        all_candidates = []
        for subset in all_subsets(item_weights.keys()):
            total_weight, u1, u2 = evaluate_set(subset)
            if total_weight <= game_instance["max_weight"]:
                all_candidates.append((frozenset(subset), u1, u2))

        # filter Pareto frontier
        pareto_sets = []
        for s, u1, u2 in all_candidates:
            dominated = False
            for _, v1, v2 in all_candidates:
                if v1 >= u1 and v2 >= u2 and (v1 > u1 or v2 > u2):
                    dominated = True
                    break
            if not dominated:
                pareto_sets.append(s)
    else:  # too many items
        pareto_sets = None

    def is_pareto(items):
        if pareto_sets is None:
            return None
        return frozenset(items) in pareto_sets

    def evaluate_set(items):
        total_weight = sum(item_weights.get(i, 0) for i in items)
        u1 = sum(p1_prefs.get(i, 0) for i in items)
        u2 = sum(p2_prefs.get(i, 0) for i in items)
        return total_weight, u1, u2

    def score_set(items):
        """Return full scoring dict for a proposal/agreement/refusal set."""
        total_weight, u1, u2 = evaluate_set(items)

        normalized_u1 = (u1 / game_instance["max_u1"] * 100) if game_instance["max_u1"] > 0 else float("NaN")
        normalized_u2 = (u2 / game_instance["max_u2"] * 100) if game_instance["max_u2"] > 0 else float("NaN")

        if normalized_u1 + normalized_u2 > 0:
            harmonic_mean = 2 * (normalized_u1 * normalized_u2) / (normalized_u1 + normalized_u2)
        else:
            harmonic_mean = float("NaN")

        normalized_harmonic_mean = (
            harmonic_mean / game_instance["max_harmonic"] * 100
            if game_instance["max_harmonic"] > 0 else float("NaN")
        )

        if game_instance["negotiation_mode"] == "equal":
            normalized_harmonic_mean = normalized_u1

        if total_weight > game_instance["max_weight"]:
            normalized_u1 = 0
            normalized_u2 = 0
            u1 = 0
            u2 = 0
            normalized_harmonic_mean = 0

        return {
            "items": sorted(list(items)),
            "weight": total_weight,
            "utility_player1": u1,
            "utility_player2": u2,
            "normalized_u1": normalized_u1,
            "normalized_u2": normalized_u2,
            "harmonic_mean": harmonic_mean,
            "normalized_harmonic_mean": normalized_harmonic_mean,
            "pareto_optimum": is_pareto(items),
        }

    # DP knapsack for individual optima
    def dp_pseudo_poly_knapsack(weights, values, max_weight):
        item_list = list(weights.keys())
        n = len(item_list)
        dp = [[0] * (max_weight + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):
            item = item_list[i - 1]
            w_i = weights[item]
            v_i = values[item]
            for w in range(max_weight + 1):
                if w_i > w:
                    dp[i][w] = dp[i - 1][w]
                else:
                    dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - w_i] + v_i)

        selected_items = set()
        w = max_weight
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i - 1][w]:
                item = item_list[i - 1]
                selected_items.add(item)
                w -= weights[item]

        max_value = dp[n][max_weight]
        return max_value, selected_items

    summary = {
        "results_folder": log_data["meta"]["results_folder"],
        "game_id": log_data["meta"]["game_id"],
        "proposals": [],
        "refusals": [],
        "agreement": None,
        "max_weight": game_instance["max_weight"],
        "scores": {}
    }

    turns = 0
    for turn in log_data["turns"]:
        turns += 1
        for event in turn:
            if (event["from"] == "GM"
                and isinstance(event["action"].get("content"), str)
            ):

                # --- Proposals ---
                response = event["action"]["content"]
                # the actor is the one who sent the message to GM so not the receiver
                actor = "Player 1" if event["to"] == "Player 2" else "Player 2"

                try:
                    proposals_found = re.findall(proposal_tag, response, flags=re.DOTALL)
                    proposals_found = [p for p in proposals_found if "{'A', 'B', 'C'" not in p]
                    for match in proposals_found:
                        set_str = match.split(":", 1)[1].strip()
                        proposal = ast.literal_eval(set_str)
                        scored = score_set(proposal)
                        scored["by"] = actor
                        scored["turn"] = turns
                        summary["proposals"].append(scored)
                except Exception:
                    pass

                # --- Refusals ---
                try:
                    refusals_found = re.findall(refusal_tag, response, flags=re.DOTALL)
                    for match in refusals_found:
                        set_str = match.split(":", 1)[1].strip()
                        refusal = ast.literal_eval(set_str)
                        scored = score_set(refusal)
                        scored["by"] = actor
                        scored["turn"] = turns
                        summary["refusals"].append(scored)
                except Exception:
                    pass

    if log_data["final deal"]:
        scored_agreement = score_set(log_data["final deal"])
        scored_agreement["by"] = "Player"
        scored_agreement["turn"] = turns
        summary["agreement"] = scored_agreement

    # Store maxima
    summary["max_u1"] = game_instance.get("max_u1")
    summary["max_u2"] = game_instance.get("max_u2")
    summary["max_harmonic"] = game_instance.get("max_harmonic")

    # Ensure proposals are sorted temporally
    summary["proposals"] = sorted(summary["proposals"], key=lambda x: x["turn"])
    summary["nb_proposals"] = len(summary["proposals"])

    # --- Compute Pareto adherence rate ---
    pareto_flags = [p.get("pareto_optimum", None) for p in summary["proposals"]]
    valid_flags = [flag for flag in pareto_flags if flag is not None]
    pareto_adherence_rate = (
        sum(flag is True for flag in valid_flags) / len(valid_flags)
        if valid_flags else None
    )

    # --- Alternation rate ---
    alternations = 0
    proposers = [p["by"] for p in summary["proposals"]]
    for i in range(1, len(proposers)):
        if proposers[i] != proposers[i - 1]:
            alternations += 1
    alternation_rate = (
        alternations / (len(proposers) - 1)
        if len(proposers) > 1 else None
    )

    # --- Normalized substitutions ---
    normalized_changes = []
    for i in range(1, len(summary["proposals"])):
        prev_items = set(summary["proposals"][i - 1]["items"])
        curr_items = set(summary["proposals"][i]["items"])
        subs = len(prev_items ^ curr_items)
        prev_size = len(prev_items) if len(prev_items) > 0 else 1
        normalized_changes.append(subs / prev_size)

    avg_normalized_subs = float(sum(normalized_changes) / len(normalized_changes)) if normalized_changes else None
    summary["normalized_substitutions_per_proposal"] = normalized_changes

    # --- Stubbornness ---
    stubbornness = {"Player 1": None, "Player 2": None}
    total_repeats = 0
    total_props = len(summary["proposals"])

    for player in ["Player 1", "Player 2"]:
        player_props = [tuple(p["items"]) for p in summary["proposals"] if p["by"] == player]
        if player_props:
            seen = set()
            repeats = 0
            for prop in player_props:
                if prop in seen:
                    repeats += 1
                else:
                    seen.add(prop)
            stubbornness[player] = repeats / len(player_props)
            total_repeats += repeats

    stubbornness_total = (total_repeats / total_props) if total_props > 0 else None

    # --- Final deal scores ---
    if log_data["final deal"]:
        summary["scores"] = {
            "player1_score": summary["agreement"]["utility_player1"],
            "player2_score": summary["agreement"]["utility_player2"],
            "normalized_u1": summary["agreement"]["normalized_u1"],
            "normalized_u2": summary["agreement"]["normalized_u2"],
            "harmonic_mean": summary["agreement"]["harmonic_mean"],
            "normalized_harmonic_mean": summary["agreement"]["normalized_harmonic_mean"],
            "episode_score": 100.0 / turns if turns > 0 else 100.0,
            "pareto_optimum": summary["agreement"]["pareto_optimum"],
            "pareto_adherence_rate": pareto_adherence_rate,
            "alternation_rate": alternation_rate,
            "avg_normalized_substitutions": avg_normalized_subs,
            "stubbornness_player1": stubbornness["Player 1"],
            "stubbornness_player2": stubbornness["Player 2"],
            "stubbornness_total": stubbornness_total,
        }
    else:
        summary["scores"].update({
            "pareto_adherence_rate": pareto_adherence_rate,
            "alternation_rate": alternation_rate,
            "avg_normalized_substitutions": avg_normalized_subs,
            "stubbornness_player1": stubbornness["Player 1"],
            "stubbornness_player2": stubbornness["Player 2"],
            "stubbornness_total": stubbornness_total,
        })

    # --- Individual optima ---
    best_u1, set_u1 = dp_pseudo_poly_knapsack(item_weights, p1_prefs, game_instance["max_weight"])
    best_u2, set_u2 = dp_pseudo_poly_knapsack(item_weights, p2_prefs, game_instance["max_weight"])

    summary["player1_optimum"] = {
        "items": sorted(list(set_u1)),
        "utility_player1": best_u1,
        "utility_player2": sum(p2_prefs[i] for i in set_u1),
        "pareto_optimum": is_pareto(set_u1),
    }
    summary["player2_optimum"] = {
        "items": sorted(list(set_u2)),
        "utility_player1": sum(p1_prefs[i] for i in set_u2),
        "utility_player2": best_u2,
        "pareto_optimum": is_pareto(set_u2),
    }

    # Save JSON
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def traverse_results(base_path: str):
    for model in os.listdir(base_path):
        model_path = os.path.join(base_path, model)
        if not os.path.isdir(model_path):
            continue

        hot_air_path = os.path.join(model_path, "hot_air_balloon")
        if not os.path.exists(hot_air_path):
            continue

        for experiment in os.listdir(hot_air_path):
            exp_path = os.path.join(hot_air_path, experiment)
            if not os.path.isdir(exp_path):
                continue

            for instance in os.listdir(exp_path):
                inst_path = os.path.join(exp_path, instance)
                if not os.path.isdir(inst_path):
                    continue

                log_file = os.path.join(inst_path, "interactions.json")
                game_instance_file = os.path.join(inst_path, "instance.json")
                output_file = os.path.join(inst_path, "summary.json")

                if os.path.exists(log_file) and os.path.exists(game_instance_file):
                    try:
                        extract_summary(log_file, game_instance_file, output_file)
                        print(f"Wrote summary for {inst_path}")
                    except Exception as e:
                        print(f"Failed at {inst_path}: {e}")


if __name__ == "__main__":
    # assumes to be run from root
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results_en"))
    traverse_results(base_path)
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results_de"))
    traverse_results(base_path)
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results_it"))
    traverse_results(base_path)