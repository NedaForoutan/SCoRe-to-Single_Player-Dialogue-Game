import os
import json
import matplotlib.pyplot as plt
import numpy as np


def plot_instance_proposal_imbalance(summary_data, save_path=None):
    proposals = summary_data["proposals"]
    indices = list(range(len(proposals)))
    u1 = [p["normalized_u1"] for p in proposals]
    u2 = [p["normalized_u2"] for p in proposals]
    imbalance = [abs(u1[i] - u2[i]) for i in range(len(proposals))]

    plt.figure(figsize=(8, 5))
    colors = {"Player 1": "tab:blue", "Player 2": "tab:orange"}

    for i, p in enumerate(proposals):
        plt.scatter(indices[i], imbalance[i],
                    color=colors[p["by"]], s=70, edgecolor="black", zorder=3)

    plt.plot(indices, imbalance, linestyle="--", color="gray", alpha=0.7)

    agreement = summary_data.get("agreement", None)
    if agreement:
        agreement_idx = len(proposals) - 1
        plt.scatter(agreement_idx,
                    abs(agreement["normalized_u1"] - agreement["normalized_u2"]),
                    color="red", s=120, edgecolor="black", marker="*", zorder=4)

    plt.xlabel("Proposal index")
    plt.ylabel("Imbalance (|u1 - u2|)")
    plt.title(f"Balance Trajectory (Game {summary_data['game_id']})")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.xticks(np.arange(0, len(proposals)))

    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label='P1',
                   markerfacecolor=colors["Player 1"], markeredgecolor="black", markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='P2',
                   markerfacecolor=colors["Player 2"], markeredgecolor="black", markersize=8),
        plt.Line2D([0], [0], marker='*', color='w', label='Agreement',
                   markerfacecolor="red", markeredgecolor="black", markersize=12)
    ]
    plt.legend(handles=legend_handles, loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

def plot_instance_total_score(summary_data, save_path=None):
    """
    Plot the normalized harmonic mean (total score) over proposals.
    X-axis = proposal index.
    """

    proposals = summary_data["proposals"]

    indices = list(range(len(proposals)))
    scores = [p["normalized_harmonic_mean"] for p in proposals]

    plt.figure(figsize=(8, 5))

    # trajectory line
    plt.plot(indices, scores, linestyle="--", color="purple", alpha=0.7)
    plt.scatter(indices, scores, color="purple", edgecolor="black", s=70, zorder=3)

    # highlight final agreement
    agreement = summary_data.get("agreement", None)
    if agreement:
        agreement_idx = len(proposals) - 1
        plt.scatter(agreement_idx, agreement["normalized_harmonic_mean"],
                    color="red", marker="*", s=120, edgecolor="black", zorder=4)

    # --- Formatting ---
    plt.xlabel("Proposal index")
    plt.ylabel("Total score (normalized harmonic mean)")
    plt.title(f"Total Score Trajectory (Game {summary_data['game_id']})")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.xticks(np.arange(0, len(proposals)))

    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label='Score',
                   markerfacecolor="purple", markeredgecolor="black", markersize=8),
        plt.Line2D([0], [0], marker='*', color='w', label='Agreement',
                   markerfacecolor="red", markeredgecolor="black", markersize=12)
    ]
    plt.legend(handles=legend_handles, loc="best")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

def plot_instance_player_scores(summary_data, save_path=None):
    proposals = summary_data["proposals"]
    indices = list(range(len(proposals)))
    u1 = [p["normalized_u1"] for p in proposals]
    u2 = [p["normalized_u2"] for p in proposals]

    plt.figure(figsize=(8, 5))
    plt.plot(indices, u1, linestyle="--", color="tab:blue", alpha=0.7)
    plt.scatter(indices, u1, color="tab:blue", edgecolor="black", s=70, zorder=3)
    plt.plot(indices, u2, linestyle="--", color="tab:orange", alpha=0.7)
    plt.scatter(indices, u2, color="tab:orange", edgecolor="black", s=70, zorder=3)

    agreement = summary_data.get("agreement", None)
    if agreement:
        agreement_idx = len(proposals) - 1
        plt.scatter(agreement_idx, agreement["normalized_u1"],
                    color="tab:blue", marker="*", s=120, edgecolor="black", zorder=4)
        plt.scatter(agreement_idx, agreement["normalized_u2"],
                    color="tab:orange", marker="*", s=120, edgecolor="black", zorder=4)

    plt.xlabel("Proposal index")
    plt.ylabel("Normalized utility")
    plt.title(f"Utility Trajectories (Game {summary_data['game_id']})")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.xticks(np.arange(0, len(proposals)))

    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label='P1',
                   markerfacecolor="tab:blue", markeredgecolor="black", markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='P2',
                   markerfacecolor="tab:orange", markeredgecolor="black", markersize=8),
        plt.Line2D([0], [0], marker='*', color='w', label='Agreement',
                   markerfacecolor="grey", markeredgecolor="black", markersize=12)
    ]
    plt.legend(handles=legend_handles, loc="best")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

def plot_nb_substitutions(summary_data, save_path=None):
    """
    Plot normalized number of substitutions per proposal.
    Substitutions = symmetric difference in items compared to previous proposal.
    Normalization = relative to previous proposal size.
    The first proposal is ignored.
    """

    proposals = summary_data["proposals"]

    indices = list(range(1, len(proposals)))  # start from second proposal
    normalized_changes = []

    for i in range(1, len(proposals)):
        prev_items = set(proposals[i-1]["items"])
        curr_items = set(proposals[i]["items"])
        subs = len(prev_items ^ curr_items)  # add + remove
        prev_size = len(prev_items) if len(prev_items) > 0 else 1
        normalized_changes.append(subs / prev_size)

    plt.figure(figsize=(8, 5))

    # Bar plot of normalized substitutions
    plt.bar(indices, normalized_changes, color="teal", edgecolor="black", alpha=0.7)

    # Highlight final agreement (if not the first proposal)
    agreement = summary_data.get("agreement", None)
    if agreement and len(proposals) > 1:
        agreement_idx = len(proposals) - 1
        plt.scatter(agreement_idx, normalized_changes[-1],
                    color="red", marker="*", s=120, edgecolor="black", zorder=4)

    # --- Formatting ---
    plt.xlabel("Proposal index")
    plt.ylabel("Normalized substitutions (changes / prev. proposal size)")
    plt.title(f"Proposal Substitutions (Game {summary_data['game_id']})")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.xticks(np.arange(1, len(proposals)))

    legend_handles = [
        plt.Line2D([0], [0], marker='s', color='w', label='Substitutions',
                   markerfacecolor="teal", markeredgecolor="black", markersize=8),
        plt.Line2D([0], [0], marker='*', color='w', label='Agreement',
                   markerfacecolor="red", markeredgecolor="black", markersize=12)
    ]
    plt.legend(handles=legend_handles, loc="best")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

def plot_summary(summary_file, make_imbalance=False):
    """Load summary.json and conditionally create plots based on instance.json."""

    with open(summary_file, "r") as f:
        summary_data = json.load(f)

    base_dir = os.path.dirname(summary_file)

    # --- Load instance.json ---
    instance_file = os.path.join(base_dir, "instance.json")
    only_individual = False
    if os.path.exists(instance_file):
        try:
            with open(instance_file, "r") as f:
                instance_data = json.load(f)
                only_individual = instance_data.get("only_individual", False)
        except Exception as e:
            print(f"Failed to load {instance_file}: {e}")

    # --- Always plot total score ---
    save_path_score = os.path.join(base_dir, "score_plot.png")
    plot_instance_total_score(summary_data, save_path=save_path_score)

    # --- Plot player scores only if instance.json says so ---
    if not only_individual:
        save_path_player = os.path.join(base_dir, "player_score_plot.png")
        plot_instance_player_scores(summary_data, save_path=save_path_player)

    # --- Only sometimes plot imbalance ---
    if make_imbalance:
        save_path_imb = os.path.join(base_dir, "imbalance_plot.png")
        plot_instance_proposal_imbalance(summary_data, save_path=save_path_imb)

    # --- Always plot substitutions ---
    save_path_subs = os.path.join(base_dir, "substitutions_plot.png")
    plot_nb_substitutions(summary_data, save_path=save_path_subs)



def traverse_and_plot(base_path: str):
    keep_experiments = {
        "air_balloon_survival_en_negotiation_hard",
        "air_balloon_survival_en_reasoning_off_hard",
    }

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

            make_imbalance = experiment in keep_experiments

            for instance in os.listdir(exp_path):
                inst_path = os.path.join(exp_path, instance)
                if not os.path.isdir(inst_path):
                    continue

                summary_file = os.path.join(inst_path, "summary.json")
                if os.path.exists(summary_file):
                    try:
                        plot_summary(summary_file, make_imbalance=make_imbalance)
                        print(f"Plotted for {summary_file} "
                              f"(imbalance={'yes' if make_imbalance else 'no'})")
                    except Exception as e:
                        print(f"Failed to plot {summary_file}: {e}")

if __name__ == "__main__":

    if __name__ == "__main__":
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        for lang in ["results_en", "results_it", "results_de"]:
            base_path = os.path.join(base_dir, lang)
            if os.path.exists(base_path):
                print(f"\n--- Processing {lang} ---")
                traverse_and_plot(base_path)
            else:
                print(f"{base_path} does not exist, skipping.")

    traverse_and_plot(base_path)