"""
Data containers passed from rollout -> loss.

An *attempt* is one full play-through of a game instance by the learner. In a
dialogue game the learner speaks on several turns, so an attempt is a list of
(prompt_ids, completion_ids) segments -- one per learner turn. REINFORCE assigns
the (shaped) episode reward to every token the learner generated.

A SCoRe *episode* is two attempts on the SAME game instance:
    y1 : first attempt
    y2 : second attempt, after a self-correction/reflection prompt
Token ids are captured at generation time (from the generation backend) so the
loss can recompute per-token log-probs on the trainable model without any
chat-template re-alignment risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TurnTokens:
    """One learner turn: the exact prompt it saw and the tokens it produced."""
    prompt_ids: list[int]
    completion_ids: list[int]


@dataclass
class Attempt:
    turns: list[TurnTokens] = field(default_factory=list)
    reward: float = 0.0        # normalized to [0, 1] by the reward layer
    success: bool = False
    aborted: bool = False
    raw_score: float = 0.0     # unnormalized BENCH_SCORE (for logging)

    def n_completion_tokens(self) -> int:
        return sum(len(t.completion_ids) for t in self.turns)

    def has_tokens(self) -> bool:
        return self.n_completion_tokens() > 0


@dataclass
class Episode:
    game: str
    game_id: tuple[str, int]  # (experiment_name, game_id) -- unique instance key
    y1: Attempt = field(default_factory=Attempt)
    y2: Attempt = field(default_factory=Attempt)

    def shaped_rewards(self, alpha: float) -> tuple[float, float]:
        """Stage II reward shaping: r2_shaped = r2 + alpha * (r2 - r1)."""
        r1 = self.y1.reward
        r2 = self.y2.reward
        return r1, r2 + alpha * (r2 - r1)

    def corrected(self) -> bool:
        """True self-correction: y1 failed, y2 succeeded."""
        return (not self.y1.success) and self.y2.success

    def regressed(self) -> bool:
        """The SCoRe failure mode: y1 succeeded, y2 broke it."""
        return self.y1.success and (not self.y2.success)
