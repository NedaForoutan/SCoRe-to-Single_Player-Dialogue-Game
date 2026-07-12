import warnings

# silence warnings
warnings.filterwarnings(
    "ignore",
    message=r".*attention mask API under `transformers\.modeling_attn_mask_utils`.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*`use_return_dict` is deprecated.*",
)

from .config import SCoReConfig, SINGLE_PLAYER_GAMES, MULTIPLAYER_GAMES

__all__ = ["SCoReConfig", "SINGLE_PLAYER_GAMES", "MULTIPLAYER_GAMES"]
