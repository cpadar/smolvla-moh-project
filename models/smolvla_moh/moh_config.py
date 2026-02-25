# models/smolvla_moh/moh_config.py
# Configuration for SmolVLA + Mixture of Horizons

from dataclasses import dataclass, field
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


@dataclass
class SmolVLAMoHConfig(SmolVLAConfig):
    """
    Extends SmolVLAConfig with Mixture of Horizons parameters.
    
    Instead of a single chunk_size, MoH runs multiple horizons in parallel
    and fuses their action predictions.
    
    Args:
        horizons: List of action chunk lengths to use (e.g. [5, 15, 50])
        weighting: How to combine horizon outputs:
            - 'uniform': equal weight to all horizons
            - 'learned': learnable scalar weights per horizon
        moh_enabled: Toggle to disable MoH and fall back to standard SmolVLA
    """
    horizons: list = field(default_factory=lambda: [5, 15, 50])
    weighting: str = "learned"
    moh_enabled: bool = True

    def __post_init__(self):
        # Set chunk_size to the largest horizon so the model can handle all of them
        self.chunk_size = max(self.horizons)
        self.n_action_steps = max(self.horizons)
        super().__post_init__()