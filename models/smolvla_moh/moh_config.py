# models/smolvla_moh/moh_config.py
from dataclasses import dataclass, field
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.configs.policies import PreTrainedConfig


@PreTrainedConfig.register_subclass("smolvla_moh")
@dataclass
class SmolVLAMoHConfig(SmolVLAConfig):
    """Extends SmolVLAConfig with Mixture of Horizons parameters."""

    horizons: list = field(default_factory=lambda: [10, 25, 50])

    def __post_init__(self):
        self.chunk_size = max(self.horizons)
        self.n_action_steps = max(self.horizons)
        # Ensure pretrained SmolVLA weights are loaded
        if not self.pretrained_path:
            self.pretrained_path = 'lerobot/smolvla_base'
        super().__post_init__()