from pydantic import BaseModel, Field

from nitrogen.flow_matching_transformer.nitrogen import NitroGen_Config
from nitrogen.mm_tokenizers import NitrogenTokenizerConfig

class ModalityConfig(BaseModel):
    frame_per_sample: int = 1 # number of context frames per sample
    frame_spacing: int | None = None # how many frames to skip between each frame. If None, use action_per_chunk
    action_per_chunk: int = 8
    action_shift: int = 1 # how many actions to skip between frame[i] and action_chunk[i]
    action_interleaving: bool = False # if True, action chunks will be interleaved with context frames and used by the model to predict the next actions
    token_set: str = "new"

    def model_post_init(self, __context):
        if self.frame_spacing is None:
            # Use object.__setattr__ because the model is frozen
            object.__setattr__(self, 'frame_spacing', self.action_per_chunk)
        assert self.action_shift >= 1, "Frame shift must be at least 1 for correct action indexing"


class CkptConfig(BaseModel):
    experiment_name: str = Field(..., description="Name of the experiment")

    model_cfg: NitroGen_Config = Field(..., description="Model configuration. This is a placeholder and should be replaced with the actual model config class.")
    tokenizer_cfg: NitrogenTokenizerConfig = Field(..., description="Tokenizer configuration. This is a placeholder and should be replaced with the actual tokenizer config class.")
    modality_cfg: ModalityConfig = Field(..., description="Modality configuration for the dataset mixture.")