"""Read-only pooled prefix features for a compact, experimental pi RLT encoder.

The action policy is unchanged. A separate frozen forward pass extracts contextual
visual/language tokens. This is a local adaptation, not a published RLT checkpoint.
"""
from types import MethodType
import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import model as model_api
from openpi.models.pi0 import make_attn_mask
from openpi.shared.nnx_utils import module_jit


def pooled_prefix(model, observation):
    observation = model_api.preprocess_observation(None, observation, train=False)
    assert tuple(observation.images) == ('base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb')
    tokens, mask, ar = model.embed_prefix(observation)
    outputs, _ = model.PaliGemma.llm([tokens, None], mask=make_attn_mask(mask, ar),
                                    positions=jnp.cumsum(mask, axis=1) - 1)
    full = outputs[0].astype(jnp.float32)
    assert full.shape[-1] == 2048 and full.shape[1] > 768
    patches = []
    for start in (0, 256):
        # Preserve a 4x4 spatial grid from each 16x16 camera patch grid.
        grid = full[:, start:start + 256].reshape(full.shape[0], 4, 4, 4, 4, 2048)
        patches.append(grid.mean(axis=(2, 4)).reshape(full.shape[0], 16, 2048))
    text_mask = mask[:, 768:, None].astype(jnp.float32)
    text = (full[:, 768:] * text_mask).sum(axis=1, keepdims=True) / jnp.maximum(text_mask.sum(axis=1, keepdims=True), 1)
    return jnp.concatenate([*patches, text], axis=1)


class PrefixFeatures:
    def __init__(self, policy):
        self.policy = policy
        self.extract = module_jit(MethodType(pooled_prefix, policy._model))

    def __call__(self, element):
        inputs = self.policy._input_transform(jax.tree.map(lambda x: x, element))
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
        value = np.asarray(self.extract(model_api.Observation.from_dict(inputs))[0], dtype=np.float32)
        if value.shape != (33, 2048) or not np.isfinite(value).all():
            raise ValueError('Invalid frozen pi prefix features')
        return value.astype(np.float16)
