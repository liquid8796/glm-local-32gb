"""Choose only a freshly generated private or sharded miniature fixture."""
from .mini_weights import MiniWeights


def open_mini_storage(stack, bundle, directory, storage):
    if storage == "private":
        return stack.enter_context(MiniWeights(bundle)), None
    if storage != "safetensors":
        raise ValueError("Miniature storage must be private or safetensors")
    from .mini_safetensors import MiniSafetensorWeights, write_mini_shards
    destination = directory / "synthetic-sharded-model"
    exported = write_mini_shards(bundle, destination)
    return stack.enter_context(MiniSafetensorWeights(destination)), exported
