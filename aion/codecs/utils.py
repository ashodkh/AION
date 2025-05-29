from huggingface_hub import hub_mixin

from aion.codecs.base import Codec
from aion.modalities import Modality

ORIGINAL_CONFIG_NAME = hub_mixin.constants.CONFIG_NAME
ORIGINAL_PYTORCH_WEIGHTS_NAME = hub_mixin.constants.PYTORCH_WEIGHTS_NAME


def _override_config_and_weights_names(modality: type[Modality]):
    hub_mixin.constants.CONFIG_NAME = f"codecs/{modality.name}/{ORIGINAL_CONFIG_NAME}"
    hub_mixin.constants.PYTORCH_WEIGHTS_NAME = (
        f"codecs/{modality.name}/{ORIGINAL_PYTORCH_WEIGHTS_NAME}"
    )


def _reset_config_and_weights_names():
    hub_mixin.constants.CONFIG_NAME = ORIGINAL_CONFIG_NAME
    hub_mixin.constants.PYTORCH_WEIGHTS_NAME = ORIGINAL_PYTORCH_WEIGHTS_NAME


class PyTorchCodecHubMixin(hub_mixin.PyTorchModelHubMixin):
    """Mixin for PyTorch models that correspond to codecs.
    Codec don't have their own model repo.
    Instead they lie in the transformer model repo as subfolders.
    """

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        modality: type[Modality],
        *model_args,
        **kwargs,
    ):
        """Load a codec model from a pretrained model repository.

        Args:
            pretrained_model_name_or_path (str): The name or path of the pretrained
                model repository.
            *model_args: Additional positional arguments to pass to the model
                constructor.
            **kwargs: Additional keyword arguments to pass to the model
                constructor.

        Returns:
            The loaded codec model.
        """
        if not issubclass(cls, Codec):
            raise ValueError("Only codecs can be loaded using this method.")
        # TODO: Check modality is valid
        # Overwrite config and pytorch weights names to load codecs stored as submodels
        _override_config_and_weights_names(modality)
        model = super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
        _reset_config_and_weights_names()
        return model

    def save_pretrained(self, save_directory, *args, **kwargs):
        """Save the codec model to a pretrained model repository.

        Args:
            save_directory (str): The directory to save the model to.
            *args: Additional positional arguments to pass to the save method.
            **kwargs: Additional keyword arguments to pass to the save method.
        """
        if not isinstance(self, Codec):
            raise ValueError("Only codecs can be saved using this method.")
        # Construct the path to the codec subfolder
        codec_path = f"{save_directory}/codecs/{self.modality.name}"
        super().save_pretrained(codec_path, *args, **kwargs)
