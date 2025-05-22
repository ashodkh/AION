import torch
import numpy as np


# Keeps track of the band indices for HSC and DES bands
band_to_index = {
    "HSC-G": 0,
    "HSC-R": 1,
    "HSC-I": 2,
    "HSC-Z": 3,
    "HSC-Y": 4,
    "DES-G": 5,
    "DES-R": 6,
    "DES-I": 7,
    "DES-Z": 8,
}


def range_compression(
    sample: torch.Tensor, div_factor: float | int = 0.01
) -> torch.Tensor:
    """Applies arcsinh compression on each band of the input."""
    return torch.arcsinh(sample / div_factor) * div_factor


def reverse_range_compression(
    sample: torch.Tensor, div_factor: float | int = 0.01
) -> torch.Tensor:
    """Undoes arcsinh compression on each band of the input."""
    return torch.sinh(sample / div_factor) * div_factor


class ImagePadder(object):
    """Formatter that pads the images to have a fixed number of bands."""

    def __init__(self, version='latest'):
        self.version = version
        self.nbands = max(band_to_index.values()) + 1

    def forward(self, sample):
        image = sample["image"]["flux"]
        ivar = sample["image"]["ivar"]
        mask = sample["image"]["mask"]
        bands = sample["image"]["band"]

        num_channels = self.nbands
        _, height, width = image.shape

        # Create a new image array with the correct number of channels
        padded_image = np.zeros((num_channels, height, width), dtype=image.dtype)
        padded_mask = np.zeros((num_channels, height, width), dtype=mask.dtype)
        padded_ivar = np.zeros((num_channels, height, width), dtype=ivar.dtype)
        padded_band = ["EMPTY"] * num_channels

        # Create a list of new channel indices based on the order of bands
        new_channel_indices = [
            band_to_index[band] for band in bands if band in band_to_index
        ]

        # Pad bands correctly
        for i, idx in enumerate(new_channel_indices):
            padded_band[idx] = bands[i]

        # Vectorized assignment of the original channels to the new positions
        padded_image[new_channel_indices, :, :] = image[
            : len(new_channel_indices), :, :
        ]
        padded_ivar[new_channel_indices, :, :] = (
            ivar[: len(new_channel_indices), :, :] if ivar is not None else None
        )

        # Handle mask
        if mask is not None:
            # if mask is one channel, broadcast it to all channels
            if len(mask.shape) == 2:
                mask = np.broadcast_to(mask, (num_channels, height, width))

            padded_mask[new_channel_indices, :, :] = mask[
                : len(new_channel_indices), :, :
            ]
        else:
            padded_mask = None

        # Get boolean mask of channels that are present
        channel_mask = np.zeros(num_channels, dtype="bool")
        channel_mask[new_channel_indices] = True

        processed = {
            "image": {
                "flux": padded_image,
                "band": padded_band,
                "channel_mask": channel_mask,
                "ivar": padded_ivar,
                "mask": padded_mask,
            }
        }

        # Get any additional keys in the image
        additional_img_keys = sample["image"].keys() - {
            "flux",
            "ivar",
            "mask",
            "band",
        }
        for key in additional_img_keys:
            processed["image"][key] = sample["image"][key]

        additional_keys = sample.keys() - {"image"}
        for key in additional_keys:
            processed[key] = sample[key]

        return processed

    def backward(self, sample):
        # Extract the padded images, ivars, masks, and bands
        padded_images = sample["image"]["flux"]
        padded_ivars = sample["image"].get("ivar", None)
        padded_masks = sample["image"].get("mask", None)
        padded_bands = sample["image"]["band"][
            0
        ]  # Assuming bands are the same across the batch
        channel_mask = sample["image"]["channel_mask"]
        device = padded_images.device

        N, _, height, width = padded_images.shape

        # Get the indices of the original bands (i.e., the bands that are not "EMPTY")
        original_band_indices = [
            i for i, band in enumerate(padded_bands) if band != "EMPTY"
        ]
        original_bands = [padded_bands[i] for i in original_band_indices]

        # Create new arrays to store the unpadded images, ivars, and masks
        num_original_channels = len(original_band_indices)
        unpadded_images = torch.zeros(
            (N, num_original_channels, height, width), dtype=padded_images.dtype
        ).to(device)
        unpadded_ivars = (
            torch.zeros(
                (N, num_original_channels, height, width), dtype=padded_ivars.dtype
            )
            if padded_ivars is not None
            else None
        ).to(device)
        unpadded_masks = (
            torch.zeros(
                (N, num_original_channels, height, width), dtype=padded_masks.dtype
            )
            if padded_masks is not None
            else None
        ).to(device)

        # Unpad the images, ivars, and masks based on the original bands
        for i in range(N):
            unpadded_images[i, :, :, :] = padded_images[i, original_band_indices, :, :]
            if padded_ivars is not None:
                unpadded_ivars[i, :, :, :] = padded_ivars[
                    i, original_band_indices, :, :
                ]
            if padded_masks is not None:
                unpadded_masks[i, :, :, :] = padded_masks[
                    i, original_band_indices, :, :
                ]

        # Recover the original band structure (undoing the previous band transposition)
        if isinstance(sample["image"]["band"][0][0], list):
            # Convert from list of lists to the original format
            unpadded_bands = list(map(list, zip(*[[original_bands] * N])))
        else:
            unpadded_bands = [[band for band in original_bands]] * N

        # Prepare the output
        unpadded_sample = {
            "image": {
                "flux": unpadded_images,
                "band": unpadded_bands,  # Recovered unpadded bands in the correct format
            }
        }

        # Add unpadded ivars and masks if present
        if unpadded_ivars is not None:
            unpadded_sample["image"]["ivar"] = unpadded_ivars
        if unpadded_masks is not None:
            unpadded_sample["image"]["mask"] = unpadded_masks

        # Handle additional keys
        additional_image_keys = sample["image"].keys() - {
            "flux",
            "ivar",
            "mask",
            "band",
            "channel_mask",
        }
        for key in additional_image_keys:
            unpadded_sample["image"][key] = sample["image"][key]

        additional_keys = sample.keys() - {"image"}
        for key in additional_keys:
            unpadded_sample[key] = sample[key]

        return unpadded_sample
