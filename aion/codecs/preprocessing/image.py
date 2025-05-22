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


class ImagePadder(object):
    """Formatter that pads the images to have a fixed number of bands."""

    def __init__(self):
        self.nbands = max(band_to_index.values()) + 1

    def forward(self, image, bands):
        num_channels = self.nbands
        batch, _, height, width = image.shape

        # Create a new image array with the correct number of channels
        padded_image = torch.zeros((batch, num_channels, height, width), dtype=image.dtype)
        padded_band = ["EMPTY"] * num_channels

        # Create a list of new channel indices based on the order of bands
        new_channel_indices = [
            band_to_index[band] for band in bands if band in band_to_index
        ]

        # Pad bands correctly
        for i, idx in enumerate(new_channel_indices):
            padded_band[idx] = bands[i]

        # Vectorized assignment of the original channels to the new positions
        padded_image[:, new_channel_indices, :, :] = image[
            :, :len(new_channel_indices), :, :
        ]

        # Get boolean mask of channels that are present
        channel_mask = np.zeros(num_channels, dtype="bool")
        channel_mask[new_channel_indices] = True
        return padded_image, padded_band, channel_mask

    def backward(self, padded_image, bands):
        # Get the indices for the requested bands
        channel_indices = [band_to_index[b] for b in bands]

        # Select those channels along dim=1
        selected_image = padded_image[:, channel_indices, :, :]

        return selected_image