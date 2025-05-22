import torch
from band_to_index import band_to_index, band_center_max


class ImagePadder(object):
    """Formatter that pads the images to have a fixed number of bands."""

    def __init__(self):
        self.nbands = max(band_to_index.values()) + 1

    def _check_bands(self, bands):
        for band in bands:
            if band not in band_to_index:
                raise ValueError(f"Invalid band: {band}. Valid bands are: {list(band_to_index.keys())}")

    def forward(self, image, bands):
        num_channels = self.nbands
        batch, _, height, width = image.shape

        # Check if bands are valid
        self._check_bands(bands)

        # Create a new image array with the correct number of channels
        padded_image = torch.zeros((batch, num_channels, height, width), dtype=image.dtype).to(image.device)

        # Create a list of new channel indices based on the order of bands
        new_channel_indices = [
            band_to_index[band] for band in bands if band in band_to_index
        ]

        # Vectorized assignment of the original channels to the new positions
        padded_image[:, new_channel_indices, :, :] = image[
            :, :len(new_channel_indices), :, :
        ]

        # Get boolean mask of channels that are present
        channel_mask = torch.zeros(num_channels, dtype=torch.bool).to(image.device)
        channel_mask[new_channel_indices] = True
        channel_mask = channel_mask.unsqueeze(0).expand(batch, -1)
        return padded_image, channel_mask

    def backward(self, padded_image, bands):
        # Check if bands are valid
        self._check_bands(bands)

        # Get the indices for the requested bands
        channel_indices = [band_to_index[b] for b in bands]

        # Select those channels along dim=1
        selected_image = padded_image[:, channel_indices, :, :]
        return selected_image
    

class CenterCrop(object):
    """Formatter that crops the images to have a fixed number of bands."""

    def __init__(self, crop_size: int = 96):
        self.crop_size = crop_size

    def __call__(self, image):
        _, _, height, width = image.shape
        start_x = (width - self.crop_size) // 2
        start_y = (height - self.crop_size) // 2
        return image[:, :, start_y : start_y + self.crop_size, start_x : start_x + self.crop_size]
    

class Clamp(object):
    """Formatter that clamps the images to a given range."""

    def __init__(self):
        self.clamp_dict = band_center_max

    def __call__(self, image):
        for i in range(image.shape[1]):
            band = image[:,i,:,:]
            image[:,i,:,:] = torch.clip(image[i], -self.clamp_dict[band], self.clamp_dict[band])
        return image
    

class RangeCompression(object):
    """Formatter that applies arcsinh compression on each band of the input."""

    def __init__(self, div_factor: float | int = 0.01, mult_factor: float | int = 10.0):
        self.div_factor = div_factor

    def forward(self, sample):
        return torch.arcsinh(sample / self.div_factor) * self.div_factor * self.mult_factor
    
    def backward(self, sample):
        return (torch.sinh(sample / self.div_factor) * self.div_factor) / self.mult_factor


class RescaleToLegacySurvey(object):
    """Formatter that rescales the images to have a fixed number of bands."""

    def __init__(self):
        pass

    def convert_zeropoint(self, zp: float) -> float:
        return 10.0 ** ((zp - 22.5) / 2.5)
    
    def reverse_zeropoint(self, scale: float) -> float:
        return 22.5 - 2.5 * torch.log10(scale)

    def forward(self, image, survey):
        zpscale = self.convert_zeropoint(27.0) if survey == "HSC" else 1.0
        image /= zpscale
        return image
    
    def backward(self, image, survey):
        zpscale = self._reverse_zeropoint(27.0) if survey == "HSC" else 1.0
        image *= zpscale
        return image