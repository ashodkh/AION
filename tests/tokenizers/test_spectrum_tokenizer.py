import torch

from aion.codecs.tokenizers.spectrum import SpectrumCodec


def test_hf_previous_predictions(data_dir):
    codec = SpectrumCodec.from_pretrained("polymathic-ai/aion-spectrum-codec")

    input_batch = torch.load(data_dir / "SPECTRUM_input_batch.pt", weights_only=False)[
        "spectrum"
    ]
    reference_encoded_output = torch.load(
        data_dir / "SPECTRUM_encoded_batch.pt", weights_only=False
    )
    reference_decoded_output = torch.load(
        data_dir / "SPECTRUM_decoded_batch.pt", weights_only=False
    )

    with torch.no_grad():
        encoded_output = codec.encode(
            input_batch["flux"],
            input_batch["ivar"],
            input_batch["mask"],
            input_batch["lambda"],
        )
        assert encoded_output.shape == reference_encoded_output.shape
        assert torch.allclose(encoded_output, reference_encoded_output)

        flux, wavelength, mask = codec.decode(encoded_output)
        assert flux.shape == reference_decoded_output["spectrum"]["flux"].shape
        assert torch.allclose(
            flux, reference_decoded_output["spectrum"]["flux"], rtol=1e-3, atol=1e-4
        )
        assert wavelength.shape == reference_decoded_output["spectrum"]["lambda"].shape
        assert torch.allclose(
            wavelength,
            reference_decoded_output["spectrum"]["lambda"],
            rtol=1e-3,
            atol=1e-4,
        )
        assert mask.shape == reference_decoded_output["spectrum"]["mask"].shape
        assert torch.allclose(
            mask, reference_decoded_output["spectrum"]["mask"], rtol=1e-3, atol=1e-4
        )
