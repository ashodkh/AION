#!/usr/bin/env python3
"""
Export Gaia Codecs Script

This script converts Gaia codecs from the old mmoma format to the new AION format,
uploads them to HuggingFace Hub, and generates test data for validation.

NOTE: This specifically requires access to the Rusty machine, and MMOMA tokenizer
training environment. It is not intended to be used for other purposes than maintenance.

Usage:
    python export_gaia_codecs.py [--test-only] [--upload-only] [--skip-upload]

Arguments:
    --test-only: Only run the testing phase (requires codecs already uploaded)
    --upload-only: Only run the conversion and upload phase
    --skip-upload: Skip the upload phase (useful for local testing)
"""

import os
import sys
import torch
import argparse
import tempfile
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path to import aion modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aion.codecs import ScalarCodec, LogScalarCodec, MultiScalarCodec
from huggingface_hub import create_repo, upload_folder


def get_codec_paths() -> Dict[str, str]:
    """Define the paths to the original codec checkpoints."""
    base_path = "/mnt/ceph/users/polymathic/mmoma/outputs/mmoma_codec_parallax_1024"

    return {
        "phot_g_mean_flux": f"{base_path}/phot_g_mean_flux_codec.pt",
        "phot_bp_mean_flux": f"{base_path}/phot_bp_mean_flux_codec.pt",
        "phot_rp_mean_flux": f"{base_path}/phot_rp_mean_flux_codec.pt",
        "parallax": f"{base_path}/parallax_codec.pt",
        "ra": f"{base_path}/ra_codec.pt",
        "dec": f"{base_path}/dec_codec.pt",
        "bp_coefficients": f"{base_path}/bp_coefficients_codec.pt",
        "rp_coefficients": f"{base_path}/rp_coefficients_codec.pt",
    }


def load_legacy_codecs(codec_paths: Dict[str, str]) -> Dict[str, Any]:
    """Load codecs from legacy checkpoint files."""
    try:
        from mmoma.codecs.common import get_codecs_from_checkpoints
    except ImportError:
        print("Error: mmoma library not found. Please ensure it's installed.")
        sys.exit(1)

    return get_codecs_from_checkpoints(codec_paths)


def convert_codec(
    name: str, legacy_codec: Any
) -> tuple[ScalarCodec | LogScalarCodec | MultiScalarCodec | None, str]:
    """
    Convert a legacy codec to AION format.

    Returns:
        Tuple of (converted_codec, error_message). If conversion fails,
        codec will be None and error_message will contain the reason.
    """
    codec_class = legacy_codec.__class__.__name__
    quantizer_class = legacy_codec._quantizer.__class__.__name__

    if codec_class != "ScalarIdentityCodec":
        return (
            None,
            f"Codec is not ScalarIdentityCodec (got {codec_class}), skipping...",
        )

    if quantizer_class == "ScalarLogReservoirQuantizer":
        # Check if it has min_log_value attribute
        min_log_value = getattr(legacy_codec.quantizer, "_min_log_value", None)
        if min_log_value is None:
            min_log_value = -3.0  # Default value

        new_codec = LogScalarCodec(
            modality=name,
            codebook_size=legacy_codec.quantizer._codebook_size,
            reservoir_size=legacy_codec.quantizer._reservoir_size,
            min_log_value=min_log_value,
        )
    elif quantizer_class == "ScalarReservoirQuantizer":
        new_codec = ScalarCodec(
            modality=name,
            codebook_size=legacy_codec.quantizer._codebook_size,
            reservoir_size=legacy_codec.quantizer._reservoir_size,
        )
    elif quantizer_class == "MultiScalarCompressedReservoirQuantizer":
        new_codec = MultiScalarCodec(
            modality=name,
            compression_fns=legacy_codec.quantizer.quantizers[0].compression_fns,
            decompression_fns=legacy_codec.quantizer.quantizers[0].decompression_fns,
            codebook_size=legacy_codec.quantizer.quantizers[0]._codebook_size,
            reservoir_size=legacy_codec.quantizer.quantizers[0]._reservoir_size,
            num_quantizers=legacy_codec.quantizer.num_quantizers,
        )
    else:
        return None, f"Unknown quantizer class: {quantizer_class}"

    # Load the state dict from the legacy codec
    new_codec.load_state_dict(legacy_codec.state_dict())

    return new_codec, ""


def test_codec_conversion(
    name: str,
    new_codec: ScalarCodec | LogScalarCodec | MultiScalarCodec,
    legacy_codec: Any,
) -> bool:
    """Test that the converted codec produces the same results as the legacy codec."""
    try:
        # For compressed codecs, we need to handle multi-channel input
        if isinstance(new_codec, MultiScalarCodec):
            # Get test values from the first channel's reservoir
            test_values = legacy_codec.quantizer.quantizers[0]._reservoir[:10]
            # Create multi-channel test data
            num_channels = legacy_codec.quantizer.num_quantizers
            test_values = test_values.unsqueeze(1).expand(-1, num_channels)
        else:
            # Get test values from the reservoir
            test_values = legacy_codec.quantizer._reservoir[:10]  # First 10 values

        # Test encoding
        encoded_values = new_codec.encode(new_codec.modality(value=test_values))

        # Test decoding
        decoded_values = new_codec.decode(encoded_values).value
        ref_decoded_values = legacy_codec.decode(encoded_values)[name]

        # Check if decoded values match (within tolerance)
        if not torch.allclose(ref_decoded_values, decoded_values, atol=1e-5):
            print(f"  ❌ Decoded values do not match for {name}")
            print(f"    Reference: {ref_decoded_values[:5]}")
            print(f"    Decoded: {decoded_values[:5]}")
            return False

        print(f"  ✅ Conversion test passed for {name}")
        return True

    except Exception as e:
        print(f"  ❌ Test failed for {name}: {e}")
        import traceback

        traceback.print_exc()
        return False


def upload_codec_to_hub(
    name: str,
    codec: ScalarCodec | LogScalarCodec | MultiScalarCodec,
    skip_upload: bool = False,
) -> bool:
    """Upload a codec to HuggingFace Hub."""
    if skip_upload:
        print(f"  ⏭️ Skipping upload for {name} (--skip-upload flag)")
        return True

    try:
        # Create temporary directory for the codec
        with tempfile.TemporaryDirectory() as temp_dir:
            codec_dir = Path(temp_dir) / f"{name}-dir"
            codec_dir.mkdir()

            # Save codec to temporary directory
            codec.save_pretrained(str(codec_dir))

            # Create repo ID
            repo_id = f"polymathic-ai/aion-gaia-{name.lower().replace('_', '-')}-codec"

            # Create repo (if it doesn't exist)
            create_repo(repo_id, exist_ok=True, private=True)

            # Upload folder
            upload_folder(
                folder_path=str(codec_dir),
                repo_id=repo_id,
                path_in_repo=".",
            )

        print(f"  ✅ Successfully uploaded {name} to {repo_id}")
        return True

    except Exception as e:
        print(f"  ❌ Failed to upload {name}: {e}")
        return False


def test_uploaded_codec(name: str, legacy_codec: Any, data_dir: Path) -> bool:
    """Test an uploaded codec by downloading and comparing with legacy codec."""
    try:
        # Determine codec class from quantizer
        quantizer_class = legacy_codec._quantizer.__class__.__name__

        if quantizer_class == "ScalarLogReservoirQuantizer":
            downloaded_codec = LogScalarCodec.from_pretrained(
                f"polymathic-ai/aion-gaia-{name.lower().replace('_', '-')}-codec"
            )
        elif quantizer_class == "ScalarReservoirQuantizer":
            downloaded_codec = ScalarCodec.from_pretrained(
                f"polymathic-ai/aion-gaia-{name.lower().replace('_', '-')}-codec"
            )
        elif quantizer_class == "MultiScalarCompressedReservoirQuantizer":
            downloaded_codec = MultiScalarCodec.from_pretrained(
                f"polymathic-ai/aion-gaia-{name.lower().replace('_', '-')}-codec"
            )
        else:
            print(f"  ❌ Unknown quantizer class for {name}: {quantizer_class}")
            return False

        # Generate test input values
        if quantizer_class == "MultiScalarCompressedReservoirQuantizer":
            # For compressed codecs, get values from first quantizer's reservoir
            input_batch = legacy_codec.quantizer.quantizers[0]._reservoir[::100]
            # Create multi-channel test data
            num_channels = legacy_codec.quantizer.num_quantizers
            input_batch = input_batch.unsqueeze(1).expand(-1, num_channels)
        else:
            # Every 100th value from reservoir
            input_batch = legacy_codec.quantizer._reservoir[::100]

        # Get reference outputs from legacy codec
        reference_encoded_batch = legacy_codec.encode({name: input_batch})
        reference_decoded_batch = legacy_codec.decode(reference_encoded_batch)[name]

        # Test downloaded codec
        with torch.no_grad():
            output = downloaded_codec.encode(
                downloaded_codec.modality(value=input_batch)
            )
            decoded_output = downloaded_codec.decode(output)

        # Remove NaN entries for comparison
        mask = ~torch.isnan(reference_decoded_batch)
        if mask.dim() > 1:
            mask = mask.all(dim=1)  # For multi-channel data

        # Verify encoding matches
        if not torch.allclose(output[mask], reference_encoded_batch[mask]):
            print(f"  ❌ Encoded output mismatch for {name}")
            return False

        # Verify decoding matches
        if not torch.allclose(
            decoded_output.value[mask], reference_decoded_batch[mask], atol=1e-5
        ):
            print(f"  ❌ Decoded output mismatch for {name}")
            return False

        # Save test data
        data_dir.mkdir(exist_ok=True)
        torch.save(input_batch[mask], data_dir / f"{name}_codec_input_batch.pt")
        torch.save(
            reference_encoded_batch[mask], data_dir / f"{name}_codec_encoded_batch.pt"
        )
        torch.save(
            reference_decoded_batch[mask], data_dir / f"{name}_codec_decoded_batch.pt"
        )

        print(f"  ✅ Downloaded codec test passed for {name}")
        return True

    except Exception as e:
        print(f"  ❌ Downloaded codec test failed for {name}: {e}")
        import traceback

        traceback.print_exc()
        return False


def convert_and_upload_codecs(skip_upload: bool = False) -> Dict[str, bool]:
    """Convert legacy codecs to AION format and upload to HuggingFace Hub."""
    print("🔄 Starting codec conversion and upload process...")

    # Get codec paths
    codec_paths = get_codec_paths()

    # Load legacy codecs
    print("📂 Loading legacy codecs...")
    legacy_codecs = load_legacy_codecs(codec_paths)

    results = {}

    # Convert and upload each codec
    for name, legacy_codec in legacy_codecs.items():
        print(f"\n🔧 Processing codec: {name}")

        # Convert codec
        new_codec, error = convert_codec(name, legacy_codec)
        if new_codec is None:
            print(f"  ⚠️ {error}")
            results[name] = False
            continue

        # Test conversion
        if not test_codec_conversion(name, new_codec, legacy_codec):
            results[name] = False
            continue

        # Upload to hub
        if not upload_codec_to_hub(name, new_codec, skip_upload):
            results[name] = False
            continue

        results[name] = True

    return results


def test_uploaded_codecs() -> Dict[str, bool]:
    """Test all uploaded codecs by downloading and validating."""
    print("🧪 Starting uploaded codec testing...")

    # Get codec paths and load legacy codecs
    codec_paths = get_codec_paths()
    legacy_codecs = load_legacy_codecs(codec_paths)

    # Test data directory
    data_dir = Path(__file__).parent.parent / "tests" / "test_data"

    results = {}

    # Test each uploaded codec
    for name, legacy_codec in legacy_codecs.items():
        print(f"\n🔍 Testing uploaded codec: {name}")

        # Skip if not a supported codec type
        codec_class = legacy_codec.__class__.__name__
        if codec_class != "ScalarIdentityCodec":
            print(f"  ⚠️ Skipping non-ScalarIdentityCodec: {codec_class}")
            results[name] = False
            continue

        results[name] = test_uploaded_codec(name, legacy_codec, data_dir)

    return results


def print_summary(results: Dict[str, bool], operation: str):
    """Print a summary of the operation results."""
    successful = sum(1 for success in results.values() if success)
    total = len(results)

    print(f"\n📊 {operation} Summary:")
    print(f"✅ Successful: {successful}/{total}")
    if successful < total:
        print("❌ Failed:")
        for name, success in results.items():
            if not success:
                print(f"  - {name}")


def main():
    """Main function to orchestrate the codec export process."""
    parser = argparse.ArgumentParser(description="Export Gaia codecs to AION format")
    parser.add_argument(
        "--test-only", action="store_true", help="Only run the testing phase"
    )
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Only run conversion and upload phase",
    )
    parser.add_argument(
        "--skip-upload", action="store_true", help="Skip the upload phase"
    )

    args = parser.parse_args()

    if args.test_only and args.upload_only:
        print("❌ Error: Cannot specify both --test-only and --upload-only")
        sys.exit(1)

    try:
        if args.test_only:
            # Only test uploaded codecs
            test_results = test_uploaded_codecs()
            print_summary(test_results, "Testing")
        elif args.upload_only:
            # Only convert and upload
            upload_results = convert_and_upload_codecs(args.skip_upload)
            print_summary(upload_results, "Conversion & Upload")
        else:
            # Run both phases
            print("🚀 Starting full export process...")

            # Phase 1: Convert and upload
            upload_results = convert_and_upload_codecs(args.skip_upload)
            print_summary(upload_results, "Conversion & Upload")

            if not args.skip_upload:
                # Phase 2: Test uploaded codecs
                test_results = test_uploaded_codecs()
                print_summary(test_results, "Testing")

        print("\n🎉 Export process completed!")

    except KeyboardInterrupt:
        print("\n⏹️ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
