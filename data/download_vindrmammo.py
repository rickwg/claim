import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from vindrmammo_py import create_dataset_downloader

load_dotenv()


def count_files(directory: Path, extension: str) -> int:
    if not directory.exists():
        return 0
    return sum(1 for _ in directory.rglob(f"*{extension}"))


def download_vindrmammo_dataset(data_dir="./vindrmammo_data", max_images=100):
    data_path = Path(data_dir).resolve()
    username = os.getenv("PHYSIONET_USERNAME")
    password = os.getenv("PHYSIONET_PASSWORD")
    if not username or not password:
        raise RuntimeError(
            "Missing PHYSIONET credentials. Set PHYSIONET_USERNAME and PHYSIONET_PASSWORD."
        )

    dataset = create_dataset_downloader(
        root_directory=str(data_path),
        username=username,
        password=password,
    )
    dataset.download_metadata_files()
    dicom_dir = data_path / "dicom"
    before_dicom_count = count_files(dicom_dir, ".dicom")
    requested_max_images = max_images if max_images and max_images > 0 else None
    successful, failed = dataset.download_complete_dataset(
        use_wget_bulk=False,
        parallel_workers=4,
        max_images=requested_max_images,
    )
    after_dicom_count = count_files(dicom_dir, ".dicom")
    new_dicom_count = after_dicom_count - before_dicom_count
    print(
        f"Download complete: {successful} successful, {failed} failed, "
        f"{new_dicom_count} new DICOM files ({after_dicom_count} total)"
    )
    if successful > 0 and new_dicom_count == 0:
        print(
            "No new DICOM files were downloaded because the requested files already existed."
        )
    if after_dicom_count == 0:
        raise RuntimeError(
            f"No DICOM files found under {dicom_dir}. Check credentials and data directory."
        )


def convert_dicom_files_to_png(data_dir="./vindrmammo_data"):
    data_path = Path(data_dir).resolve()
    dataset = create_dataset_downloader(str(data_path))
    dicom_dir = data_path / "dicom"
    png_dir = data_path / "png"
    before_png_count = count_files(png_dir, ".png")
    print("Converting DICOM files to PNG...")
    conv_successful, conv_failed = dataset.convert_all_dicom_files_to_png()
    after_png_count = count_files(png_dir, ".png")
    new_png_count = after_png_count - before_png_count
    print(
        f"Conversion complete: {conv_successful} successful, {conv_failed} failed, "
        f"{new_png_count} new PNG files ({after_png_count} total)"
    )
    if conv_successful > 0 and new_png_count == 0:
        print(
            "No new PNG files were created because PNG files for those DICOMs already existed."
        )
    if count_files(dicom_dir, ".dicom") > 0 and after_png_count == 0:
        raise RuntimeError(
            f"No PNG files found under {png_dir} after conversion despite available DICOM files."
        )


def main(data_dir="./vindrmammo_data", max_images=20):
    resolved_data_dir = str(Path(data_dir).resolve())
    print(f"Using data directory: {resolved_data_dir}")
    download_vindrmammo_dataset(data_dir=resolved_data_dir, max_images=max_images)
    convert_dicom_files_to_png(data_dir=resolved_data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and convert VinDr-Mammo dataset"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./vindrmammo_data",
        help="Directory to store the VinDr-Mammo dataset (default: ./vindrmammo_data)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=20,
        help="Number of images to download. Use 0 to download all images from metadata CSV.",
    )

    args = parser.parse_args()
    main(data_dir=args.data_dir, max_images=args.max_images)
