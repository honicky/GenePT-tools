import gzip
import pickle
import zipfile
from pathlib import Path
from shutil import move
from typing import Union

import anndata as ad
import h5py
import pandas as pd
import requests
from scipy import sparse


def download_file(
    url: str, output_path: Union[Path, str, None] = None, chunk_size: int = 8192
) -> Path:
    """
    Download a file in chunks and return the Path to the downloaded file.
    If output_path is not provided, the filename will be extracted from the URL.

    Args:
        url: URL of the file to download
        output_path: Path where the file should be saved (optional)
        chunk_size: Size of chunks to download (default: 8192 bytes)

    Returns:
        Path: Path object pointing to the downloaded file

    Raises:
        requests.exceptions.RequestException: If download fails
        ValueError: If URL doesn't contain a filename and output_path is not provided
    """
    # If no output path is provided, extract filename from URL
    if output_path is None:
        # Get the filename from the URL
        filename = url.split("/")[-1].split("?")[0]  # Remove query parameters
        if not filename:
            raise ValueError(
                "Could not determine filename from URL. Please provide output_path."
            )
        output_path = Path(__file__).parent.parent / "data" / filename
    else:
        output_path = Path(output_path)

    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Download the file if it doesn't exist
    if not output_path.exists():
        print(f"Downloading file to {output_path}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
        print("Download complete!")
    else:
        print(f"File already exists at {output_path}")

    return output_path


def extract_gz(
    gz_path: Union[Path, str], extract_dir: Union[Path, str] = Path("data")
) -> Path:
    """
    Extract a gz file to the specified directory, skipping if the file already exists with the same size.

    Args:
        gz_path: Path to the gz file
        extract_dir: Directory where file should be extracted

    Returns:
        None
    """
    # Convert to Path objects
    gz_path = Path(gz_path)
    extract_dir = Path(extract_dir)

    # Create extraction directory if it doesn't exist
    extract_dir.mkdir(parents=True, exist_ok=True)

    # Determine output filename (remove .gz extension)
    output_path = extract_dir / gz_path.stem

    # Check if file needs to be extracted
    if output_path.exists():
        # Compare file sizes (uncompressed vs compressed)
        with gzip.open(gz_path, "rb") as f:
            # Seek to end to get uncompressed size
            f.seek(0, 2)
            uncompressed_size = f.tell()

        if output_path.stat().st_size == uncompressed_size:
            print(f"Skipping {gz_path.name} - already exists with same size")
            return output_path

    print(f"Extracting {gz_path.name}")
    with gzip.open(gz_path, "rb") as f_in:
        with open(output_path, "wb") as f_out:
            f_out.write(f_in.read())

    print("Extraction complete!")

    return output_path


def extract_zip(zip_path: Union[Path, str], extract_dir: Union[Path, str]) -> None:
    """
    Extract a zip file, skipping files that already exist with the same size.

    Args:
        zip_path: Path to the zip file
        extract_dir: Directory where files should be extracted

    Returns:
        None
    """
    # Convert to Path objects
    zip_path = Path(zip_path)
    extract_dir = Path(extract_dir)

    # Create extraction directory if it doesn't exist
    extract_dir.mkdir(parents=True, exist_ok=True)

    print("Extracting files...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        # Get list of files in archive
        for file_info in zip_ref.infolist():
            target_path = extract_dir / file_info.filename

            # Check if file needs to be extracted
            extract_file = True
            if target_path.exists():
                # Compare file sizes
                if target_path.stat().st_size == file_info.file_size:
                    extract_file = False
                    print(
                        f"Skipping {file_info.filename} - already exists with same size"
                    )

            if extract_file:
                print(f"Extracting {file_info.filename}")
                zip_ref.extract(file_info, extract_dir)

    print("Extraction complete!")


def setup_data_dir():
    """
    Set up the data directory by downloading and extracting required files.

    Downloads the GenePT embedding file from Zenodo and extracts it into
    the 'data' directory. If files already exist, downloading and extraction
    will be skipped.
    """

    # Create data directory if it doesn't exist
    repo_path = Path().absolute().parent
    data_dir = repo_path / "data"
    data_dir.mkdir(exist_ok=True)

    # Download URL
    url = "https://zenodo.org/records/10833191/files/GenePT_emebdding_v2.zip?download=1"
    zip_path = download_file(url)

    extract_zip(zip_path, data_dir)

    print("Setup finished!")


def get_gene_embeddings(model_name) -> dict:
    """
    Get the GenePT embeddings for a given model.
    """
    model_path_map = {
        "text-embedding-ada-002": "data/GenePT_emebdding_v2/GenePT_gene_embedding_ada_text.pickle",
        "text-embedding-3-large": "data/GenePT_emebdding_v2/GenePT_gene_protein_embedding_model_3_text.pickle.",
    }

    model_path = Path(__file__).parent.parent / model_path_map[model_name]
    with open(model_path, "rb") as f:
        return pickle.load(f)


def download_gdrive_file(url: str, output_path: Union[Path, str]) -> Path:
    """
    Download a file from Google Drive using gdown and optionally rename/move it.

    Args:
        url: Google Drive sharing URL
        output_path: Path where the file should be saved (optional)

    Returns:
        Path: Path object pointing to the downloaded file
    """
    import gdown

    output_path = Path(output_path)
    if not output_path.exists():

        # Download to current directory first
        downloaded_path = gdown.download(url, fuzzy=True)
        if not downloaded_path:
            raise RuntimeError("Download failed")
        temp_path = Path(downloaded_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Move file to final location if it's different from where it was downloaded
        if temp_path != output_path:
            print(f"Moving file to {output_path}...")
            move(str(temp_path), str(output_path))

    return output_path


def load_subset_anndata(file_path, start_row=0, n_rows=None, obs_columns=None):
    """
    Load a subset of rows from an h5ad file as an AnnData object efficiently.

    Args:
        file_path: Path to h5ad file
        start_row: Starting row index
        n_rows: Number of rows to load
        obs_columns: List of obs (cell metadata) columns to include. If None, includes all.

    Returns:
        AnnData object with the subset of data and selected obs metadata.
    """
    with h5py.File(file_path, "r") as f:
        # Determine total rows and number of rows to load
        total_rows = len(f["X"]["indptr"]) - 1
        if n_rows is None:
            n_rows = total_rows - start_row

        # Load components
        data, indices, indptr = _load_csr_matrix_components(f, start_row, n_rows)
        var_df = _load_var_metadata(f)
        obs_df = _load_obs_metadata(f, start_row, n_rows, obs_columns)

        # Create sparse matrix
        X_subset = sparse.csr_matrix(
            (data, indices, indptr), shape=(n_rows, len(var_df))
        )

    return ad.AnnData(X=X_subset, obs=obs_df, var=var_df)


def _load_csr_matrix_components(f, start_row, n_rows):
    """Helper function to load CSR matrix components from h5ad file."""
    indptr = f["X"]["indptr"][start_row : start_row + n_rows + 1]
    start_idx, end_idx = indptr[0], indptr[-1]

    data = f["X"]["data"][start_idx:end_idx]
    indices = f["X"]["indices"][start_idx:end_idx]
    indptr = indptr - start_idx  # Adjust indptr to start at 0

    return data, indices, indptr


def _load_var_metadata(f):
    """Helper function to load variable (gene) metadata."""
    var_dict = {}
    for key in f["var"].keys():
        item = f["var"][key]
        if isinstance(item, h5py.Dataset):
            var_dict[key] = item[:]
        elif isinstance(item, h5py.Group) and "categories" in item and "codes" in item:
            categories = [
                cat.decode("utf-8") if isinstance(cat, bytes) else cat
                for cat in item["categories"][:]
            ]
            codes = item["codes"][:]
            var_dict[key] = pd.Categorical.from_codes(codes, categories=categories)

    var_df = pd.DataFrame(var_dict)

    # Convert bytes to strings
    for col in var_df.columns:
        if var_df[col].dtype == object:
            var_df[col] = var_df[col].apply(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )

    if "feature_name" in var_df:
        var_df.index = var_df["feature_name"]

    return var_df


def _load_obs_metadata(f, start_row, n_rows, obs_columns=None):
    """Helper function to load observation (cell) metadata."""
    selected_obs_keys = obs_columns if obs_columns else list(f["obs"].keys())
    obs_dict = {}

    for key in selected_obs_keys:
        if key not in f["obs"]:
            continue

        item = f["obs"][key]
        if isinstance(item, h5py.Dataset):
            obs_dict[key] = item[start_row : start_row + n_rows]
        elif isinstance(item, h5py.Group) and "categories" in item and "codes" in item:
            categories = [
                cat.decode("utf-8") if isinstance(cat, bytes) else cat
                for cat in item["categories"][:]
            ]
            codes = item["codes"][start_row : start_row + n_rows]
            obs_dict[key] = pd.Categorical.from_codes(codes, categories=categories)

    return pd.DataFrame(obs_dict)
