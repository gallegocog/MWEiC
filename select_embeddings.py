"""
Embedding Filtering for Word-in-Context (WiC) dataset with MWEs.

Filters pre-computed sentence embeddings to extract paired embeddings for
head and dependent tokens in sentence pairs. Processes safetensors files
containing layerwise embeddings and creates filtered versions based on
sentence ID pairs from datasets.
"""

import argparse
import ast
import logging
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

from utils import setup_logging, load_dataset

# ================================================================================
#   Constants
# ================================================================================

EMBEDDING_TYPES = ['head', 'dep', 'comp', 'cont', 'cls']


# ================================================================================
#   Functions
# ================================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Filter embeddings from MWEs study to extract head and '
                    'dep embeddings.'
    )
    parser.add_argument('--datasets_folder', type=str, default="datasets/",
                        help='Folder containing TSV datasets (default: "datasets/")')
    parser.add_argument('--embeddings_folder', type=str, default="full_embeddings/",
                        help='Folder containing embedding files (default: "full_embeddings/")')
    parser.add_argument('--output_folder', type=str, default="embeddings/",
                        help='Output directory (default: "embeddings/")')
    parser.add_argument('--log_level', type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help='Logging level (default: INFO)')
    return parser.parse_args()


def build_paired_tensor(
    embs: torch.Tensor,
    available_sent_ids: list[str],
    sent_id_pairs: list[tuple[str, str]],
) -> torch.Tensor:
    """Build paired embeddings tensor from sentence ID pairs.
    
    Args:
        embs: Tensor of shape (n_layers, n_sentences, vector_dim)
        available_sent_ids: List of sentence IDs in embeddings tensor
        sent_id_pairs: List of (sent_id_1, sent_id_2) tuples to extract
    
    Returns:
        Tensor of shape (n_layers, n_pairs, 2, vector_dim) with paired embeddings
    """
    # Map sentence IDs to tensor indices
    id_to_idx = {sid: i for i, sid in enumerate(available_sent_ids)}

    # Convert sentence ID pairs to tensor indices
    try:
        idx1 = torch.tensor(
            [id_to_idx[sid1] for sid1, _ in sent_id_pairs],
            dtype=torch.long,
            device=embs.device,
        )

        idx2 = torch.tensor(
            [id_to_idx[sid2] for _, sid2 in sent_id_pairs],
            dtype=torch.long,
            device=embs.device,
        )

    except KeyError as e:
        raise ValueError(f"Sentence ID not found in tensor: {e}")

    emb1 = embs[:, idx1, :]  # (n_layers, n_pairs, vector_dim)
    emb2 = embs[:, idx2, :]  # (n_layers, n_pairs, vector_dim)
    paired = torch.stack([emb1, emb2], dim=2)  # (n_layers, n_pairs, 2, vector_dim)

    return paired


def main():
    """Load datasets and filter embeddings by sentence pairs."""
    start_time = time.time()

    args = parse_arguments()
    setup_logging(args.log_level)

    logging.info("Starting embedding filtering")
    logging.info(f"Dataset: {args.datasets_folder}, Embeddings: {args.embeddings_folder}, Output: {args.output_folder}")

    # Load datasets
    logging.info(f"Loading datasets from {args.datasets_folder}")
    dataset_files = list(Path(args.datasets_folder).rglob("*.tsv"))
    
    if not dataset_files:
        logging.error(f"No .tsv files found in {args.datasets_folder}")
        return
    
    datasets = {
        file.stem: load_dataset(str(file), sep="\t")
        for file in dataset_files
    }

    # Process embeddings for each model
    embs_path = Path(args.embeddings_folder)
    embs_files = list(embs_path.rglob("*mean_pooled.safetensors"))

    for emb_file in tqdm(embs_files, desc="Filtering embeddings"):
        model_name = emb_file.parent.name

        try: 
            # Load embeddings and metadata
            embs = {}
            with safe_open(emb_file, framework="pt") as f:
                file_metadata = f.metadata()
                for key in EMBEDDING_TYPES:
                    embs[key] = f.get_tensor(key)
            available_sent_ids = ast.literal_eval(file_metadata["sent_ids"])
            
            # Extract dimensions and validate shape
            first_emb = embs[EMBEDDING_TYPES[0]]
            if first_emb.ndim != 3:
                raise ValueError(
                    f"Expected 3D embeddings (layers, sentences, features), "
                    f"got {first_emb.ndim}D in {emb_file}"
                )
            n_layers, vector_dim = first_emb.shape[0], first_emb.shape[2]

            # Filter embeddings for each dataset
            for dataset_name, dataset in datasets.items():
                sent_id_pairs = list(
                    zip(dataset["sent1_id"], dataset["sent2_id"])
                )
                filtered_embs = {
                    key: build_paired_tensor(
                        embs[key], available_sent_ids, sent_id_pairs
                    )
                    for key in EMBEDDING_TYPES
                }
                
                # Save filtered embeddings
                output_dir = Path(args.output_folder) / model_name
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / f"{dataset_name}.safetensors"

                metadata = {
                    "dataset": dataset_name,
                    "model_name": model_name,
                    "n_layers": str(n_layers),
                    "n_pairs": str(len(sent_id_pairs)),
                    "vector_dim": str(vector_dim),
                    "sent_ids": str(sent_id_pairs)
                }
                save_file(filtered_embs, str(output_file), metadata=metadata)
                logging.info(f"Saved embeddings to {output_file}")

        except Exception as e:
            logging.error(f"Error processing {model_name}: {e}")
            continue

    elapsed_time = time.time() - start_time
    logging.info(f"Embedding filtering completed in {elapsed_time:.2f}s")


if __name__ == "__main__":
    main()