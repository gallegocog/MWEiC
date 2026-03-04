import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file


# ================================================================================
#   Configuration
# ================================================================================

def setup_logging(log_level: str) -> None:
    """Configure logging.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# ================================================================================
#   Dataset and embedding loading
# ================================================================================

def load_dataset(file: str, sep: str = ',') -> pd.DataFrame:
    """Load dataset from CSV/TSV file removing unnamed columns.
    
    Args:
        file: Path to the CSV/TSV file.
        sep: Column separator (default: ',').
        
    Returns:
        DataFrame with unnamed columns removed.
        
    Raises:
        FileNotFoundError: If dataset file does not exist.
    """
    try:
        if not Path(file).exists():
            raise FileNotFoundError(f"Dataset file not found: {file}")

        df = pd.read_csv(file, sep=sep)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        logging.info(f"Successfully loaded {file} with {len(df)} entries")
        
        return df

    except Exception as e:
        logging.error(f"Error loading dataset from {file}: {str(e)}")
        raise


def load_embeddings(file: str) -> dict[str, torch.Tensor]:
    """
    Load embeddings from .safetensor file.
    """
    try:
        if not Path(file).exists():
            raise FileNotFoundError(f"Embeddings file not found: {file}")

        embeddings = load_file(file)

        logging.info(f"Successfully loaded {file} with {len(embeddings)} tensors")
        logging.debug(f"Loaded embeddings shapes: "f"{ {k: v.shape for k, v in embeddings.items()} }")

        return embeddings

    except Exception as e:
        logging.error(f"Error loading embeddings from {file}: {e}")
        raise