from __future__ import annotations
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd

from data.dataset import Dataset
from models import Model

AMINO_ACID_DICT = {"-": 0,
                   "A": 1,
                   "R": 2,
                   "N": 3,
                   "D": 4,
                   "C": 5,
                   "E": 6,
                   "Q": 7,
                   "G": 8,
                   "H": 9,
                   "I": 10,
                   "L": 11,
                   "K": 12,
                   "M": 13,
                   "F": 14,
                   "P": 15,
                   "S": 16,
                   "T": 17,
                   "W": 18,
                   "Y": 19,
                   "V": 20}

N_AMINO_ACID_STATES = len(AMINO_ACID_DICT)

_EXTENSION_TO_FORMAT = {
    ".fasta": "fasta", ".fa": "fasta", ".fna": "fasta",
    ".sto": "stockholm", ".stk": "stockholm",
    ".aln": "clustal", ".clustal": "clustal",
    ".phy": "phylip", ".phylip": "phylip",
    ".a3m": "a3m",
    ".a2m": "a2m",
}

_UNKNOWN = -1


def _build_lookup_table() -> np.ndarray:
    """Build a length-128 ASCII lookup table mapping amino-acid characters to their integer state (unknown chars -> -1)."""
    table = np.full(128, _UNKNOWN, dtype=np.int64)
    for char, state in AMINO_ACID_DICT.items():
        table[ord(char)] = state
    return table


_LOOKUP_TABLE = _build_lookup_table()

def _guess_format(path: str) -> str:
    """Infer the MSA file format from `path`'s extension, raising ValueError if it's not recognized."""
    suffix = Path(path).suffix.lower()
    if suffix not in _EXTENSION_TO_FORMAT:
        raise ValueError(
            f"could not infer MSA format from extension {suffix!r}; pass fmt explicitly. "
            f"Known extensions: {sorted(_EXTENSION_TO_FORMAT)}"
        )
    return _EXTENSION_TO_FORMAT[suffix]


def _read_a3m(path: str) -> list[str]:
    """Parse an a3m/a2m file into aligned sequences, dropping insert-state (lowercase/'.') columns."""
    sequences = []
    current = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                current = []
            else:
                current.append(line)
    if current:
        sequences.append("".join(current))

    return ["".join(ch for ch in seq if not (ch.islower() or ch == ".")) for seq in sequences]


def _sequences_to_array(sequences: Sequence[str]) -> np.ndarray:
    """Encode equal-length amino-acid sequences into an integer array via AMINO_ACID_DICT, raising ValueError on inconsistent lengths or unrecognized characters."""
    lengths = {len(seq) for seq in sequences}
    if len(lengths) != 1:
        raise ValueError(f"sequences have inconsistent lengths after parsing: {sorted(lengths)}")

    byte_array = np.array([list(seq.upper().encode("ascii")) for seq in sequences], dtype=np.uint8)
    out = _LOOKUP_TABLE[byte_array]

    if np.any(out == _UNKNOWN):
        bad_chars = sorted(chr(b) for b in np.unique(byte_array[out == _UNKNOWN]))
        raise ValueError(f"unrecognized characters in alignment: {bad_chars}")

    return out


def load_convert_MSA_file(MSA_path: str, fmt: str | None = None) -> np.ndarray:
    """Load an MSA file (format guessed from extension unless `fmt` is given) and return it as an integer-encoded array."""
    fmt = fmt or _guess_format(MSA_path)

    if fmt in ("a3m", "a2m"):
        sequences = _read_a3m(MSA_path)
    else:
        from Bio import AlignIO
        alignment = AlignIO.read(MSA_path, fmt)
        sequences = [str(record.seq) for record in alignment]

    return _sequences_to_array(sequences)


def potts_dataset_from_msa(model: Model, MSA_path: str, fmt: str | None = None) -> Dataset:
    """Load an MSA file and wrap it as a Dataset for `model`."""
    samples = load_convert_MSA_file(MSA_path, fmt=fmt)
    return Dataset(samples=samples, model=model)
