from __future__ import annotations
import gzip
from pathlib import Path
from typing import Sequence
import numpy as np

from spininfer.data.dataset import Dataset
from spininfer.models.potts import PottsModel
from spininfer.models import Model

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
    """Build a length-128 ASCII lookup table mapping amino-acid characters to their integer state.

    Ambiguous residues (X, B, Z, J, U, O) map to the gap state; any other unknown character maps to -1.
    """
    table = np.full(128, _UNKNOWN, dtype=np.int64)
    for char, state in AMINO_ACID_DICT.items():
        table[ord(char)] = state
    for char in "XBZJUO":
        table[ord(char)] = AMINO_ACID_DICT["-"]
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


def _open_text(path: str):
    """Open `path` for reading as text, decompressing it first if it ends in .gz."""
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def _read_a3m(path: str) -> list[str]:
    """Parse an a3m/a2m file into its raw aligned sequences."""
    sequences = []
    current = []
    with _open_text(path) as f:
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

    return sequences


def _read_stockholm(path: str) -> list[str]:
    """Parse a Stockholm file into its raw aligned sequences, keeping '.' and lowercase so inserts can be removed."""
    sequences: dict[str, list[str]] = {}
    with _open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line == "//":
                continue
            name, seq = line.split(None, 1)
            sequences.setdefault(name, []).append(seq.strip())
    return ["".join(parts) for parts in sequences.values()]


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
    """Load an MSA file (format guessed from extension unless `fmt` is given) and return it as an integer-encoded array.

    Insert states (lowercase letters and '.', as in Pfam/HMMER alignments) are removed, so only the match columns remain.
    """
    fmt = fmt or _guess_format(MSA_path)

    if fmt in ("a3m", "a2m"):
        sequences = _read_a3m(MSA_path)
    elif fmt == "stockholm":
        sequences = _read_stockholm(MSA_path)
    else:
        from Bio import AlignIO
        with _open_text(MSA_path) as handle:
            alignment = AlignIO.read(handle, fmt)
        sequences = [str(record.seq) for record in alignment]

    sequences = ["".join(ch for ch in seq if not (ch.islower() or ch == ".")) for seq in sequences]
    return _sequences_to_array(sequences)


def potts_dataset_from_msa(MSA_path: str, fmt: str | None = None, backend: str = "numba") -> tuple[PottsModel, Dataset]:
    """Load an MSA file and return a PottsModel sized to it, together with its Dataset."""
    samples = load_convert_MSA_file(MSA_path, fmt=fmt)
    model = PottsModel(n_sites=samples.shape[1], n_states=N_AMINO_ACID_STATES, backend=backend)
    return model, Dataset(samples=samples, model=model)
