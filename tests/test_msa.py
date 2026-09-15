import numpy as np
import pytest

from models.potts import PottsModel
from data.MSA import (AMINO_ACID_DICT, load_convert_MSA_file, potts_dataset_from_msa)


FASTA = """\
>seq1
AC-D
>seq2
-CVD
"""

STOCKHOLM = """\
# STOCKHOLM 1.0
seq1 AC-D
seq2 -CVD
//
"""

A3M = """\
>seq1
ACde-D
>seq2
A--D
"""


def _write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content)
    return str(path)


def test_fasta_matches_expected_states(tmp_path):
    path = _write(tmp_path, "align.fasta", FASTA)
    out = load_convert_MSA_file(path)

    expected = np.array([
        [AMINO_ACID_DICT["A"], AMINO_ACID_DICT["C"], AMINO_ACID_DICT["-"], AMINO_ACID_DICT["D"]],
        [AMINO_ACID_DICT["-"], AMINO_ACID_DICT["C"], AMINO_ACID_DICT["V"], AMINO_ACID_DICT["D"]],
    ])
    assert np.array_equal(out, expected)


def test_stockholm_parses_via_extension(tmp_path):
    path = _write(tmp_path, "align.sto", STOCKHOLM)
    out = load_convert_MSA_file(path)
    assert out.shape == (2, 4)


def test_explicit_format_overrides_extension(tmp_path):
    path = _write(tmp_path, "align.txt", FASTA)
    out = load_convert_MSA_file(path, fmt="fasta")
    assert out.shape == (2, 4)


def test_unknown_extension_without_fmt(tmp_path):
    path = _write(tmp_path, "align.mystery", FASTA)
    with pytest.raises(ValueError):
        load_convert_MSA_file(path)


def test_a3m_strips_insert_states(tmp_path):
    path = _write(tmp_path, "align.a3m", A3M)
    out = load_convert_MSA_file(path)

    expected = np.array([
        [AMINO_ACID_DICT["A"], AMINO_ACID_DICT["C"], AMINO_ACID_DICT["-"], AMINO_ACID_DICT["D"]],
        [AMINO_ACID_DICT["A"], AMINO_ACID_DICT["-"], AMINO_ACID_DICT["-"], AMINO_ACID_DICT["D"]],
    ])
    assert np.array_equal(out, expected)


def test_wrong_character_raises(tmp_path):
    bad_fasta = ">seq1\nACXD\n>seq2\nAC-D\n"
    path = _write(tmp_path, "align.fasta", bad_fasta)
    with pytest.raises(ValueError, match="unrecognized"):
        load_convert_MSA_file(path)


def test_potts_dataset_from_msa_(tmp_path):
    path = _write(tmp_path, "align.fasta", FASTA)
    model = PottsModel(n_sites=4, n_states=len(AMINO_ACID_DICT), backend="numpy")

    dataset = potts_dataset_from_msa(model, path)

    assert dataset.samples.shape == (2, 4)
    assert dataset.moments.mean_s.shape == (4, len(AMINO_ACID_DICT))
