import numpy as np
import pytest

from spininfer.models.potts import PottsModel
from spininfer.data.msa import (AMINO_ACID_DICT, load_convert_msa_file, potts_dataset_from_msa)


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
    out = load_convert_msa_file(path)

    expected = np.array([
        [AMINO_ACID_DICT["A"], AMINO_ACID_DICT["C"], AMINO_ACID_DICT["-"], AMINO_ACID_DICT["D"]],
        [AMINO_ACID_DICT["-"], AMINO_ACID_DICT["C"], AMINO_ACID_DICT["V"], AMINO_ACID_DICT["D"]],
    ])
    assert np.array_equal(out, expected)


def test_stockholm_parses_via_extension(tmp_path):
    path = _write(tmp_path, "align.sto", STOCKHOLM)
    out = load_convert_msa_file(path)
    assert out.shape == (2, 4)


def test_explicit_format_overrides_extension(tmp_path):
    path = _write(tmp_path, "align.txt", FASTA)
    out = load_convert_msa_file(path, fmt="fasta")
    assert out.shape == (2, 4)


def test_unknown_extension_without_fmt(tmp_path):
    path = _write(tmp_path, "align.mystery", FASTA)
    with pytest.raises(ValueError):
        load_convert_msa_file(path)


def test_a3m_strips_insert_states(tmp_path):
    path = _write(tmp_path, "align.a3m", A3M)
    out = load_convert_msa_file(path)

    expected = np.array([
        [AMINO_ACID_DICT["A"], AMINO_ACID_DICT["C"], AMINO_ACID_DICT["-"], AMINO_ACID_DICT["D"]],
        [AMINO_ACID_DICT["A"], AMINO_ACID_DICT["-"], AMINO_ACID_DICT["-"], AMINO_ACID_DICT["D"]],
    ])
    assert np.array_equal(out, expected)


def test_wrong_character_raises(tmp_path):
    bad_fasta = ">seq1\nAC*D\n>seq2\nAC-D\n"
    path = _write(tmp_path, "align.fasta", bad_fasta)
    with pytest.raises(ValueError, match="unrecognized"):
        load_convert_msa_file(path)


def test_ambiguous_residues_become_gaps(tmp_path):
    path = _write(tmp_path, "align.fasta", ">seq1\nAXBD\n>seq2\nAC-D\n")
    out = load_convert_msa_file(path)
    assert out[0, 1] == AMINO_ACID_DICT["-"]
    assert out[0, 2] == AMINO_ACID_DICT["-"]


def test_stockholm_strips_insert_states(tmp_path):
    stockholm_with_inserts = "# STOCKHOLM 1.0\nseq1 AC..DE\nseq2 ACgkDE\nseq3 A-..DE\n//\n"
    path = _write(tmp_path, "align.sto", stockholm_with_inserts)
    out = load_convert_msa_file(path)

    expected = np.array([[AMINO_ACID_DICT[c] for c in row] for row in ["ACDE", "ACDE", "A-DE"]])
    assert np.array_equal(out, expected)


def test_gzipped_file_matches_uncompressed(tmp_path):
    import gzip
    plain = _write(tmp_path, "align.sto", STOCKHOLM)
    zipped = tmp_path / "align.sto.gz"
    with gzip.open(zipped, "wt") as f:
        f.write(STOCKHOLM)
    assert np.array_equal(load_convert_msa_file(str(zipped), fmt="stockholm"), load_convert_msa_file(plain))


def test_potts_dataset_from_msa_(tmp_path):
    path = _write(tmp_path, "align.fasta", FASTA)
    
    model, dataset = potts_dataset_from_msa(path, backend="numpy")

    assert isinstance(model, PottsModel)
    assert (model.n_sites, model.n_states) == (4, len(AMINO_ACID_DICT))
    assert dataset.samples.shape == (2, 4)
    assert dataset.moments.mean_s.shape == (4, len(AMINO_ACID_DICT))
