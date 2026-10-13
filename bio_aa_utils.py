"""Compatibility shim for removed Biopython helpers.

``Bio.PDB.Polypeptide.three_to_one`` (and the Bio.SeqUtils equivalents) were
removed in Biopython >= 1.84 while DiffSBDD still references them. This module
reimplements the function on top of the surviving IUPACData mapping tables so
the code runs on both old and new Biopython.

``is_aa`` still exists in ``Bio.PDB.Polypeptide`` (its recent signature accepts
a residue name string or a Residue object, matching how DiffSBDD calls it), so
it is imported from there directly in the call sites.
"""
from Bio.Data.IUPACData import protein_letters_3to1


def three_to_one(aa, undef_code='X'):
    """Convert a three-letter residue code to a one-letter code.

    Mirrors the behaviour of ``Bio.PDB.Polypeptide.three_to_one`` (removed in
    newer Biopython): match is case-insensitive, non-standard codes map to
    ``undef_code`` ('X' by default).
    """
    if not isinstance(aa, str) or not aa.strip():
        return undef_code
    return protein_letters_3to1.get(aa.capitalize(), undef_code)