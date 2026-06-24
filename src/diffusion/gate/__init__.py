"""diffusion.gate — unified chemical-validity gate for the DiffLinker -> MD pipeline.

This package consolidates the project's two previously-separate ad-hoc gate
scripts into one versioned, importable + CLI tool (Wave-0 task 0.2):

  * ``outputs/h4/eval_3dstrict.py``      -> :func:`gate_usable3d`
  * ``md/chem_validity_gate.py``         -> :func:`gate_ligand` (Level A) and
                                             :func:`gate_topology_valence` (Level B)

Behaviour/chemistry/thresholds/SMARTS are preserved exactly.  Notably,
``native_connected`` is measured on the RAW obabel-perceived SDF graph with NO
MST closure (hard project requirement, Master §19.16).

Public API
----------
- :func:`gate_ligand`            Level A: sanitize / single_fragment /
                                 undefined_stereo / maleimide_state / druglike
- :func:`gate_topology_valence`  Level B: per-atom valence + retained-thiol-H (§19.28)
- :func:`gate_usable3d`          3D-strict: parsed / native_connected / mmff_ok /
                                 has_valcit / has_urea / usable_3d
- :func:`gate_all`               fail-closed convenience: pass/fail + reasons
- :data:`MAX_VAL`, :func:`neutralise`, :data:`SMARTS_UREA`, :data:`SMARTS_VALCIT`

CLI: ``python -m diffusion.gate.chem_gate {ligand|topology|usable3d} ...``
"""
from __future__ import annotations

from .chem_gate import (
    MAX_VAL,
    SMARTS_UREA,
    SMARTS_VALCIT,
    gate_all,
    gate_ligand,
    gate_topology_valence,
    gate_usable3d,
    neutralise,
)

__version__ = "1.1.0"  # 1.1.0: Level-B RETAINED-THIOL-H (S-H) detection (§19.28)

__all__ = [
    "__version__",
    "gate_ligand",
    "gate_topology_valence",
    "gate_usable3d",
    "gate_all",
    "neutralise",
    "MAX_VAL",
    "SMARTS_UREA",
    "SMARTS_VALCIT",
]
