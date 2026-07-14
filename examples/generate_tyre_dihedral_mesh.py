"""Edit the two division counts below, then run this file in fenicsx-env."""

from pathlib import Path

from cofebem.mesh.tyre_dihedral_hex import generate_tyre_mesh


AXIAL_DIVISIONS = 24
CIRCUMFERENTIAL_DIVISIONS = 32


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    generate_tyre_mesh(
        root / "geo_files" / "geometry_v2.geo",
        root / "results" / "tyre_dihedral" / "tyre_dihedral.msh",
        axial_divisions=AXIAL_DIVISIONS,
        circumferential_divisions=CIRCUMFERENTIAL_DIVISIONS,
        scale=1.0e-3,
    )
