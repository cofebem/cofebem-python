# Graded road-facing tyre mesh

`cofebem.mesh.tyre_dihedral_hex` supports two circumferential layouts:

- `uniform`: the original equally spaced, globally `D_n`-symmetric mesh;
- `graded`: a locally fine road-facing sector and a coarse back side.

The graded layout follows `geo_files/improved_mesh.svg`:

```text
                 240 deg coarse zone
             /-------------------------\
        30 deg transition         30 deg transition
              \                       /
               \--- 60 deg fine -----/
                    road / floor
```

The fine zone is centered on the lowest tyre meridian. Its surface grid is
regular and mirror-symmetric about the road-facing centreline. Each adjacent
30-degree zone uses a monotone target-size progression. The remaining 240
degrees is unstructured and coarse.

## Input semantics

For a graded mesh:

```json
{
  "mesh": {
    "circumferential_layout": "graded",
    "circumferential_divisions": 118,
    "coarsening_factor": 6.0
  },
  "compliance": {
    "strategy": "hmatrix",
    "load": null
  },
  "hmatrix": {
    "local_symmetry_tag": 204,
    "local_symmetry_validation_columns": 4,
    "local_symmetry_tolerance": 0.05,
    "local_symmetry_strict": false
  }
}
```

`circumferential_divisions` is the number of surface intervals in the
60-degree fine zone, not the full-circle count. `axial_divisions` is likewise
enforced on that fine road-facing surface. `coarsening_factor` is an upper
bound on the coarse-to-fine isotropic tetrahedral target size. The transition
target varies linearly; a curvature-safety cap can make the achieved ratio
smaller on deliberately tiny test meshes.

To preserve the old local angular resolution of a uniform `N`-sector mesh,
use an even fine-zone count close to `N / 6`. Thus 700 uniform sectors,
whose angular size is approximately 0.514 degrees, correspond to 118 fine
divisions, whose angular size is approximately 0.508 degrees. The final
tetrahedron/node count depends on Gmsh's transition and quality optimisation
and is printed and recorded in the mesh manifest.

## Cell topology and quality

DOLFINx 0.9 can store an experimental mixed topology at C++ level, but its
supported Python function-space/form path cannot assemble one elasticity
problem over connected hexahedra and tetrahedra. The graded tyre therefore
uses one conforming tetrahedral volume topology. This is intentional: it keeps
every cell in the FE solve and avoids the Gmsh importer silently selecting one
of two volume cell families.

The 60-degree road-facing boundary is still transfinite. It has exactly
`axial_divisions + 1` nodes on each of
`circumferential_divisions + 1` meridians, so the requested contact resolution
and reflection symmetry are preserved. The adjacent 30-degree zones and the
remaining rear surface are triangular and unstructured. Their axial density
is free to decrease with the target size; it is no longer inherited from a
full-width hexahedral extrusion.

Gmsh's tetrahedral optimiser is run before export. Generation reports minimum,
one-percentile, and mean `minSICN` quality and rejects inverted elements. The
same data and the requested density are written to `<mesh>.msh.json`. That
manifest is used to decide whether an existing mesh can be reused.

## Compliance strategy

Local refinement breaks the full-circle discrete rotational symmetry. The
fine zone is regular and reflection-symmetric, but rotating it into the coarse
zone does not map the finite-element mesh onto itself. Consequently, the
reference-meridian sampled H-matrix cannot reconstruct the exact compliance of
the graded FE discretization.

The generated boundary therefore uses two disjoint physical tags: `201` for
the transition/coarse road surface and `204` for the regular fine patch. With
`hmatrix.local_symmetry_tag: 204`, only the `(fine divisions + 1) x (axial
divisions + 1)` patch is ordered, sampled and passed to ACA. Angular offsets
are open: they never wrap through the coarse rear surface. Reverse entries are
supplied by Maxwell--Betti reciprocity.

Local regularity does not make the globally condensed FE operator rotationally
invariant. The coarse remainder still affects patch displacement, so the code
compares selected reconstructed columns with direct FE back-solves and reports
the relative error. `local_symmetry_strict: true` makes an error above
`local_symmetry_tolerance` fatal; otherwise it produces a visible warning.

The warning-distance zone must remain inside tag 204. This is certified first
from all road-surface free gaps and again after every final FE recovery solve.
Contact outside the patch stops the run. The local patch uses a nodal-area
diagonal PPCG preconditioner and a non-periodic nearest-neighbour halo.

Use `compliance.strategy: "fe_matrix_free"` when exact graded-mesh compliance
is required. It treats tags 201 and 204 as one physical contact surface and
applies compliance through the reused PETSc LU factorization.

## Command line

The equivalent command-line controls are:

```bash
python examples/tyre_dihedral_contact.py \
  --circumferential-layout graded \
  --circumferential-divisions 118 \
  --coarsening-factor 6 \
  --compliance-strategy hmatrix \
  --local-symmetry-tag 204 \
  --no-load-compliance \
  --regenerate
```

The standard large-case input is already configured, so the normal entry point
remains:

```bash
python examples/tyre_dihedral_contact.py -in examples/sm_input.json
```
