# Moving-floor tyre contact

`examples/tyre_dihedral_contact.py` can keep the tyre FE model fixed while a
flat or rough rigid floor approaches, tilts, twists, and slides. This keeps the
elastic stiffness, Dirichlet conditions, contact DOF ordering, compliance
samples, and PETSc direct factorization unchanged throughout a load history.

## Static motion controls

The undeformed tyre is aligned once so its lowest tread point is tangent to
`--floor-level`. Indentation is then an upward global-z translation of the
floor, not a translation of the tyre mesh:

```bash
conda run -n fenicsx-env python examples/tyre_dihedral_contact.py \
  --axial-divisions 100 --circumferential-divisions 200 \
  --indentation 1e-2 --rotate-floor 3 --torsion-floor 2 \
  --floor-translation-x 1e-3 --floor-translation-y 2e-3
```

`--rotate-floor` is a rotation in degrees about global `OY`, through the point
on the undeformed floor below the tyre centre. It creates a slope along `OX`.
`--torsion-floor` rotates about global `OZ`. The implemented composition is

```text
R = Rz(torsion) @ Ry(slope)
x_floor = pivot + R @ (x_reference - pivot) + (dx, dy, indentation)
```

The displayed floor mesh uses this exact rigid transformation. For a rough
floor, vertical tyre rays are intersected with the transformed bilinear height
field by a vectorized Newton solve; the same transformed geometry defines the
gap and the visualization.

The contact operator remains the existing global-z force/global-z displacement
operator. A tilted floor is therefore a vertical unilateral constraint, not a
local-normal frictionless-contact formulation. `Rz` changes the orientation of
roughness and of an already tilted plane; rotating a perfectly horizontal,
flat plane about `OZ` alone has no geometrical effect.

## JSON histories

The recommended entry point embeds this motion object in the complete
`examples/input.json` described in
[`tyre_contact_input.md`](tyre_contact_input.md). The standalone
`--motion-file` form remains supported for compatibility.

Pass `--motion-file examples/tyre_floor_motion.json`. The example file is:

```json
{
  "time": [0, 1, 2, 3],
  "interval_steps": [10, 10, 10],
  "indentation": [0.0, 0.01, 0.01, 0.01],
  "floor_rotation_y_deg": [0.0, 3.0, 0.0, -3.0],
  "floor_rotation_z_deg": [0.0, 0.0, 3.0, 0.0],
  "floor_translation_x": [0.0, 0.0, 0.001, 0.001],
  "floor_translation_y": [0.0, 0.001, 0.001, 0.0]
}
```

`interval_steps[i]` is the number of equal increments between `time[i]` and
`time[i + 1]`. End points shared by two intervals are not duplicated, so this
example produces `1 + 10 + 10 + 10 = 31` states. All six state components are
linearly interpolated. Lengths are metres, rotations are degrees, and time is
an arbitrary strictly increasing load parameter. A scalar motion field is
broadcast to every key time; an omitted field uses its corresponding command
line value.

At setup, the minimum gap over every scheduled state defines a union potential
contact zone. One restricted H-matrix or factorized-FE operator is reused until
full-surface verification requests an expansion. Each LCP is warm-started from
the previous force vector. Expanding the contact zone does not rebuild or
refactorize the bulk stiffness.

## Reuse and factor persistence

The H-matrix and `fe_matrix_free` strategies call `create_lu_solver` exactly
once. PETSc `PREONLY+LU` setup factorizes the stiffness on that call, and all
inflation, compliance-sampling, matrix-free PPCG, and result-recovery solves
reuse the same KSP. `fe_iterative` similarly reuses one KSP/PC setup.
`mumps_schur` first uses a temporary LU to obtain the inflation displacement,
then releases it and reuses one exact motion-union selected-Schur factor for
all contact and recovery solves; its reported factorization count is two.

PETSc 3.24.3 in `fenicsx-env` was tested with its PETSc, MUMPS, SuperLU,
UMFPACK, and KLU factor matrices. None could be written with a PETSc binary
viewer: each factor `MatView` returned PETSc error 73. PETSc's supported
`MatLoad` path restores an unfactorized matrix and therefore still needs a new
numeric factorization after restart.

A backend-specific fallback was also benchmarked by storing SciPy/SuperLU
`L`, `U`, row permutation, and column permutation. On the 24 axial by 32 sector
tyre (`7200` displacement DOFs, `441504` stiffness nonzeros):

| operation | time |
| --- | ---: |
| PETSc factorization | 0.188 s |
| 20 native PETSc solves | 0.040 s |
| stored SuperLU factor load | 0.038 s |
| 20 restored sparse triangular solves | 1.390 s |

The restored solve agreed with SuperLU to `8.6e-18` relative error, but was
about 35 times slower per right-hand side. Its 0.15 s startup saving was lost
after roughly three solves, whereas even one contact state needs several and a
matrix-free history needs many. The implementation therefore deliberately
factorizes once at process startup instead of offering a slower factor-cache
mode. `--factor-solver-type` remains available for machine- and mesh-specific
backend selection.

## Outputs

- `floor_motion.pvd` references one transformed floor VTU per state under
  `floor_motion/`.
- `tyre_dihedral_contact_<strategy>.pvd` contains the tyre fields at the JSON
  time values.
- `motion_steps/contact_result_XXXXX.npz` stores complete contact arrays for
  each state.
- `motion_history.npz` stores the expanded schedule, convergence/resultant
  histories, timings, peak RSS, and the backend-dependent factorization count.
- `contact_result.npz` and the strategy-specific result archive contain the
  final state for compatibility with one-step post-processing.
