# Complete tyre-contact JSON input

The complete tyre workflow can be launched from one structured JSON file:

```bash
conda run -n fenicsx-env python examples/tyre_dihedral_contact.py \
  -in examples/input.json
```

With `fenicsx-env` already activated, the shorter command is:

```bash
python examples/tyre_dihedral_contact.py -in examples/input.json
```

`examples/input.json` is a complete 200 axial by 400 circumferential rough-floor
case using the existing compliance archive. It embeds the motion key frames, so
no second JSON is required.

## Structure

The input accepts only the sections and fields below. Unknown names are errors,
which catches misspellings instead of silently using a default.

| Section | Fields |
| --- | --- |
| `mesh` | `template`, `file`, `axial_divisions`, `circumferential_divisions`, `scale`, `regenerate` |
| `material` | `young_modulus`, `poisson_ratio`, `inflation_pressure` |
| `floor` | `kind`, `level`, `grid_size`, `margin` |
| `roughness` | `rms`, `hurst`, `k_low`, `k_high`, `seed`, `plateau`, `noise` |
| `motion` | `time`, `interval_steps`, `indentation`, `floor_rotation_y_deg`, `floor_rotation_z_deg`, `floor_translation_x`, `floor_translation_y` |
| `compliance` | `strategy`, `load`, `factor_solver_type`, `memory_map` |
| `solver` | `contact_method`, `max_iterations`, `tolerance`, `pcg_preconditioner`, `pcg_zero_mode_factor`, `pcg_beta_method` |
| `hmatrix` | `leaf_size`, `eta`, `tolerance`, `max_rank`, `split` |
| `potential_contact` | `warning_distance`, `halo`, `max_rounds`, `verification_tolerance` |
| `execution` | `sampling_only`, `show_progress` |
| `postprocessing` | `stress_projection`, `write_vtk`, `save_volume_fields` |

The choices have the same meaning as their CLI equivalents:

- `floor.kind`: `flat` or `rough`;
- `compliance.strategy`: `hmatrix` or `fe_matrix_free`;
- `solver.contact_method`: `ppcg`, `ccg_v2`, or `ccg`;
- `solver.pcg_preconditioner`: `spectral` or `none`;
- `solver.pcg_beta_method`: `pr_plus` or `fletcher_reeves`;
- `hmatrix.split`: `pca` or `kd`.

Set `compliance.load` to `null` to sample compliance in the current run. Set it
to an NPZ path to reuse reference-meridian samples. `factor_solver_type` may be
`null` for PETSc's default or a backend name available in the environment.
Keep `memory_map: true` for large cases. Samples are stored in
`compliance_samples.npy`, while `compliance.npz` contains metadata and a
relative sidecar reference. The H-matrix entry source then reads the tensor
through a NumPy memory map instead of pinning the entire array in resident RAM.
Only the three combinations needed by the scalar normal operator are stored;
this is 25% smaller than the legacy complete 2x2 transverse tensor. Loading a
legacy archive converts it automatically. There is only one contact H-matrix,
for normal compliance. For the 300 by 700 case, the raw sample data decrease
from about 1.89 GiB to 1.42 GiB, saving approximately 484 MiB.

## Large-case postprocessing

The recommended low-memory settings are:

```json
{
  "compliance": {
    "memory_map": true
  },
  "postprocessing": {
    "stress_projection": "lumped",
    "write_vtk": true,
    "save_volume_fields": false
  }
}
```

`lumped` assembles the signed normal-stress load and divides it by consistent
nodal surface area. It avoids the additional surface-mass LU required by
`consistent`, which can exceed available memory while the elasticity LU is
resident. Both recover constant stress exactly. Use `consistent` only when the
extra factor memory is affordable.

Volume displacement functions are still streamed to VTK when `write_vtk` is
true. `save_volume_fields: false` prevents duplicating the same large arrays in
every step NPZ. Set `write_vtk: false` as a further reduction when only contact
forces, clearances, and the two contact-pressure arrays are required.

## Paths

`mesh.template`, `mesh.file`, and `compliance.load` are resolved relative to
the input JSON, not the shell working directory. Thus the repository example
uses:

```json
{
  "mesh": {
    "template": "../geo_files/geometry_v2.geo",
    "file": "../results/tyre_dihedral/tyre_dihedral.msh"
  },
  "compliance": {
    "load": "../results/tyre_dihedral/compliance.npz"
  }
}
```

Absolute paths and `~` expansion are also supported.

## Motion

The embedded `motion` section follows the interpolation rules in
[`floor_motion.md`](floor_motion.md). With a `time` array, every motion field
must be a scalar or have `len(time)` values, while `interval_steps` must have
`len(time) - 1` positive integers.

A static solve may omit `time` and use scalar values:

```json
{
  "motion": {
    "indentation": 0.01,
    "floor_rotation_y_deg": 3.0,
    "floor_rotation_z_deg": 0.0,
    "floor_translation_x": 0.0,
    "floor_translation_y": 0.0
  }
}
```

## CLI overrides

Explicit CLI options override values from the non-motion JSON sections. This
is useful for a small variation without editing the case:

```bash
python examples/tyre_dihedral_contact.py -in examples/input.json \
  --max-iter 20000 --warning-distance 0.03 --no-progress
```

Boolean reversals include `--no-regenerate`, `--no-roughness-plateau`,
`--no-roughness-noise`, `--no-sampling-only`, and `--progress`.
`--no-load-compliance` overrides a configured archive and forces new sampling.
The corresponding memory flags are `--[no-]mmap-compliance`,
`--stress-projection`, `--[no-]write-vtk`, and
`--[no-]save-volume-fields`.

An explicit `--motion-file` replaces the embedded `motion` section. Individual
CLI motion scalars serve as defaults only for fields omitted from a motion
schedule; they do not replace arrays explicitly present in that schedule.
