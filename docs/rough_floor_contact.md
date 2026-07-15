# Regular flat and rough floors

`examples/tyre_dihedral_contact.py` supports a rigid floor represented by a
regular tensor-product height grid. Both floor types use the same projection
and output path:

```text
projected point = (x_tyre, y_tyre, h_floor(x_tyre, y_tyre))
g0              = z_tyre - h_floor(x_tyre, y_tyre)
g               = g0 + inflation displacement
```

The height is bilinearly interpolated at every ordered tyre contact node. The
current dihedral compliance maps global-z force to global-z displacement, so
this is a vertical contact formulation. Floor slopes are visualized through
their normals but are not yet used as varying contact directions. A true
local-normal rough-contact formulation would require vector force/response
reconstruction and a different scalar operator.

## Flat floor

The default remains a flat floor at `--floor-level 0`:

```bash
conda run -n fenicsx-env python examples/tyre_dihedral_contact.py \
  --axial-divisions 100 --circumferential-divisions 200 \
  --indentation 1e-2 --floor flat --floor-grid-size 256
```

The tyre translation interprets indentation relative to `floor-level`, so this
reproduces the previous `z=0` gap when the default level is used.

## rfgen rough floor

The rough floor uses `rfgen.selfaffine_field(dim=2, ...)`, following the API in
<https://github.com/vyastreb/rfgen> and the Hcontact rough-contact example. The
field is periodic, normalized to the requested RMS height, and shifted so its
highest asperity equals `floor-level`. Thus a rough floor is nowhere higher
than the corresponding flat datum.

```bash
conda run -n fenicsx-env python examples/tyre_dihedral_contact.py \
  --axial-divisions 100 --circumferential-divisions 200 \
  --indentation 1e-2 --floor rough --floor-grid-size 256 \
  --roughness-rms 2e-4 --roughness-hurst 0.8 \
  --roughness-k-low 0.03 --roughness-k-high 0.3 \
  --roughness-seed 42
```

The grid is a centered square containing the complete tyre projection plus
`--floor-margin`. `floor-grid-size` is the number of cells in both directions
and the rfgen sample size. `k-low` and `k-high` use rfgen's normalized
wavenumber convention and must satisfy `0 < k_low < k_high <= 0.5`.
`--roughness-plateau` enables spectral roll-off, and
`--no-roughness-noise` selects the ideal-spectrum random-phase mode.

## Contact pressure fields

Two positive-compression conventions are saved on the tyre contact surface:

```text
contact_pressure_force_based[i] = nodal_force[i] / associated_area[i]
contact_pressure_stress          = L2 projection of -n . sigma(u_contact) . n
u_contact                        = u_final - u_inflation
```

The associated area is the consistent surface lump
`A_i = integral_Gamma N_i dA`, assembled with the same CG1 scalar basis. The
force-based field therefore preserves the contact resultant to roundoff:

```text
sum_i contact_pressure_force_based[i] * A_i = sum_i nodal_force[i].
```

The stress field deliberately uses only the contact displacement increment,
excluding the inflation preload. It is a raw signed FE stress recovery. It can
oscillate and differ substantially from force/area near nodal loads, especially
for nearly incompressible CG1 elements; this difference is diagnostic rather
than silently clipped or rescaled.

## Outputs

The tyre PVD/VTU contains:

- `contact_pressure_stress`;
- `contact_pressure_force_based`;
- `contact_associated_area`;
- `initial_gap`;
- `floor_height_projection`;
- total, inflation, and contact displacement fields;
- nodal contact force and potential-zone indicator.

The result NPZ stores the same contact arrays in sector-major ordering.
`floor_flat.vtu` or `floor_rough.vtu` contains the regular floor geometry with
`floor_height` and `floor_normal`; `floor.npz` stores the exact grid, height
array, and generation parameters for reproducibility.

## Small validation

For the 24-by-32 validation mesh at 5 mm indentation, the flat case produced a
5.3475 kN total force and the 0.2 mm RMS rough case produced 4.7281 kN. Both had
nine positive-force nodes. In both cases the integrated force-based pressure
matched the LCP force to machine precision, and the rough field was periodic
with exactly the requested RMS height.
