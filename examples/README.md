python examples/tyre_dihedral_contact.py --axial-divisions 100
  │ --circumferential-divisions 200 --indentation 1e-2 --warning-distance 2e-2 --compliance-strategy
  │ fe_matrix_free --no-progress

conda run -n fenicsx-env 

--compliance-strategy
  │ hmatrix

python tyre_dihedral_contact.py --axial-divisions 100 --circumferential-divisions 200 --indentation 2e-2 --warning-distance 3e-2 --compliance-strategy hmatrix

conda run -n fenicsx-env python examples/tyre_dihedral_contact.py \
 --axial-divisions 200 \
 --circumferential-divisions 400 \
 --indentation 1e-2 \
 --floor rough \
 --floor-grid-size 256 \
 --roughness-rms 2e-4 \
 --roughness-hurst 0.8 \
 --roughness-k-low 0.03 \
 --roughness-k-high 0.3 \
 --roughness-seed 42
