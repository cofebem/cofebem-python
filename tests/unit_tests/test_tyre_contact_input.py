import argparse
import json
from pathlib import Path

import pytest

from cofebem.fenics.tyre_contact_input import (
    load_tyre_contact_input,
    validated_argument_defaults,
)


def test_structured_input_flattens_sections_and_resolves_paths(tmp_path):
    input_path = tmp_path / "case" / "input.json"
    input_path.parent.mkdir()
    input_path.write_text(
        json.dumps(
            {
                "mesh": {
                    "template": "../geometry.geo",
                    "file": "mesh.msh",
                    "axial_divisions": 20,
                    "regenerate": True,
                },
                "floor": {"kind": "rough", "grid_size": 64},
                "compliance": {"load": None},
                "execution": {"sampling_only": False, "show_progress": False},
                "motion": {"indentation": 0.01},
            }
        )
    )

    data = load_tyre_contact_input(input_path)

    assert data.path == input_path.resolve()
    assert data.argument_defaults["template"] == (tmp_path / "geometry.geo").resolve()
    assert data.argument_defaults["mesh"] == (input_path.parent / "mesh.msh").resolve()
    assert data.argument_defaults["axial_divisions"] == 20
    assert data.argument_defaults["floor_kind"] == "rough"
    assert data.argument_defaults["no_progress"] is True
    assert data.motion == {"indentation": 0.01}


def test_structured_input_rejects_unknown_section_and_field(tmp_path):
    path = tmp_path / "input.json"
    path.write_text(json.dumps({"unknown": {}}))
    with pytest.raises(ValueError, match="Unknown tyre contact input section"):
        load_tyre_contact_input(path)

    path.write_text(json.dumps({"mesh": {"axial_division": 20}}))
    with pytest.raises(ValueError, match="Unknown field.*axial_division"):
        load_tyre_contact_input(path)


def test_argument_defaults_apply_types_choices_and_booleans():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int)
    parser.add_argument("--method", choices=("a", "b"))
    parser.add_argument("--enabled", action=argparse.BooleanOptionalAction)

    values = validated_argument_defaults(
        parser, {"count": 12, "method": "b", "enabled": True}
    )
    assert values == {"count": 12, "method": "b", "enabled": True}

    with pytest.raises(ValueError, match="choose a, b"):
        validated_argument_defaults(parser, {"method": "c"})


def test_repository_example_is_complete_and_loadable():
    root = Path(__file__).resolve().parents[2]
    data = load_tyre_contact_input(root / "examples" / "input.json")

    assert set(data.argument_defaults) == {
        "template",
        "mesh",
        "axial_divisions",
        "circumferential_divisions",
        "scale",
        "regenerate",
        "young_modulus",
        "poisson_ratio",
        "inflation_pressure",
        "floor_kind",
        "floor_level",
        "floor_grid_size",
        "floor_margin",
        "roughness_rms",
        "roughness_hurst",
        "roughness_k_low",
        "roughness_k_high",
        "roughness_seed",
        "roughness_plateau",
        "roughness_noise",
        "compliance_strategy",
        "load_compliance",
        "factor_solver_type",
        "contact_solver",
        "max_iter",
        "tol",
        "pcg_preconditioner",
        "pcg_zero_mode_factor",
        "pcg_beta_method",
        "h_leaf_size",
        "h_eta",
        "h_tol",
        "h_max_rank",
        "h_split",
        "warning_distance",
        "warning_halo",
        "warning_max_rounds",
        "warning_verification_tol",
        "sampling_only",
        "no_progress",
    }
    assert data.motion is not None
