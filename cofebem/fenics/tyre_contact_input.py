"""Structured JSON input for the tyre dihedral contact example."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TyreContactInput:
    """Argument defaults and an optional embedded motion schedule."""

    argument_defaults: dict[str, Any]
    motion: dict[str, Any] | None
    path: Path


_SECTIONS: dict[str, dict[str, str]] = {
    "mesh": {
        "template": "template",
        "file": "mesh",
        "axial_divisions": "axial_divisions",
        "circumferential_divisions": "circumferential_divisions",
        "scale": "scale",
        "regenerate": "regenerate",
    },
    "material": {
        "young_modulus": "young_modulus",
        "poisson_ratio": "poisson_ratio",
        "inflation_pressure": "inflation_pressure",
    },
    "floor": {
        "kind": "floor_kind",
        "level": "floor_level",
        "grid_size": "floor_grid_size",
        "margin": "floor_margin",
    },
    "roughness": {
        "rms": "roughness_rms",
        "hurst": "roughness_hurst",
        "k_low": "roughness_k_low",
        "k_high": "roughness_k_high",
        "seed": "roughness_seed",
        "plateau": "roughness_plateau",
        "noise": "roughness_noise",
    },
    "compliance": {
        "strategy": "compliance_strategy",
        "load": "load_compliance",
        "factor_solver_type": "factor_solver_type",
    },
    "solver": {
        "contact_method": "contact_solver",
        "max_iterations": "max_iter",
        "tolerance": "tol",
        "pcg_preconditioner": "pcg_preconditioner",
        "pcg_zero_mode_factor": "pcg_zero_mode_factor",
        "pcg_beta_method": "pcg_beta_method",
    },
    "hmatrix": {
        "leaf_size": "h_leaf_size",
        "eta": "h_eta",
        "tolerance": "h_tol",
        "max_rank": "h_max_rank",
        "split": "h_split",
    },
    "potential_contact": {
        "warning_distance": "warning_distance",
        "halo": "warning_halo",
        "max_rounds": "warning_max_rounds",
        "verification_tolerance": "warning_verification_tol",
    },
    "execution": {
        "sampling_only": "sampling_only",
        "show_progress": "show_progress",
    },
}

_PATH_DESTINATIONS = {"template", "mesh", "load_compliance"}


def load_tyre_contact_input(path: str | Path) -> TyreContactInput:
    """Load a complete structured input and reject unknown fields."""
    input_path = Path(path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Tyre contact input does not exist: {input_path}")
    try:
        with input_path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid tyre contact JSON {input_path}: line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("Tyre contact input root must be a JSON object")

    allowed_sections = set(_SECTIONS) | {"motion"}
    unknown_sections = set(payload) - allowed_sections
    if unknown_sections:
        raise ValueError(
            "Unknown tyre contact input section(s): "
            + ", ".join(sorted(unknown_sections))
        )

    defaults: dict[str, Any] = {}
    for section_name, field_mapping in _SECTIONS.items():
        section = payload.get(section_name, {})
        if not isinstance(section, dict):
            raise ValueError(f"Input section {section_name!r} must be an object")
        unknown_fields = set(section) - set(field_mapping)
        if unknown_fields:
            raise ValueError(
                f"Unknown field(s) in {section_name!r}: "
                + ", ".join(sorted(unknown_fields))
            )
        for field_name, value in section.items():
            destination = field_mapping[field_name]
            if destination == "show_progress":
                if not isinstance(value, bool):
                    raise ValueError("execution.show_progress must be boolean")
                defaults["no_progress"] = not value
                continue
            if destination in defaults:
                raise ValueError(f"Duplicate input destination {destination!r}")
            if destination in _PATH_DESTINATIONS and value is not None:
                if not isinstance(value, str):
                    raise ValueError(
                        f"{section_name}.{field_name} must be a path string or null"
                    )
                candidate = Path(value).expanduser()
                if not candidate.is_absolute():
                    candidate = input_path.parent / candidate
                value = candidate.resolve()
            defaults[destination] = value

    motion = payload.get("motion")
    if motion is not None and not isinstance(motion, dict):
        raise ValueError("Input section 'motion' must be an object")
    return TyreContactInput(defaults, motion, input_path)


def validated_argument_defaults(
    parser: argparse.ArgumentParser, defaults: dict[str, Any]
) -> dict[str, Any]:
    """Apply argparse conversions and choices to values originating in JSON."""
    actions = {action.dest: action for action in parser._actions}
    validated: dict[str, Any] = {}
    for destination, value in defaults.items():
        if destination not in actions:
            raise ValueError(f"Input destination {destination!r} has no CLI argument")
        action = actions[destination]
        option = action.option_strings[-1] if action.option_strings else destination
        if isinstance(action, argparse.BooleanOptionalAction) or isinstance(
            action, (argparse._StoreTrueAction, argparse._StoreFalseAction)
        ):
            if not isinstance(value, bool):
                raise ValueError(f"Input value for {option} must be boolean")
            converted = value
        elif value is None:
            converted = None
        elif action.type is not None:
            try:
                converted = action.type(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid input value for {option}: {value!r}") from exc
        else:
            converted = value
        if action.choices is not None and converted not in action.choices:
            choices = ", ".join(str(choice) for choice in action.choices)
            raise ValueError(
                f"Invalid input value for {option}: {converted!r}; choose {choices}"
            )
        validated[destination] = converted
    return validated
