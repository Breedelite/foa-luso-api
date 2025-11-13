# app.py
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --------------------------------------------------------------------
# 1. Wire in the original LUSO code
# --------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

# Expecting:
#   BASE_DIR / "files"   -> lusofuncs.py, readers.py, etc.
#   BASE_DIR / "inputs"  -> _LUSdetails_used.csv, _parameters_used.csv, ...
files_dir = BASE_DIR / "files"
inputs_dir = BASE_DIR / "inputs"

if not files_dir.exists():
    raise RuntimeError(f"'files' directory not found at {files_dir}")

if not inputs_dir.exists():
    raise RuntimeError(f"'inputs' directory not found at {inputs_dir}")

# Make the LUSO "files" folder importable (so `from readers import ...` works)
sys.path.append(str(files_dir))

from readers import (  # type: ignore[import]
    readinLUlist,
    readinparams,
    readoptionalparams,
)
from lusofuncs import profit, convertLUnameToLUI  # type: ignore[import]

# --------------------------------------------------------------------
# 2. Data models (request / response)
# --------------------------------------------------------------------


class RotationInput(BaseModel):
    """
    Rotation specification.

    You can send either:
      - index_sequence: [6,0,0,1,0,0]
        (LUI indices, exactly as used in the original scripts)
      - name_sequence: ["nugpasture","wheat","wheat","canola","wheat","wheat"]

    If both are provided, index_sequence is used and name_sequence is ignored.
    """
    index_sequence: Optional[List[int]] = None
    name_sequence: Optional[List[str]] = None
    label: Optional[str] = None  # optional user label for the rotation


class EvaluateRequest(BaseModel):
    """
    Request body for /evaluate.
    """
    rotation: RotationInput
    # Future flags (for now deterministic only):
    annualise: bool = True


class RotationEconomics(BaseModel):
    years: int
    profit_total: float
    profit_per_year: float


class EvaluateResponse(BaseModel):
    rotation_label: str
    sequence_indices: List[int]
    sequence_names: List[str]
    economics: RotationEconomics


# --------------------------------------------------------------------
# 3. FastAPI app + global LUSO data
# --------------------------------------------------------------------

app = FastAPI(
    title="FOA LUSO Deterministic API",
    description=(
        "Thin wrapper around the LUSO model for deterministic "
        "single-rotation evaluation using default seasons."
    ),
    version="0.1.0",
)

@app.get("/health")
def health():
    return {"status": "ok"}

# Global caches so we only read CSVs once
LULIST = None
PARAMETERS = None
OPTIONALPARAMS = None


@app.on_event("startup")
def startup_event() -> None:
    """
    Load LUSO CSV inputs once, when the server starts.
    Mirrors the original 'single rotation with default season' script.
    """
    global LULIST, PARAMETERS, OPTIONALPARAMS

    # These filenames come straight from your uploaded project
    lus_filename = "_LUSdetails_used.csv"
    params_filename = "_parameters_used.csv"

    # readers.py expects to open "inputs/<filename>" relative to CWD.
    # Ensure the current working directory is the same as BASE_DIR.
    os.chdir(BASE_DIR)

    print(f"[startup] Reading LUSO inputs from 'inputs/{lus_filename}' and 'inputs/{params_filename}'")

    LULIST = readinLUlist(lus_filename)
    PARAMETERS = readinparams(params_filename)
    OPTIONALPARAMS = readoptionalparams(LULIST)

    print(f"[startup] Loaded {len(LULIST)} land-use entries.")


# --------------------------------------------------------------------
# 4. Helper functions
# --------------------------------------------------------------------


def ensure_loaded():
    if LULIST is None or PARAMETERS is None or OPTIONALPARAMS is None:
        raise HTTPException(status_code=500, detail="LUSO inputs not loaded.")


def indices_from_rotation(rot: RotationInput) -> List[int]:
    """
    Convert RotationInput into a list of LUI indices.
    """
    ensure_loaded()
    assert LULIST is not None

    if rot.index_sequence and len(rot.index_sequence) > 0:
        # Use indices directly
        indices = rot.index_sequence
    elif rot.name_sequence and len(rot.name_sequence) > 0:
        # Map names -> indices using convertLUnameToLUI
        indices = []
        for name in rot.name_sequence:
            idx = convertLUnameToLUI(name, LULIST)
            # Basic sanity check: ensure we really matched the name
            if idx >= len(LULIST) or LULIST[idx]["name"] != name:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown land use name '{name}' in rotation.",
                )
            indices.append(idx)
    else:
        raise HTTPException(
            status_code=400,
            detail="rotation.index_sequence or rotation.name_sequence must be provided.",
        )

    if len(indices) == 0:
        raise HTTPException(status_code=400, detail="Rotation sequence cannot be empty.")

    return indices


def names_from_indices(indices: List[int]) -> List[str]:
    ensure_loaded()
    assert LULIST is not None

    names: List[str] = []
    for i in indices:
        if i < 0 or i >= len(LULIST):
            raise HTTPException(
                status_code=400,
                detail=f"Rotation index {i} is out of range (0–{len(LULIST)-1}).",
            )
        names.append(LULIST[i]["name"])
    return names


# --------------------------------------------------------------------
# 5. Endpoints
# --------------------------------------------------------------------


@app.get("/ping")
def ping() -> dict:
    """
    Simple health check.
    """
    return {"status": "ok"}


@app.get("/")
def root() -> dict:
    """
    Basic info + list of available land uses (by name).
    """
    ensure_loaded()
    assert LULIST is not None

    return {
        "message": "FOA LUSO Deterministic API",
        "land_uses": [lu["name"] for lu in LULIST],
    }


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate_rotation(req: EvaluateRequest) -> EvaluateResponse:
    """
    Deterministic evaluation of a single rotation using the default season.
    This mirrors the behaviour of 'single rotation with default season.py'.

    Body example:

    {
      "rotation": {
        "index_sequence": [6,0,0,1,0,0],
        "label": "nugpasture-wheat-wheat-canola-wheat-wheat"
      },
      "annualise": true
    }

    or

    {
      "rotation": {
        "name_sequence": ["nugpasture","wheat","wheat","canola","wheat","wheat"]
      }
    }
    """
    ensure_loaded()
    assert LULIST is not None and PARAMETERS is not None and OPTIONALPARAMS is not None

    indices = indices_from_rotation(req.rotation)
    names = names_from_indices(indices)

    # Use the original profit() just like
    #  "single rotation with default season.py" does.
    #
    # In that script they do:
    #     prof = profit(cropRotation, parameters, lulist, annualise=True)
    #
    # So here we do the same, and compute total-from-annual and vice versa.
    if req.annualise:
        profit_per_year = profit(indices, PARAMETERS, LULIST,
                                 getDetails=False,
                                 optionalparams=OPTIONALPARAMS,
                                 annualise=True)
        years = len(indices)
        profit_total = profit_per_year * years
    else:
        profit_total = profit(indices, PARAMETERS, LULIST,
                              getDetails=False,
                              optionalparams=OPTIONALPARAMS,
                              annualise=False)
        years = len(indices)
        profit_per_year = profit_total / years

    label = (
        req.rotation.label
        or "-".join(names)
    )

    econ = RotationEconomics(
        years=years,
        profit_total=float(profit_total),
        profit_per_year=float(profit_per_year),
    )

    return EvaluateResponse(
        rotation_label=label,
        sequence_indices=indices,
        sequence_names=names,
        economics=econ,
    )