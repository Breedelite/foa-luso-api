# app.py
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from copy import deepcopy

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --------------------------------------------------------------------
# 1. Wire in LUSO model code
# --------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
files_dir = BASE_DIR / "files"
inputs_dir = BASE_DIR / "inputs"

if not files_dir.exists():
    raise RuntimeError(f"'files' directory not found at {files_dir}")
if not inputs_dir.exists():
    raise RuntimeError(f"'inputs' directory not found at {inputs_dir}")

sys.path.append(str(files_dir))

from readers import readinLUlist, readinparams, readoptionalparams  # type: ignore
from lusofuncs import profit, convertLUnameToLUI  # type: ignore


# --------------------------------------------------------------------
# 2. Pydantic Models
# --------------------------------------------------------------------

class RotationInput(BaseModel):
    index_sequence: Optional[List[int]] = None
    name_sequence: Optional[List[str]] = None
    label: Optional[str] = None


class EvaluateRequest(BaseModel):
    rotation: RotationInput
    annualise: bool = True
    scenario: Optional[Dict[str, Any]] = None


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

app = FastAPI(title="FOA LUSO Deterministic API", version="0.2")

@app.get("/health")
def health():
    return {"status": "ok"}

LULIST = None
PARAMETERS = None
OPTIONALPARAMS = None


@app.on_event("startup")
def startup_event():
    global LULIST, PARAMETERS, OPTIONALPARAMS

    os.chdir(BASE_DIR)

    print("[startup] Loading LUSO input files...")
    LULIST = readinLUlist("_LUSdetails_used.csv")
    PARAMETERS = readinparams("_parameters_used.csv")
    OPTIONALPARAMS = readoptionalparams(LULIST)
    print(f"[startup] Loaded {len(LULIST)} land uses.")


# --------------------------------------------------------------------
# 4. Helper functions
# --------------------------------------------------------------------

def ensure_loaded():
    if LULIST is None or PARAMETERS is None or OPTIONALPARAMS is None:
        raise HTTPException(status_code=500, detail="LUSO inputs not loaded.")


def indices_from_rotation(rot: RotationInput) -> List[int]:
    ensure_loaded()

    if rot.index_sequence:
        return rot.index_sequence

    if not rot.name_sequence:
        raise HTTPException(400, "Provide index_sequence OR name_sequence")

    seq = []
    for name in rot.name_sequence:
        idx = convertLUnameToLUI(name, LULIST)
        if LULIST[idx]["name"] != name:
            raise HTTPException(400, f"Unknown land-use {name}")
        seq.append(idx)

    return seq


def names_from_indices(indices: List[int]) -> List[str]:
    ensure_loaded()
    return [LULIST[i]["name"] for i in indices]


# --------------------------------------------------------------------
# 5. Core: override economic inputs
# --------------------------------------------------------------------


from copy import deepcopy

def build_scenario_adjusted_inputs(
    scenario: dict | None,
) -> tuple[list[dict], dict]:
    """
    Take the global LULIST & PARAMETERS and return copies with
    simple overrides applied from the 'scenario' block.

    Rules (v3, with costs):

      - If scenario is None or empty: return originals unchanged.

      - Cropping overheads (global):
          scenario["overheads_per_ha"]  -> PARAMETERS["fixedcosts"]

      - N price:
          scenario["N_price_per_kgN"]   -> PARAMETERS["Ncost"]

      - Wheat / Barley / Canola:
          scenario["wheat"]["price_per_t"]     -> lu["price"]
          scenario["wheat"]["yield_t_ha"]      -> lu["yield"]
          scenario["wheat"]["var_cost_per_ha"] -> lu["cost"]

      - Pasture:
          scenario["pasture"] contains lamb + wool production, plus:
            var_cost_per_ha
            overheads_per_ha

          We:
            * convert wool + lamb into a single gross revenue/ha
              and set yield=1.0, price=revenue *for pasture LUs*
              (vpasture, ipasture, npasture)

            * set lu["cost"] = pasture.var_cost_per_ha (if given)

            * adjust lu["cost"] again so that *effective* overhead
              for pasture years == pasture.overheads_per_ha

              (we do this by backing out the difference between
               cropping overheads and pasture overheads).
    """

    # If nothing to do, just hand back the originals
    if not scenario:
        return LULIST, PARAMETERS  # type: ignore[return-value]

    # Work on copies so the globals stay as the default bundle
    assert LULIST is not None and PARAMETERS is not None
    lulist = deepcopy(LULIST)
    params = dict(PARAMETERS)

    # --- 1) Cropping overhead cost override ---------------------------
    oh = scenario.get("overheads_per_ha")
    if isinstance(oh, (int, float)):
        try:
            params["fixedcosts"] = float(oh)
        except Exception:
            # if the key isn't there or something odd happens, just ignore
            pass

    # --- 2) N price override -----------------------------------------
    n_price = scenario.get("N_price_per_kgN")
    if isinstance(n_price, (int, float)):
        try:
            params["Ncost"] = float(n_price)
        except Exception:
            pass

    # helper to apply grain overrides, including variable cost where provided
    def apply_grain_override(lu_name: str, scenario_key: str) -> None:
        block = scenario.get(scenario_key)
        if not isinstance(block, dict):
            return

        price = block.get("price_per_t")
        yld = block.get("yield_t_ha")
        var_cost = block.get("var_cost_per_ha")

        for lu in lulist:
            if lu.get("name") != lu_name:
                continue

            if price is not None:
                lu["price"] = float(price)
            if yld is not None:
                lu["yield"] = float(yld)
            if var_cost is not None:
                # LUSO uses 'cost' as variable cost per ha
                lu["cost"] = float(var_cost)

    # --- 3) Crops: wheat, barley, canola -----------------------------
    apply_grain_override("wheat", "wheat")
    apply_grain_override("barley", "barley")
    apply_grain_override("canola", "canola")

    # --- 4) Pasture revenue + variable + overheads -------------------
    past = scenario.get("pasture")
    if isinstance(past, dict):
        lamb_kg    = past.get("lamb_kg_cwt_ha") or 0
        lamb_price = past.get("lamb_price_per_kg") or 0
        wool_kg    = past.get("wool_cfw_kg_ha") or 0
        wool_price = past.get("wool_price_per_kg") or 0

        past_var_cost = past.get("var_cost_per_ha")
        past_oh       = past.get("overheads_per_ha")

        # compute combined wool + lamb revenue / ha
        try:
            revenue = (
                float(lamb_kg) * float(lamb_price)
                + float(wool_kg) * float(wool_price)
            )
        except Exception:
            revenue = 0.0

        # current cropping overhead (after scenario override), if numeric
        cropping_oh = params.get("fixedcosts")
        cropping_oh = float(cropping_oh) if isinstance(cropping_oh, (int, float)) else None

        if revenue > 0 or past_var_cost is not None or past_oh is not None:
            for lu in lulist:
                if lu.get("name") in ("vpasture", "ipasture", "npasture"):

                    # 4a. overwrite revenue if provided
                    if revenue > 0:
                        lu["yield"] = 1.0
                        lu["price"] = float(revenue)

                    # 4b. set pasture variable cost per ha (if given)
                    if past_var_cost is not None:
                        lu["cost"] = float(past_var_cost)

                    # 4c. adjust effective overhead via cost difference
                    if (
                        past_oh is not None
                        and isinstance(past_oh, (int, float))
                        and cropping_oh is not None
                    ):
                        # we want pasture overheads to be lower (or higher)
                        # than the global fixedcosts.
                        delta = cropping_oh - float(past_oh)

                        # reduce (or increase) the LU's per-ha cost
                        # to reflect this overhead difference.
                        current_cost = float(lu.get("cost", 0.0))
                        adjusted_cost = current_cost - delta

                        # don't let it go negative
                        if adjusted_cost < 0:
                            adjusted_cost = 0.0

                        lu["cost"] = adjusted_cost

    return lulist, params

# --------------------------------------------------------------------
# 6. Deterministic evaluation endpoint
# --------------------------------------------------------------------

@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate_rotation(req: EvaluateRequest):

    ensure_loaded()

    indices = indices_from_rotation(req.rotation)
    names = names_from_indices(indices)

    # Adjust inputs for scenario
    lul_use, params_use = build_scenario_adjusted_inputs(req.scenario)

    # Call the LUSO profit function
    if req.annualise:
        per_year = profit(
            indices,
            params_use,
            lul_use,
            getDetails=False,
            optionalparams=OPTIONALPARAMS,
            annualise=True,
        )
        years = len(indices)
        total = per_year * years
    else:
        total = profit(
            indices,
            params_use,
            lul_use,
            getDetails=False,
            optionalparams=OPTIONALPARAMS,
            annualise=False,
        )
        years = len(indices)
        per_year = total / years

    return EvaluateResponse(
        rotation_label=req.rotation.label or "-".join(names),
        sequence_indices=indices,
        sequence_names=names,
        economics=RotationEconomics(
            years=years,
            profit_total=float(total),
            profit_per_year=float(per_year),
        ),
    )