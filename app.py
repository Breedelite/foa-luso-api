# app.py
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from copy import deepcopy

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from numpy import generic as np_generic  # for normalising numpy scalars

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
from lusofuncs import profit, convertLUnameToLUI, testallMultiSols  # type: ignore


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
    # LEGACY path (kept for backward compatibility): the old hard-coded
    # scenario-override block against the startup CSV land uses.
    scenario: Optional[Dict[str, Any]] = None
    # STATELESS path (preferred): FileMaker supplies the finished inputs.
    # Each land_use uses engine-native keys: name, label, yield, price, cost,
    # costCont, Nreq, IEprevcrop, DEcrop, NboostperTonne, weedsurvival,
    # compindex, sowdensity, weedseedreturn, watermult, extracostperextrayield,
    # overhead. When land_uses is provided the CSV globals are NOT used.
    land_uses: Optional[List[Dict[str, Any]]] = None
    parameters: Optional[Dict[str, Any]] = None
    additional_effects: Optional[List[Dict[str, Any]]] = None
    disallowed_combos: Optional[List[List[Any]]] = None


class RotationEconomics(BaseModel):
    years: int
    profit_total: float
    profit_per_year: float


class EvaluateResponse(BaseModel):
    rotation_label: str
    sequence_indices: List[int]
    sequence_names: List[str]
    economics: RotationEconomics
    raw_details: Optional[List[Dict[str, Any]]] = None


# --------------------------------------------------------------------
# 3. FastAPI app + global LUSO data
# --------------------------------------------------------------------

app = FastAPI(title="FOA LUSO Decision API", version="0.3")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/landuses")
def landuses():
    """Return the land-use roster currently loaded from the startup CSV, so
    FileMaker can sync names/indices. Stateless calls supply their own
    land_uses and do not depend on this."""
    ensure_loaded()
    return {
        "count": len(LULIST),
        "land_uses": [
            {"index": i, "name": lu["name"], "label": lu.get("label")}
            for i, lu in enumerate(LULIST)
        ],
    }

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


def indices_from_rotation(rot: RotationInput, lulist: List[dict]) -> List[int]:
    """Resolve a rotation to land-use indices against the supplied lulist
    (works for both the stateless land_uses and the CSV globals)."""
    if rot.index_sequence:
        return rot.index_sequence

    if not rot.name_sequence:
        raise HTTPException(400, "Provide index_sequence OR name_sequence")

    name_to_idx = {lu["name"]: i for i, lu in enumerate(lulist)}
    seq: List[int] = []
    for name in rot.name_sequence:
        if name not in name_to_idx:
            raise HTTPException(400, f"Unknown land-use {name}")
        seq.append(name_to_idx[name])

    return seq


def names_from_indices(indices: List[int], lulist: List[dict]) -> List[str]:
    return [lulist[i]["name"] for i in indices]


def _normalise_value(v: Any) -> Any:
    """
    Convert numpy scalars to plain Python types so FastAPI can JSON-encode them.
    """
    if isinstance(v, np_generic):
        return float(v)
    return v


# --------------------------------------------------------------------
# 4b. Stateless input builders (FileMaker supplies the finished inputs)
# --------------------------------------------------------------------

# Engine-native numeric keys expected on each supplied land use.
_LU_NUMERIC_KEYS = [
    "yield", "price", "cost", "costCont", "Nreq", "IEprevcrop", "DEcrop",
    "NboostperTonne", "weedsurvival", "compindex", "sowdensity",
    "weedseedreturn", "watermult", "extracostperextrayield", "overhead",
]

# Safe defaults so a missing key never crashes the engine.
_PARAM_DEFAULTS: Dict[str, float] = {
    "season": 1.0, "fixedcosts": 0.0, "discountrate": 0.0, "Ncost": 0.0,
    "seedbank0": 0.0, "N0": 0.0, "DI0": 0.0, "DImin": 0.0,
    "weedgermination": 0.0, "weedcompindex": 0.0, "weedmaxseedset": 0.0,
    "IEprevinc": 0.0, "IErandom": 0.0, "DEinc": 0.0, "DErandom": 0.0,
    "costperweedseed": 0.0,
}


def _coerce_land_uses(land_uses: List[Dict[str, Any]]) -> List[dict]:
    """Normalise a supplied land_uses list into engine-shaped dicts (floats)."""
    out: List[dict] = []
    for lu in land_uses:
        d = dict(lu)
        d["name"] = str(d.get("name", ""))
        d.setdefault("label", d["name"])
        for k in _LU_NUMERIC_KEYS:
            try:
                d[k] = float(d.get(k, 0) or 0)
            except (TypeError, ValueError):
                d[k] = 0.0
        out.append(d)
    return out


def _build_parameters(parameters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge supplied parameters over safe defaults; force a default season."""
    p: Dict[str, Any] = dict(_PARAM_DEFAULTS)
    if parameters:
        for k, v in parameters.items():
            try:
                p[k] = float(v)
            except (TypeError, ValueError):
                p[k] = v
    if not p.get("season"):
        p["season"] = 1.0
    return p


def _build_optionalparams(
    lulist: List[dict],
    additional_effects: Optional[List[Dict[str, Any]]],
    disallowed_combos: Optional[List[List[Any]]],
) -> Dict[str, Any]:
    """Build the engine's optionalparams from name-based request data,
    converting land-use names to indices against the supplied lulist."""
    name_to_idx = {lu["name"]: i for i, lu in enumerate(lulist)}

    addefflist: List[dict] = []
    for ae in (additional_effects or []):
        il = name_to_idx.get(ae.get("initiallu"))
        ll = name_to_idx.get(ae.get("laterlu"))
        if il is None or ll is None:
            continue
        try:
            addefflist.append({
                "initiallu": il, "laterlu": ll,
                "yearsbetween": int(ae["yearsbetween"]),
                "effectonlaterlu": float(ae["effectonlaterlu"]),
            })
        except (KeyError, TypeError, ValueError):
            continue

    combos: List[list] = []
    for c in (disallowed_combos or []):
        if not c:
            continue
        try:
            conv = [int(c[0])] + [name_to_idx[n] for n in c[1:] if n in name_to_idx]
        except (TypeError, ValueError):
            continue
        combos.append(conv)

    return {"disallowedcombos": combos, "addefflist": addefflist, "pricevarlist": []}


# --------------------------------------------------------------------
# 5. Core: override economic inputs (scenarios)
# --------------------------------------------------------------------

def build_scenario_adjusted_inputs(
    scenario: dict | None,
) -> tuple[List[dict], Dict[str, Any]]:
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

              (by backing out the difference between cropping overheads
               and pasture overheads).
    """

    # If nothing to do, just hand back the originals
    if not scenario:
        return LULIST, PARAMETERS  # type: ignore[return-value]

    # Work on copies so the globals stay as the default bundle
    assert LULIST is not None and PARAMETERS is not None
    lulist = deepcopy(LULIST)
    params: Dict[str, Any] = dict(PARAMETERS)

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
def evaluate_rotation(req: EvaluateRequest) -> EvaluateResponse:
    """
    Deterministic evaluation of a single rotation using the default season.

    - Uses original LUSO profit() for headline numbers
      (so it matches Roger's script).
    - Adds raw per-year details from profit(getDetails=True) as `raw_details`.
    """
    # Resolve inputs. Two modes:
    #  - STATELESS (preferred): FileMaker supplies the finished land_uses[]
    #    (library + scenario overrides + grazing economics already applied),
    #    plus parameters and optional effects. The API just evaluates them.
    #  - LEGACY: no land_uses -> use the startup CSV globals + the old
    #    hard-coded scenario-override block (kept for backward compatibility).
    if req.land_uses:
        lulist_use = _coerce_land_uses(req.land_uses)
        params_use = _build_parameters(req.parameters)
        optional_use = _build_optionalparams(
            lulist_use, req.additional_effects, req.disallowed_combos
        )
    else:
        ensure_loaded()
        lulist_use, params_use = build_scenario_adjusted_inputs(
            getattr(req, "scenario", None)
        )
        optional_use = OPTIONALPARAMS

    indices = indices_from_rotation(req.rotation, lulist_use)
    names = names_from_indices(indices, lulist_use)

    # ----- Headline economics (net profit; profit() now nets overhead) ----
    if req.annualise:
        profit_per_year = profit(
            indices, params_use, lulist_use,
            getDetails=False, optionalparams=optional_use, annualise=True,
        )
        years = len(indices)
        profit_total = profit_per_year * years
    else:
        profit_total = profit(
            indices, params_use, lulist_use,
            getDetails=False, optionalparams=optional_use, annualise=False,
        )
        years = len(indices)
        profit_per_year = profit_total / years

    # ----- Per-year details -----
    raw = profit(
        indices, params_use, lulist_use,
        getDetails=True, optionalparams=optional_use, annualise=False,
    )

    if not isinstance(raw, list):
        raise HTTPException(
            status_code=500,
            detail="LUSO profit(getDetails=True) returned an unexpected structure.",
        )

    details_clean: List[Dict[str, Any]] = []
    for year_dict in raw:
        if isinstance(year_dict, dict):
            cleaned = {k: _normalise_value(v) for k, v in year_dict.items()}
            details_clean.append(cleaned)

    label = req.rotation.label or "-".join(names)

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
        raw_details=details_clean,
    )


# --------------------------------------------------------------------
# 7. Optimiser endpoint — find the most profitable rotation(s)
# --------------------------------------------------------------------

class OptimiseRequest(BaseModel):
    years: Optional[int] = None          # rotation length; defaults to parameters.nyears
    nsols: int = 10                      # how many top rotations to return
    land_uses: Optional[List[Dict[str, Any]]] = None
    parameters: Optional[Dict[str, Any]] = None
    additional_effects: Optional[List[Dict[str, Any]]] = None
    disallowed_combos: Optional[List[List[Any]]] = None


# Exhaustive search evaluates nlu**years rotations. Cap it so a large
# library can't hang the request. (6 land uses x 6 yrs = 46,656; fine.)
EXHAUSTIVE_CAP = 200_000


@app.post("/optimise")
def optimise(req: OptimiseRequest):
    """Exhaustively find the top-N most profitable rotations for the supplied
    land uses / parameters. Returns them ranked best-first. Uses the same
    net-profit engine as /evaluate (incl. the end-of-rotation seed-bank
    penalty), so a rotation's profit here matches evaluating it directly."""
    if req.land_uses:
        lulist_use = _coerce_land_uses(req.land_uses)
        params_use = _build_parameters(req.parameters)
        optional_use = _build_optionalparams(
            lulist_use, req.additional_effects, req.disallowed_combos
        )
    else:
        ensure_loaded()
        lulist_use = LULIST
        params_use = dict(PARAMETERS)
        optional_use = OPTIONALPARAMS

    nlu = len(lulist_use)
    ny = req.years or int(params_use.get("nyears") or 0)
    if nlu < 1:
        raise HTTPException(400, "No land uses supplied to optimise over.")
    if ny < 1:
        raise HTTPException(400, "Provide 'years' (rotation length) or parameters.nyears >= 1.")

    combos = nlu ** ny
    if combos > EXHAUSTIVE_CAP:
        raise HTTPException(
            400,
            f"Search space too large: {nlu} land uses ^ {ny} years = {combos:,} "
            f"combinations (cap {EXHAUSTIVE_CAP:,}). Reduce the number of land uses "
            f"or the rotation length.",
        )

    nsols = max(1, min(req.nsols, 50))
    best = testallMultiSols(ny, nlu, nsols, params_use, lulist_use, optionalparams=optional_use)

    # testallMultiSols returns [[profit, [indices]], ...] ascending, with
    # possible [-99999, 'invalid'] placeholders. Keep real ones, best-first.
    rows = [b for b in best if isinstance(b[1], list)]
    rows.sort(key=lambda b: b[0], reverse=True)

    rotations = []
    for rank, (p, lus) in enumerate(rows, start=1):
        p = float(_normalise_value(p))
        rotations.append({
            "rank": rank,
            "profit_total": p,
            "profit_per_year": p / ny,
            "sequence_indices": list(lus),
            "sequence_names": [lulist_use[i]["name"] for i in lus],
            "label": "-".join(lulist_use[i].get("label") or lulist_use[i]["name"] for i in lus),
        })

    return {
        "method": "exhaustive",
        "years": ny,
        "land_uses_count": nlu,
        "combinations_evaluated": combos,
        "rotations": rotations,
    }
