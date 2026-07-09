# app.py
from __future__ import annotations

import os
import sys
import json
import math
import random
from pathlib import Path
from typing import List, Optional, Dict, Any
from copy import deepcopy

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
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

from readers import readinLUlist, readinparams, readoptionalparams, readinstochMults  # type: ignore
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
STOCHMULTS = []   # default season table for Monte-Carlo (overridable per request)
APSOIL_INDEX = []  # APSoil DB summary for soil matching (apsoil, lat, lon, soiltype, pawc_per_m)
APSOIL_PROFILES = {}  # per-layer water profiles keyed by str(apsoil) - drives the depth chart


@app.on_event("startup")
def startup_event():
    global LULIST, PARAMETERS, OPTIONALPARAMS, STOCHMULTS, APSOIL_INDEX, APSOIL_PROFILES

    os.chdir(BASE_DIR)

    print("[startup] Loading LUSO input files...")
    LULIST = readinLUlist("_LUSdetails_used.csv")
    PARAMETERS = readinparams("_parameters_used.csv")
    OPTIONALPARAMS = readoptionalparams(LULIST)
    try:
        STOCHMULTS = readinstochMults("_stochasticParameters_used.csv")
        print(f"[startup] Loaded {len(STOCHMULTS)} default season types.")
    except Exception as e:
        STOCHMULTS = []
        print(f"[startup] No default season table loaded ({e}).")
    try:
        APSOIL_INDEX = json.load(open(BASE_DIR / "apsoil_index.json"))
        print(f"[startup] Loaded {len(APSOIL_INDEX)} APSoil profiles for soil matching.")
    except Exception as e:
        APSOIL_INDEX = []
        print(f"[startup] No APSoil index loaded ({e}).")
    try:
        APSOIL_PROFILES = json.load(open(BASE_DIR / "apsoil_profiles.json"))
        print(f"[startup] Loaded {len(APSOIL_PROFILES)} APSoil water profiles.")
    except Exception as e:
        APSOIL_PROFILES = {}
        print(f"[startup] No APSoil profiles loaded ({e}).")
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


# Exhaustive search evaluates nlu**years rotations. Above this many combos we
# switch to a fast sampled (random) search so a long rotation / large library
# can't make the request grind. (6 land uses x 6 yrs = 46,656 -> still exact.)
EXHAUSTIVE_CAP = 50_000


def _heuristic_search(ny, nlu, nsols, params, lulist, optionalparams,
                      eval_budget=30_000, max_sweeps=6):
    """Scalable optimiser for large search spaces (long rotations / many land
    uses) where exhaustive search is infeasible. Random-restart coordinate
    ascent: from each start, repeatedly set each year to the land use that
    maximises net profit given the others, sweeping until stable. Collects the
    distinct local optima and returns the top-N. Deterministic (fixed seed).
    Uses the same net-profit engine as exhaustive, so profits are directly
    comparable, and total work is bounded regardless of rotation length."""
    rng = random.Random(12345)

    def pf(lus):
        return float(_normalise_value(
            profit(lus, params, lulist, optionalparams=optionalparams)
        ))

    # Bound total work: each restart costs up to max_sweeps*ny*(nlu-1) evals,
    # so longer rotations simply get fewer restarts (wall time stays ~constant).
    per_restart = max(1, max_sweeps * ny * max(1, nlu - 1))
    restarts = max(8, min(40, eval_budget // per_restart))

    results = {}  # tuple(lus) -> profit
    for r in range(restarts):
        # First restart = single-land-use baseline; the rest are random.
        lus = [0] * ny if r == 0 else [rng.randint(0, nlu - 1) for _ in range(ny)]
        cur = pf(lus)
        for _ in range(max_sweeps):
            improved = False
            for i in range(ny):
                orig = lus[i]
                best_c, best_p = orig, cur
                for c in range(nlu):
                    if c == orig:
                        continue
                    lus[i] = c
                    p = pf(lus)
                    if p > best_p:
                        best_p, best_c = p, c
                lus[i] = best_c
                if best_c != orig:
                    cur, improved = best_p, True
            if not improved:
                break
        results[tuple(lus)] = cur

    ranked = sorted(([p, list(k)] for k, p in results.items()),
                    key=lambda b: b[0], reverse=True)
    return ranked[:nsols]


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

    nsols = max(1, min(req.nsols, 50))
    combos = nlu ** ny
    if combos <= EXHAUSTIVE_CAP:
        best = testallMultiSols(ny, nlu, nsols, params_use, lulist_use, optionalparams=optional_use)
        method = "exhaustive"
    else:
        best = _heuristic_search(ny, nlu, nsols, params_use, lulist_use, optional_use)
        method = "heuristic"

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
        "method": method,
        "years": ny,
        "land_uses_count": nlu,
        "rotations": rotations,
    }


# --------------------------------------------------------------------
# 8. Monte-Carlo simulation endpoint — distribution of outcomes
# --------------------------------------------------------------------

class SimulateRequest(BaseModel):
    rotation: RotationInput
    reps: int = 1000                       # number of simulated season sequences
    with_replacement: bool = False         # sample season sequence with/without replacement
    seed: Optional[int] = None             # fix for reproducible distributions
    land_uses: Optional[List[Dict[str, Any]]] = None
    parameters: Optional[Dict[str, Any]] = None
    additional_effects: Optional[List[Dict[str, Any]]] = None
    disallowed_combos: Optional[List[List[Any]]] = None
    # Per-season stochastic multipliers. Each season is a dict with keys
    # IEseason, DEseason, weedSeed, weedComp, NReq, Nlost, plus a yield
    # multiplier per land-use name. If omitted, the bundled default table is used.
    seasons: Optional[List[Dict[str, Any]]] = None
    # Alternative (preferred, member-grounded): a reference (wheat) yield-
    # multiplier per historical year, e.g. APSIM FrostHeatYield / mean. The API
    # expands each into a per-crop season via land_uses[].beta:
    #   crop_mult = 1 + beta * (m - 1)   (weed/disease/N held neutral).
    yield_multipliers: Optional[List[float]] = None
    # Paired price+yield: per-crop detrended price multiplier per historical year,
    # aligned index-for-index with yield_multipliers (same years). Keyed by land-use
    # name, e.g. {"wheat":[0.9,1.1,...], "canola":[...]}. Each season then carries its
    # year's yield AND price together, preserving the real yield-price relationship.
    price_multipliers: Optional[Dict[str, List[float]]] = None
    # Preferred paired mode: raw series, API does the year-alignment. FileMaker passes
    # the member's yield BY YEAR (FrostHeatYield) and the prices block (each crop's
    # decile-5 "price" + full "annual" {year: price}). The API intersects years,
    # builds yield_multipliers + per-crop price_multipliers over the common years.
    yield_by_year: Optional[Dict[str, float]] = None
    prices: Optional[Dict[str, Any]] = None


def _paired_from_series(yield_by_year: Dict[str, float], prices: Optional[Dict[str, Any]]):
    """From the member's FrostHeatYield-by-year and the prices block, build the paired
    yield_multipliers (list over common years) + per-crop price_multipliers. Season pool =
    the yield years intersected with wheat's price years (the reference), so wheat is always
    fully paired; other crops vary where their history covers the year, else sit at median
    (multiplier 1.0). Returns (years, yield_multipliers, price_multipliers)."""
    prices = prices or {}
    yby = {}
    for y, v in yield_by_year.items():
        try:
            yby[int(y)] = float(v)
        except (TypeError, ValueError):
            continue
    if not yby:
        return [], [], {}
    wheat_yrs = set()
    wa = (prices.get("wheat") or {}).get("annual") or {}
    for y in wa:
        try:
            wheat_yrs.add(int(y))
        except (TypeError, ValueError):
            pass
    pool = sorted(y for y in yby if (not wheat_yrs or y in wheat_yrs))
    if len(pool) < 4:            # too little overlap -> fall back to all yield years
        pool = sorted(yby)
    ymean = sum(yby[y] for y in pool) / len(pool)
    if ymean <= 0:
        return [], [], {}
    yield_multipliers = [yby[y] / ymean for y in pool]
    price_multipliers: Dict[str, List[float]] = {}
    for crop, pd in prices.items():
        if not isinstance(pd, dict):
            continue
        try:
            med = float(pd.get("price"))
        except (TypeError, ValueError):
            continue
        if med <= 0:
            continue
        annual = pd.get("annual") or {}
        row = []
        for y in pool:
            a = annual.get(str(y))
            try:
                row.append(float(a) / med if a is not None else 1.0)
            except (TypeError, ValueError):
                row.append(1.0)
        price_multipliers[crop] = row
    return pool, yield_multipliers, price_multipliers


def _coerce_season(s: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in s.items():
        if k == "label":
            out[k] = v
            continue
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            out[k] = v
    return out


def _seasons_from_multipliers(mults: List[Any], lulist: List[dict],
                              price_mults: Optional[Dict[str, List[float]]] = None) -> List[dict]:
    """Expand a reference (wheat) yield-multiplier series into per-crop season
    dicts using each land use's beta. Weed/disease/N effects held neutral. When
    price_mults is given, each season also carries that year's per-crop price
    multiplier (key "<name>_price"), so yield and price stay paired by year."""
    price_mults = price_mults or {}
    seasons: List[dict] = []
    for i, m in enumerate(mults):
        try:
            m = float(m)
        except (TypeError, ValueError):
            continue
        s: Dict[str, Any] = {
            "label": str(i + 1),  # profit() reads season['label']
            "IEseason": 1.0, "DEseason": 1.0, "weedSeed": 1.0,
            "weedComp": 1.0, "NReq": 1.0, "Nlost": 0.0,
        }
        for lu in lulist:
            try:
                beta = float(lu.get("beta") or 1.0)
            except (TypeError, ValueError):
                beta = 1.0
            s[lu["name"]] = 1.0 + beta * (m - 1.0)
            pm = price_mults.get(lu["name"])
            if pm and i < len(pm):
                try:
                    s[lu["name"] + "_price"] = float(pm[i])
                except (TypeError, ValueError):
                    pass
        seasons.append(s)
    return seasons


def _yield_price_correlation(yield_mults, price_mults):
    """Pearson correlation between the reference (wheat) yield multiplier series
    and the wheat price multiplier series, over the paired years. A read-out for
    the member: negative = a natural hedge (poor yields tend to meet firmer prices).
    Returns None if wheat prices weren't supplied or the series is too short."""
    if not price_mults:
        return None
    wp = price_mults.get("wheat")
    if not wp or not yield_mults:
        return None
    n = min(len(yield_mults), len(wp))
    if n < 4:
        return None
    try:
        y = np.array([float(x) for x in yield_mults[:n]], dtype=float)
        p = np.array([float(x) for x in wp[:n]], dtype=float)
    except (TypeError, ValueError):
        return None
    if y.std() == 0 or p.std() == 0:
        return None
    return float(np.corrcoef(y, p)[0, 1])


@app.post("/simulate")
def simulate(req: SimulateRequest):
    """Monte-Carlo the rotation across many sampled season sequences and return
    the distribution of net-profit outcomes (mean, median, p10-p90, probability
    of loss, risk-adjusted ratio, histogram). Uses the same net-profit engine as
    /evaluate. Season variability comes from `seasons` (or the bundled default)."""
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

    indices = indices_from_rotation(req.rotation, lulist_use)
    names = names_from_indices(indices, lulist_use)

    ym_used, pm_used = None, None
    if req.yield_by_year:
        _pool, ym_used, pm_used = _paired_from_series(req.yield_by_year, req.prices)
        seasons = _seasons_from_multipliers(ym_used, lulist_use, pm_used)
    elif req.yield_multipliers:
        ym_used, pm_used = req.yield_multipliers, req.price_multipliers
        seasons = _seasons_from_multipliers(ym_used, lulist_use, pm_used)
    elif req.seasons:
        seasons = [_coerce_season(s) for s in req.seasons]
    else:
        seasons = STOCHMULTS
    if not seasons:
        raise HTTPException(400, "No season data: provide 'yield_multipliers', 'seasons', or bundle a default table.")

    nseason = len(seasons)
    ny = len(indices)
    reps = max(1, min(int(req.reps), 5000))
    random.seed(req.seed if req.seed is not None else 12345)

    profits: List[float] = []
    for _ in range(reps):
        if req.with_replacement or ny > nseason:
            seq = [random.choice(seasons) for _ in range(ny)]
        else:
            seq = random.sample(seasons, ny)
        p = profit(
            indices, params_use, lulist_use,
            optionalparams=optional_use, stochMultsUsed=seq, pureRandomEffects=True,
        )
        profits.append(float(_normalise_value(p)))

    arr = np.array(profits, dtype=float)
    p10, p25, p50, p75, p90 = [float(x) for x in np.percentile(arr, [10, 25, 50, 75, 90])]
    counts, edges = np.histogram(arr, bins=20)
    std = float(arr.std())
    mean = float(arr.mean())

    price_varied = bool(pm_used)
    yp_corr = _yield_price_correlation(ym_used, pm_used) if price_varied else None

    return {
        "rotation_label": req.rotation.label or "-".join(names),
        "sequence_names": names,
        "season_types": nseason,
        "price_varied": price_varied,
        "yield_price_correlation": yp_corr,
        "distribution": {
            "reps": reps,
            "years": ny,
            "mean_total": mean,
            "mean_per_year": mean / ny,
            "median_total": float(np.median(arr)),
            "median_per_year": float(np.median(arr)) / ny,
            "std_dev": std,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "p10": p10, "p25": p25, "p50": p50, "p75": p75, "p90": p90,
            "prob_negative": float((arr < 0).mean()),
            "sharpe_like": (mean / std if std != 0 else None),
            "histogram": {
                "bin_edges": [float(e) for e in edges],
                "bin_counts": [int(c) for c in counts],
            },
        },
    }


# --------------------------------------------------------------------
# 8. Soil matching — pick the AgriYield-Z soil that best matches the
#    member's actual local soil, by soil type then water-holding/metre.
# --------------------------------------------------------------------

class SoilMatchRequest(BaseModel):
    latitude: float
    longitude: float
    available: List[int]              # APSoil numbers simulated at the member's grid cell
    cultivar: Optional[str] = None    # accepted for convenience; not used in the match


def _nearest_apsoil(lat, lon):
    coslat = math.cos(math.radians(lat))
    best, bestd = None, None
    for s in APSOIL_INDEX:
        d = (s["lat"] - lat) ** 2 + ((s["lon"] - lon) * coslat) ** 2
        if bestd is None or d < bestd:
            best, bestd = s, d
    return best


@app.post("/soil-match")
def soil_match(req: SoilMatchRequest):
    """Deterministic base-case soil: find the member's nearest actual soil in
    the APSoil DB, then choose the soil (from those simulated at their grid
    cell) that best matches it by SOIL TYPE first, then water-holding per metre
    (PAWC/m) - so a bleached sand maps to a deep sand, not a shallow clay loam
    that only shares a total-PAWC number because it is shallower."""
    if not APSOIL_INDEX:
        raise HTTPException(500, "APSoil index not loaded on the server.")
    by = {s["apsoil"]: s for s in APSOIL_INDEX}
    cands = [by[a] for a in req.available if a in by]
    if not cands:
        raise HTTPException(400, "None of the supplied 'available' soils are in the APSoil database.")

    local = _nearest_apsoil(req.latitude, req.longitude)
    coslat = math.cos(math.radians(req.latitude))
    local_km = round(math.hypot(local["lat"] - req.latitude,
                                (local["lon"] - req.longitude) * coslat) * 111, 1)
    lt = (local["soiltype"] or "").lower()
    ranked = sorted(cands, key=lambda s: ((s["soiltype"] or "").lower() != lt,
                                          abs(s["pawc_per_m"] - local["pawc_per_m"])))
    best = ranked[0]
    same = (best["soiltype"] or "").lower() == lt
    reason = (
        f"Your nearest mapped soil is {local['name']} ({local['soiltype']}, "
        f"{local['pawc_per_m']} mm/m plant-available water, {local_km} km away). "
        + (f"{best['name']} is the closest available match - same soil type "
           f"({best['soiltype']}) with similar water-holding ({best['pawc_per_m']} mm/m)."
           if same else
           f"No exact soil-type match is simulated here, so {best['name']} "
           f"({best['soiltype']}, {best['pawc_per_m']} mm/m) is the closest by water-holding.")
    )
    return {
        "apsoil": best["apsoil"],
        "name": best["name"],
        "soiltype": best["soiltype"],
        "pawc": best["pawc"],
        "pawc_per_m": best["pawc_per_m"],
        "depth_mm": best["depth_mm"],
        "same_type": same,
        "local": {
            "apsoil": local["apsoil"], "name": local["name"],
            "soiltype": local["soiltype"], "pawc_per_m": local["pawc_per_m"],
            "distance_km": local_km,
        },
        "reason": reason,
        "profile": APSOIL_PROFILES.get(str(best["apsoil"])),
        "candidates": [
            {"apsoil": s["apsoil"], "name": s["name"],
             "soiltype": s["soiltype"], "pawc_per_m": s["pawc_per_m"],
             "pawc": s["pawc"], "depth_mm": s["depth_mm"],
             "profile": APSOIL_PROFILES.get(str(s["apsoil"]))}
            for s in ranked
        ],
    }


@app.get("/soil-profile/{apsoil}")
def soil_profile(apsoil: int):
    """Per-layer water profile (Thickness, AirDry, LL15, DUL, SAT, crop LLs +
    PAWC) for one APSoil - the data behind the volumetric-water-over-depth
    chart. Used both for the auto-selected base case and the override picker."""
    prof = APSOIL_PROFILES.get(str(apsoil))
    if prof is None:
        raise HTTPException(404, f"No water profile for APSoil {apsoil}.")
    meta = next((s for s in APSOIL_INDEX if s["apsoil"] == apsoil), None)
    return {
        "apsoil": apsoil,
        "name": meta["name"] if meta else str(apsoil),
        "soiltype": meta["soiltype"] if meta else prof.get("soiltype"),
        "pawc": meta["pawc"] if meta else None,
        "pawc_per_m": meta["pawc_per_m"] if meta else None,
        "depth_mm": meta["depth_mm"] if meta else None,
        "profile": prof,
    }
