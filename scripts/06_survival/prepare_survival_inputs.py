#!/usr/bin/env python3
"""Prepare harmonized MAPK3 and SMNDC1 tables for patient-level survival analysis."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CONDITIONS = ("Exon4", "NoExon4")


def normalize_case_id(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    match = re.fullmatch(r"(\d+)\.0+", text)
    if match:
        return match.group(1)
    try:
        numeric = float(text)
        if np.isfinite(numeric) and abs(numeric - round(numeric)) < 1e-9:
            return str(int(round(numeric)))
    except ValueError:
        pass
    return text


def normalize_core(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().replace(".ome", "")
    match = re.search(r"(\d+)[_-](\d+)", text)
    if not match:
        return None
    return f"{int(match.group(1))}_{int(match.group(2))}"


def normalize_condition(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower().replace("_", "").replace("-", "")
    if text in {"exon", "exon4", "e"}:
        return "Exon4"
    if text in {"noexon", "noexon4", "noe", "n"}:
        return "NoExon4"
    return None


def core_name(core_norm: str) -> str:
    return f"{core_norm}.ome"


def core_dash(core_norm: str) -> str:
    return core_norm.replace("_", "-")


def require_columns(frame: pd.DataFrame, columns: Iterable[str], source: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def read_grid(xls: pd.ExcelFile, sheet: str) -> dict[tuple[int, int], str]:
    frame = pd.read_excel(xls, sheet_name=sheet)
    frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if frame.empty or frame.shape[1] < 2:
        return {}

    row_column = frame.columns[0]
    numeric_columns: list[object] = []
    for column in frame.columns[1:]:
        try:
            int(str(column).strip())
            numeric_columns.append(column)
        except ValueError:
            continue

    output: dict[tuple[int, int], str] = {}
    for _, row in frame[[row_column] + numeric_columns].iterrows():
        try:
            row_number = int(str(row[row_column]).strip())
        except ValueError:
            continue
        for column in numeric_columns:
            try:
                column_number = int(str(column).strip())
            except ValueError:
                continue
            value = row[column]
            if pd.isna(value):
                continue
            output[(row_number, column_number)] = str(value).strip()
    return output


def build_core_to_cpid(tma_xlsx: str) -> dict[str, str]:
    xls = pd.ExcelFile(tma_xlsx)
    required_sheets = {"TMA_ID", "TMA_Num"}
    missing_sheets = required_sheets.difference(xls.sheet_names)
    if missing_sheets:
        raise ValueError(
            f"{tma_xlsx} is missing required sheets: {', '.join(sorted(missing_sheets))}"
        )

    ids = read_grid(xls, "TMA_ID")
    numbers = read_grid(xls, "TMA_Num")
    mapping: dict[str, str] = {}

    for position, raw_cpid in ids.items():
        cpid = normalize_case_id(raw_cpid)
        if cpid is None:
            continue
        raw_core = numbers.get(position)
        core = normalize_core(raw_core) if raw_core is not None else None
        if core is None:
            row_number, column_number = position
            core = f"{row_number}_{column_number}"
        previous = mapping.get(core)
        if previous is not None and previous != cpid:
            raise ValueError(f"Core {core} maps to more than one CPID: {previous}, {cpid}")
        mapping[core] = cpid
    return mapping


def load_survival_table(
    tumor_xlsx: str,
    cpid_col: str,
    censor_col: str,
    diagnosis_col: str,
    followup_col: str,
    os_col: str,
    os_col_unit: str,
) -> tuple[pd.DataFrame, str]:
    clinical = pd.read_excel(tumor_xlsx)
    require_columns(clinical, [cpid_col, censor_col], "Tumor annotations")

    clinical = clinical.copy()
    clinical["CPID"] = clinical[cpid_col].map(normalize_case_id)
    censor = pd.to_numeric(clinical[censor_col], errors="coerce")
    clinical["event"] = np.where(censor == 0, 1.0, np.where(censor == 1, 0.0, np.nan))

    method = ""
    if diagnosis_col in clinical.columns and followup_col in clinical.columns:
        diagnosis = pd.to_datetime(clinical[diagnosis_col], errors="coerce")
        followup = pd.to_datetime(clinical[followup_col], errors="coerce")
        os_days = (followup - diagnosis).dt.days.astype(float)
        valid_dates = np.isfinite(os_days) & (os_days >= 0)
    else:
        os_days = pd.Series(np.nan, index=clinical.index, dtype=float)
        valid_dates = pd.Series(False, index=clinical.index)

    if valid_dates.sum() > 0:
        clinical["OS_months"] = os_days / 30.44
        method = f"dates:{diagnosis_col}->{followup_col}"
    elif os_col and os_col in clinical.columns:
        values = pd.to_numeric(clinical[os_col], errors="coerce")
        if os_col_unit == "days":
            clinical["OS_months"] = values / 30.44
            method = f"column:{os_col}:days"
        elif os_col_unit == "months":
            clinical["OS_months"] = values
            method = f"column:{os_col}:months"
        else:
            median = float(np.nanmedian(values)) if np.isfinite(values).any() else np.nan
            if np.isfinite(median) and median > 200:
                clinical["OS_months"] = values / 30.44
                method = f"column:{os_col}:auto_days"
            else:
                clinical["OS_months"] = values
                method = f"column:{os_col}:auto_months"
    else:
        raise ValueError(
            "Overall survival could not be derived from diagnosis/follow-up dates, "
            f"and fallback column {os_col!r} was not available."
        )

    survival = clinical[["CPID", "OS_months", "event"]].copy()
    survival["OS_months"] = pd.to_numeric(survival["OS_months"], errors="coerce")
    survival["event"] = pd.to_numeric(survival["event"], errors="coerce")
    survival = survival[
        survival["CPID"].notna()
        & np.isfinite(survival["OS_months"])
        & (survival["OS_months"] >= 0)
        & survival["event"].isin([0, 1])
    ].copy()

    duplicated = survival[survival["CPID"].duplicated(keep=False)]
    if not duplicated.empty:
        inconsistent = (
            duplicated.groupby("CPID")[["OS_months", "event"]]
            .nunique(dropna=False)
            .max(axis=1)
            > 1
        )
        if inconsistent.any():
            bad = inconsistent[inconsistent].index.tolist()[:10]
            raise ValueError(f"Conflicting survival records for CPID(s): {bad}")
    survival = survival.drop_duplicates("CPID", keep="first").reset_index(drop=True)
    return survival, method


def standardize_identity(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    frame = frame.copy()
    if "core_norm" not in frame.columns:
        for alias in ("core_key", "core"):
            if alias in frame.columns:
                frame = frame.rename(columns={alias: "core_norm"})
                break
    if "condition" not in frame.columns:
        for alias in ("class_label", "group", "cohort"):
            if alias in frame.columns:
                frame = frame.rename(columns={alias: "condition"})
                break
    require_columns(frame, ["core_norm", "condition"], source)
    frame["core_norm"] = frame["core_norm"].map(normalize_core)
    frame["condition"] = frame["condition"].map(normalize_condition)
    if frame["core_norm"].isna().any():
        raise ValueError(f"{source} contains core identifiers that cannot be normalized.")
    if frame["condition"].isna().any():
        raise ValueError(f"{source} contains unrecognized condition labels.")
    if "CPID" in frame.columns:
        frame["CPID"] = frame["CPID"].map(normalize_case_id)
    else:
        frame["CPID"] = None
    return frame


def attach_cpid(
    frame: pd.DataFrame,
    core_to_cpid: dict[str, str],
    source: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = frame.copy()
    frame["CPID_from_tma"] = frame["core_norm"].map(core_to_cpid)
    conflicts = frame[
        frame["CPID"].notna()
        & frame["CPID_from_tma"].notna()
        & (frame["CPID"] != frame["CPID_from_tma"])
    ][["core_norm", "CPID", "CPID_from_tma"]].drop_duplicates()

    if not conflicts.empty:
        preview = conflicts.head(10).to_dict("records")
        raise ValueError(f"CPID conflicts in {source}: {preview}")

    frame["CPID"] = frame["CPID"].fillna(frame["CPID_from_tma"])
    frame = frame.drop(columns=["CPID_from_tma"])
    return frame, conflicts


def make_unit_table(
    frame: pd.DataFrame,
    protein: str,
    population: str,
    unit_id_col: str,
    area_col: str,
    positive_area_col: str | None,
    ratio_col: str,
) -> pd.DataFrame:
    require_columns(frame, ["core_norm", "condition", area_col, ratio_col], f"{protein} {population}")
    output = frame.copy()

    if unit_id_col not in output.columns:
        output[unit_id_col] = np.arange(1, len(output) + 1, dtype=np.int64)

    output["unit_area_px"] = pd.to_numeric(output[area_col], errors="coerce")
    output["protein_ratio"] = pd.to_numeric(output[ratio_col], errors="coerce")

    if positive_area_col and positive_area_col in output.columns:
        output["protein_positive_area_px"] = pd.to_numeric(
            output[positive_area_col], errors="coerce"
        )
    else:
        output["protein_positive_area_px"] = (
            output["protein_ratio"] * output["unit_area_px"]
        )

    output["unit_id"] = output[unit_id_col]
    output["protein"] = protein
    output["population"] = population
    output["class_label"] = output["condition"]
    output["core_name"] = output["core_norm"].map(core_name)
    output["core_dash"] = output["core_norm"].map(core_dash)

    keep = [
        "protein",
        "population",
        "condition",
        "class_label",
        "core_norm",
        "core_name",
        "core_dash",
        "CPID",
        "unit_id",
        "unit_area_px",
        "protein_positive_area_px",
        "protein_ratio",
    ]
    output = output[keep].copy()
    output = output[
        output["CPID"].notna()
        & np.isfinite(output["unit_area_px"])
        & (output["unit_area_px"] > 0)
        & np.isfinite(output["protein_positive_area_px"])
        & (output["protein_positive_area_px"] >= 0)
        & np.isfinite(output["protein_ratio"])
        & output["protein_ratio"].between(0, 1, inclusive="both")
    ].copy()

    output = output.drop_duplicates(
        ["condition", "core_norm", "unit_id"], keep="first"
    ).reset_index(drop=True)
    return output


def attach_survival(units: pd.DataFrame, survival: pd.DataFrame) -> pd.DataFrame:
    merged = units.merge(survival, on="CPID", how="left", validate="many_to_one")
    return merged[
        np.isfinite(pd.to_numeric(merged["OS_months"], errors="coerce"))
        & pd.to_numeric(merged["event"], errors="coerce").isin([0, 1])
    ].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare harmonized MAPK3 and SMNDC1 unit-level inputs for survival analyses."
    )
    parser.add_argument("--mapk3_global_csv", required=True)
    parser.add_argument("--smndc1_global_csv", required=True)
    parser.add_argument("--registered_csv", required=True)
    parser.add_argument("--tma_xlsx", required=True)
    parser.add_argument("--tumor_xlsx", required=True)
    parser.add_argument("--outdir", required=True)

    parser.add_argument("--cpid_col", default="CPID")
    parser.add_argument("--censor_col", default="OS_censor")
    parser.add_argument("--diagnosis_col", default="Date_of_diagnosis")
    parser.add_argument("--followup_col", default="Date_merged_last_followup_death")
    parser.add_argument("--os_col", default="OS")
    parser.add_argument("--os_col_unit", choices=["auto", "days", "months"], default="auto")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    core_to_cpid = build_core_to_cpid(args.tma_xlsx)
    survival, survival_method = load_survival_table(
        args.tumor_xlsx,
        cpid_col=args.cpid_col,
        censor_col=args.censor_col,
        diagnosis_col=args.diagnosis_col,
        followup_col=args.followup_col,
        os_col=args.os_col,
        os_col_unit=args.os_col_unit,
    )

    mapk3_global = standardize_identity(pd.read_csv(args.mapk3_global_csv), "MAPK3 Global")
    smndc1_global = standardize_identity(pd.read_csv(args.smndc1_global_csv), "SMNDC1 Global")
    registered = standardize_identity(pd.read_csv(args.registered_csv), "Registered table")

    mapk3_global, _ = attach_cpid(mapk3_global, core_to_cpid, "MAPK3 Global")
    smndc1_global, _ = attach_cpid(smndc1_global, core_to_cpid, "SMNDC1 Global")
    registered, _ = attach_cpid(registered, core_to_cpid, "Registered table")

    if "mapk3_positive_area_px" not in mapk3_global.columns and "area_protein_in_cell_px" in mapk3_global.columns:
        mapk3_global["mapk3_positive_area_px"] = pd.to_numeric(
            mapk3_global["area_protein_in_cell_px"], errors="coerce"
        )

    if "smndc1_positive_area_px" not in smndc1_global.columns:
        if {"smndc1_positive_area_fraction", "area_nuc_px"}.issubset(smndc1_global.columns):
            smndc1_global["smndc1_positive_area_px"] = (
                pd.to_numeric(smndc1_global["smndc1_positive_area_fraction"], errors="coerce")
                * pd.to_numeric(smndc1_global["area_nuc_px"], errors="coerce")
            )
        elif {"ratio_S_nuc", "area_nuc_px"}.issubset(smndc1_global.columns):
            smndc1_global["smndc1_positive_area_px"] = (
                pd.to_numeric(smndc1_global["ratio_S_nuc"], errors="coerce")
                * pd.to_numeric(smndc1_global["area_nuc_px"], errors="coerce")
            )

    if "ratio_S_nuc" not in smndc1_global.columns and "smndc1_positive_area_fraction" in smndc1_global.columns:
        smndc1_global["ratio_S_nuc"] = pd.to_numeric(
            smndc1_global["smndc1_positive_area_fraction"], errors="coerce"
        )

    if "smndc1_positive_area_px" not in registered.columns:
        if {"smndc1_positive_area_fraction", "area_nuc_px"}.issubset(registered.columns):
            registered["smndc1_positive_area_px"] = (
                pd.to_numeric(registered["smndc1_positive_area_fraction"], errors="coerce")
                * pd.to_numeric(registered["area_nuc_px"], errors="coerce")
            )
        elif {"ratio_S_nuc", "area_nuc_px"}.issubset(registered.columns):
            registered["smndc1_positive_area_px"] = (
                pd.to_numeric(registered["ratio_S_nuc"], errors="coerce")
                * pd.to_numeric(registered["area_nuc_px"], errors="coerce")
            )

    if "mapk3_positive_area_px" not in registered.columns and "area_protein_in_cell_px" in registered.columns:
        registered["mapk3_positive_area_px"] = pd.to_numeric(
            registered["area_protein_in_cell_px"], errors="coerce"
        )

    mapk3_global_units = make_unit_table(
        mapk3_global,
        protein="MAPK3",
        population="Global",
        unit_id_col="cell_id",
        area_col="area_cyto_px",
        positive_area_col="mapk3_positive_area_px",
        ratio_col="ratio_M_cyto",
    )
    smndc1_global_units = make_unit_table(
        smndc1_global,
        protein="SMNDC1",
        population="Global",
        unit_id_col="nucleus_id",
        area_col="area_nuc_px",
        positive_area_col="smndc1_positive_area_px",
        ratio_col="ratio_S_nuc",
    )
    mapk3_registered_units = make_unit_table(
        registered,
        protein="MAPK3",
        population="Registered",
        unit_id_col="mapk3_cell_id",
        area_col="area_cyto_px",
        positive_area_col="mapk3_positive_area_px",
        ratio_col="ratio_M_cyto",
    )
    smndc1_registered_units = make_unit_table(
        registered,
        protein="SMNDC1",
        population="Registered",
        unit_id_col="nucleus_id",
        area_col="area_nuc_px",
        positive_area_col="smndc1_positive_area_px",
        ratio_col="ratio_S_nuc",
    )

    tables = {
        "MAPK3_Global": attach_survival(mapk3_global_units, survival),
        "MAPK3_Registered": attach_survival(mapk3_registered_units, survival),
        "SMNDC1_Global": attach_survival(smndc1_global_units, survival),
        "SMNDC1_Registered": attach_survival(smndc1_registered_units, survival),
    }

    output_paths: dict[str, str] = {}
    for name, table in tables.items():
        path = outdir / f"survival_units_{name}.csv"
        table.to_csv(path, index=False)
        output_paths[name] = str(path)

    survival_by_patient = survival.sort_values("CPID").reset_index(drop=True)
    survival_by_patient.to_csv(outdir / "survival_by_patient.csv", index=False)

    core_rows = []
    for core, cpid in sorted(core_to_cpid.items()):
        core_rows.append({"core_norm": core, "core_name": core_name(core), "core_dash": core_dash(core), "CPID": cpid})
    survival_by_core = pd.DataFrame(core_rows).merge(
        survival_by_patient, on="CPID", how="left", validate="many_to_one"
    )
    survival_by_core.to_csv(outdir / "survival_by_core.csv", index=False)

    summary_rows = []
    for name, table in tables.items():
        summary_rows.append(
            {
                "dataset": name,
                "n_units": int(len(table)),
                "n_cores": int(table["core_norm"].nunique()),
                "n_patients": int(table["CPID"].nunique()),
                "n_exon4_units": int((table["condition"] == "Exon4").sum()),
                "n_noexon4_units": int((table["condition"] == "NoExon4").sum()),
            }
        )
    pd.DataFrame(summary_rows).to_csv(outdir / "preparation_summary.csv", index=False)

    metadata = {
        "survival_method": survival_method,
        "event_definition": f"event=1 when {args.censor_col}=0; event=0 when {args.censor_col}=1",
        "outputs": output_paths,
        "n_survival_patients": int(len(survival_by_patient)),
    }
    with open(outdir / "preparation_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"[OK] Prepared survival inputs in: {outdir}")
    for name, table in tables.items():
        print(f"  {name}: {len(table):,} units, {table['CPID'].nunique():,} patients")


if __name__ == "__main__":
    main()
