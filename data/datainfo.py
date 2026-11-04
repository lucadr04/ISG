import pandas as pd
import numpy as np
import json
import os

OUTPUTS = ["Activity", "org:resource", "time:timestamp"]


def get_redundant_event_attributes(tab_all, categorical_columns, outputs):
    """Individua gli event attribute dinamici (che cambiano all'interno di un caso),
    che vengono non generati dal modello ed esclusi dal grafo."""
    redundant = set()
    for col in categorical_columns:
        if col in outputs:
            continue
        case_unique = tab_all.groupby("CaseID")[col].nunique()
        if (case_unique > 1).any():
            redundant.add(col)
    return redundant


def compute_avg_duration(tab_all):
    """Gestisce sia timestamp già normalizzati (float) sia date vere."""
    ts = tab_all["time:timestamp"]
    if pd.api.types.is_numeric_dtype(ts):
        grouped = tab_all.groupby("CaseID")["time:timestamp"]
        durations = grouped.max() - grouped.min()
    else:
        parsed = pd.to_datetime(ts, errors="coerce")
        tmp = tab_all.copy()
        tmp["_ts"] = parsed
        grouped = tmp.groupby("CaseID")["_ts"]
        durations = (grouped.max() - grouped.min()).dt.total_seconds() / 86400
    return durations.mean()


def compute_dataset_stats(dataset_name, root_path):
    processed_path = f"{root_path}/data/datasets/processed/{dataset_name}"

    with open(f"{root_path}/data/dataset_features.json") as f:
        dataset_info = json.load(f)[dataset_name]

    categorical_columns = dataset_info["categorical"]
    real_value_columns = dataset_info["numerical"]

    tab_all = pd.read_csv(f"{processed_path}/{dataset_name}_processed_all.csv")
    tab_train = pd.read_csv(f"{processed_path}/{dataset_name}_processed_train.csv")
    tab_valid = pd.read_csv(f"{processed_path}/{dataset_name}_processed_valid.csv")
    tab_test = pd.read_csv(f"{processed_path}/{dataset_name}_processed_test.csv")

    for k in categorical_columns:
        tab_all[k] = tab_all[k].astype("object")

    # --- PRE-PROCESSING TEMPORALE ---
    # Creiamo una colonna temporale sicura e ordiniamo per CaseID e tempo 
    # per garantire che i calcoli sequenziali siano corretti
    if pd.api.types.is_numeric_dtype(tab_all["time:timestamp"]):
        tab_all["_ts"] = tab_all["time:timestamp"]
    else:
        tab_all["_ts"] = pd.to_datetime(tab_all["time:timestamp"], errors="coerce")
        
    tab_all = tab_all.sort_values(by=["CaseID", "_ts"]).reset_index(drop=True)

    # --- Conteggi di base ---
    n_cases_total = tab_all["CaseID"].nunique()
    n_cases_train = tab_train["CaseID"].nunique()
    n_cases_valid = tab_valid["CaseID"].nunique()
    n_cases_test = tab_test["CaseID"].nunique()
    n_events_total = len(tab_all)

    n_activities = tab_all["Activity"].nunique() if "Activity" in tab_all.columns else None
    n_resources = tab_all["org:resource"].nunique() if "org:resource" in tab_all.columns else None

    # --- Lunghezza tracce ---
    trace_lens = tab_all.groupby("CaseID").size()
    avg_len = trace_lens.mean()
    median_len = int(np.median(trace_lens))
    min_len = int(trace_lens.min())
    max_len = int(trace_lens.max())

    # --- Durata media e Tempi di Intermezzo ---
    try:
        avg_duration_days = compute_avg_duration(tab_all)
    except Exception as e:
        avg_duration_days = None
        print(f"  [!] durata non calcolabile per {dataset_name}: {e}")

    # Gap temporale tra gli eventi in ore
    tab_all["prev_time"] = tab_all.groupby("CaseID")["_ts"].shift(1)
    if pd.api.types.is_numeric_dtype(tab_all["_ts"]):
        # Se numerico, assumiamo sia in secondi (adattalo se sono ms o giorni)
        event_gaps_hours = (tab_all["_ts"] - tab_all["prev_time"]) / 3600
    else:
        event_gaps_hours = (tab_all["_ts"] - tab_all["prev_time"]).dt.total_seconds() / 3600
    avg_event_gap_hours = event_gaps_hours.mean()

    # --- Analisi del Flusso di Controllo ---
    # Varianti (Sequenze uniche di attività)
    traces = tab_all.groupby("CaseID")["Activity"].apply(tuple)
    n_variants = traces.nunique()
    top_5_variant_coverage_perc = (traces.value_counts().head(5).sum() / n_cases_total) * 100

    # Attività di Start ed End
    n_start_activities = tab_all.groupby("CaseID")["Activity"].first().nunique()
    n_end_activities = tab_all.groupby("CaseID")["Activity"].last().nunique()

    # Rework (Self-Loops immediati)
    is_self_loop = (tab_all["Activity"] == tab_all.groupby("CaseID")["Activity"].shift(1))
    self_loop_perc = is_self_loop.mean() * 100

    # --- Attributi statici vs esclusi (dinamici) ---
    excluded_event_attrs = get_redundant_event_attributes(tab_all, categorical_columns, OUTPUTS)
    
    static_attrs = [
        c for c in (categorical_columns + real_value_columns)
        if c not in OUTPUTS and c not in excluded_event_attrs
    ]
    
    n_trace_attrs = len(static_attrs)
    n_excluded_event_attrs = len(excluded_event_attrs)
    n_total_attrs = len(categorical_columns) + len(real_value_columns)

    # --- Dimensione test set per i due protocolli ---
    test_trace_lens = tab_test.groupby("CaseID").size()
    n_test_prefix2 = int((test_trace_lens > 2).sum())
    n_test_median = int((test_trace_lens > median_len).sum())

    return {
        "dataset": dataset_name,
        "n_cases_total": n_cases_total,
        "n_cases_train": n_cases_train,
        "n_cases_valid": n_cases_valid,
        "n_cases_test": n_cases_test,
        "n_events_total": n_events_total,
        "n_activities": n_activities,
        "n_resources": n_resources,
        # Traces Lengths
        "avg_trace_len": round(avg_len, 2),
        "median_trace_len": median_len,
        "min_trace_len": min_len,
        "max_trace_len": max_len,
        # Temporal Stats
        "avg_duration_days": round(avg_duration_days, 2) if avg_duration_days is not None else None,
        "avg_event_gap_hours": round(avg_event_gap_hours, 2) if pd.notna(avg_event_gap_hours) else None,
        # Control-Flow Stats
        "n_variants": n_variants,
        "top_5_variant_coverage_perc": round(top_5_variant_coverage_perc, 2),
        "n_start_activities": n_start_activities,
        "n_end_activities": n_end_activities,
        "self_loop_perc": round(self_loop_perc, 2),
        # Attributes Stats
        "n_trace_attrs": n_trace_attrs,
        "n_excluded_event_attrs": n_excluded_event_attrs,
        "excluded_event_attrs_list": sorted(excluded_event_attrs),
        "n_total_attrs": n_total_attrs,
        # Evaluation Stats
        "n_test_prefix2": n_test_prefix2,
        "n_test_median": n_test_median,
    }


# --------------------------------------------------------------------- #
# Esecuzione su tutti i dataset
# --------------------------------------------------------------------- #
ROOT_PATH = os.path.dirname(os.getcwd())
DATASETS = ["bpi_2012", "bpi_2013", "sp2020", "BPI20_RequestForPayment"]

all_stats = []
for ds in DATASETS:
    try:
        s = compute_dataset_stats(ds, ROOT_PATH)
        all_stats.append(s)
        print(f"[OK]   {ds}")
    except Exception as e:
        print(f"[FAIL] {ds}: {e}")

df_stats = pd.DataFrame(all_stats)
with open('stats.txt', 'w') as f:
    f.write(df_stats.to_string(index=False))