#!/usr/bin/env python  
"""filter_resolution_from_api.py  
  
Fetch resolution for each PDB ID via RCSB GraphQL API,  
then filter CSV files to remove entries with resolution > cutoff.  
  
Usage:  
    python scripts/filter_resolution_from_api.py \  
        --output_dir ~/dataset/RCSB_PDB/blob_fixed \  
        --resolution_cutoff 3.5  
"""  
  
import argparse  
import json  
import logging  
import os  
import time  
from pathlib import Path  
  
import pandas as pd  
import requests  
  
logging.basicConfig(  
    level=logging.INFO,  
    format="%(asctime)s [%(levelname)s] %(message)s",  
    datefmt="%Y-%m-%d %H:%M:%S",  
)  
log = logging.getLogger(__name__)  
  
GRAPHQL_URL = "https://data.rcsb.org/graphql"  
  
GRAPHQL_QUERY = """  
query($ids: [String!]!) {  
  entries(entry_ids: $ids) {  
    rcsb_id  
    rcsb_entry_info {  
      resolution_combined  
    }  
  }  
}  
"""  
  
  
def fetch_resolutions_batch(pdb_ids: list[str], max_retries: int = 3) -> dict[str, float | None]:  
    """Fetch resolution for a batch of PDB IDs via RCSB GraphQL API."""  
    # RCSB expects uppercase  
    upper_ids = [pid.upper() for pid in pdb_ids]  
    result = {}  
  
    for attempt in range(1, max_retries + 1):  
        try:  
            resp = requests.post(  
                GRAPHQL_URL,  
                json={"query": GRAPHQL_QUERY, "variables": {"ids": upper_ids}},  
                timeout=60,  
            )  
            resp.raise_for_status()  
            data = resp.json()  
  
            if "data" not in data or data["data"] is None:  
                log.warning("GraphQL returned no data (attempt %d/%d)", attempt, max_retries)  
                if attempt < max_retries:  
                    time.sleep(2 ** attempt)  
                    continue  
                # give up, mark all as None  
                for pid in pdb_ids:  
                    result.setdefault(pid.lower(), None)  
                return result  
  
            for entry in data["data"].get("entries") or []:  
                if entry is None:  
                    continue  
                rcsb_id = entry["rcsb_id"].lower()  
                info = entry.get("rcsb_entry_info") or {}  
                res_list = info.get("resolution_combined")  
                if res_list and len(res_list) > 0:  
                    result[rcsb_id] = res_list[0]  
                else:  
                    result[rcsb_id] = None  
  
            # mark any missing IDs as None  
            for pid in pdb_ids:  
                result.setdefault(pid.lower(), None)  
            return result  
  
        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:  
            log.warning("Request failed (attempt %d/%d): %s", attempt, max_retries, e)  
            if attempt < max_retries:  
                time.sleep(2 ** attempt)  
  
    # all retries exhausted  
    for pid in pdb_ids:  
        result.setdefault(pid.lower(), None)  
    return result  
  
  
def fetch_all_resolutions(  
    pdb_ids: list[str],  
    batch_size: int = 200,  
    cache_path: str | None = None,  
) -> dict[str, float | None]:  
    """Fetch resolution for all PDB IDs, with caching."""  
    cache: dict[str, float | None] = {}  
    if cache_path and os.path.exists(cache_path):  
        log.info("Loading resolution cache from %s", cache_path)  
        with open(cache_path) as f:  
            cache = json.load(f)  
        log.info("  Cached entries: %d", len(cache))  
  
    # figure out which IDs still need fetching  
    to_fetch = [pid for pid in pdb_ids if pid.lower() not in cache]  
    log.info("Total unique PDB IDs: %d, already cached: %d, to fetch: %d",  
             len(pdb_ids), len(pdb_ids) - len(to_fetch), len(to_fetch))  
  
    for i in range(0, len(to_fetch), batch_size):  
        batch = to_fetch[i : i + batch_size]  
        batch_num = i // batch_size + 1  
        total_batches = (len(to_fetch) + batch_size - 1) // batch_size  
        log.info("  Fetching batch %d/%d (%d IDs)...", batch_num, total_batches, len(batch))  
        res = fetch_resolutions_batch(batch)  
        cache.update(res)  
  
        # save cache periodically (every 10 batches)  
        if cache_path and batch_num % 10 == 0:  
            with open(cache_path, "w") as f:  
                json.dump(cache, f)  
  
        # be polite to the API  
        time.sleep(0.2)  
  
    # final cache save  
    if cache_path:  
        with open(cache_path, "w") as f:  
            json.dump(cache, f)  
        log.info("Resolution cache saved to %s (%d entries)", cache_path, len(cache))  
  
    return cache  
  
  
def main():  
    parser = argparse.ArgumentParser(description="Fetch resolution and filter CSVs")  
    parser.add_argument("--output_dir", required=True, help="Directory containing the CSV files")  
    parser.add_argument("--resolution_cutoff", type=float, default=3.5,  
                        help="Remove entries with resolution > this value (default: 3.5)")  
    parser.add_argument("--batch_size", type=int, default=200,  
                        help="Number of PDB IDs per GraphQL request (default: 200)")  
    parser.add_argument("--cache_file", default=None,  
                        help="Path to resolution cache JSON (default: {output_dir}/resolution_cache.json)")  
    args = parser.parse_args()  
  
    output_dir = Path(args.output_dir)  
    cache_path = args.cache_file or str(output_dir / "resolution_cache.json")  
  
    csv_names = ["all_entries.csv", "train_pdb_data.csv", "valid_pdb_data.csv", "test_pdb_data.csv"]  
    csv_paths = {name: output_dir / name for name in csv_names}  
  
    # check existence  
    for name, path in csv_paths.items():  
        if not path.exists():  
            log.warning("CSV not found: %s (will skip)", path)  
  
    # ---- Step 1: Read all_entries.csv to get unique PDB IDs ----  
    all_csv_path = csv_paths["all_entries.csv"]  
    if not all_csv_path.exists():  
        log.error("all_entries.csv not found in %s", output_dir)  
        return  
  
    df_all = pd.read_csv(all_csv_path)  
    unique_ids = df_all["pdb_id"].dropna().unique().tolist()  
    log.info("Unique PDB IDs in all_entries.csv: %d", len(unique_ids))  
  
    # ---- Step 2: Fetch resolutions ----  
    resolution_map = fetch_all_resolutions(unique_ids, batch_size=args.batch_size, cache_path=cache_path)  
  
    # ---- Step 3: Process each CSV ----  
    all_removed = []  
  
    for name, path in csv_paths.items():  
        if not path.exists():  
            continue  
  
        df = pd.read_csv(path)  
        n_before = len(df)  
  
        # fill in resolution from API  
        df["resolution"] = df["pdb_id"].map(  
            lambda pid: resolution_map.get(str(pid).lower())  
        )  
  
        # identify rows to remove: resolution known and > cutoff  
        mask_remove = df["resolution"].notna() & (df["resolution"] > args.resolution_cutoff)  
        removed = df[mask_remove].copy()  
        removed["reason"] = f"resolution > {args.resolution_cutoff}"  
        removed["source_csv"] = name  
        all_removed.append(removed)  
  
        # identify rows with unknown resolution  
        mask_unknown = df["resolution"].isna()  
        n_unknown = mask_unknown.sum()  
  
        # keep rows that pass  
        df_filtered = df[~mask_remove]  
        n_after = len(df_filtered)  
        n_removed = n_before - n_after  
  
        log.info("[%s] before=%d, removed=%d (res>%.1f), remaining=%d, unknown_resolution=%d",  
                 name, n_before, n_removed, args.resolution_cutoff, n_after, n_unknown)  
  
        # backup original  
        backup_path = path.with_name(path.stem + "_backup.csv")  
        if not backup_path.exists():  
            df_orig = pd.read_csv(path)  
            df_orig.to_csv(backup_path, index=False)  
            log.info("  Backup saved: %s", backup_path)  
  
        # overwrite  
        df_filtered.to_csv(path, index=False)  
  
    # ---- Step 4: Save removed entries ----  
    if all_removed:  
        df_removed = pd.concat(all_removed, ignore_index=True)  
        # deduplicate (same entry appears in all_entries + train/valid/test)  
        removed_path = output_dir / "removed_by_resolution.csv"  
        df_removed.to_csv(removed_path, index=False)  
        log.info("Removed entries saved to %s (%d rows total across all CSVs)", removed_path, len(df_removed))  
  
        # unique PDB IDs removed  
        unique_removed = df_removed["pdb_id"].nunique()  
        log.info("Unique PDB IDs removed: %d", unique_removed)  
    else:  
        log.info("No entries removed.")  
  
    # ---- Step 5: Summary ----  
    n_total = len(resolution_map)  
    n_has_res = sum(1 for v in resolution_map.values() if v is not None)  
    n_no_res = n_total - n_has_res  
    log.info("=== Summary ===")  
    log.info("  Total unique PDB IDs: %d", n_total)  
    log.info("  With resolution: %d", n_has_res)  
    log.info("  Without resolution (kept): %d", n_no_res)  
    log.info("  Resolution cutoff: %.1f", args.resolution_cutoff)  
    log.info("Done.")  
  
  
if __name__ == "__main__":  
    main()