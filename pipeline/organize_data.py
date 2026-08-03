import os
import glob
import csv
import datetime
import concurrent.futures
from collections import defaultdict
import cdflib
import pandas as pd
from tqdm import tqdm

DATA_ROOT = "../data"
DOSE_RATES_FILE = os.path.join(DATA_ROOT, "dose-rates/dose-rates.txt")
GOES_SEISS_DIR = os.path.join(DATA_ROOT, "gcr-data/goes_seiss_data")
SIS_DIR = os.path.join(DATA_ROOT, "sep-data/sis-data")
EPAM_DIR = os.path.join(DATA_ROOT, "sep-data/epam-data")
ACE_CRIS_DIR = os.path.join(DATA_ROOT, "gcr-data/ace-cris-data")
ALBEDO_FILE = os.path.join(DATA_ROOT, "albedo-map.csv")
OUTPUT_FILE = "../merged_lunar_radiation.csv"
MAX_WORKERS = 32

PRELOADED_ACE = {}

def log(msg):
    print(f"[LOG {datetime.datetime.now()}] {msg}")

def julian_date_to_datetime(jd):
    jd = float(jd)
    JD_UNIX_EPOCH = 2440587.5
    SECONDS_PER_DAY = 86400
    seconds_since_epoch = (jd - JD_UNIX_EPOCH) * SECONDS_PER_DAY
    dt = datetime.datetime.utcfromtimestamp(seconds_since_epoch)
    return dt

def parse_doserate_line(row):
    try:
        timestamp = julian_date_to_datetime(row[0])
        timestamp = timestamp.replace(minute=0, second=0, microsecond=0)
        values = list(map(str.strip, row))
        return timestamp, {
            "dose_julian": values[0],
            "dose_year_fraction": values[3],
            "dose_h2o_factor": values[4],
            "dose_altitude_factor": values[5],
            "dose_good_event_factor": values[6],
            "dose_D12": values[7],
            "dose_D34": values[8],
            "dose_D56": values[9],
            "dose_D1": values[10],
            "dose_D2": values[11],
            "dose_D3": values[12],
            "dose_D4": values[13],
            "dose_D5": values[14],
            "dose_D6": values[15],
        }
    except Exception as e:
        log(f"Failed to parse dose rate line: {e}, line: {row}")
        return None, None

def read_dose_rates():
    log("Reading CRaTER dose rate data...")
    dose_rates = []
    with open(DOSE_RATES_FILE, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or row[0].startswith("#") or row[0].lower().startswith("julian"):
                continue
            ts, data = parse_doserate_line(row)
            if ts and data:
                dose_rates.append((ts, data))
    log(f"Loaded {len(dose_rates)} dose rate entries.")
    return dose_rates

def get_epam_data_for_hour(hour):
    year = str(hour.year)
    folder = os.path.join(EPAM_DIR, year)
    if not os.path.exists(folder):
        return []

    rows = []
    for file in glob.glob(os.path.join(folder, "*.txt")):
        with open(file, "r") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.strip().split()
                try:
                    ts = datetime.datetime.strptime(parts[0], "%Y-%m-%dT%H:%M:%S")
                    if ts.replace(minute=0, second=0, microsecond=0) == hour:
                        row_data = {"epam_col" + str(i): val for i, val in enumerate(parts[1:], start=1)}
                        row_data["timestamp"] = ts.isoformat()
                        rows.append(row_data)
                except:
                    continue
    return rows

def get_goes_data_for_hour(hour):
    year = str(hour.year)
    month = f"{hour.month:02d}"
    folder = os.path.join(GOES_SEISS_DIR, year, month)
    if not os.path.exists(folder):
        return []

    rows = []
    for file in glob.glob(os.path.join(folder, "*.csv")):
        with open(file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts = datetime.datetime.strptime(row["time_tag"], "%Y-%m-%dT%H:%M:%SZ")
                    if ts.replace(minute=0, second=0, microsecond=0) == hour:
                        row_clean = {"goes_" + k: v for k, v in row.items() if k != "time_tag"}
                        row_clean["timestamp"] = ts.isoformat()
                        rows.append(row_clean)
                except:
                    continue
    return rows

def flatten_array_to_str(arr):
    if isinstance(arr, (list, tuple)):
        return ",".join(str(x) for x in arr)
    try:
        import numpy as np
        if isinstance(arr, np.ndarray):
            return ",".join(str(x) for x in arr.flatten())
    except ImportError:
        pass
    return str(arr)

def get_sis_data_for_hour(hour):
    year = str(hour.year)
    folder = os.path.join(SIS_DIR, year)
    if not os.path.exists(folder):
        return []

    rows = []
    for file in glob.glob(os.path.join(folder, "*.cdf")):
        if f"{hour.year}{hour.month:02d}{hour.day:02d}" not in file:
            continue
        try:
            print("SIS file found!: " + f"{hour.year}{hour.month:02d}{hour.day:02d}")
            cdf = cdflib.CDF(file)
            epoch_vals = cdf.varget("Epoch")
            timestamps = [cdflib.cdfepoch.to_datetime(ev) for ev in epoch_vals]

            for i, ts in enumerate(timestamps):
                if pd.to_datetime(ts[0]).to_pydatetime().replace(minute=0, second=0, microsecond=0) == hour:
                    print("SIS Match found!")
                    row = {}
                    for varname in cdf.cdf_info().zVariables:
                        if varname == "Epoch" or varname == "cnt_Al":
                            continue
                        val = cdf.varget(varname)
                        try:
                            val_i = val[i]
                            if hasattr(val_i, "__len__") and not isinstance(val_i, str):
                                val_i = ",".join(str(x) for x in val_i)
                            row[f"sis_{varname}"] = val_i
                        except Exception:
                            row[f"sis_{varname}"] = val
                    row["timestamp"] = pd.to_datetime(ts[0]).to_pydatetime().replace(minute=0, second=0, microsecond=0).isoformat()
                    rows.append(row)
        except Exception as e:
            print(f"[SIS] Failed to read {file}: {e}")
    return rows

def doy_to_month_day(year, doy):
    doy = int(doy)
    date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=doy - 1)
    return date.month, date.day

def preload_ace_data():
    log("Preloading ACE CRIS data...")
    file_path = os.path.join(ACE_CRIS_DIR, 'ASCBCrOp3.txt')
    if not os.path.exists(file_path):
        log(f"[ACE] File not found: {file_path}")
        return

    element_list = ["B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S",
                    "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni"]
    num_energies = 7

    with open(file_path, "r") as f:
        in_metadata_section = True
        lines = f.readlines()
        for line in tqdm(lines, desc="Loading ACE data"):
            if line.strip().startswith("BEGIN DATA"):
                in_metadata_section = False
                continue
            if in_metadata_section or not line.strip() or line.startswith("#"):
                continue

            parts = line.strip().split()
            if len(parts) < 331:
                continue

            try:
                year = int(parts[0])
                doy = int(parts[1])
                hr = int(parts[2])
                minute = int(parts[3])
                second = int(float(parts[4]))
                month, day = doy_to_month_day(year, doy)
                ts = datetime.datetime(year, month, day, hr, minute, second)
                hour_key = ts.replace(minute=0, second=0, microsecond=0)

                row = {"timestamp": ts.isoformat(), "ace_fp_year": parts[5], "ace_fp_doy": parts[6], "ace_epoch": parts[7]}
                idx = 8
                for el in element_list:
                    for e in range(num_energies):
                        row[f"ace_flux_{el}_{e+1}"] = parts[idx]
                        idx += 1
                for el in element_list:
                    for e in range(num_energies):
                        row[f"ace_cnt_{el}_{e+1}"] = parts[idx]
                        idx += 1
                row["ace_livetime"] = parts[idx] if idx < len(parts) else ""

                PRELOADED_ACE.setdefault(hour_key, []).append(row)
            except Exception:
                continue
    log("Finished preloading ACE data.")

def get_ace_cris_data_for_hour(hour):
    return PRELOADED_ACE.get(hour, [])

def get_albedo_data_for_hour(hour):
    if not os.path.exists(ALBEDO_FILE):
        return {}

    try:
        with open(ALBEDO_FILE, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                return {"albedo_" + k: v for k, v in row.items()}
    except Exception as e:
        log(f"Failed to read albedo data: {e}")
    return {}

def process_hour(entry):
    hour, dose_data = entry
    log(f"Processing hour {hour}...")

    sis_rows = get_sis_data_for_hour(hour)
    ace_rows = get_ace_cris_data_for_hour(hour)

    if not sis_rows or not ace_rows:
        log(f"Skipping {hour} due to missing SIS or ACE data.")
        return []

    combined_rows = []
    for sis in sis_rows:
        sis_ts = sis["timestamp"]
        matching_ace = next((a for a in ace_rows if a["timestamp"] == sis_ts), None)
        if not matching_ace:
            continue

        combined = {
            "timestamp": sis_ts,
            **dose_data,
            **sis,
            **matching_ace
        }
        combined_rows.append(combined)

    if not combined_rows:
        log(f"No matching timestamps found for SIS and ACE at hour {hour}")

    return combined_rows

def merge_and_write():
    dose_data = read_dose_rates()
    preload_ace_data()
    log("Beginning concurrent processing of each hour...")
    all_rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = list(executor.map(process_hour, dose_data))
        for result in futures:
            all_rows.extend(result)

    if not all_rows:
        log("No data merged, exiting.")
        return

    log(f"Writing {len(all_rows)} merged records to {OUTPUT_FILE}...")
    all_fieldnames = sorted({k for row in all_rows for k in row.keys()})
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    log("Write complete.")

if __name__ == "__main__":
    merge_and_write()