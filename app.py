import re
import datetime as dt
from typing import Optional, Dict, List, Set, Tuple

import pandas as pd
import requests
import streamlit as st
from requests.auth import HTTPBasicAuth

RTT_BASE = "https://api.rtt.io/api/v1"

# Railway day codes (as you described)
# SO=Saturday, Su=Sunday, MO=Monday, TO=Tuesday, WO=Wednesday, ThO=Thursday, FO=Friday
RAIL_DAY_TO_WEEKDAY = {
    "MO": 0,
    "TO": 1,
    "WO": 2,
    "THO": 3,
    "FO": 4,
    "SO": 5,
    "SU": 6,   # allow SU
    "SU ": 6,
}
WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# -------------------------
# Helpers
# -------------------------
def norm(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())

def find_col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    """Loose column match: ignores case/spaces/punctuation."""
    cols = list(df.columns)
    km = {key(c): c for c in cols}

    for w in candidates:
        kw = key(w)
        if kw in km:
            return km[kw]

    for w in candidates:
        kw = key(w)
        for c in cols:
            if kw and kw in key(c):
                return c
    return None

def parse_ddmmyyyy(x) -> Optional[dt.date]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None

def parse_hhmm(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    m = re.match(r"^(\d{1,2}):(\d{2})$", s)
    if not m:
        return None
    hh = int(m.group(1))
    mm = int(m.group(2))
    if 0 <= hh <= 23 and 0 <= mm <= 59:
        return f"{hh:02d}:{mm:02d}"
    return None

def hhmm_to_seconds(hhmm: str) -> int:
    hh, mm = hhmm.split(":")
    return int(hh) * 3600 + int(mm) * 60

def within_30_seconds(expected_hhmm: Optional[str], actual_hhmm: Optional[str], allow_30s: bool) -> bool:
    if expected_hhmm is None or actual_hhmm is None:
        return False
    if not allow_30s:
        return expected_hhmm == actual_hhmm
    # Treat HH:MM as HH:MM:00 and allow +/- 30 seconds
    return abs(hhmm_to_seconds(expected_hhmm) - hhmm_to_seconds(actual_hhmm)) <= 30

def daterange(d1: dt.date, d2: dt.date):
    cur = d1
    while cur <= d2:
        yield cur
        cur += dt.timedelta(days=1)

def parse_journey_days_run(value) -> Set[int]:
    """
    Parses railway day codes like:
      "MO", "TO", "WO", "ThO", "FO", "SO", "Su" and combinations.
    Returns set of weekday indexes (Mon=0..Sun=6).

    Accepts:
      - "MO TO WO ThO FO" (space)
      - "MO,TO,WO" (comma)
      - "MOTO" (rare; we still try)
      - Any mixture, case-insensitive
    """
    if pd.isna(value):
        return set()
    raw = str(value).strip()
    if not raw:
        return set()

    raw_up = raw.upper()

    # normalise variants of Sunday code "Su"
    raw_up = raw_up.replace("SU", "SU")  # keep
    # handle "THO" written as "THO" or "TH0"? (just in case)
    raw_up = raw_up.replace("TH0", "THO")

    # Tokenise on comma/space/semicolon/slash
    tokens = re.split(r"[,\s;/]+", raw_up)
    tokens = [t.strip() for t in tokens if t.strip()]

    # If it came as one blob, try to extract known codes by regex
    if len(tokens) == 1 and len(tokens[0]) > 3:
        blob = tokens[0]
        found = re.findall(r"THO|MO|TO|WO|FO|SO|SU", blob)
        tokens = found if found else tokens

    out: Set[int] = set()
    for t in tokens:
        t = t.strip().upper()
        if t == "SU":  # Sunday
            out.add(6)
        elif t in RAIL_DAY_TO_WEEKDAY:
            out.add(RAIL_DAY_TO_WEEKDAY[t])
        elif t == "THO":
            out.add(3)
        # allow "Su" passed in mixed case
        elif t == "SU":
            out.add(6)

    return out


# -------------------------
# RailReferences
# -------------------------
@st.cache_data
def load_railrefs_from_path(path: str) -> pd.DataFrame:
    # RailReferences.csv: TIPLOC, CRS, Description (no header)
    return pd.read_csv(path, header=None, names=["tiploc", "crs", "description"])

@st.cache_data
def load_railrefs_from_upload(uploaded) -> pd.DataFrame:
    return pd.read_csv(uploaded, header=None, names=["tiploc", "crs", "description"])

def build_desc_to_crs(rail_refs: pd.DataFrame) -> Dict[str, str]:
    d: Dict[str, str] = {}
    for _, r in rail_refs.iterrows():
        desc = norm(r["description"])
        crs = str(r["crs"]).strip() if not pd.isna(r["crs"]) else ""
        if desc and crs and desc not in d:
            d[desc] = crs

    # A few useful aliases (extend anytime)
    aliases = {
        "kings cross": "KGX",
        "london kings cross": "KGX",
        "kings x": "KGX",
        "edinburgh waverley": "EDB",
        "edinburgh": "EDB",
    }
    for k, v in aliases.items():
        d.setdefault(norm(k), v)
    return d


# -------------------------
# RTT calls
# -------------------------
def rtt_location_services(crs_or_tiploc: str, run_date: dt.date, auth: HTTPBasicAuth) -> dict:
    y = run_date.year
    m = f"{run_date.month:02d}"
    d = f"{run_date.day:02d}"
    url = f"{RTT_BASE}/json/search/{crs_or_tiploc}/{y}/{m}/{d}"
    r = requests.get(url, auth=auth, timeout=30)
    r.raise_for_status()
    return r.json()

def flatten_location_services(payload: dict) -> pd.DataFrame:
    services = payload.get("services", []) or []
    rows = []
    for s in services:
        ld = (s.get("locationDetail") or {})
        rows.append({
            "trainIdentity": (s.get("trainIdentity") or "").strip(),
            "serviceUid": s.get("serviceUid"),
            "runDate": s.get("runDate"),
            "atocName": s.get("atocName") or "",
            "plannedCancel": bool(s.get("plannedCancel", False)),
            "gbttBookedDeparture": ld.get("gbttBookedDeparture"),
        })
    return pd.DataFrame(rows)

def rtt_service_detail(service_uid: str, run_date_iso: str, auth: HTTPBasicAuth) -> dict:
    url = f"{RTT_BASE}/json/service/{service_uid}/{run_date_iso}"
    r = requests.get(url, auth=auth, timeout=30)
    r.raise_for_status()
    return r.json()

def extract_origin_dest_times_from_detail(payload: dict) -> dict:
    locs = payload.get("locations", []) or []
    if not locs:
        return {"origin_name": None, "dest_name": None, "book_dep": None, "book_arr": None}

    origin = locs[0]
    dest = locs[-1]

    def booked_dep(loc):
        ld = loc.get("locationDetail") or {}
        return ld.get("gbttBookedDeparture") or ld.get("gbttBookedArrival")

    def booked_arr(loc):
        ld = loc.get("locationDetail") or {}
        return ld.get("gbttBookedArrival") or ld.get("gbttBookedDeparture")

    return {
        "origin_name": origin.get("description"),
        "dest_name": dest.get("description"),
        "book_dep": booked_dep(origin),
        "book_arr": booked_arr(dest),
    }


# -------------------------
# App
# -------------------------
st.set_page_config(page_title="RTT No-Brand Checker", layout="wide")
st.title("RTT No-Brand Checker")
st.caption("Checks PASS trips where Brand is blank. Expected run dates are derived from selected date range + JourneyDays Run railway day codes.")

# RTT creds (from secrets)
rtt_user = st.secrets.get("RTT_USER", "")
rtt_pass = st.secrets.get("RTT_PASS", "")
auth = HTTPBasicAuth(rtt_user, rtt_pass) if rtt_user and rtt_pass else None
if not auth:
    st.warning("RTT credentials missing. Add RTT_USER and RTT_PASS in Streamlit Cloud → App Settings → Secrets.")

with st.sidebar:
    st.header("1) Date range")
    today = dt.date.today()
    date_from = st.date_input("From", value=today)
    date_to = st.date_input("To", value=today + dt.timedelta(days=7))
    if date_to < date_from:
        date_from, date_to = date_to, date_from

    st.header("2) Options")
    allow_30s = st.checkbox("Allow time tolerance (±30 seconds)", value=False)

    st.header("3) RailReferences")
    st.caption("Loaded once by default. Tick to upload an updated copy.")
    update_refs = st.checkbox("Update RailReferences now", value=False)

    st.header("4) Upload pass-trips.csv")
    pass_file = st.file_uploader("pass-trips.csv", type=["csv"])

# Load RailReferences
# Put RailReferences.csv in your repo root (same folder as app.py).
DEFAULT_RAILREFS_PATH = "RailReferences.csv"

rail_refs = None
if update_refs:
    uploaded_refs = st.file_uploader("Upload RailReferences.csv (override)", type=["csv"], key="railrefs_uploader")
    if uploaded_refs:
        rail_refs = load_railrefs_from_upload(uploaded_refs)
        st.success("Using uploaded RailReferences.csv for this session.")
    else:
        st.info("Ticked update, but no file uploaded yet.")
        st.stop()
else:
    try:
        rail_refs = load_railrefs_from_path(DEFAULT_RAILREFS_PATH)
    except Exception as e:
        st.error(f"Couldn't load default RailReferences.csv from repo path '{DEFAULT_RAILREFS_PATH}'. Upload it using the update option. Details: {e}")
        st.stop()

desc_to_crs = build_desc_to_crs(rail_refs)

if not pass_file:
    st.info("Upload pass-trips.csv to begin.")
    st.stop()

# Load pass-trips (always metadata first 3 rows, headers on row 4)
df = pd.read_csv(pass_file, skiprows=3)

with st.expander("Debug: detected columns in pass-trips.csv"):
    st.write(list(df.columns))

# Detect needed columns (loose matching)
col_brand = find_col(df, "Brand")
col_headcode = find_col(df, "Headcode")
col_origin = find_col(df, "JourneyOrigin", "Journey Origin", "Origin")
col_dest = find_col(df, "JourneyDestination", "Journey Destination", "Destination")
col_dep = find_col(df, "JourneyDeparture", "Journey Departure", "Departure")
col_arr = find_col(df, "JourneyArrival", "Journey Arrival", "Arrival")
col_jdays = find_col(df, "JourneyDays Run", "Journey Days Run", "JourneyDaysRun", "JourneyDays")
# diagram info columns for reporting
col_resource = find_col(df, "Resource")
col_plan_type = find_col(df, "DiagramPlan Type", "Diagram Plan Type", "Plan Type")
col_depot = find_col(df, "DiagramDepot", "Diagram Depot", "Depot")
col_did = find_col(df, "DiagramID", "Diagram Id", "Diagram ID")
col_ddays = find_col(df, "DiagramDays Run", "Diagram Days Run", "DiagramDaysRun")

required = {
    "Brand": col_brand,
    "Headcode": col_headcode,
    "JourneyOrigin": col_origin,
    "JourneyDestination": col_dest,
    "JourneyDeparture": col_dep,
    "JourneyArrival": col_arr,
    "JourneyDays Run": col_jdays,
}
missing = [k for k, v in required.items() if v is None]
with st.expander("Debug: column mapping used"):
    st.write({
        **required,
        "Resource": col_resource,
        "DiagramPlan Type": col_plan_type,
        "DiagramDepot": col_depot,
        "DiagramID": col_did,
        "DiagramDays Run": col_ddays,
    })

if missing:
    st.error("Missing required columns: " + ", ".join(missing) + ". Check the Debug columns expander above.")
    st.stop()

# Filter to Brand blank only
df_nb = df[df[col_brand].isna() | (df[col_brand].astype(str).str.strip() == "")].copy()

# Pre-parse fields
df_nb["dep_hhmm"] = df_nb[col_dep].apply(parse_hhmm)
df_nb["arr_hhmm"] = df_nb[col_arr].apply(parse_hhmm)
df_nb["jdays_set"] = df_nb[col_jdays].apply(parse_journey_days_run)

df_nb["origin_crs"] = df_nb[col_origin].apply(lambda x: desc_to_crs.get(norm(x)))
df_nb["dest_crs"] = df_nb[col_dest].apply(lambda x: desc_to_crs.get(norm(x)))

st.subheader("Input summary (Brand blank only)")
st.write(f"Rows with Brand blank: **{len(df_nb)}** | Unique headcodes: **{df_nb[col_headcode].nunique()}**")

with st.expander("Preview (Brand blank only)"):
    show_cols = [col_headcode, col_origin, col_dest, col_dep, col_arr, col_jdays]
    for c in [col_resource, col_plan_type, col_depot, col_did, col_ddays]:
        if c:
            show_cols.append(c)
    show_cols = list(dict.fromkeys(show_cols))
    st.dataframe(df_nb[show_cols + ["origin_crs", "dest_crs", "dep_hhmm", "arr_hhmm"]], use_container_width=True)

with st.expander("Debug: JourneyDays Run values (top 30)"):
    st.write(df_nb[col_jdays].astype(str).value_counts().head(30))

# Build expected services by date based on selected range + JourneyDays Run
expected_rows: List[dict] = []
for d in daterange(date_from, date_to):
    wd = d.weekday()  # Mon=0..Sun=6
    day_subset = df_nb[df_nb["jdays_set"].apply(lambda s: wd in s if isinstance(s, set) else False)]
    if day_subset.empty:
        continue

    for _, r in day_subset.iterrows():
        # Diagram string for reporting (your required format)
        diagram_str_parts = []
        if col_resource: diagram_str_parts.append(str(r.get(col_resource, "")).strip())
        if col_plan_type: diagram_str_parts.append(str(r.get(col_plan_type, "")).strip())
        depot = str(r.get(col_depot, "")).strip() if col_depot else ""
        did = str(r.get(col_did, "")).strip() if col_did else ""
        ddays = str(r.get(col_ddays, "")).strip() if col_ddays else ""
        # Format exactly: "Resource" - "DiagramPlan Type" "DiagramDepot"."DiagramID" "DiagramDays Run"
        diagram_info = ""
        if diagram_str_parts:
            resource = diagram_str_parts[0]
            plan = diagram_str_parts[1] if len(diagram_str_parts) > 1 else ""
            diagram_info = f'{resource} - {plan} {depot}.{did} {ddays}'.strip()
        else:
            diagram_info = f'{depot}.{did} {ddays}'.strip()

        expected_rows.append({
            "date": d,
            "weekday": WEEKDAY_NAMES[wd],
            "headcode": str(r[col_headcode]).strip(),
            "from": str(r[col_origin]).strip(),
            "to": str(r[col_dest]).strip(),
            "dep": r["dep_hhmm"],
            "arr": r["arr_hhmm"],
            "origin_crs": r["origin_crs"],
            "dest_crs": r["dest_crs"],
            "diagram_info": diagram_info,
        })

expected = pd.DataFrame(expected_rows)

st.subheader("Expected services in selected date range")
st.write(f"Date range: **{date_from} → {date_to}** | Expected checks: **{len(expected)}**")

if expected.empty:
    st.warning("No expected services found for that date range using JourneyDays Run. Widen the range or check JourneyDays Run values.")
    st.stop()

# Run check button
run = st.button("Run RTT check", type="primary")
if not run:
    st.stop()

if not auth:
    st.error("RTT credentials are missing. Add RTT_USER/RTT_PASS in Streamlit Cloud Secrets.")
    st.stop()

# Cache RTT location searches
@st.cache_data(ttl=300)
def cached_location(crs: str, d: dt.date, user: str, pw: str) -> dict:
    return rtt_location_services(crs, d, HTTPBasicAuth(user, pw))

results: List[dict] = []
progress = st.progress(0)
total = len(expected)
done = 0

# Group by (date, origin_crs) to minimise RTT calls
expected["date_iso"] = expected["date"].apply(lambda x: x.isoformat())
groups = expected.groupby(["date_iso", "origin_crs"], dropna=False)

for (date_iso, origin_crs), grp in groups:
    run_date = dt.date.fromisoformat(date_iso)

    # If we can't map CRS, mark as not checked
    if not isinstance(origin_crs, str) or not origin_crs.strip():
        for _, row in grp.iterrows():
            results.append({
                **row.to_dict(),
                "status": "NOT CHECKED",
                "error": "Origin CRS not mapped (check RailReferences or add alias)",
                "operator": "",
                "planned_cancel": "",
            })
            done += 1
            progress.progress(min(1.0, done / total))
        continue

    # Fetch RTT for this origin/date
    try:
        payload = cached_location(origin_crs.strip(), run_date, rtt_user, rtt_pass)
        loc_df = flatten_location_services(payload)
    except Exception as ex:
        for _, row in grp.iterrows():
            results.append({
                **row.to_dict(),
                "status": "FAIL",
                "error": f"RTT location query failed: {ex}",
                "operator": "",
                "planned_cancel": "",
            })
            done += 1
            progress.progress(min(1.0, done / total))
        continue

    # Check each expected row
    for _, row in grp.iterrows():
        hc = str(row["headcode"]).strip()
        exp_dep = row["dep"]
        exp_arr = row["arr"]
        exp_to = row["to"]

        candidates = loc_df[loc_df["trainIdentity"] == hc].copy()
        if candidates.empty:
            results.append({
                **row.to_dict(),
                "status": "FAIL",
                "error": "Headcode not found at origin on date",
                "operator": "",
                "planned_cancel": "",
            })
            done += 1
            progress.progress(min(1.0, done / total))
            continue

        # Departure match at origin
        if exp_dep:
            candidates["dep_ok"] = candidates["gbttBookedDeparture"].apply(lambda t: within_30_seconds(exp_dep, t, allow_30s))
            candidates = candidates[candidates["dep_ok"] == True]
            if candidates.empty:
                results.append({
                    **row.to_dict(),
                    "status": "FAIL",
                    "error": "Headcode found but departure time differs",
                    "operator": "",
                    "planned_cancel": "",
                })
                done += 1
                progress.progress(min(1.0, done / total))
                continue

        # Use first candidate and confirm destination + arrival via service detail
        cand = candidates.iloc[0]
        uid = cand.get("serviceUid")
        rtt_run_date = cand.get("runDate")
        op = cand.get("atocName") or ""
        pc = "Y" if bool(cand.get("plannedCancel", False)) else ""

        if not uid or not rtt_run_date:
            results.append({
                **row.to_dict(),
                "status": "FAIL",
                "error": "Insufficient RTT data to verify destination/arrival",
                "operator": op,
                "planned_cancel": pc,
            })
            done += 1
            progress.progress(min(1.0, done / total))
            continue

        try:
            detail = rtt_service_detail(uid, rtt_run_date, auth)
            det = extract_origin_dest_times_from_detail(detail)
        except Exception:
            results.append({
                **row.to_dict(),
                "status": "FAIL",
                "error": "RTT service detail lookup failed",
                "operator": op,
                "planned_cancel": pc,
            })
            done += 1
            progress.progress(min(1.0, done / total))
            continue

        # Destination name check (soft match)
        det_dest = det.get("dest_name") or ""
        if exp_to and det_dest:
            if norm(exp_to) not in norm(det_dest) and norm(det_dest) not in norm(exp_to):
                results.append({
                    **row.to_dict(),
                    "status": "FAIL",
                    "error": f"Destination differs (RTT: {det_dest})",
                    "operator": op,
                    "planned_cancel": pc,
                })
                done += 1
                progress.progress(min(1.0, done / total))
                continue

        # Arrival time check at destination
        rtt_arr = det.get("book_arr")
        if exp_arr and rtt_arr:
            if not within_30_seconds(exp_arr, rtt_arr, allow_30s):
                results.append({
                    **row.to_dict(),
                    "status": "FAIL",
                    "error": f"Arrival time differs (RTT: {rtt_arr})",
                    "operator": op,
                    "planned_cancel": pc,
                })
                done += 1
                progress.progress(min(1.0, done / total))
                continue

        # OK
        results.append({
            **row.to_dict(),
            "status": "OK",
            "error": "",
            "operator": op,
            "planned_cancel": pc,
        })
        done += 1
        progress.progress(min(1.0, done / total))

progress.empty()

report = pd.DataFrame(results)

ok = report[report["status"] == "OK"]
fail = report[report["status"] == "FAIL"]
nc = report[report["status"] == "NOT CHECKED"]

st.subheader("Outcome")
st.write(f"Checked: **{len(report)}** | OK: **{len(ok)}** | Errors: **{len(fail)}** | Not checked: **{len(nc)}**")

# Info message as requested
if len(fail) == 0 and len(nc) == 0:
    st.success("All trains are running as booked for the selected date range (based on RTT checks).")
else:
    st.error("Some trains are NOT running as booked, or could not be checked. You can export a CSV report with details.")

# Simplified list for OK trains
with st.expander("Trains running as booked (simplified)"):
    simp = ok.copy()
    simp["line"] = simp.apply(lambda r: f"{r['headcode']} {r['dep']} {r['from']} - {r['to']} {r['arr']} ({r['date_iso']})", axis=1)
    st.write("\n".join(simp.sort_values(["date_iso", "dep", "headcode"])["line"].tolist()) or "None")

# Errors list
with st.expander("Errors / not checked (details)"):
    if report[(report["status"] != "OK")].empty:
        st.write("None")
    else:
        st.dataframe(
            report[report["status"] != "OK"][[
                "date_iso", "headcode", "from", "to", "dep", "arr", "status", "error", "diagram_info", "operator", "planned_cancel"
            ]].sort_values(["date_iso", "status", "dep", "headcode"]),
            use_container_width=True
        )

# Ask user if they want CSV report
st.subheader("CSV report")
create_csv = st.checkbox("Create CSV report", value=False)

if create_csv:
    # Build CSV output:
    # - OK rows: simplified info
    # - FAIL/NOT CHECKED: include error + diagram_info
    out = report.copy()
    out["date"] = out["date_iso"]
    out["diagram"] = out["diagram_info"]

    # Simplify OK
    out.loc[out["status"] == "OK", "error"] = ""

    cols_out = [
        "date", "headcode", "from", "to", "dep", "arr", "status",
        "error", "diagram", "operator", "planned_cancel"
    ]
    csv_bytes = out[cols_out].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV report",
        data=csv_bytes,
        file_name=f"rtt_no_brand_report_{date_from}_{date_to}.csv",
        mime="text/csv",
    )
