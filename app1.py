import re
import datetime as dt
from typing import Optional, Dict, Set, List, Tuple

import pandas as pd
import requests
import streamlit as st
from requests.auth import HTTPBasicAuth

RTT_BASE = "https://secure.realtimetrains.co.uk/api"


def key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())


def norm_name(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def find_col(df: pd.DataFrame, *candidates: str, allow_contains: bool = True) -> Optional[str]:
    cols = list(df.columns)
    km = {key(c): c for c in cols}

    for w in candidates:
        kw = key(w)
        if kw in km:
            return km[kw]

    if not allow_contains:
        return None

    for w in candidates:
        kw = key(w)
        for c in cols:
            if kw and kw in key(c):
                return c
    return None


def parse_pass_hhmm(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    m = re.match(r"^(\d{1,2}):(\d{2})$", s)
    if not m:
        return None
    hh, mm = int(m.group(1)), int(m.group(2))
    if 0 <= hh <= 23 and 0 <= mm <= 59:
        return f"{hh:02d}:{mm:02d}"
    return None


def rtt_public_to_hhmm(x) -> Optional[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    if not s or s.lower() == "null":
        return None
    s = re.sub(r"\D", "", s)
    if len(s) == 4:
        return f"{s[:2]}:{s[2:]}"
    if len(s) == 6:
        return f"{s[:2]}:{s[2:4]}"
    return None


def parse_ddmmyyyy_to_date(x) -> Optional[dt.date]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return pd.to_datetime(s, dayfirst=True, errors="raise").date()
    except Exception:
        return None


def rail_day_index(day: dt.date) -> int:
    # Sat=1, Sun=2, Mon=3, Tue=4, Wed=5, Thu=6, Fri=7
    return ((day.weekday() + 2) % 7) + 1


def next_rail_day(rd: int) -> int:
    return 1 if rd == 7 else rd + 1


RAIL_ORDER = ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
RAIL_BASE_MON_SAT = {1, 3, 4, 5, 6, 7}
RAIL_BASE_ALL_DAYS = {1, 2, 3, 4, 5, 6, 7}


def parse_days_run_to_rail(value) -> Set[int]:
    """
    Returns rail-day indexes: Sat=1, Sun=2, Mon=3, Tue=4, Wed=5, Thu=6, Fri=7.

    Supports:
      - Railway tokens: MO TO WO ThO FO SO Su
      - Compact: M T W Th F S Su with optional trailing O / X

    Dataset rule:
      - Sunday runs appear explicitly as Su rows.
      - Therefore ...X patterns default to a Mon–Sat base (Sunday not implied).
    """
    if pd.isna(value):
        return set()
    raw = str(value).strip()
    if not raw:
        return set()

    rail_map = {"SO": 1, "SU": 2, "MO": 3, "TO": 4, "WO": 5, "THO": 6, "FO": 7}
    tokens = re.split(r"[,\s;/]+", raw, flags=re.IGNORECASE)
    days = set()
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        t_up = t.upper().replace("TH0", "THO")
        if t_up in rail_map:
            days.add(rail_map[t_up])
    if days:
        return days

    blob = re.sub(r"[\s,;/]+", "", raw)
    if not blob:
        return set()

    mode = None
    if blob[-1].upper() in ("O", "X"):
        mode = blob[-1].upper()
        blob = blob[:-1]

    mentions_sunday = "su" in blob.lower()
    base = RAIL_BASE_ALL_DAYS if mentions_sunday else RAIL_BASE_MON_SAT

    i = 0
    picked: Set[int] = set()
    while i < len(blob):
        part2 = blob[i : i + 2].lower()
        part1 = blob[i : i + 1].upper()

        if part2 == "th":
            picked.add(6)
            i += 2
            continue
        if part2 == "su":
            picked.add(2)
            i += 2
            continue

        if part1 == "M":
            picked.add(3)
        elif part1 == "T":
            picked.add(4)
        elif part1 == "W":
            picked.add(5)
        elif part1 == "F":
            picked.add(7)
        elif part1 == "S":
            picked.add(1)
        i += 1

    if mode == "O":
        return picked
    if mode == "X":
        return (base - picked) if picked else set(base)
    return picked


@st.cache_data(show_spinner=False)
def load_railrefs_from_repo(path: str) -> pd.DataFrame:
    return pd.read_csv(path, header=None, names=["tiploc", "crs", "description"])


@st.cache_data(show_spinner=False)
def load_railrefs_from_upload(uploaded) -> pd.DataFrame:
    return pd.read_csv(uploaded, header=None, names=["tiploc", "crs", "description"])


def build_desc_to_crs(rail_refs: pd.DataFrame) -> Dict[str, str]:
    d: Dict[str, str] = {}
    for _, r in rail_refs.iterrows():
        desc = norm_name(r["description"])
        crs = str(r["crs"]).strip().upper() if not pd.isna(r["crs"]) else ""
        if desc and crs and desc not in d:
            d[desc] = crs

    aliases = {
        "kings cross": "KGX",
        "london kings cross": "KGX",
        "kings x": "KGX",
        "edinburgh": "EDB",
        "edinburgh waverley": "EDB",
    }
    for k, v in aliases.items():
        d.setdefault(norm_name(k), v)
    return d


def rtt_location_services(crs_or_tiploc: str, run_date: dt.date, auth: HTTPBasicAuth) -> dict:
    url = f"{RTT_BASE}/json/search/{crs_or_tiploc}/{run_date.year}/{run_date.month:02d}/{run_date.day:02d}"
    r = requests.get(url, auth=auth, timeout=30)
    r.raise_for_status()
    return r.json()


def flatten_location_services(payload: dict) -> pd.DataFrame:
    services = payload.get("services", []) or []
    rows = []
    for s in services:
        ld = (s.get("locationDetail") or {})
        rows.append(
            {
                "trainIdentity": (s.get("trainIdentity") or "").strip(),
                "serviceUid": s.get("serviceUid"),
                "runDate": s.get("runDate"),
                "operator": s.get("atocName") or "",
                "plannedCancel": bool(s.get("plannedCancel", False)),
                "realtimeActivated": bool(s.get("realtimeActivated", False)),
                "pub_dep_raw": ld.get("gbttBookedDeparture")
                or ld.get("publicTime")
                or ld.get("realtimeDeparture"),
            }
        )
    return pd.DataFrame(rows)


def rtt_service_detail(service_uid: str, run_date_iso: str, auth: HTTPBasicAuth) -> dict:
    y, m, d = str(run_date_iso).split("-")
    url = f"{RTT_BASE}/json/service/{service_uid}/{y}/{m}/{d}"
    r = requests.get(url, auth=auth, timeout=30)
    r.raise_for_status()
    return r.json()


def extract_point(detail_payload: dict, crs: str) -> Optional[dict]:
    crs = (crs or "").strip().upper()
    for loc in detail_payload.get("locations", []) or []:
        if (loc.get("crs") or "").strip().upper() == crs:
            return loc
    return None


def get_time_fields(loc: dict) -> dict:
    ld = loc.get("locationDetail")
    src = ld if isinstance(ld, dict) else loc
    return {
        "isCall": bool(loc.get("isCall", False)),
        "isCallPublic": bool(loc.get("isCallPublic", False)),
        "gbtt_arr": rtt_public_to_hhmm(src.get("gbttBookedArrival")),
        "gbtt_dep": rtt_public_to_hhmm(src.get("gbttBookedDeparture")),
        "rt_arr": rtt_public_to_hhmm(src.get("realtimeArrival")),
        "rt_dep": rtt_public_to_hhmm(src.get("realtimeDeparture")),
        "wtt_arr": rtt_public_to_hhmm(src.get("wttBookedArrival")),
        "wtt_dep": rtt_public_to_hhmm(src.get("wttBookedDeparture")),
        "desc": loc.get("description") or "",
        "displayAs": loc.get("displayAs") or "",
    }


def derive_validity_window_row(row: pd.Series) -> Tuple[Optional[dt.date], Optional[dt.date]]:
    ts = row.get("train_start")
    te = row.get("train_end")
    ds = row.get("diag_start")
    de = row.get("diag_end")

    starts = [d for d in (ts, ds) if isinstance(d, dt.date)]
    ends = [d for d in (te, de) if isinstance(d, dt.date)]
    start = max(starts) if starts else None
    end = min(ends) if ends else None
    return start, end


@st.cache_data(ttl=300, show_spinner=False)
def cached_location_search(crs: str, run_date: dt.date, user: str, pw: str) -> dict:
    return rtt_location_services(crs, run_date, HTTPBasicAuth(user, pw))


@st.cache_data(ttl=3600, show_spinner=False)
def cached_service_detail(service_uid: str, run_date_iso: str, user: str, pw: str) -> dict:
    return rtt_service_detail(service_uid, run_date_iso, HTTPBasicAuth(user, pw))


# ----------------------------
# App
# ----------------------------
st.set_page_config(page_title="LNER (TRENT) - PASS RIDE CHECKER", layout="wide")
st.title("LNER (TRENT) - PASS RIDE CHECKER")
st.caption(
    "Uses the raw pass-trips.csv file to check whether other operators’ trains are running as diagrammed.\n"
    "Shift logic: if the diagram day does not exist in JourneyDays, but the *next* rail-day exists in JourneyDays, "
    "the check is performed on day+1 (shift_flag=8)."
)

rtt_user = st.secrets.get("RTT_USER", "")
rtt_pass = st.secrets.get("RTT_PASS", "")
auth = HTTPBasicAuth(rtt_user, rtt_pass) if rtt_user and rtt_pass else None
DEFAULT_RAILREFS_PATH = "RailReferences.csv"

today = dt.date.today()
if "date_from" not in st.session_state:
    st.session_state["date_from"] = today
if "date_to" not in st.session_state:
    st.session_state["date_to"] = today
if "last_from" not in st.session_state:
    st.session_state["last_from"] = st.session_state["date_from"]


def reset_run_state():
    st.session_state.pop("report", None)
    st.session_state.pop("expected", None)
    st.session_state["date_from"] = dt.date.today()
    st.session_state["date_to"] = dt.date.today()
    st.session_state["last_from"] = st.session_state["date_from"]


with st.sidebar:
    st.header("New search?")
    if st.button("Clear previous results", use_container_width=True):
        reset_run_state()
        st.success("Cleared. You can run another check now.")

    st.divider()

    st.subheader("1) Upload pass-trips.csv")
    pass_file = st.file_uploader("pass-trips.csv", type=["csv"])

    st.subheader("2) Date range")
    _ = st.date_input("From", key="date_from")
    if st.session_state.get("last_from") != st.session_state["date_from"]:
        st.session_state["date_to"] = st.session_state["date_from"]
        st.session_state["last_from"] = st.session_state["date_from"]
    _ = st.date_input("To", key="date_to")
    if st.session_state["date_to"] < st.session_state["date_from"]:
        st.session_state["date_to"] = st.session_state["date_from"]

    date_from = st.session_state["date_from"]
    date_to = st.session_state["date_to"]

    st.subheader("3) RailReferences (optional)")
    update_refs = st.checkbox("Upload updated RailReferences.csv", value=False)
    uploaded_refs = st.file_uploader("RailReferences.csv", type=["csv"], key="railrefs_upload") if update_refs else None


# ----------------------------
# Load RailReferences
# ----------------------------
if update_refs:
    if not uploaded_refs:
        st.info("Upload RailReferences.csv (or untick the option).")
        st.stop()
    rail_refs = load_railrefs_from_upload(uploaded_refs)
else:
    try:
        rail_refs = load_railrefs_from_repo(DEFAULT_RAILREFS_PATH)
    except Exception as e:
        st.error(
            f"Couldn't load '{DEFAULT_RAILREFS_PATH}'. Tick 'Upload updated RailReferences.csv' and upload it. Details: {e}"
        )
        st.stop()

desc_to_crs = build_desc_to_crs(rail_refs)

# ----------------------------
# Load pass-trips.csv
# ----------------------------
if not pass_file:
    st.info("Upload pass-trips.csv to begin.")
    st.stop()

df = pd.read_csv(pass_file, skiprows=3)

col_brand = find_col(df, "Brand")
col_headcode = find_col(df, "Headcode")
col_origin = find_col(df, "JourneyOrigin", "Journey Origin", "Origin")
col_dest = find_col(df, "JourneyDestination", "Journey Destination", "Destination")
col_dep = find_col(df, "JourneyDeparture", "Journey Departure", "Departure")
col_arr = find_col(df, "JourneyArrival", "Journey Arrival", "Arrival")

col_jdays = find_col(df, "JourneyDays Run", "Journey Days Run", "JourneyDaysRun", allow_contains=False)
col_ddays = find_col(df, "DiagramDays Run", "Diagram Days Run", "DiagramDaysRun")

col_train_start = find_col(df, "TrainStart Date", "Train Start Date", "TrainStartDate")
col_train_end = find_col(df, "TrainEnd Date", "Train End Date", "TrainEndDate")
col_diag_start = find_col(df, "DiagramStart Date", "Diagram Start Date", "DiagramStartDate")
col_diag_end = find_col(df, "DiagramEnd Date", "Diagram End Date", "DiagramEndDate")

col_resource = find_col(df, "Resource")
col_plan_type = find_col(df, "DiagramPlan Type", "Diagram Plan Type", "Plan Type")
col_depot = find_col(df, "DiagramDepot", "Diagram Depot", "Depot")
col_did = find_col(df, "DiagramID", "Diagram Id", "Diagram ID")

missing = []
for name, col in {
    "Brand": col_brand,
    "Headcode": col_headcode,
    "JourneyOrigin": col_origin,
    "JourneyDestination": col_dest,
    "JourneyDeparture": col_dep,
    "JourneyArrival": col_arr,
    "JourneyDays Run": col_jdays,
}.items():
    if col is None:
        missing.append(name)

if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    with st.expander("Detected columns"):
        st.write(list(df.columns))
    st.stop()

df_nb = df[df[col_brand].isna() | (df[col_brand].astype(str).str.strip() == "")].copy()

df_nb["exp_dep"] = df_nb[col_dep].apply(parse_pass_hhmm)
df_nb["exp_arr"] = df_nb[col_arr].apply(parse_pass_hhmm)

df_nb["jdays_set"] = df_nb[col_jdays].apply(parse_days_run_to_rail)
df_nb["ddays_set"] = df_nb[col_ddays].apply(parse_days_run_to_rail) if col_ddays else df_nb["jdays_set"]

df_nb["origin_crs"] = df_nb[col_origin].apply(lambda x: desc_to_crs.get(norm_name(x)))
df_nb["dest_crs"] = df_nb[col_dest].apply(lambda x: desc_to_crs.get(norm_name(x)))

df_nb["train_start"] = df_nb[col_train_start].apply(parse_ddmmyyyy_to_date) if col_train_start else None
df_nb["train_end"] = df_nb[col_train_end].apply(parse_ddmmyyyy_to_date) if col_train_end else None
df_nb["diag_start"] = df_nb[col_diag_start].apply(parse_ddmmyyyy_to_date) if col_diag_start else None
df_nb["diag_end"] = df_nb[col_diag_end].apply(parse_ddmmyyyy_to_date) if col_diag_end else None

vb = df_nb.apply(derive_validity_window_row, axis=1, result_type="expand")
df_nb["valid_start"] = vb[0]
df_nb["valid_end"] = vb[1]

total_rows = len(df)
blank_brand_rows = len(df_nb)

st.subheader("Input summary")
st.write(
    f"Total rows in file: **{total_rows}** | "
    f"Rows with Brand blank: **{blank_brand_rows}** | "
    f"Unique headcodes (Brand blank): **{df_nb[col_headcode].nunique()}**"
)

with st.expander("Input preview (first 200 rows)"):
    base_cols = [col_headcode, col_origin, col_dest, col_dep, col_arr, col_jdays]
    diag_cols = [
        c
        for c in [
            col_ddays,
            col_resource,
            col_plan_type,
            col_depot,
            col_did,
            col_train_start,
            col_train_end,
            col_diag_start,
            col_diag_end,
        ]
        if c
    ]
    show_cols = list(dict.fromkeys(base_cols + diag_cols))
    preview_cols = show_cols + [
        "origin_crs",
        "dest_crs",
        "exp_dep",
        "exp_arr",
        "jdays_set",
        "ddays_set",
        "valid_start",
        "valid_end",
    ]
    preview = df_nb[preview_cols].head(200)
    st.caption(f"Showing **{len(preview)}** row(s) (max 200) out of **{blank_brand_rows}** Brand-blank row(s).")
    st.dataframe(preview, use_container_width=True)

# ----------------------------
# Build expected checks (SHIFT LOGIC: per-day)
# ----------------------------
expected_rows: List[dict] = []

for i in range((date_to - date_from).days + 1):
    day = date_from + dt.timedelta(days=i)
    rd = rail_day_index(day)

    mask_day = df_nb["ddays_set"].apply(lambda s: rd in s if isinstance(s, set) else False)
    mask_valid = (
        (df_nb["valid_start"].isna() | (df_nb["valid_start"] <= day))
        & (df_nb["valid_end"].isna() | (df_nb["valid_end"] >= day))
    )
    subset = df_nb[mask_day & mask_valid]
    if subset.empty:
        continue

    for _, r in subset.iterrows():
        journey_set = r["jdays_set"] if isinstance(r["jdays_set"], set) else set()

        # STRICT day alignment:
        # If diagram runs today but journey runs tomorrow -> check tomorrow (shift_flag=8)
        # If diagram runs today and journey runs today -> check today
        # Otherwise mismatch
        if rd in journey_set:
            shift_flag = 0
            rtt_day = day
            day_alignment = "MATCH"
        elif next_rail_day(rd) in journey_set:
            shift_flag = 8
            rtt_day = day + dt.timedelta(days=1)
            day_alignment = "AFTER_MIDNIGHT"
        else:
            shift_flag = 0
            rtt_day = day
            day_alignment = "MISMATCH"

        resource = str(r[col_resource]).strip() if col_resource else ""
        plan = str(r[col_plan_type]).strip() if col_plan_type else ""
        depot = str(r[col_depot]).strip() if col_depot else ""
        did = str(r[col_did]).strip() if col_did else ""
        ddays_txt = str(r[col_ddays]).strip() if col_ddays else ""

        diagram_info = (
            f"{resource} - {plan} {depot}.{did} {ddays_txt}".strip()
            if (resource or plan)
            else f"{depot}.{did} {ddays_txt}".strip()
        )

        expected_rows.append(
            {
                "date": day.isoformat(),
                "rtt_date": rtt_day.isoformat(),
                "Shift flag": shift_flag,
                "day_alignment": day_alignment,
                "Day of the week": day.strftime("%A"),
                "Rail day index": rd,
                "headcode": str(r[col_headcode]).strip(),
                "exp_from": str(r[col_origin]).strip(),
                "exp_to": str(r[col_dest]).strip(),
                "exp_dep": r["exp_dep"],
                "exp_arr": r["exp_arr"],
                "origin_crs": r["origin_crs"],
                "dest_crs": r["dest_crs"],
                "diagram_info": diagram_info,
            }
        )

expected = pd.DataFrame(expected_rows)
st.session_state["expected"] = expected

st.subheader("Headcodes by day")
st.write(f"Date range: **{date_from} → {date_to}** | Expected checks: **{len(expected)}**")

with st.expander("Day alignment (counts)"):
    if expected.empty:
        st.write("None")
    else:
        st.dataframe(expected["day_alignment"].value_counts())

if expected.empty:
    st.warning("No expected services found in that date range.")
    st.stop()

# ----------------------------
# Run RTT check
# ----------------------------
run = st.button("Run RTT check", type="primary")
existing_report = st.session_state.get("report")

if not run and existing_report is None:
    st.stop()

if run:
    if not auth:
        st.error("RTT credentials missing. Add RTT_USER / RTT_PASS in Streamlit Secrets.")
        st.stop()

    results: List[dict] = []
    total = len(expected)
    done = 0
    progress = st.progress(0)

    grouped = expected.groupby(["rtt_date", "origin_crs"], dropna=False)

    for (rtt_date_iso, origin_crs), grp in grouped:
        run_date = dt.date.fromisoformat(rtt_date_iso)

        if not isinstance(origin_crs, str) or not origin_crs.strip():
            for _, row in grp.iterrows():
                results.append(
                    {
                        **row.to_dict(),
                        "status": "NOT CHECKED",
                        "error": "Origin CRS not mapped",
                        "operator": "",
                        "planned_cancel": "",
                        "realtimeActivated": "NA",
                        "serviceUid": "",
                        "runDate": "",
                        "origin_displayAs": "",
                        "dest_displayAs": "",
                        "origin_isCall": "",
                        "dest_isCall": "",
                        "origin_isCallPublic": "",
                        "dest_isCallPublic": "",
                        "act_dep": None,
                        "act_arr": None,
                        "origin_rtt_deps": "",
                        "variants_count": 0,
                        "variants_deps": "",
                        "has_variants": "",
                    }
                )
                done += 1
                progress.progress(min(1.0, done / total))
            continue

        try:
            payload = cached_location_search(origin_crs.strip(), run_date, rtt_user, rtt_pass)
            loc_df = flatten_location_services(payload)
        except Exception as ex:
            for _, row in grp.iterrows():
                results.append(
                    {
                        **row.to_dict(),
                        "status": "FAIL",
                        "error": f"RTT location query failed: {ex}",
                        "operator": "",
                        "planned_cancel": "",
                        "realtimeActivated": "NA",
                        "serviceUid": "",
                        "runDate": "",
                        "origin_displayAs": "",
                        "dest_displayAs": "",
                        "origin_isCall": "",
                        "dest_isCall": "",
                        "origin_isCallPublic": "",
                        "dest_isCallPublic": "",
                        "act_dep": None,
                        "act_arr": None,
                        "origin_rtt_deps": "",
                        "variants_count": 0,
                        "variants_deps": "",
                        "has_variants": "",
                    }
                )
                done += 1
                progress.progress(min(1.0, done / total))
            continue

        if loc_df.empty:
            for _, row in grp.iterrows():
                results.append(
                    {
                        **row.to_dict(),
                        "status": "FAIL",
                        "error": f"No RTT services returned (origin CRS: {origin_crs})",
                        "operator": "",
                        "planned_cancel": "",
                        "realtimeActivated": "NA",
                        "serviceUid": "",
                        "runDate": "",
                        "origin_displayAs": "",
                        "dest_displayAs": "",
                        "origin_isCall": "",
                        "dest_isCall": "",
                        "origin_isCallPublic": "",
                        "dest_isCallPublic": "",
                        "act_dep": None,
                        "act_arr": None,
                        "origin_rtt_deps": "",
                        "variants_count": 0,
                        "variants_deps": "",
                        "has_variants": "",
                    }
                )
                done += 1
                progress.progress(min(1.0, done / total))
            continue

        for _, row in grp.iterrows():
            hc = str(row["headcode"]).strip()
            exp_dep = row.get("exp_dep")
            exp_arr = row.get("exp_arr")
            dest_crs = row.get("dest_crs")

            if not isinstance(dest_crs, str) or not dest_crs.strip():
                results.append(
                    {
                        **row.to_dict(),
                        "status": "NOT CHECKED",
                        "error": "Destination CRS not mapped",
                        "operator": "",
                        "planned_cancel": "",
                        "realtimeActivated": "NA",
                        "serviceUid": "",
                        "runDate": "",
                        "origin_displayAs": "",
                        "dest_displayAs": "",
                        "origin_isCall": "",
                        "dest_isCall": "",
                        "origin_isCallPublic": "",
                        "dest_isCallPublic": "",
                        "act_dep": None,
                        "act_arr": None,
                        "origin_rtt_deps": "",
                        "variants_count": 0,
                        "variants_deps": "",
                        "has_variants": "",
                    }
                )
                done += 1
                progress.progress(min(1.0, done / total))
                continue

            cand_all = loc_df[loc_df["trainIdentity"] == hc].copy()
            variants_count = len(cand_all)

            if cand_all.empty:
                results.append(
                    {
                        **row.to_dict(),
                        "status": "FAIL",
                        "error": "Headcode not found at origin on RTT date",
                        "operator": "",
                        "planned_cancel": "",
                        "realtimeActivated": "NA",
                        "serviceUid": "",
                        "runDate": "",
                        "origin_displayAs": "",
                        "dest_displayAs": "",
                        "origin_isCall": "",
                        "dest_isCall": "",
                        "origin_isCallPublic": "",
                        "dest_isCallPublic": "",
                        "act_dep": None,
                        "act_arr": None,
                        "origin_rtt_deps": "",
                        "variants_count": variants_count,
                        "variants_deps": "",
                        "has_variants": "",
                    }
                )
                done += 1
                progress.progress(min(1.0, done / total))
                continue

            cand_all["dep_pub_hhmm"] = cand_all["pub_dep_raw"].apply(rtt_public_to_hhmm)
            deps_list = sorted(set(cand_all["dep_pub_hhmm"].dropna().astype(str).tolist()))
            variants_deps = ", ".join(deps_list)
            has_variants = "Y" if len(deps_list) >= 2 else ""
            origin_rtt_deps = variants_deps

            ops = sorted(set(cand_all["operator"].dropna().astype(str).tolist()))

            # STRICT selection:
            # if exp_dep is provided, require exact match at origin
            chosen = None
            if exp_dep:
                exact = cand_all[cand_all["dep_pub_hhmm"] == exp_dep]
                if not exact.empty:
                    chosen = exact.iloc[0]
                else:
                    results.append(
                        {
                            **row.to_dict(),
                            "status": "FAIL",
                            "error": f"Origin departure mismatch (expected {exp_dep})",
                            "operator": ", ".join(ops) if ops else "",
                            "planned_cancel": "",
                            "realtimeActivated": "NA",
                            "serviceUid": "",
                            "runDate": "",
                            "origin_displayAs": "",
                            "dest_displayAs": "",
                            "origin_isCall": "",
                            "dest_isCall": "",
                            "origin_isCallPublic": "",
                            "dest_isCallPublic": "",
                            "act_dep": deps_list[0] if deps_list else None,
                            "act_arr": None,
                            "origin_rtt_deps": origin_rtt_deps,
                            "variants_count": variants_count,
                            "variants_deps": variants_deps,
                            "has_variants": has_variants,
                        }
                    )
                    done += 1
                    progress.progress(min(1.0, done / total))
                    continue
            else:
                chosen = cand_all.iloc[0]

            cand = chosen
            uid = cand.get("serviceUid") or ""
            rtt_run_date = cand.get("runDate") or ""
            op = cand.get("operator") or (", ".join(ops) if ops else "")
            planned_cancel = bool(cand.get("plannedCancel", False))
            realtime_activated_bool = bool(cand.get("realtimeActivated", False))
            realtimeActivated = "Y" if realtime_activated_bool else "N"

            if not uid or not rtt_run_date:
                results.append(
                    {
                        **row.to_dict(),
                        "status": "FAIL",
                        "error": "Insufficient RTT data to verify calling points (missing serviceUid/runDate)",
                        "operator": op,
                        "planned_cancel": "Y" if planned_cancel else "",
                        "realtimeActivated": realtimeActivated,
                        "serviceUid": uid,
                        "runDate": rtt_run_date,
                        "origin_displayAs": "",
                        "dest_displayAs": "",
                        "origin_isCall": "",
                        "dest_isCall": "",
                        "origin_isCallPublic": "",
                        "dest_isCallPublic": "",
                        "act_dep": cand.get("dep_pub_hhmm"),
                        "act_arr": None,
                        "origin_rtt_deps": origin_rtt_deps,
                        "variants_count": variants_count,
                        "variants_deps": variants_deps,
                        "has_variants": has_variants,
                    }
                )
                done += 1
                progress.progress(min(1.0, done / total))
                continue

            try:
                detail = cached_service_detail(uid, rtt_run_date, rtt_user, rtt_pass)
            except Exception as ex:
                results.append(
                    {
                        **row.to_dict(),
                        "status": "FAIL",
                        "error": f"RTT service detail lookup failed: {ex}",
                        "operator": op,
                        "planned_cancel": "Y" if planned_cancel else "",
                        "realtimeActivated": realtimeActivated,
                        "serviceUid": uid,
                        "runDate": rtt_run_date,
                        "origin_displayAs": "",
                        "dest_displayAs": "",
                        "origin_isCall": "",
                        "dest_isCall": "",
                        "origin_isCallPublic": "",
                        "dest_isCallPublic": "",
                        "act_dep": cand.get("dep_pub_hhmm"),
                        "act_arr": None,
                        "origin_rtt_deps": origin_rtt_deps,
                        "variants_count": variants_count,
                        "variants_deps": variants_deps,
                        "has_variants": has_variants,
                    }
                )
                done += 1
                progress.progress(min(1.0, done / total))
                continue

            origin_loc = extract_point(detail, row["origin_crs"])
            dest_loc = extract_point(detail, row["dest_crs"])

            if not origin_loc:
                results.append(
                    {
                        **row.to_dict(),
                        "status": "FAIL",
                        "error": f"Origin CRS not found in RTT calling points ({row['origin_crs']})",
                        "operator": op,
                        "planned_cancel": "Y" if planned_cancel else "",
                        "realtimeActivated": realtimeActivated,
                        "serviceUid": uid,
                        "runDate": rtt_run_date,
                        "origin_displayAs": "",
                        "dest_displayAs": "",
                        "origin_isCall": "",
                        "dest_isCall": "",
                        "origin_isCallPublic": "",
                        "dest_isCallPublic": "",
                        "act_dep": None,
                        "act_arr": None,
                        "origin_rtt_deps": origin_rtt_deps,
                        "variants_count": variants_count,
                        "variants_deps": variants_deps,
                        "has_variants": has_variants,
                    }
                )
                done += 1
                progress.progress(min(1.0, done / total))
                continue

            if not dest_loc:
                results.append(
                    {
                        **row.to_dict(),
                        "status": "FAIL",
                        "error": f"Destination CRS not found in RTT calling points ({row['dest_crs']})",
                        "operator": op,
                        "planned_cancel": "Y" if planned_cancel else "",
                        "realtimeActivated": realtimeActivated,
                        "serviceUid": uid,
                        "runDate": rtt_run_date,
                        "origin_displayAs": "",
                        "dest_displayAs": "",
                        "origin_isCall": "",
                        "dest_isCall": "",
                        "origin_isCallPublic": "",
                        "dest_isCallPublic": "",
                        "act_dep": None,
                        "act_arr": None,
                        "origin_rtt_deps": origin_rtt_deps,
                        "variants_count": variants_count,
                        "variants_deps": variants_deps,
                        "has_variants": has_variants,
                    }
                )
                done += 1
                progress.progress(min(1.0, done / total))
                continue

            o = get_time_fields(origin_loc)
            d = get_time_fields(dest_loc)

            origin_isCall = o["isCall"]
            origin_isCallPublic = o["isCallPublic"]
            dest_isCall = d["isCall"]
            dest_isCallPublic = d["isCallPublic"]

            act_dep = o["gbtt_dep"] or o["rt_dep"] or o["wtt_dep"]
            act_arr = d["gbtt_arr"] or d["rt_arr"] or d["gbtt_dep"] or d["rt_dep"] or d["wtt_arr"] or d["wtt_dep"]

            errs: List[str] = []

            if planned_cancel:
                errs.append("Planned cancel (RTT)")

            # STRICT time checks
            if exp_dep and act_dep and exp_dep != act_dep:
                errs.append(f"Departure mismatch (expected {exp_dep}, RTT {act_dep})")
            elif exp_dep and not act_dep:
                errs.append("Departure time not available in RTT at origin")

            if exp_arr and act_arr and exp_arr != act_arr:
                errs.append(f"Arrival mismatch (expected {exp_arr}, RTT {act_arr})")
            elif exp_arr and not act_arr:
                errs.append("Arrival time not available in RTT at destination")

            if not origin_isCall:
                errs.append("Origin is PASS (not a call)")
            if not origin_isCallPublic:
                errs.append("Origin is not a public call")
            if not dest_isCall:
                errs.append("Destination is PASS (not a call)")
            if not dest_isCallPublic:
                errs.append("Destination is not a public call")

            status = "OK" if not errs else "FAIL"

            results.append(
                {
                    **row.to_dict(),
                    "status": status,
                    "error": " | ".join(errs),
                    "operator": op,
                    "planned_cancel": "Y" if planned_cancel else "",
                    "realtimeActivated": realtimeActivated,
                    "serviceUid": uid,
                    "runDate": rtt_run_date,
                    "origin_displayAs": o["displayAs"],
                    "dest_displayAs": d["displayAs"],
                    "origin_isCall": "Y" if origin_isCall else "",
                    "dest_isCall": "Y" if dest_isCall else "",
                    "origin_isCallPublic": "Y" if origin_isCallPublic else "",
                    "dest_isCallPublic": "Y" if dest_isCallPublic else "",
                    "act_dep": act_dep,
                    "act_arr": act_arr,
                    "origin_rtt_deps": origin_rtt_deps,
                    "variants_count": variants_count,
                    "variants_deps": variants_deps,
                    "has_variants": has_variants,
                }
            )

            done += 1
            progress.progress(min(1.0, done / total))

    progress.empty()
    report = pd.DataFrame(results)
    st.session_state["report"] = report
else:
    report = existing_report

# ----------------------------
# Output
# ----------------------------
if report is None or report.empty:
    st.warning("No report available. Run the check to generate one.")
    st.stop()

ok = report[report["status"] == "OK"]
fail = report[report["status"] == "FAIL"]
nc = report[report["status"] == "NOT CHECKED"]

st.subheader("Outcome")
st.write(f"Checked: **{len(report)}** | OK: **{len(ok)}** | FAIL: **{len(fail)}** | NOT CHECKED: **{len(nc)}**")

st.subheader("Visual report")
view_cols = [
    "date",
    "rtt_date",
    "Day of the week",
    "Rail day index",
    "Shift flag",
    "day_alignment",
    "headcode",
    "origin_crs",
    "dest_crs",
    "exp_dep",
    "act_dep",
    "exp_arr",
    "act_arr",
    "origin_isCall",
    "origin_isCallPublic",
    "dest_isCall",
    "dest_isCallPublic",
    "origin_displayAs",
    "dest_displayAs",
    "operator",
    "planned_cancel",
    "realtimeActivated",
    "has_variants",
    "variants_count",
    "variants_deps",
    "status",
    "error",
    "diagram_info",
    "serviceUid",
    "runDate",
    "origin_rtt_deps",
]
view_cols = [c for c in view_cols if c in report.columns]

st.dataframe(
    report[view_cols].sort_values(["rtt_date", "origin_crs", "headcode", "exp_dep"], na_position="last"),
    use_container_width=True,
)

with st.expander("Only FAIL / NOT CHECKED"):
    bad = report[report["status"].isin(["FAIL", "NOT CHECKED"])].copy()
    if bad.empty:
        st.write("None")
    else:
        st.dataframe(
            bad[view_cols].sort_values(["rtt_date", "status", "headcode"], na_position="last"),
            use_container_width=True,
        )

st.subheader("CSV report")
c1, c2 = st.columns(2)

with c1:
    csv_all = report[view_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV (all results)",
        data=csv_all,
        file_name=f"pass_ride_checker_ALL_{date_from}_{date_to}.csv",
        mime="text/csv",
        use_container_width=True,
    )

with c2:
    failures_only = report[report["status"].isin(["FAIL", "NOT CHECKED"])].copy()
    csv_fail = failures_only[view_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV (failures only)",
        data=csv_fail,
        file_name=f"pass_ride_checker_FAIL_ONLY_{date_from}_{date_to}.csv",
        mime="text/csv",
        use_container_width=True,
    )
