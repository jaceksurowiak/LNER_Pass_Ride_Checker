import re
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Dict, Set, List, Tuple

import pandas as pd
import requests
import streamlit as st
from requests.auth import HTTPBasicAuth


RTT_BASE = "https://api.rtt.io/api/v1"

WEEKDAYS = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
DOW_TO_IDX = {d: i for i, d in enumerate(WEEKDAYS)}  # MON=0 .. SUN=6


# ----------------------------
# Normalisation / parsing
# ----------------------------
def norm(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def parse_ddmmyyyy(s: str) -> Optional[dt.date]:
    if pd.isna(s):
        return None
    s = str(s).strip()
    if not s:
        return None
    try:
        return dt.datetime.strptime(s, "%d/%m/%Y").date()
    except ValueError:
        return None


def parse_hhmm(s: str) -> Optional[str]:
    if pd.isna(s):
        return None
    s = str(s).strip()
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


def hhmm_to_minutes(hhmm: str) -> int:
    hh, mm = hhmm.split(":")
    return int(hh) * 60 + int(mm)


def within_tolerance(expected: Optional[str], actual: Optional[str], tol_min: int) -> bool:
    if expected is None or actual is None:
        return False
    if tol_min <= 0:
        return expected == actual
    return abs(hhmm_to_minutes(expected) - hhmm_to_minutes(actual)) <= tol_min


def daterange(start: dt.date, end: dt.date):
    cur = start
    while cur <= end:
        yield cur
        cur += dt.timedelta(days=1)


def dow_code(d: dt.date) -> str:
    return WEEKDAYS[d.weekday()]


def parse_days_run(value: str) -> Set[str]:
    """
    Convert common 'Diagram Days Run' formats into weekday set: {"MON",...}
    """
    if pd.isna(value):
        return set()
    raw = str(value).strip().upper()
    if not raw:
        return set()

    raw = raw.replace("&", " ").replace("/", " ")
    raw = re.sub(r"\s+", " ", raw)

    if raw in {"DAILY", "EVERY DAY", "EVERYDAY"}:
        return set(WEEKDAYS)
    if raw in {"WEEKDAYS", "MON-FRI"}:
        return {"MON", "TUE", "WED", "THU", "FRI"}
    if raw in {"WEEKENDS"}:
        return {"SAT", "SUN"}

    # Remove common noise words
    raw = raw.replace("ONLY", "").strip()

    # Range like MON-FRI
    m = re.match(r"^(MON|TUE|WED|THU|FRI|SAT|SUN)\s*-\s*(MON|TUE|WED|THU|FRI|SAT|SUN)$", raw)
    if m:
        a, b = m.group(1), m.group(2)
        ia, ib = DOW_TO_IDX[a], DOW_TO_IDX[b]
        if ia <= ib:
            return set(WEEKDAYS[ia:ib+1])
        return set(WEEKDAYS[ia:] + WEEKDAYS[:ib+1])

    # Comma/space list like MON,TUE,WED
    tokens = re.split(r"[,\s]+", raw)
    tokens = [t for t in tokens if t]
    if tokens and all(t in WEEKDAYS for t in tokens):
        return set(tokens)

    # Condensed letters heuristic e.g. MTWTFSS
    compact = re.sub(r"[^MTWFSU]", "", raw)
    if len(compact) >= 5:
        allowed = set()
        if "M" in compact: allowed.add("MON")
        if "W" in compact: allowed.add("WED")
        if "F" in compact: allowed.add("FRI")
        if "T" in compact: allowed.update({"TUE", "THU"})  # safe assumption
        if "S" in compact: allowed.update({"SAT", "SUN"})
        if "U" in compact: allowed.add("SUN")
        return allowed

    # Single day mention
    for d in WEEKDAYS:
        if d in raw:
            return {d}

    return set()


# ----------------------------
# Data loading
# ----------------------------
@st.cache_data
def load_pass_trips(uploaded_file) -> pd.DataFrame:
    # pass-trips.csv has 3 metadata lines before the header row
    return pd.read_csv(uploaded_file, skiprows=3)


@st.cache_data
def load_rail_refs(uploaded_file) -> pd.DataFrame:
    # RailReferences.csv: TIPLOC, CRS, Description (no header)
    return pd.read_csv(uploaded_file, header=None, names=["tiploc", "crs", "description"])


def build_desc_to_crs(rail_refs: pd.DataFrame) -> Dict[str, str]:
    d: Dict[str, str] = {}
    for _, r in rail_refs.iterrows():
        desc = norm(r["description"])
        crs = str(r["crs"]).strip() if not pd.isna(r["crs"]) else ""
        if desc and crs and desc not in d:
            d[desc] = crs

    # Some helpful aliases (extend as needed)
    aliases = {
        "kings cross": "KGX",
        "london kings cross": "KGX",
        "kings x": "KGX",
        "edinburgh": "EDB",
        "edinburgh waverley": "EDB",
    }
    for k, v in aliases.items():
        d.setdefault(norm(k), v)
    return d


# ----------------------------
# RTT calls
# ----------------------------
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
            "atocName": s.get("atocName"),
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


# ----------------------------
# Matching structures
# ----------------------------
@dataclass
class ExpectedService:
    headcode: str
    origin_name: str
    dest_name: str
    dep_hhmm: Optional[str]
    arr_hhmm: Optional[str]
    origin_crs: Optional[str]
    dest_crs: Optional[str]
    diagram_id: str


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="No-Brand RTT Checker", layout="wide")
st.title("No-Brand RTT Checker")
st.caption("Checks services where Brand is blank. Validates Headcode + From/To + booked dep/arr on eligible dates.")

with st.sidebar:
    st.header("1) Upload files")

    pass_trips_file = st.file_uploader("Upload pass-trips.csv", type=["csv"])
    rail_refs_file = st.file_uploader("Upload RailReferences.csv (optional but recommended)", type=["csv"])

    st.header("2) Select date range")
    today = dt.date.today()
    date_from = st.date_input("From", value=today)
    date_to = st.date_input("To", value=today + dt.timedelta(days=7))

    st.header("3) Rules")
    use_diagram_window = st.checkbox("Use Diagram Start/End as guardrail", value=True)
    tol = st.selectbox("Time tolerance (minutes)", options=[0, 1, 2, 3, 5], index=0)

    st.header("4) Run")
    run_btn = st.button("Run check", type="primary")

# Credentials from Streamlit secrets
rtt_user = st.secrets.get("RTT_USER", "")
rtt_pass = st.secrets.get("RTT_PASS", "")
if not rtt_user or not rtt_pass:
    st.warning("RTT credentials missing. Add RTT_USER / RTT_PASS in Streamlit Secrets.")
auth = HTTPBasicAuth(rtt_user, rtt_pass) if rtt_user and rtt_pass else None

if not pass_trips_file:
    st.info("Upload pass-trips.csv to begin.")
    st.stop()

# Load data
df = load_pass_trips(pass_trips_file)

if rail_refs_file:
    rail_refs = load_rail_refs(rail_refs_file)
    desc_to_crs = build_desc_to_crs(rail_refs)
    st.success("Using uploaded RailReferences.csv")
else:
    st.warning("RailReferences.csv not uploaded. Station CRS mapping may be weaker without it.")
    desc_to_crs = {}  # still runs; origin/dest checks will be name-based

# Filter: Brand blank
df["Brand_missing"] = df["Brand"].isna() | (df["Brand"].astype(str).str.strip() == "")
df_nb = df[df["Brand_missing"]].copy()

# Parse relevant fields
df_nb["DiagramStart_dt"] = df_nb["DiagramStart Date"].apply(parse_ddmmyyyy)
df_nb["DiagramEnd_dt"] = df_nb["DiagramEnd Date"].apply(parse_ddmmyyyy)
df_nb["dep_hhmm"] = df_nb["JourneyDeparture"].apply(parse_hhmm)
df_nb["arr_hhmm"] = df_nb["JourneyArrival"].apply(parse_hhmm)
df_nb["days_set"] = df_nb["Diagram Days Run"].apply(parse_days_run)

df_nb["origin_crs"] = df_nb["JourneyOrigin"].apply(lambda x: desc_to_crs.get(norm(x)) if desc_to_crs else None)
df_nb["dest_crs"] = df_nb["JourneyDestination"].apply(lambda x: desc_to_crs.get(norm(x)) if desc_to_crs else None)

st.subheader("Rows considered (Brand blank)")
st.write(f"Rows: **{len(df_nb)}** | Unique headcodes: **{df_nb['Headcode'].nunique()}**")

with st.expander("Preview filtered input rows"):
    cols = [
        "Headcode", "JourneyOrigin", "JourneyDestination", "JourneyDeparture", "JourneyArrival",
        "DiagramStart Date", "DiagramEnd Date", "Diagram Days Run", "DiagramID", "origin_crs", "dest_crs"
    ]
    st.dataframe(df_nb[cols], use_container_width=True)

if not run_btn:
    st.stop()

if auth is None:
    st.error("RTT credentials not configured (Secrets).")
    st.stop()

# Build expected services per date
date_from = min(date_from, date_to)
date_to = max(date_from, date_to)

expected_by_date: Dict[dt.date, List[ExpectedService]] = {}
for d in daterange(date_from, date_to):
    dc = dow_code(d)
    subset = df_nb[df_nb["days_set"].apply(lambda s: dc in s if isinstance(s, set) else False)].copy()

    if use_diagram_window:
        subset = subset[
            subset.apply(
                lambda r: (r["DiagramStart_dt"] is None or d >= r["DiagramStart_dt"]) and
                          (r["DiagramEnd_dt"] is None or d <= r["DiagramEnd_dt"]),
                axis=1
            )
        ]

    exp_list: List[ExpectedService] = []
    for _, r in subset.iterrows():
        exp_list.append(
            ExpectedService(
                headcode=str(r["Headcode"]).strip(),
                origin_name=str(r["JourneyOrigin"]).strip(),
                dest_name=str(r["JourneyDestination"]).strip(),
                dep_hhmm=r["dep_hhmm"],
                arr_hhmm=r["arr_hhmm"],
                origin_crs=r.get("origin_crs"),
                dest_crs=r.get("dest_crs"),
                diagram_id=str(r.get("DiagramID", "")).strip()
            )
        )
    expected_by_date[d] = exp_list

total_expected = sum(len(v) for v in expected_by_date.values())
st.subheader("Expected checks in selected date range")
st.write(f"Date range: **{date_from} → {date_to}** | Expected service-checks: **{total_expected}**")

# Efficient RTT queries: (origin_crs, date) once
@st.cache_data(ttl=300)
def cached_location_search(origin: str, d: dt.date, user: str, pw: str) -> dict:
    a = HTTPBasicAuth(user, pw)
    return rtt_location_services(origin, d, a)

def check_one(expected: ExpectedService, d: dt.date, origin_services: pd.DataFrame) -> Tuple[bool, str, Optional[bool], Optional[str]]:
    """
    Returns:
      ok, reason, planned_cancel, atoc_name
    """
    # Filter by headcode + departure time at origin
    headcode = expected.headcode
    dep = expected.dep_hhmm

    candidates = origin_services[origin_services["trainIdentity"] == headcode].copy()
    if candidates.empty:
        return False, "Headcode not found at origin on date", None, None

    # Validate departure time
    if dep:
        candidates["dep_ok"] = candidates["gbttBookedDeparture"].apply(lambda t: within_tolerance(dep, t, tol))
        candidates = candidates[candidates["dep_ok"] == True]
        if candidates.empty:
            return False, "Headcode found but departure time differs", None, None

    # Take first candidate, then confirm destination + arrival using service detail
    cand = candidates.iloc[0]
    uid = cand.get("serviceUid")
    run_date_iso = cand.get("runDate")
    planned_cancel = bool(cand.get("plannedCancel", False))
    atoc_name = cand.get("atocName")

    if not uid or not run_date_iso:
        # If detail missing, do a weaker match (headcode+dep already checked)
        # Destination/arrival cannot be verified reliably.
        return False, "Insufficient RTT data to verify destination/arrival", planned_cancel, atoc_name

    try:
        detail = rtt_service_detail(uid, run_date_iso, auth)
        det = extract_origin_dest_times_from_detail(detail)
    except Exception:
        return False, "RTT service detail lookup failed", planned_cancel, atoc_name

    # Destination match (name-based; CRS-based could be added if detail includes CRS)
    if expected.dest_name and det.get("dest_name"):
        if norm(expected.dest_name) not in norm(det["dest_name"]) and norm(det["dest_name"]) not in norm(expected.dest_name):
            return False, f"Destination differs (RTT: {det.get('dest_name')})", planned_cancel, atoc_name

    # Arrival time match
    exp_arr = expected.arr_hhmm
    rtt_arr = det.get("book_arr")
    if exp_arr and rtt_arr:
        if not within_tolerance(exp_arr, rtt_arr, tol):
            return False, f"Arrival time differs (RTT: {rtt_arr})", planned_cancel, atoc_name

    # Origin name sanity check (optional, light)
    if expected.origin_name and det.get("origin_name"):
        # don’t fail hard on minor naming differences
        pass

    return True, "Matched", planned_cancel, atoc_name


report_rows = []
progress = st.progress(0)
done = 0

for d, exp_list in expected_by_date.items():
    if not exp_list:
        continue

    # For reliability, prefer CRS. If CRS missing, we cannot query RTT properly.
    # In that case, mark as unmapped.
    # We group expected services by origin CRS for that date.
    by_origin: Dict[str, List[ExpectedService]] = {}
    unmapped: List[ExpectedService] = []

    for e in exp_list:
        if e.origin_crs:
            by_origin.setdefault(e.origin_crs, []).append(e)
        else:
            unmapped.append(e)

    # Handle unmapped
    for e in unmapped:
        report_rows.append({
            "date": d.isoformat(),
            "headcode": e.headcode,
            "from": e.origin_name,
            "to": e.dest_name,
            "dep": e.dep_hhmm,
            "arr": e.arr_hhmm,
            "status": "NOT CHECKED",
            "reason": "Origin CRS not mapped (upload RailReferences.csv or add alias)",
            "planned_cancel": "",
            "operator": "",
            "diagram_id": e.diagram_id,
        })
        done += 1
        progress.progress(min(1.0, done / max(1, total_expected)))

    # Query RTT per origin CRS once
    for origin_crs, services in by_origin.items():
        try:
            payload = cached_location_search(origin_crs, d, rtt_user, rtt_pass)
            origin_services = flatten_location_services(payload)
        except Exception as ex:
            for e in services:
                report_rows.append({
                    "date": d.isoformat(),
                    "headcode": e.headcode,
                    "from": e.origin_name,
                    "to": e.dest_name,
                    "dep": e.dep_hhmm,
                    "arr": e.arr_hhmm,
                    "status": "FAIL",
                    "reason": f"RTT location query failed: {ex}",
                    "planned_cancel": "",
                    "operator": "",
                    "diagram_id": e.diagram_id,
                })
                done += 1
                progress.progress(min(1.0, done / max(1, total_expected)))
            continue

        for e in services:
            ok, reason, planned_cancel, atoc_name = check_one(e, d, origin_services)
            report_rows.append({
                "date": d.isoformat(),
                "headcode": e.headcode,
                "from": e.origin_name,
                "to": e.dest_name,
                "dep": e.dep_hhmm,
                "arr": e.arr_hhmm,
                "status": "OK" if ok else "FAIL",
                "reason": "" if ok else reason,
                "planned_cancel": "Y" if planned_cancel else "",
                "operator": atoc_name or "",
                "diagram_id": e.diagram_id,
            })
            done += 1
            progress.progress(min(1.0, done / max(1, total_expected)))

progress.empty()

report = pd.DataFrame(report_rows)

# Summary + “screenshot-style” report
fails = report[report["status"] == "FAIL"]
oks = report[report["status"] == "OK"]
not_checked = report[report["status"] == "NOT CHECKED"]

st.subheader("Result summary")
st.write(f"OK: **{len(oks)}** | FAIL: **{len(fails)}** | Not checked: **{len(not_checked)}**")

def format_line(r) -> str:
    ddmmyyyy = dt.date.fromisoformat(r["date"]).strftime("%d/%m/%Y")
    return f"{r['headcode']}: {r['dep']} {r['from']} - {r['to']} {r['arr']} on {ddmmyyyy}"

# “Screenshot-like” message block
st.markdown("### Daily-style report")
if len(fails) == 0 and len(not_checked) == 0:
    st.success("All services found to be running!")
    st.write("The following services were checked and running as expected:")
    for _, r in oks.sort_values(["date", "dep", "headcode"]).iterrows():
        st.write(f"- {format_line(r)}")
else:
    st.error("Some services did not match.")
    if len(oks) > 0:
        st.write("**Matched:**")
        for _, r in oks.sort_values(["date", "dep", "headcode"]).iterrows():
            st.write(f"- {format_line(r)}")
    if len(fails) > 0:
        st.write("**Not matched / differences:**")
        for _, r in fails.sort_values(["date", "dep", "headcode"]).iterrows():
            st.write(f"- {format_line(r)} — {r['reason']}")
    if len(not_checked) > 0:
        st.write("**Not checked:**")
        for _, r in not_checked.sort_values(["date", "dep", "headcode"]).iterrows():
            st.write(f"- {format_line(r)} — {r['reason']}")

st.subheader("Detailed table")
st.dataframe(report.sort_values(["date", "status", "dep", "headcode"]), use_container_width=True)

st.download_button(
    "Download report as CSV",
    data=report.to_csv(index=False).encode("utf-8"),
    file_name=f"rtt_no_brand_report_{date_from}_{date_to}.csv",
    mime="text/csv",
)
