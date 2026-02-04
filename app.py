import re
import datetime as dt
from typing import Optional, Dict, Set, List

import pandas as pd
import requests
import streamlit as st
from requests.auth import HTTPBasicAuth

RTT_BASE = "https://api.rtt.io/api/v1"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())


def find_col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    cols = list(df.columns)
    km = {key(c): c for c in cols}
    for w in candidates:
        if key(w) in km:
            return km[key(w)]
    for w in candidates:
        for c in cols:
            if key(w) in key(c):
                return c
    return None


def norm_name(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]+", "", str(s).lower())
    return re.sub(r"\s+", " ", s).strip()


# ------------------------------------------------------------
# Time parsing
# ------------------------------------------------------------
def parse_pass_hhmm(x) -> Optional[str]:
    if pd.isna(x):
        return None
    m = re.match(r"^(\d{1,2}):(\d{2})$", str(x).strip())
    if not m:
        return None
    hh, mm = int(m.group(1)), int(m.group(2))
    if 0 <= hh <= 23 and 0 <= mm <= 59:
        return f"{hh:02d}:{mm:02d}"
    return None


def rtt_public_to_hhmm(x) -> Optional[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = re.sub(r"\D", "", str(x))
    if len(s) == 4:
        return f"{s[:2]}:{s[2:]}"
    if len(s) == 6:
        return f"{s[:2]}:{s[2:4]}"
    return None


def is_overnight(dep: Optional[str], arr: Optional[str]) -> bool:
    if not dep or not arr:
        return False
    return arr < dep


# ------------------------------------------------------------
# Days run parsing (JourneyDays Run ONLY)
# ------------------------------------------------------------
def parse_journey_days_run(value) -> Set[int]:
    if pd.isna(value):
        return set()

    raw = str(value).strip()
    if not raw:
        return set()

    rail_map = {"MO": 0, "TO": 1, "WO": 2, "THO": 3, "FO": 4, "SO": 5, "SU": 6}
    tokens = re.split(r"[,\s;/]+", raw.upper())
    days = {rail_map[t] for t in tokens if t in rail_map}
    if days:
        return days

    blob = re.sub(r"[,\s;/]+", "", raw)
    mode = blob[-1].upper() if blob[-1:].upper() in ("O", "X") else None
    if mode:
        blob = blob[:-1]

    parsed: Set[int] = set()
    i = 0
    while i < len(blob):
        if blob[i : i + 2].lower() == "th":
            parsed.add(3)
            i += 2
        elif blob[i : i + 2].lower() == "su":
            parsed.add(6)
            i += 2
        else:
            parsed |= {
                "M": {0},
                "T": {1},
                "W": {2},
                "F": {4},
                "S": {5},
            }.get(blob[i].upper(), set())
            i += 1

    if mode == "X":
        return set(range(7)) - parsed
    return parsed


# ------------------------------------------------------------
# RailReferences
# ------------------------------------------------------------
@st.cache_data
def load_railrefs(path_or_file):
    return pd.read_csv(path_or_file, header=None, names=["tiploc", "crs", "description"])


def build_desc_to_crs(df: pd.DataFrame) -> Dict[str, str]:
    d = {}
    for _, r in df.iterrows():
        if pd.notna(r["description"]) and pd.notna(r["crs"]):
            d.setdefault(norm_name(r["description"]), str(r["crs"]).upper())
    d.update(
        {
            "kings cross": "KGX",
            "london kings cross": "KGX",
            "edinburgh": "EDB",
            "edinburgh waverley": "EDB",
        }
    )
    return d


# ------------------------------------------------------------
# RTT
# ------------------------------------------------------------
def rtt_location_services(crs: str, date: dt.date, auth) -> dict:
    url = f"{RTT_BASE}/json/search/{crs}/{date:%Y/%m/%d}"
    r = requests.get(url, auth=auth, timeout=30)
    r.raise_for_status()
    return r.json()


def flatten_location_services(payload: dict) -> pd.DataFrame:
    rows = []
    for s in payload.get("services", []):
        ld = s.get("locationDetail") or {}
        rows.append(
            {
                "trainIdentity": s.get("trainIdentity", "").strip(),
                "serviceUid": s.get("serviceUid"),
                "runDate": s.get("runDate"),
                "operator": s.get("atocName", ""),
                "plannedCancel": bool(s.get("plannedCancel")),
                "realtimeActivated": bool(s.get("realtimeActivated")),
                "pub_dep_raw": ld.get("gbttBookedDeparture")
                or ld.get("publicTime")
                or ld.get("realtimeDeparture"),
            }
        )
    return pd.DataFrame(rows)


def rtt_service_detail(uid: str, run_date: str, auth) -> dict:
    y, m, d = run_date.split("-")
    url = f"{RTT_BASE}/json/service/{uid}/{y}/{m}/{d}"
    r = requests.get(url, auth=auth, timeout=30)
    r.raise_for_status()
    return r.json()


def extract_segment_times(detail, origin, dest) -> dict:
    found_from = found_to = False
    dep = arr = None
    act_from = act_to = None

    for loc in detail.get("locations", []):
        ld = loc.get("locationDetail") or {}
        crs = (loc.get("crs") or "").upper()

        if not found_from and crs == origin:
            found_from = True
            act_from = loc.get("description")
            dep = rtt_public_to_hhmm(
                ld.get("gbttBookedDeparture") or ld.get("publicTime")
            )

        elif found_from and not found_to and crs == dest:
            found_to = True
            act_to = loc.get("description")
            arr = rtt_public_to_hhmm(
                ld.get("gbttBookedArrival") or ld.get("publicTime")
            )

    return {
        "from_found": found_from,
        "to_found": found_to,
        "act_from": act_from,
        "act_to": act_to,
        "act_dep": dep,
        "act_arr": arr,
    }


# ------------------------------------------------------------
# App
# ------------------------------------------------------------
st.set_page_config("LNER (TRENT) - PASS RIDE CHECKER", layout="wide")
st.title("LNER (TRENT) - PASS RIDE CHECKER")

auth = HTTPBasicAuth(
    st.secrets.get("RTT_USER", ""),
    st.secrets.get("RTT_PASS", ""),
)

pass_file = st.file_uploader("Upload pass-trips.csv", type="csv")
if not pass_file:
    st.stop()

df = pd.read_csv(pass_file, skiprows=3)

col_headcode = find_col(df, "Headcode")
col_origin = find_col(df, "Journey Origin")
col_dest = find_col(df, "Journey Destination")
col_dep = find_col(df, "Journey Departure")
col_arr = find_col(df, "Journey Arrival")
col_jdays = find_col(df, "JourneyDays Run")
col_ddays = find_col(df, "DiagramDays Run")

df = df[df[find_col(df, "Brand")].isna()].copy()

df["exp_dep"] = df[col_dep].apply(parse_pass_hhmm)
df["exp_arr"] = df[col_arr].apply(parse_pass_hhmm)
df["is_overnight"] = df.apply(lambda r: is_overnight(r["exp_dep"], r["exp_arr"]), axis=1)
df["jdays_set"] = df[col_jdays].apply(parse_journey_days_run)

railrefs = load_railrefs("RailReferences.csv")
desc_to_crs = build_desc_to_crs(railrefs)

df["origin_crs"] = df[col_origin].apply(lambda x: desc_to_crs.get(norm_name(x)))
df["dest_crs"] = df[col_dest].apply(lambda x: desc_to_crs.get(norm_name(x)))

date_from = st.date_input("From", dt.date.today())
date_to = st.date_input("To", dt.date.today() + dt.timedelta(days=7))

expected = []
for i in range((date_to - date_from).days + 1):
    day = date_from + dt.timedelta(days=i)
    wd = day.weekday()
    for _, r in df[df["jdays_set"].apply(lambda s: wd in s)].iterrows():
        expected.append(
            {
                "date": day.isoformat(),
                "Day": day.strftime("%A"),
                "headcode": r[col_headcode],
                "origin_crs": r["origin_crs"],
                "dest_crs": r["dest_crs"],
                "exp_dep": r["exp_dep"],
                "exp_arr": r["exp_arr"],
                "overnight": "Y" if r["is_overnight"] else "",
            }
        )

expected = pd.DataFrame(expected)
st.dataframe(expected, use_container_width=True)
