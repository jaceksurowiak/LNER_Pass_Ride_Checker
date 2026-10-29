# Pass Ride Checker – LNER (TRENT)

Checks the other operators' trains (Brand blank) in a TRACS **pass-trips.csv** export
against Realtime Trains, and produces **WKxx PASS RIDES.xlsx**.

## Files

| File | What it is |
|---|---|
| `app.py` | The app for the **old** RTT API – use until 31 March 2027 |
| `app2.py` | The same app for the **new** RTT API (token) – use from 31 March 2027 |
| `requirements.txt` | Python packages Streamlit installs |
| `RailReferences.csv` | Station names → CRS codes (can be replaced from the app's sidebar for one run) |
| `.streamlit/config.toml` | Colours, matching the Traincrew Planning Hub |
| `.gitignore` | Keeps your RTT password out of GitHub |

## Put it on GitHub

1. On github.com, open your existing app repository (or **New repository**, e.g. `pass-ride-checker`, Private).
2. **Add file → Upload files**, then drag in everything from this folder, including the `.streamlit` folder.
   - If the `.streamlit` folder doesn't upload by dragging, use **Add file → Create new file**,
     type `.streamlit/config.toml` as the name and paste its contents.
3. **Commit changes**.

## Run it on Streamlit Community Cloud

1. Go to share.streamlit.io and sign in with GitHub.
   - **Existing app:** it redeploys by itself after the commit. If not, open the app's menu → **Reboot**.
   - **New app:** **Create app** → pick the repository, branch `main`, main file `app.py` → **Deploy**.
2. Add your Realtime Trains login: app menu → **Settings → Secrets**.

   **Now (`app.py`, old API):**

   ```toml
   RTT_USER = "rttapi_yourname"
   RTT_PASS = "your-old-api-password"
   ```

   **From March 2027 (`app2.py`, new API):** request a token at https://api-portal.rtt.io/ and add:

   ```toml
   RTT_TOKEN = "your-token-from-the-rtt-api-portal"
   ```

   Optional app-wide default for parallel requests (1–16): `RTT_WORKERS = 8`.
   Each person can also save their own default in the app (**Advanced → Save as my default**); it is remembered in their browser.

## Switching to the new API (before 31 March 2027)

1. Get a token at https://api-portal.rtt.io/ and add `RTT_TOKEN` to Secrets (keep the old lines for now).
2. Either change the existing app's main file to `app2.py` (app menu → **Settings**, or delete and recreate the app),
   or deploy a second app from the same repository with main file `app2.py` and test it side by side first.
3. Realtime Trains rate-limits the new API. If a run reports a rate limit, lower `RTT_WORKERS`.

## Using it

1. Upload **pass-trips.csv**. The railway week is read from the file's diagram dates.
2. Optional: upload an updated **RailReferences.csv** for this run.
3. Check the **railway week** shown in the sidebar and pick the **diagram days** (Sat–Fri).
4. **Run check**, review the Issues tab, then **Download WKxx PASS RIDES.xlsx**.

### How it decides things

- Only rows with a blank **Brand** are checked.
- **Railway weeks** follow Network Rail planning weeks: Saturday–Friday, Week 1 starting on the Saturday nearest 1 April.
- **After midnight:** when JourneyDays Run is DiagramDays Run moved forward one day (e.g. FO → SO), the train is checked on the next calendar day.
- **Picking days:** a day includes that day's diagrams **and** any after-midnight trains that run on that day from the previous day's diagram. Picking only Saturday therefore also checks a Friday diagram's 00:17 train on the following Saturday.
- Each train is looked up in RTT once (8 requests at a time; change under **Advanced**) and its result is copied to every diagram it appears on.

## Run it on your own computer (optional)

```bash
pip install -r requirements.txt
streamlit run app.py
```

For the RTT login locally, create `.streamlit/secrets.toml` with the two lines above. It is ignored by git.
