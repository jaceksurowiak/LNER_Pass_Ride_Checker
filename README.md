# Pass Ride Checker – LNER (TRENT)

Checks the other operators' trains (Brand blank) in a TRACS **pass-trips.csv** export
against Realtime Trains, and produces **WKxx PASS RIDES.xlsx**.

## Files

| File | What it is |
|---|---|
| `app.py` | The app |
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
2. Add your Realtime Trains **API** login (the `rttapi_…` username from the RTT API portal, not your website login). The app won't start without it: app menu → **Settings → Secrets**, and paste:

   ```toml
   RTT_USER = "your-rtt-api-username"
   RTT_PASS = "your-rtt-api-password"
   ```

## Using it

1. Upload **pass-trips.csv**. The railway week is read from the file's diagram dates.
2. Optional: upload an updated **RailReferences.csv** for this run.
3. Check the **railway week** and pick the **diagram days** (Sat–Fri).
4. **Run check**, review the Issues tab, then **Download WKxx PASS RIDES.xlsx**.

### How it decides things

- Only rows with a blank **Brand** are checked.
- **Railway weeks** follow Network Rail planning weeks: Saturday–Friday, Week 1 starting on the Saturday nearest 1 April.
- **After midnight:** when JourneyDays Run is DiagramDays Run moved forward one day (e.g. FO → SO), the train is checked on the next calendar day.
- Each train is looked up in RTT once (8 requests at a time; change under **Advanced**) and its result is copied to every diagram it appears on.

## Run it on your own computer (optional)

```bash
pip install -r requirements.txt
streamlit run app.py
```

For the RTT login locally, create `.streamlit/secrets.toml` with the two lines above. It is ignored by git.
