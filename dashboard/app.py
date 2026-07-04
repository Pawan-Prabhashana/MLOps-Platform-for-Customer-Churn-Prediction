# ruff: noqa: E402
"""dashboard/app.py — ChurnOps business dashboard (Streamlit).

Reads ONLY through src/churnops/dashboard/data_access.py, which itself reads
ONLY through src/churnops/monitoring/store.py — no SQL lives in this file.
Works against whatever DATABASE_URL churnops.config.Settings resolves to:
the local docker churnops Postgres by default, or a real Supabase project
with zero code changes if DATABASE_URL is swapped in .env.

Usage
-----
    streamlit run dashboard/app.py --server.port 8501
    python pipelines/serve_dashboard.py          # convenience wrapper
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import pandas as pd
import streamlit as st
import yaml

from churnops.config import get_settings
from churnops.dashboard import charts, data_access
from churnops.dashboard.data_access import DashboardDataError

# ── Config ────────────────────────────────────────────────────────────────────

_DASHBOARD_YAML = _REPO_ROOT / "configs" / "dashboard.yaml"


@st.cache_data
def _load_config() -> dict:
    with _DASHBOARD_YAML.open() as f:
        return yaml.safe_load(f)


CFG = _load_config()
TTL = CFG["refresh"]["cache_ttl_seconds"]
THEME = CFG.get("theme", {})

st.set_page_config(page_title="ChurnOps — Business Dashboard", layout="wide", page_icon="📉")


# ── Cached data-access wrappers (short TTL; sidebar "Refresh now" bypasses) ───

_cached_recent_predictions = st.cache_data(ttl=TTL, show_spinner=False)(data_access.fetch_recent_predictions)
_cached_churn_by_contract = st.cache_data(ttl=TTL, show_spinner=False)(data_access.fetch_churn_rate_by_contract)
_cached_top_k = st.cache_data(ttl=TTL, show_spinner=False)(data_access.fetch_top_k)
_cached_volume = st.cache_data(ttl=TTL, show_spinner=False)(data_access.fetch_prediction_volume)
_cached_drift = st.cache_data(ttl=TTL, show_spinner=False)(data_access.fetch_latest_drift)
_cached_performance = st.cache_data(ttl=TTL, show_spinner=False)(data_access.fetch_latest_performance)
_cached_alerts = st.cache_data(ttl=TTL, show_spinner=False)(data_access.fetch_active_alerts)
_cached_monitoring_run = st.cache_data(ttl=TTL, show_spinner=False)(data_access.fetch_latest_monitoring_run)
_cached_prediction_count = st.cache_data(ttl=TTL, show_spinner=False)(data_access.fetch_prediction_count)
_cached_overview_kpis = st.cache_data(ttl=TTL, show_spinner=False)(data_access.fetch_overview_kpis)
_cached_production_model = st.cache_data(ttl=TTL, show_spinner=False)(data_access.fetch_production_model_info)


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ── Header + sidebar ───────────────────────────────────────────────────────────

st.title("📉 ChurnOps — Business Dashboard")
st.caption("Churn risk, model health, and drift — read live from the monitoring store.")

settings = get_settings()
database_url = settings.monitoring_database_url

with st.sidebar:
    st.header("Filters")
    if st.button("🔄 Refresh now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    window = st.selectbox(
        "Window (most recent N predictions)",
        options=CFG["refresh"]["window_options"],
        index=CFG["refresh"]["window_options"].index(CFG["refresh"]["default_window"])
        if CFG["refresh"]["default_window"] in CFG["refresh"]["window_options"] else 0,
    )
    top_k = st.slider("Top-K at-risk customers", min_value=1, max_value=CFG["top_k"]["max"], value=CFG["top_k"]["default"])

    st.divider()
    db_host = database_url.split("@")[-1] if "@" in database_url else database_url
    st.caption(f"Store: `{db_host}`")


# ── Connectivity + empty-state guard ──────────────────────────────────────────

try:
    total_rows = _cached_prediction_count(database_url)
except DashboardDataError as exc:
    st.error(
        "**Can't reach the monitoring store.**\n\n"
        f"{exc}\n\n"
        "Check that Postgres is running (`docker compose ps`) and that `DATABASE_URL` "
        "(or the local defaults in `.env`) point at a reachable database."
    )
    st.stop()

if total_rows == 0:
    st.info(
        "**No data yet.** The monitoring store is reachable but empty.\n\n"
        "Seed it end-to-end, then refresh this page:\n"
        "```bash\n"
        "python producer.py --mode batch --limit 500\n"
        "python consumer.py --mode batch --max-records 500\n"
        "python pipelines/run_monitoring.py --source kafka --window 500\n"
        "```"
    )
    st.stop()


# ── Contract filter (populated from real data) ────────────────────────────────

churn_by_contract_full = _cached_churn_by_contract(database_url, limit=5000)
available_contracts = sorted(churn_by_contract_full["contract"].tolist()) if not churn_by_contract_full.empty else []

with st.sidebar:
    selected_contracts = st.multiselect(
        "Contract type", options=available_contracts, default=available_contracts,
    ) if available_contracts else []


def _apply_contract_filter(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "contract" not in df.columns or not selected_contracts or not available_contracts:
        return df
    if set(selected_contracts) == set(available_contracts):
        return df
    return df[df["contract"].isin(selected_contracts)]


# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_overview, tab_segment, tab_topk, tab_health, tab_score = st.tabs(
    ["Overview", "Churn by Segment", "Top-K At-Risk", "Model Health & Drift", "Score a Customer"]
)

risk_high = CFG["risk"]["high_threshold"]
risk_mid = CFG["risk"]["mid_threshold"]

# ── 1. Overview ────────────────────────────────────────────────────────────────

with tab_overview:
    kpis = _cached_overview_kpis(database_url, window=window, risk_threshold=risk_high)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers scored", f"{kpis['total']:,}", delta=kpis["delta_total"])
    c2.metric(
        "Predicted churn rate",
        f"{kpis['churn_rate_pct']:.1f}%" if kpis["churn_rate_pct"] is not None else "—",
        delta=f"{kpis['delta_churn_rate_pct']:.1f} pts" if kpis["delta_churn_rate_pct"] is not None else None,
    )
    c3.metric(
        "Avg. churn probability",
        f"{kpis['avg_probability']:.1%}" if kpis["avg_probability"] is not None else "—",
        delta=f"{kpis['delta_avg_probability']:.1%}" if kpis["delta_avg_probability"] is not None else None,
    )
    c4.metric(
        f"High-risk (> {risk_high:.0%})", f"{kpis['high_risk_count']:,}",
        delta=kpis["delta_high_risk_count"],
    )
    st.caption("Deltas compare this window against the immediately preceding window of the same size.")

    recent_df = _apply_contract_filter(_cached_recent_predictions(database_url, limit=window))
    st.plotly_chart(charts.probability_histogram(recent_df), use_container_width=True, key="overview_prob_hist")


# ── 2. Churn by segment ────────────────────────────────────────────────────────

with tab_segment:
    st.plotly_chart(charts.churn_rate_bar(churn_by_contract_full), use_container_width=True, key="segment_churn_bar")

    st.caption(
        f"Secondary cuts not shown — {', '.join(data_access.UNAVAILABLE_SEGMENT_FEATURES)} are not "
        "stored in the current `predictions` schema (only `contract` is enriched onto scored records; "
        "see `src/churnops/monitoring/db.py`)."
    )

    st.plotly_chart(charts.probability_histogram(recent_df), use_container_width=True, key="segment_prob_hist")


# ── 3. Top-K at-risk customers ─────────────────────────────────────────────────

with tab_topk:
    st.subheader(f"Top {top_k} at-risk customers")

    pool = _cached_top_k(database_url, k=max(top_k * 5, 50), limit=5000)
    filtered = _apply_contract_filter(pool).sort_values("churn_probability", ascending=False).head(top_k)

    if filtered.empty:
        st.info("No customers match the current filters.")
    else:
        display_df = filtered.copy()
        display_df["risk"] = display_df["churn_probability"].apply(
            lambda p: data_access.risk_band(p, risk_mid, risk_high)
        )

        def _risk_style(row: pd.Series) -> list[str]:
            color = data_access.risk_color(row["risk"], THEME)
            return [f"background-color: {color}22"] * len(row)

        cols = [c for c in ["customer_id", "churn_probability", "prediction", "risk", "contract", "processed_ts"] if c in display_df.columns]
        st.dataframe(
            display_df[cols].style.apply(_risk_style, axis=1).format({"churn_probability": "{:.2%}"}),
            use_container_width=True, hide_index=True, key="topk_table",
        )

        st.download_button(
            "⬇ Download top-K as CSV",
            data=data_access.to_csv_bytes(display_df[cols]),
            file_name="top_at_risk_customers.csv",
            mime="text/csv",
            key="topk_download",
        )


# ── 4. Model health & drift ─────────────────────────────────────────────────────

with tab_health:
    model_info = _cached_production_model(database_url)
    st.caption(
        f"Production model: **{model_info['version'] or 'unknown'}** "
        f"(source: {model_info['source']}" + (f", name: {model_info['model_name']}" if model_info["model_name"] else "") + ")"
    )

    perf_df = _cached_performance(database_url, limit=20)
    st.markdown("**Latest performance metrics**")
    if perf_df.empty:
        st.info("No performance metrics yet — labeled predictions (`actual_churn`) are required. Run `pipelines/run_monitoring.py`.")
    else:
        latest_computed_at = perf_df["computed_at"].max()
        latest = perf_df[perf_df["computed_at"] == latest_computed_at]
        cols = st.columns(len(latest)) if len(latest) <= 6 else st.columns(6)
        for i, (_, row) in enumerate(latest.iterrows()):
            with cols[i % len(cols)]:
                st.metric(row["metric_name"].replace("_", " ").upper(), f"{row['metric_value']:.3f}")

    st.markdown("**Prediction volume over time**")
    volume_df = _cached_volume(database_url, bucket="hour", limit=5000)
    st.plotly_chart(charts.volume_timeseries(volume_df), use_container_width=True, key="health_volume")

    st.markdown("**Drift**")
    drift_df = _cached_drift(database_url, limit=50)
    if drift_df.empty:
        st.info("No drift metrics yet. Run `pipelines/run_monitoring.py`.")
    else:
        overall_psi = drift_df[(drift_df["feature_name"].isna()) & (drift_df["statistic_name"] == "psi")]
        if not overall_psi.empty:
            psi_val = overall_psi.iloc[0]["statistic_value"]
            band = "significant" if psi_val >= CFG["drift"]["psi_significant"] else (
                "moderate" if psi_val >= CFG["drift"]["psi_moderate"] else "none"
            )
            st.metric("Overall prediction PSI", f"{psi_val:.4f}", help=f"Band: {band}")
        st.caption(
            f"PSI bands — < {CFG['drift']['psi_moderate']:.2f}: ok · "
            f"{CFG['drift']['psi_moderate']:.2f}–{CFG['drift']['psi_significant']:.2f}: watch · "
            f"> {CFG['drift']['psi_significant']:.2f}: drift"
        )
        st.plotly_chart(charts.drift_heatmap(drift_df), use_container_width=True, key="health_drift_heatmap")

    st.markdown("**Active alerts**")
    alerts_df = _cached_alerts(database_url, limit=50, unacknowledged_only=True)
    if alerts_df.empty:
        st.success("No active alerts.")
    else:
        severity_color = {"critical": "🔴", "warning": "🟠", "info": "🔵"}
        alerts_df = alerts_df.copy()
        alerts_df["severity"] = alerts_df["severity"].apply(lambda s: f"{severity_color.get(s, '⚪')} {s}")
        st.dataframe(
            alerts_df[["fired_at", "severity", "category", "message"]],
            use_container_width=True, hide_index=True, key="alerts_table",
        )

    latest_run = _cached_monitoring_run(database_url)
    if latest_run:
        st.caption(
            f"Last monitoring cycle: {latest_run['run_ts']} · "
            f"records={latest_run['records_processed']} · "
            f"drift_detected={latest_run['drift_detected']} · "
            f"performance_degraded={latest_run['performance_degraded']} · "
            f"status={latest_run['status']}"
        )


# ── 5. Score a customer (optional live-scoring widget) ────────────────────────

with tab_score:
    live_cfg = CFG.get("live_scoring", {})
    if not live_cfg.get("enabled", False):
        st.info("Live scoring is disabled in `configs/dashboard.yaml` (`live_scoring.enabled: false`).")
    else:
        st.caption(
            f"POSTs a hypothetical customer to the live FastAPI service at "
            f"`{live_cfg['api_base_url']}{live_cfg['predict_path']}` — separate from, and never blocking, the read-only analytics above."
        )
        with st.form("score_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                gender = st.selectbox("Gender", ["Female", "Male"])
                senior = st.selectbox("Senior citizen", [0, 1])
                partner = st.selectbox("Partner", ["Yes", "No"])
                dependents = st.selectbox("Dependents", ["Yes", "No"])
                tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
                phone_service = st.selectbox("Phone service", ["Yes", "No"])
            with col2:
                multiple_lines = st.selectbox("Multiple lines", ["Yes", "No", "No phone service"])
                internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
                online_security = st.selectbox("Online security", ["Yes", "No", "No internet service"])
                online_backup = st.selectbox("Online backup", ["Yes", "No", "No internet service"])
                device_protection = st.selectbox("Device protection", ["Yes", "No", "No internet service"])
                tech_support = st.selectbox("Tech support", ["Yes", "No", "No internet service"])
            with col3:
                streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
                streaming_movies = st.selectbox("Streaming movies", ["Yes", "No", "No internet service"])
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])
                payment_method = st.selectbox(
                    "Payment method",
                    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                )
                monthly_charges = st.number_input("Monthly charges", min_value=0.0, value=70.0, step=0.5)
            total_charges = st.number_input("Total charges", min_value=0.0, value=840.0, step=1.0)
            submitted = st.form_submit_button("Score customer")

        if submitted:
            payload = {
                "customerID": "DASHBOARD-DEMO",
                "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
                "tenure": tenure, "PhoneService": phone_service, "MultipleLines": multiple_lines,
                "InternetService": internet_service, "OnlineSecurity": online_security,
                "OnlineBackup": online_backup, "DeviceProtection": device_protection,
                "TechSupport": tech_support, "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
                "Contract": contract, "PaperlessBilling": paperless_billing, "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
            }
            api_url = live_cfg["api_base_url"].rstrip("/") + live_cfg["predict_path"]
            try:
                result = _post_json(payload=payload, url=api_url, timeout=live_cfg.get("timeout_s", 5))
                col_a, col_b = st.columns([1, 1])
                with col_a:
                    st.plotly_chart(
                        charts.risk_gauge(result["churn_probability"], risk_mid, risk_high),
                        use_container_width=True, key="score_risk_gauge",
                    )
                with col_b:
                    st.metric("Prediction", result["prediction"])
                    st.metric("Probability", f"{result['churn_probability']:.1%}")
                    st.caption(f"model_source={result.get('model_source')}  model_version={result.get('model_version')}")
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                st.error(f"Could not reach the FastAPI service at `{api_url}`: {exc}")
