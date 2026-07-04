"""Business dashboard: reads the monitoring store and presents churn risk,
model health, and drift/alert status in business terms.

Deliberately framework-agnostic — data_access.py and charts.py have no
Streamlit import, so they're plain-Python testable. dashboard/app.py (repo
root, not in this package) is the only place Streamlit is imported.
"""
