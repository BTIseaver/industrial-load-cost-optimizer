"""Streamlit Community Cloud launcher for the industrial load optimizer."""

import runpy

# A second entrypoint lets Community Cloud deploy this project with a fresh
# runtime configuration while executing the primary app unchanged.
runpy.run_path("app.py", run_name="__main__")
