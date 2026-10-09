"""Streamlit Community Cloud launcher for the industrial load optimizer."""

# A second entrypoint lets Community Cloud deploy this project with a fresh
# runtime configuration while sharing the primary app's implementation.
from app import main

main()
