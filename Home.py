import streamlit as st

st.set_page_config(layout="wide")
st.title("Ultimate Sanity Check ✅")

st.success("If you can see this message, then Streamlit and your basic configuration (runtime.txt, requirements.txt) are working correctly.")

st.info(
    "This proves that the error is a SyntaxError or ImportError inside one of your project files, such as: \n"
    "- `utils/visualizations.py`\n"
    "- `utils/xai_techniques.py`\n"
    "- `utils/prediction.py`\n"
    "- `model/fetalnet.py`"
)

st.warning("Please now revert this file to its previous version and check the logs in 'Manage app' for the specific traceback.")