# 🏠_Home.py (Temporary Debugging Version)

import streamlit as st

try:
    from utils.xai_techniques import test_function
    
    st.title("Debugging XAI Import")
    st.success("Successfully imported `test_function` from `utils/xai_techniques.py`.")
    
    result = test_function()
    st.write(result)
    
except ImportError as e:
    st.error(f"An ImportError occurred. This means the file path or __init__.py is likely the issue.")
    st.error(f"Details: {e}")

except SyntaxError as e:
    st.error(f"A SyntaxError occurred! This confirms the problem is a typo or error INSIDE the utils/xai_techniques.py file.")
    st.error(f"Details: {e}")

except Exception as e:
    st.error(f"An unexpected error occurred during import.")
    st.error(f"Details: {e}")