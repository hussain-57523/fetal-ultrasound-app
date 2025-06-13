# pages/1_Model_Performance.py

import streamlit as st
import pandas as pd
# === CHANGE: Import the new function ===
from utils.visualizations import plot_detailed_confusion_matrix

st.set_page_config(page_title="Model Performance", layout="wide")

st.title("Final Model Performance")
st.markdown("This page details the performance of the **best fine-tuned model** on the unseen test set.")

st.markdown("---")

# --- Display Headline Metrics ---
col1, col2, col3 = st.columns(3)
col1.metric("Final Test Accuracy", "93.01%", delta="1.24% vs Baseline")
col2.metric("Weighted F1-Score", "0.93", delta="0.01 vs Baseline")
col3.metric("Macro F1-Score", "0.91", delta="0.01 vs Baseline")

st.markdown("---")

# --- Display Confusion Matrix and Classification Report Side-by-Side ---
st.header("Detailed Test Set Analysis")
col_cm, col_report = st.columns(2)

with col_cm:
    st.subheader("Confusion Matrix")
    # === CHANGE: Call the new detailed plot function ===
    st.pyplot(plot_detailed_confusion_matrix())

with col_report:
    st.subheader("Classification Report")
    st.markdown("The report below shows the main classification metrics on a per-class basis for the test set.")
    
    report_data = {
        'Class': ['Fetal abdomen', 'Fetal brain', 'Fetal femur', 'Fetal thorax', 'Maternal cervix', 'Other'],
        'Precision': [0.90, 0.98, 0.85, 0.89, 1.00, 0.93],
        'Recall': [0.79, 0.97, 0.83, 0.89, 1.00, 0.95],
        'F1-Score': [0.84, 0.98, 0.84, 0.89, 1.00, 0.94],
        'Support': [106, 464, 156, 258, 244, 632]
    }
    df_report = pd.DataFrame(report_data)
    st.dataframe(df_report, use_container_width=True, hide_index=True)


st.info(
    "**How to Read the Confusion Matrix:**\n"
    "- Each row represents an **Actual** class, while each column represents a **Predicted** class.\n"
    "- The main diagonal shows the percentage of correct predictions for each class (Recall).\n"
    "- Off-diagonal cells show where the model made mistakes (e.g., the percentage of 'Actual Abdomen' images that were incorrectly predicted as 'Other')."
)