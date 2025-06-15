import streamlit as st
import pandas as pd
from utils.visualizations import plot_detailed_confusion_matrix, plot_class_distribution # Assuming you might want this too

st.set_page_config(page_title="Model Performance", layout="wide")

st.title("📊 Final Model Performance Analysis")
st.markdown("This page details the performance of the **best fine-tuned model**, including its performance on the validation set used during training and the final evaluation on the unseen test set.")

st.markdown("---")

# --- Display Headline Metrics ---
st.header("Key Performance Metrics")

# Create columns for Test and Validation metrics
col1, col2 = st.columns(2)

with col1:
    st.subheader("Test Set (Final Evaluation)")
    # Data from your final analysis document
    st.metric("Test Accuracy", "93.01%", delta="1.24% vs Baseline", help="Overall percentage of correct predictions on the test set.")
    st.metric("Weighted F1-Score", "0.93", delta="0.01 vs Baseline", help="The balanced score between precision and recall, weighted by class support.")

with col2:
    st.subheader("Validation Set (During Training)")
    # Data from your final analysis document
    st.metric("Best Validation Accuracy", "93.34%", help="The highest accuracy achieved on the validation set across all 20 training epochs.")
    st.metric("Lowest Validation Loss", "0.2290", help="The lowest loss value achieved on the validation set, indicating the point of best fit.")


st.markdown("---")

# --- Display Detailed Test Set Analysis ---
st.header("Detailed Test Set Analysis")
st.markdown("The following charts and tables show a detailed breakdown of the model's performance on the unseen test data.")

col_cm, col_report = st.columns([1.2, 1]) # Make the confusion matrix column slightly wider

with col_cm:
    st.subheader("Confusion Matrix")
    # This calls the function from your visualizations utility file
    st.pyplot(plot_detailed_confusion_matrix())

with col_report:
    st.subheader("Classification Report")
    
    # Data from your final analysis document and fine-tuning run
    report_data = {
        'Class': ['Fetal abdomen', 'Fetal brain', 'Fetal femur', 'Fetal thorax', 'Maternal cervix', 'Other'],
        'Precision': [0.90, 0.98, 0.85, 0.89, 1.00, 0.93],
        'Recall': [0.79, 0.97, 0.83, 0.89, 1.00, 0.95],
        'F1-Score': [0.84, 0.98, 0.84, 0.89, 1.00, 0.94]
    }
    df_report = pd.DataFrame(report_data)
    st.dataframe(df_report, use_container_width=True, hide_index=True)

st.info(
    "**How to Read These Metrics:**\n"
    "- The **Confusion Matrix** shows where the model made correct vs. incorrect predictions for each class.\n"
    "- The **Classification Report** provides the Precision, Recall, and F1-Score, offering a deep dive into the model's predictive quality for each specific fetal plane."
)