import streamlit as st
import pandas as pd
from utils.visualizations import plot_confusion_matrix

st.set_page_config(page_title="Model Performance", layout="wide")

st.title("📊 Final Model Performance")
st.markdown("This page details the performance of the **best fine-tuned model** on both the unseen **test set** and the **validation set** used during training.")

st.markdown("---")

# --- Metrics for the Test Set ---
st.subheader("Test Set Performance (Final Evaluation)")
st.markdown("These metrics represent the model's performance on completely unseen data.")

col1, col2, col3 = st.columns(3)
# Data from your final analysis document
col1.metric("Final Test Accuracy", "93.01%", delta="1.24% vs Baseline", help="Overall percentage of correct predictions on the test set.")
col2.metric("Weighted F1-Score", "0.93", delta="0.01 vs Baseline", help="The balanced score between precision and recall, weighted by the number of samples in each class.")
col3.metric("Macro F1-Score", "0.91", delta="0.01 vs Baseline", help="The balanced score between precision and recall, where each class gets equal weight.")


# === NEW SECTION START ===
st.markdown("---")
st.subheader("Validation Set Performance (During Training)")
st.markdown("These metrics show the model's peak performance on the validation set, which was used to select the best model during the training process.")

col_val1, col_val2 = st.columns(2)
# Data from your final analysis document
col_val1.metric("Best Validation Accuracy", "93.34%", help="The highest accuracy achieved on the validation set across all training epochs.")
col_val2.metric("Lowest Validation Loss", "0.2290", help="The lowest loss value achieved on the validation set, indicating the point of best fit before potential overfitting.")
# === NEW SECTION END ===


st.markdown("---")

# --- Display Confusion Matrix and Classification Report Side-by-Side ---
st.header("Detailed Test Set Analysis")
col_cm, col_report = st.columns(2)

with col_cm:
    st.subheader("Confusion Matrix")
    st.pyplot(plot_confusion_matrix())

with col_report:
    st.subheader("Classification Report")
    st.markdown("The report below shows the main classification metrics on a per-class basis for the test set.")
    
    # Data from your final analysis document
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
    "**How to Read These Metrics:**\n"
    "- **Precision:** Of all the times the model predicted a class, how often was it correct?\n"
    "- **Recall:** Of all the actual instances of a class, how many did the model correctly identify?\n"
    "- **F1-Score:** The balanced average of Precision and Recall."
)