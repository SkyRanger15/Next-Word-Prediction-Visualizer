import streamlit as st
import time
from backend import load_model_and_assets, predict_next_word, list_available_models, plot_training_history, get_model_summary,plot_confusion_matrix, evaluate_model

# Title
st.title("Next Word Prediction using LSTM Viusaliser")

# Dynamically list available models
model_choices = list_available_models()
selected_model = st.selectbox("Select Model", model_choices)

st.markdown(
    """
    <style>
    .model-details-header {
        font-size: 38px;
        font-weight: bold;
        color: #333;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if selected_model:
    # Dynamic model path
    model_path = f"./{selected_model}"
    tokenizer, model, history, X_test, y_test = load_model_and_assets(model_path)

    # Input field
    user_input = st.text_input("Enter text:")

    if user_input:
        if st.button("Predict"):
            eg = user_input
            placeholder = st.empty()

            for _ in range(10):  # Generate 10 words
                predicted_word, confidence = predict_next_word(eg, model, tokenizer)

                if predicted_word == "No prediction":
                    break
                
                eg = eg + " " + predicted_word
                placeholder.write(f"**{eg}**")
                st.write(f"Confidence: `{confidence * 100:.2f}%`")
                time.sleep(1)  # Add delay for typing effect


    with st.container(border=True):
        st.markdown('<div class="model-details-header">Model Details</div>', unsafe_allow_html=True)

        # Display model summary
        st.markdown("### Model Summary")
        summary_text = get_model_summary(model)
        st.code(summary_text, language="plaintext")

        # Display Training History
        st.markdown("### Training History")
        fig = plot_training_history(history)
        st.pyplot(fig)

        # Evaluate the model
        st.markdown("### Model Evaluation Metrics")
        metrics = evaluate_model(model, tokenizer, X_test, y_test)

        # Display accuracy
        st.write(f"**Test Accuracy:** `{metrics['accuracy'] * 100:.2f}%`")

        # Display perplexity
        st.write(f"**Perplexity:** `{metrics['perplexity']:.2f}`")

        # Display classification report
        st.markdown("### Classification Report")
        st.dataframe(metrics["classification_report_df"])

        # Display Confusion Matrix
        st.markdown("### Confusion Matrix")
        if "X_test" in locals() and "y_test" in locals():  # Ensure test data exists
            fig_cm = plot_confusion_matrix(model, tokenizer, X_test, y_test)
            st.pyplot(fig_cm)
        else:
            st.warning("Confusion matrix cannot be displayed (test data missing).")
