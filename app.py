from pathlib import Path
import streamlit as st
import joblib

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "sentiment_pipeline.joblib"

st.set_page_config(page_title="AI Sentiment Analysis", page_icon="💬", layout="centered")
st.title("💬 AI-Powered Sentiment Analysis")
st.write("Enter a customer review and predict whether its sentiment is Positive, Negative, or Neutral.")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

model = load_model()
if model is None:
    st.warning("The trained model was not found. First run `python train_model.py` from the project folder.")
    st.stop()

review = st.text_area(
    "Customer review",
    placeholder="Example: The product arrived quickly and works perfectly.",
    height=140
)

if st.button("Analyze sentiment", type="primary", use_container_width=True):
    if not review.strip():
        st.error("Please enter a review first.")
    else:
        prediction = model.predict([review.strip()])[0]
        st.subheader("Predicted sentiment")
        if prediction.lower() == "positive":
            st.success(f"😊 {prediction}")
        elif prediction.lower() == "negative":
            st.error(f"☹️ {prediction}")
        else:
            st.info(f"😐 {prediction}")

        # LogisticRegression exposes predict_proba; other models may not.
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([review.strip()])[0]
            classes = model.classes_
            st.subheader("Model probability scores")
            for label, probability in sorted(
                zip(classes, probabilities), key=lambda item: item[1], reverse=True
            ):
                st.write(f"**{label}:** {probability:.1%}")
                st.progress(float(probability))
        else:
            st.caption("Probability scores are unavailable for this model.")
st.divider()
st.caption("Educational project. Predictions can be wrong; review the model evaluation before relying on results.")
