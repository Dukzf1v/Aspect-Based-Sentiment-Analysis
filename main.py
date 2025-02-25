import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_pipelines():
    token_classifier = pipeline(
        task="token-classification",
        model="abte-restaurants-distilbert-base-uncased/checkpoint-344",
        aggregation_strategy="simple"
    )
    classifier = pipeline(
        task="text-classification",
        model="absa-restaurants-albert-base-v2/checkpoint-630"
    )
    return token_classifier, classifier

token_classifier, classifier = load_pipelines()

st.title("🍽️ Aspect-Based Sentiment Analysis (ABSA)")

user_input = st.text_area("Enter sentence:")

if st.button("Analysis"):
    if user_input:
        results = token_classifier(user_input)

        aspects = [result['word'] for result in results if result['entity_group'] == 'Term']
        sentence_tags = " ".join(aspects)

        st.subheader("🔍 Term:")
        if aspects:
            st.write(", ".join(aspects))
        else:
            st.write("Not found")

        combined_input = f"{user_input} [SEP] {sentence_tags}"
        pred_label = classifier(combined_input)

        st.subheader("💡 General Sentiment:")
        label = pred_label[0]['label']
        score = pred_label[0]['score']
        st.write(f"**{label}** (Trust: {score:.2f})")
    else:
        st.warning("Please enter sentence")
