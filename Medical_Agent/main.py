import streamlit as st
from dotenv import load_dotenv
import os
from google import generativeai as genai
from google.api_core import retry
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import DuckDuckGoSearchException
import time

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Retry logic for Gemini
is_retriable = lambda e: isinstance(e, genai.types.generation_types.APIError) and e.code in {429, 503}
genai.GenerativeModel.generate_content = retry.Retry(predicate=is_retriable)(genai.GenerativeModel.generate_content)

generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
}

# Prompts
doc_prompt = """
You are a highly experienced and specialized medical doctor with expertise in diagnosing diseases and prescribing appropriate treatments or medications. Please analyze the symptoms or condition provided and offer a professional medical opinion, including possible diagnoses and suggested treatments or medications. Ensure your advice is detailed, medically accurate, and based on current medical guidelines.
"""

structure_prompt = """
You are an expert medical doctor with years of experience in diagnosing and treating a wide range of diseases. When given any medical condition, your task is to:

1. Act as a real doctor and suggest **prescription medications** (including dosage, frequency, and duration) for the condition in a **well-structured table**.
2. Provide a separate table for **over-the-counter (OTC) medicines**, if applicable.
3. Provide a separate table for **effective home remedies**, including ingredients and instructions.
4. List **precautions and lifestyle advice** in bullet points to help manage or prevent the condition.

---

### 🩺 Prescription Medications

| Medicine Name | Dosage | Frequency | Duration | Notes |
|---------------|--------|-----------|----------|-------|
|               |        |           |          |       |

---

### 💊 OTC Medications (if applicable)

| Medicine Name | Dosage | Frequency | Duration | Notes |
|---------------|--------|-----------|----------|-------|
|               |        |           |          |       |

---

### 🏡 Home Remedies

| Remedy Name | Ingredients | Preparation Method | How to Use |
|-------------|-------------|--------------------|------------|
|             |             |                    |            |

---

### ⚠️ Precautions and Lifestyle Advice

- Bullet point 1  
- Bullet point 2  
- Bullet point 3  
*(Add as many relevant points as needed)*

---

Start with a brief explanation of the disease in simple terms before diving into the tables. Be clear, professional, and helpful. Always include a safety reminder to consult a healthcare provider before starting any treatment.
"""

combined_prompt = doc_prompt.strip() + "\n\n" + structure_prompt.strip()

# RAG: Fetch medical content
def retrieve_medical_data(query):
    with DDGS() as ddgs:
        results = ddgs.text(query + " site:mayoclinic.org OR site:medlineplus.gov OR site:who.int", max_results=3)
        docs, sources = [], []
        for res in results:
            try:
                url = res['href']
                page = requests.get(url, timeout=5)
                soup = BeautifulSoup(page.text, "html.parser")
                text = " ".join(p.get_text() for p in soup.find_all("p"))
                if text:
                    docs.append(text[:2000])
                    sources.append(url)
            except Exception:
                continue
        return "\n\n".join(docs), sources

# RAG: Get nearby pharmacies with retry & rate-limit handling
def get_nearby_pharmacies(location, retries=3, delay=2):
    query = f"pharmacies near {location}"
    for attempt in range(retries):
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=5)
                return [(r["title"], r["href"]) for r in results if "href" in r]
        except DuckDuckGoSearchException as e:
            if "202 Ratelimit" in str(e):
                time.sleep(delay * (attempt + 1))  # Exponential backoff
            else:
                break
        except Exception:
            break
    return []

# Streamlit UI
st.set_page_config(page_title="AI Medical Assistant", layout="centered")
st.title("🧠 AI Medical Assistant")
st.markdown("Enter your medical condition and location to receive expert-style advice and find local pharmacies.")

# Initialize model
model = genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=generation_config)
if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])

# Inputs
col1, col2 = st.columns(2)
with col1:
    condition = st.text_input("Enter medical condition (e.g., 'flu'):")
with col2:
    location = st.text_input("Enter your location (e.g., 'Delhi'):")

# Button Action
if st.button("Get Advice & Pharmacies") and condition:
    # Medical RAG
    with st.spinner("Retrieving medical info..."):
        rag_text, rag_sources = retrieve_medical_data(condition)

    if rag_text:
        source_list = "\n".join(f"- [{url}]({url})" for url in rag_sources)
        full_prompt = (
            f"The following medical content was retrieved from trusted sources to enhance accuracy:\n\n"
            f"{rag_text}\n\n"
            f"---\n\n"
            f"{combined_prompt}\n\n"
            f"Condition: {condition}"
        )

        try:
            response = st.session_state.chat.send_message(full_prompt)

            # 🧠 1. Medical response
            st.markdown("### 🧠 Medical Advice")
            st.markdown(response.text)

            # 🌐 2. Sources used
            if rag_sources:
                st.markdown("### 🌐 Sources Used for Medical Accuracy")
                st.markdown(source_list)

        except Exception as e:
            st.error(f"Error generating medical advice: {e}")
    else:
        st.warning("No medical data found for that condition.")

    # 🏥 3. Pharmacies nearby
    if location:
        with st.spinner("Finding nearby pharmacies..."):
            pharmacies = get_nearby_pharmacies(location)
        if pharmacies:
            st.markdown("### 🏥 Nearby Pharmacies")
            for name, link in pharmacies:
                st.markdown(f"- [{name}]({link})")
        else:
            st.warning("No pharmacies found for that location or rate limit reached. Try again later.")



# streamlit run main.py    -> To run the app, use the command above in your terminal.