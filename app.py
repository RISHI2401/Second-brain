import streamlit as st
from PIL import Image
import os
import uuid
import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import chromadb


load_dotenv()
genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))


def generate_summary(text):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(f"Summarize this memory: {text}")
        return response.text.strip()
    except Exception as e:
        return f"(⚠️Gemini Error: {e})"


# --- Setup Directories ---
os.makedirs("data/images", exist_ok=True)

# --- Chroma Setup ---
chroma_client = chromadb.PersistentClient(path="./chroma_store")
collection = chroma_client.get_or_create_collection("second_brain")

# --- Embedding Model ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- Streamlit UI ---
st.set_page_config(page_title="Second Brain", layout="centered")
st.title("🧠 Second Brain - Memory & Search")

tab1, tab2, tab3 = st.tabs(["📥 Remember", "🔍 Search", "All Logs"])

# ========== 📥 Remember Tab ==========
with tab1:
    st.subheader("Add a Memory")

    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    user_text = st.text_area("Describe this memory or just enter a note (required)")
    

    # Asking user for ai summary
    use_ai_summary = st.checkbox("Use AI to summarize this memory", value=True)
    if use_ai_summary:
        summary = generate_summary(user_text)
    else:
        summary = None

    # Reminder Date Input
    reminder_date = st.date_input(
        "Set a reminder date (optional)",
        value=datetime.date.today() + datetime.timedelta(days=7)
    )

    if st.button("Remember"):
        if not user_text:
            st.warning("You must enter a description or reminder text.")
        else:
            image_id = str(uuid.uuid4())

            # Save image if provided
            image_path = None
            if uploaded_image:
                image_path = f"data/images/{image_id}.jpg"
                img = Image.open(uploaded_image)
                img.save(image_path)

            # Embed memory text 
            memory_text = user_text


            embedding = embedder.encode([memory_text])[0]

            # Build metadata dict safely
            metadata = {
                "reminder_date": str(reminder_date),
                "summary" : summary
            }
            if image_path:
                metadata["image_path"] = image_path

            # Save to Chroma
            collection.add(
                documents=[memory_text],
                embeddings=[embedding.tolist()],
                ids=[image_id],
                metadatas=[metadata]
            )

            msg = f"✅ Memory saved!" + (f" Reminder set for {reminder_date}" if reminder_date else "")
            st.success(msg)


# ========== 🔍 Search Tab ==========
with tab2:
    st.subheader("Search Your Memories")

    query = st.text_input("Ask me anything...")

    st.markdown("**Optional:** Filter by reminder date")
    use_date_filter = st.checkbox("Only show reminders due by a specific date")

    if use_date_filter:
        due_by = st.date_input("Show reminders due on or before:", datetime.date.today())


    if st.button("Search"):
        if not query and not use_date_filter:
            st.warning("Please enter a query or enable date filter.")
        else:
            results = {"documents": [[]], "metadatas": [[]]}

            if query:
                query_embedding = embedder.encode([query])[0]
                results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=10)
            else:
                # No query → fetch all documents
                fetched = collection.get()
                results["documents"] = [fetched["documents"]]
                results["metadatas"] = [fetched["metadatas"]]

            if results["documents"]:
                shown = False
                for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                    # Safe parse of reminder date
                    reminder_str = meta.get("reminder_date")
                    reminder_date = None
                    if reminder_str:
                        try:
                            reminder_date = datetime.datetime.strptime(reminder_str, "%Y-%m-%d").date()
                        except ValueError:
                            pass

                    # Date filter
                    if use_date_filter:
                        if reminder_date is None or reminder_date > due_by:
                            continue

                    # Show memory
                    st.markdown(f"📘 **Memory:** {doc}")
                    if reminder_date:
                        st.markdown(f"📅 **Reminder Date:** `{reminder_date}`")

                    # Show image
                    image_path = meta.get("image_path")
                    if image_path and os.path.exists(image_path):
                        st.image(image_path, width=150)
                    else:
                        st.info("No image available for this memory.")

                    # Show Gemini summary
                    summary = meta.get("summary")
                    if summary:
                        st.markdown(f"📝 **Gemini Summary:** _{summary}_")

                    st.markdown("---")
                    shown = True

                if not shown:
                    st.info("No memories match the query and filter.")
            else:
                st.info("No relevant memories found.")


with tab3:
    if st.checkbox("Show all stored memories (debug)"):
        all_results = collection.get()
        for doc, meta in zip(all_results["documents"], all_results["metadatas"]):
            st.write(f"📝 {doc}")
            st.write(f"📅 {meta.get('reminder_date')}")
            
            image_path = meta.get("image_path")
            if image_path and os.path.exists(image_path):
                st.image(image_path, width=150)
            else:
                st.info("No image available for this memory.")
            
            st.markdown("---")

