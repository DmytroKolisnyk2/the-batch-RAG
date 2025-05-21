import streamlit as st

# Streamlit page configuration
st.set_page_config(page_title="Batch Newsletter RAG Demo", layout="centered")
st.title("Batch Newsletter RAG Demo")

# User input
query = st.text_input("Enter your question about 'The Batch' newsletter:")

# Retrieval trigger
if st.button("Retrieve"):
    if not query.strip():
        st.warning("Please enter a question to retrieve results.")
    else:
        with st.spinner("Retrieving results..."):
            # Import and call your RAG function
            from rag_module import answer_with_rag

            answer, passages, images = answer_with_rag(query)

        # Display answer
        st.subheader("Answer")
        st.write(answer)

        # Display relevant text passages
        if passages:
            st.subheader("Text Passages")
            for p in passages:
                title = p.get("title", "Untitled")
                chunk = p.get("chunk", "")
                slug = p.get("slug")
                url = f"https://www.deeplearning.ai/the-batch/{slug}/" if slug else None

                if url:
                    st.markdown(f"[**{title}**]({url})")
                else:
                    st.markdown(f"**{title}**")

                st.write(chunk)

        # Display images and captions
        if images:
            st.subheader("Images")
            cols = st.columns(2)
            for idx, img in enumerate(images):
                col = cols[idx % 2]
                col.image(img["url"], caption=img.get("caption", ""))

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit & Qdrant-powered RAG*")
