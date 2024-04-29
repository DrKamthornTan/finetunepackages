import argparse
from dataclasses import dataclass
import re
from translate import Translator
import streamlit as st
import csv

from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

st.set_page_config(page_title=None, page_icon=None, layout="wide")

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    st.title("DHV AI Startup Packages Query Demo")
    query_text = st.text_input("กรุณาลงข้อมูล")

    if query_text:
        # Google Translate
        try:
            translator = Translator(from_lang='th', to_lang='en')
            translated_text = translator.translate(query_text)
        except Exception as e:
            st.write(f"Error translating: {str(e)}")
            return

        # Rest of the code...
        st.write(f"Translated Text: {translated_text}")

        # Prepare the DB.
        openai_api_key = ""  # Replace with your actual OpenAI API key
        if not openai_api_key:
            st.write("OpenAI API key is not provided.")
            return

        embedding_function = OpenAIEmbeddings(openai_api_key=openai_api_key)
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        # Search the DB.
        results = db.similarity_search_with_relevance_scores(translated_text, k=3)
        if not results or (results and results[0][1] < 0.7):
            st.write("ไม่สามารถค้นหาคำตอบขณะนี้ได้ โปรดติดต่อแพทย์ของท่าน")
            return

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=translated_text)
        st.write(prompt)

        model = ChatOpenAI()
        response_text = model.predict(prompt)

        sources = [doc.metadata.get("source", None) for doc, _score in results]
        formatted_response = f"<span style='color:red'>{response_text}</span>\nSources: {sources}"
        st.write(formatted_response, unsafe_allow_html=True)

        
        import difflib

        # Assuming you have a "pack.csv" file containing the data
        pack_file_path = "C:\\Users\\kamth\\Hpackage\\Hpackage\\pack.csv"

        # Read the pack.csv file and extract the name and urls fields
        sources_dict = {}
        with open(pack_file_path, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                name = row["name"]
                urls = row["urls"]
                sources_dict[name] = urls

        # Assuming you want to search for similar names in the pack.csv file
        source = [doc.metadata.get("source", None) for doc, _score in results]
        matching_sources = []
        for s in source:
            matching_names = difflib.get_close_matches(s, sources_dict.keys(), n=1, cutoff=0.8)
            if matching_names:
                matching_sources.append(matching_names[0])

        formatted_response2 = f"<span style='color:red'></span>\n"

        if matching_sources:
            formatted_response2 += "Matching packages links:\n"
            for matching_source in matching_sources:
                urls = sources_dict[matching_source]
                formatted_response2 += f"<a href='{urls}'></a>: {urls}\n"
        else:
            formatted_response2 += "No matching sources found."

        st.write(formatted_response2, unsafe_allow_html=True)

if __name__ == "__main__":
    main()