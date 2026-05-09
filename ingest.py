import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

print("🚀 Starting 2-Million Token Extraction...")

# 1. Load the Massive File we just inflated
file_path = "final_enron.txt"
if not os.path.exists(file_path):
    print(f"❌ Error: Could not find {file_path}. Make sure it is in this folder!")
    exit()

print("📂 Loading massive document...")
loader = TextLoader(file_path, encoding="utf-8")
docs = loader.load()

# 2. Chunk the Data
print("✂️ Chunking data into readable pieces...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)
print(f"✅ Created {len(chunks)} chunks of data.")

# 3. Create Embeddings & Build Vector Database
print("🧠 Extracting to Vector Database (This may take 5-15 minutes on your laptop...)")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

# 4. Save to Disk
print("💾 Saving database to disk...")
vectorstore.save_local("faiss_index")

print("🎉 EXTRACTION COMPLETE! You can now run your Streamlit app.")