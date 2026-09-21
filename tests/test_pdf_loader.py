from ingestion.pdf_loader import PDFLoader


loader = PDFLoader()

docs = loader.load(
    "data/document.pdf"
)


for doc in docs: python --version python --version
    print(doc.content[:200])
    print("----------------")