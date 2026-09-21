from pypdf import PdfReader

from core.interfaces.document_loader import DocumentLoader
from core.interfaces.models.document import Document



class PDFLoader(DocumentLoader):


    def load(self,file_path):

        reader = PdfReader(file_path)

        documents=[]


        for page_number,page in enumerate(reader.pages):

            text = page.extract_text()


            if text:

                documents.append(
                    Document(
                        content=text,
                        metadata={
                            "page":page_number+1,
                            "source":file_path
                        }
                    )
                )


        return documents 