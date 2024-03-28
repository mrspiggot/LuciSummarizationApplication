import os
import tempfile
from langchain.document_loaders import PyPDFLoader



class LuciDocumentProcessor:
    """
    A class to process document files, specifically for extracting text from PDF files.
    """

    def __init__(self, file_data):
        """
        Initializes the LuciDocumentProcessor with the file data.

        :param file_data: The binary data of the file to be processed.
        """
        self.file_data = file_data

    def get_text(self):
        """
        Extracts and returns the text from the loaded PDF file.

        :return: A string containing all the text extracted from the PDF file.
        """
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(self.file_data.getvalue())
            loader = PyPDFLoader(os.path.abspath(temp_file.name))
            pages = loader.load_and_split()
            return "".join(t.page_content for t in pages)

    def get_text2(self):
        """
        Extracts and returns the text from the loaded PDF file.

        :return: A string containing all the text extracted from the PDF file.
        """
        text = ""
        temp_file_path = None  # Initialize the variable outside of the try block

        try:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(self.file_data.getvalue())
                temp_file_path = temp_file.name  # Keep the file path to delete later

                # Load the PDF and extract text
                loader = PyPDFLoader(temp_file_path)
                documents = loader.load_and_split()
                text = "".join(doc.page_content for doc in documents)

        except Exception as e:
            # Implement logging or handle specific exceptions as needed
            print(f"An error occurred: {e}")

        finally:
            # Clean up the temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        print(f"Text = {text}")
        return text
