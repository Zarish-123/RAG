class Document:

    def __init__(
            self,
            content,
            metadata=None
    ):

        self.content = content
        self.metadata = metadata or {}


    def __repr__(self):

        return f"Document({self.metadata})"