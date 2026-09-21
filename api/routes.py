from fastapi import APIRouter, HTTPException

from api.schemas import QuestionRequest, AnswerResponse

from app import create_rag


router = APIRouter()

rag = create_rag()


@router.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):

    try:

        result = rag.ask(
            request.question,
            k=request.k
        )

        sources = []

        for document in result["documents"]:

            sources.append(
                {
                    "content": document.content,
                    "metadata": document.metadata
                }
            )

        return {
            "question": request.question,
            "answer": result["answer"],
            "sources": sources
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Error while generating answer: {str(e)}"
        )