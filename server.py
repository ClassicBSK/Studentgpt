from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn 
from summarize import get_final_answers
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

@app.get("/ds/{query}")
async def get_response(query:str):

    response_text = get_final_answers(query)
    code_response = f"Code-related output for: {query}"
    # print('-----------')
    # print(response_text)
    # print('-----------')
    ans={"response": response_text, "codeResponse": code_response}
    # print(ans)

    return ans 
    # return response_text


if __name__ == "__main__":
    uvicorn.run(app)