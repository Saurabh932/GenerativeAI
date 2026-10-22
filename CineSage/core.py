from typing import List, Optional
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

from pydantic import BaseModel

load_dotenv()

class Movie(BaseModel):
    title: str
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: float
    summary: str

parser = PydanticOutputParser(pydantic_object=Movie)

llm = HuggingFacePipeline.from_model_id(
    model_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens":512,
        "do_sample":False,
        "repetition_penalty":1.03,
        "return_full_text":False,
        "clean_up_tokenization_spaces":False
    }
)
model = ChatHuggingFace(llm=llm)

prompt = ChatPromptTemplate.from_messages(
    [
        ('system',
            """
            Extract movie information from the paragraph
                {format_instructions}
            """),
        
        ('human',
            """
            {paragraph}
            """)
    ]
)

paragraph = input("Give your Paragraph: ")

final_prompt = prompt.invoke(
    {
        "paragraph": paragraph,
        "format_instructions": parser.get_format_instructions()
    }
)

response = model.invoke(final_prompt)
print(response.content)