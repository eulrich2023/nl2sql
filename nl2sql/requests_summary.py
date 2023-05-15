"""Build a summary for a sequence of requests using OpenAI's GPT-3 API."""
from typing import Sequence, Type, TypeVar

from langchain import LLMChain, OpenAI, PromptTemplate

REQUST_SUMMARY_TEMPLATE = """
Given a sequence of requests, create a condensed request based on the last one.

Begin!

{requests_history}

Condensed Request:
"""

# Create a generic variable that can be 'SummaryBuilder', or any subclass.
T = TypeVar("T", bound="SummaryBuilder")


class SummaryBuilder:
    """Builds a summary for a sequence of requests using OpenAI's GPT-3 API."""

    @classmethod
    def build_summary(cls: Type[T], requests: Sequence[str]) -> str:
        """Build a summary for a sequence of requests using OpenAI's GPT-3 API.

        Args:
            requests: A sequence of requests.

        Returns:
            A summary of the last request.
        """
        llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")
        llm_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(REQUST_SUMMARY_TEMPLATE))
        requests_history = "\n".join(map(lambda req: f"Request: {req}", requests))
        return llm_chain.predict(requests_history=requests_history)
