from typing import Tuple

from pydantic import BaseModel
from dotenv import load_dotenv
from langchain import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI

from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from output_parsers import summary_parser, Summary

# Define your Summary model correctly
class Summary(BaseModel):
    """
    Represents a summary with a descriptive text and a list of factual statements.

    Attributes:
        summary (str): A concise description or overview.
        facts (list[str]): A collection of factual statements related to the summary.
    """
    summary: str
    facts: list[str]
    
def parse_summary_output(output: str) -> Summary:
    """
    Parses the output string to extract a summary and a list of facts.

    Args:
        output (str): The output string containing summary and facts.

    Returns:
        Summary: An instance of the Summary class with extracted summary and facts.
    """
    try:
        lines = output.strip().split('\n')
        summary_text = lines[0].split(":")[1].strip()
        facts = [line.split(":")[1].strip() for line in lines[1:] if line.startswith("Fact")]
        return Summary(summary=summary_text, facts=facts)
    except IndexError:
        print(f"Error parsing output: {output}")
        return Summary(summary="Could not parse summary", facts=[]) # Return a default object

def ice_break_with(name: str) -> Tuple[Summary, str]:
    linkedin_username = linkedin_lookup_agent(name=name)
    linkedin_data = scrape_linkedin_profile(
        linkedin_profile_url=linkedin_username, mock=True
    )
    """
    Generates an icebreaker summary and retrieves LinkedIn profile picture URL for a given name.

    Args:
        name (str): The full name of the person to look up.

    Returns:
        Tuple[Summary, str]: A tuple containing the summary and the profile picture URL.
    """

    summary_template = """
    given the information about a person from linkedin {information},
    I want you to create:
    Summary: <short summary here>
    Fact 1: <first interesting fact>
    Fact 2: <second interesting fact>
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-4") # Use a full model for best results

    chain = LLMChain(llm=llm, prompt=summary_prompt_template) # Use LLMChain

    llm_output = chain.run(information=linkedin_data)
    res = parse_summary_output(llm_output)

    return res, linkedin_data.get("profile_pic_url")


if __name__ == "__main__":
    load_dotenv()

    print("Ice Breaker Enter")
    ice_break_with(name="Harrison Chase")
