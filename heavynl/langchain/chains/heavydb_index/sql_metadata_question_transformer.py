from typing import Any, Optional

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.prompts.example_selector import MaxMarginalRelevanceExampleSelector
from langchain.vectorstores import Chroma
from langchain.base_language import BaseLanguageModel
from pydantic import BaseModel, Extra

from heavynl.config import config
from heavynl.langchain.llms import get_llm

rephrase_question_examples = [
    {
        "input": "What is the elevation of the highest airport in Europe?",
        "output": "Which table contains information on airports? What are the relevant columns for elevation and continent?",
    },
    {
        "input": "Which firewall rule has the highest number of blocked traffic events?",
        "output": "Which table contains information on firewall events? What are the relevant columns for rule number and action?",
    },
    {
        "input": "How many heliports are located in the United States?",
        "output": "Which table contains information on airports? What are the relevant columns for airport type and country?",
    },
    {
        "input": "What was the average Time To Live (TTL) value for traffic blocked by firewall rule number 5?",
        "output": "Which table contains information on firewall events? What are the relevant columns for TTL, rule number, and action?",
    },
    {
        "input": "What are the names of the bus stops located at the highest latitudes?",
        "output": "Which table contains information on bus stops? What are the relevant columns for stop name and latitude?",
    },
    {
        "input": "Which airport has the longest scheduled flights in Asia?",
        "output": "Which table contains information on airports? What are the relevant columns for airport name, scheduled flights, and continent?",
    },
    {
        "input": "What are the top 5 most common protocols used in incoming traffic according to firewall logs?",
        "output": "Which table contains information on firewall events? What are the relevant columns for protocol text and direction?",
    },
    {
        "input": "Which bus stop has the highest number of routes passing through it?",
        "output": "Which table contains information on bus stops? What are the relevant columns for stop name and number of routes?",
    },
    {
        "input": "What was the highest trading volume for a single minute for the stock symbol 'AAPL'?",
        "output": "Which table contains information on stock data? What are the relevant columns for stock symbol, trading volume, and minute timestamp?",
    },
    {
        "input": "What was the average closing price for the stock symbol 'GOOGL' in October 2019?",
        "output": "Which table contains information on stock data? What are the relevant columns for stock symbol, closing price, and timestamp?",
    },
    {
        "input": "How many companies in the S&P 500 belong to the Technology sector?",
        "output": "Which table contains information on S&P 500 companies? What are the relevant columns for sector and company name or symbol?",
    },
    {
        "input": "Which company in the S&P 500 had the highest average closing price in 2020?",
        "output": "Which table contains information on stock data and S&P 500 companies? What are the relevant columns for company name or symbol, closing price, and timestamp?",
    },
    {
        "input": "What was the total trading volume for the S&P 500 index on December 15, 2020?",
        "output": "Which table contains information on stock data? What are the relevant columns for trading volume and timestamp?",
    },
    {
        "input": "Which vessel had the highest speed recorded in the aishub_2021_06_16 table?",
        "output": "In the aishub_2021_06_16 table what are the relevant columns for vessel name or identifier and speed over ground?",
    },
    {
        "input": "What was the destination of the vessel with MMSI 123456789 on June 10, 2021?",
        "output": "Which table contains information on vessel movements? What are the relevant columns for vessel MMSI, destination, and timestamp?",
    },
    {
        "input": "How many vessels with a position accuracy of 1 were in the area defined by latitudes 40-45 and longitudes -10 to -5 on June 12, 2021?",
        "output": "Which table contains information on vessel movements? What are the relevant columns for latitude, longitude, position accuracy, and timestamp?",
    },
    {
        "input": "What is the average speed of cargo ships in the aishub_2021_06_16 table?",
        "output": "In the aishub_2021_06_16 table what are the relevant columns for vessel type and speed over ground?",
    },
    {
        "input": "Which ships in the world_ships table have a deadweight of more than 200,000 metric tonnes?",
        "output": "In the world_ships table what are the relevant columns for ship name or identifier and deadweight?",
    },
    {
        "input": "What is the average signal strength for LTE cell towers in the United States?",
        "output": "Which table contains information on cell towers in the United States? What are the relevant columns for radio access technology and average signal strength?",
    },
    {
        "input": "How many GSM cell towers are there in a specific area code?",
        "output": "Which table contains information on cell towers? What are the relevant columns for radio access technology and area code?",
    },
    {
        "input": "What is the distribution of cell towers across different network operators in the United States?",
        "output": "Which table contains information on cell towers in the United States? What are the relevant columns for network code and cell tower count?",
    },
    {
        "input": "Which cell towers have a range of over 1000 meters?",
        "output": "Which table contains information on cell towers? What are the relevant columns for cell tower identifier and range?",
    },
    {
        "input": "What is the total number of cell towers using UMTS technology in a specific country?",
        "output": "Which table contains information on cell towers across the globe? What are the relevant columns for radio access technology and mobile country code?",
    },
    {
        "input": "What are the most popular hashtags in the California region according to the ca_tweets table?",
        "output": "In the ca_tweets table, what is the relevant column for hashtags? How can we rank the popularity of hashtags?",
    },
    {
        "input": "What is the average number of followers for users who tweeted about the hashtag #Earthquake?",
        "output": "Which table contains information on tweets? What are the relevant columns for hashtags and number of followers?",
    },
    {
        "input": "How many tweets were in English and originated from iOS devices in the last month?",
        "output": "Which table contains information on tweets? What are the relevant columns for language, device, and timestamp?",
    },
    {
        "input": "What is the daily average of new COVID-19 cases in Los Angeles County during the last month using nyt_covid_counties_v2 table?",
        "output": "In the nyt_covid_counties_v2 table, what are the relevant columns for county, new cases, and reporting date?",
    },
    {
        "input": "Who are the top 3 actors in the movie 'The Matrix'?",
        "output": "Which table contains information on movie actors? What are the relevant columns for movie title, actor name, and appearance order?",
    },
    {
        "input": "How many female actors are in the movie 'Wonder Woman'?",
        "output": "Which table contains information on movie actors? What are the relevant columns for movie title, actor gender, and actor name?",
    },
    {
        "input": "Which movie has the largest cast?",
        "output": "Which table contains information on movie actors? What are the relevant columns for movie title and total number of actors?",
    },
    {
        "input": "List all the movies starring Scarlett Johansson.",
        "output": "Which table contains information on movie actors? What is the relevant column for actor name and movie title?",
    },
    {
        "input": "What is the gender distribution among actors in the movie 'Inception'?",
        "output": "Which table contains information on movie actors? What are the relevant columns for movie title, actor gender, and actor name?",
    },
]
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)

_example_selector: MaxMarginalRelevanceExampleSelector = None


def get_example_selector() -> MaxMarginalRelevanceExampleSelector:
    global _example_selector
    if _example_selector is None:
        print("Initting rephrase example selector. This will only happen once.")
        _example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
            rephrase_question_examples,
            HuggingFaceEmbeddings(model_name=config.huggingface_embed_model),
            Chroma,
        )
    return _example_selector


class SQLMetadataQuestionTransformerChain(Chain, BaseModel):
    """The HeavyDB Metadata Index performs best when the query input is about database tables and columns.
    This chain rephrases input to be more database-centric.

    Example:
        input: 'What is the average signal strength for LTE cell towers in the United States?'
        sql_metadata_question: 'Which table contains information on cell towers in the United States? What are the relevant columns for radio access technology and average signal strength?'
    """

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    llm: BaseLanguageModel = get_llm(["chain", "sql_metadata_question_transformer_chain"], temperature=0)
    """LLM wrapper to use."""
    input_key: str = "input"  #: :meta private:
    output_key: str = "sql_metadata_question"  #: :meta private:

    @property
    def _chain_type(self) -> str:
        return "sql_metadata_question_transformer"

    @property
    def input_keys(self) -> list[str]:
        """Return the singular input key.
        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> list[str]:
        """Return the output keys.
        :meta private:
        """
        return [self.output_key]

    def _call(self, inputs: dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None) -> dict[str, Any]:
        if run_manager:
            run_manager.on_text(
                f"Rephrasing input to be more database-centric: {inputs[self.input_key]}",
                color="blue",
                verbose=self.verbose,
            )
        rephrase_question_prompt = FewShotPromptTemplate(
            example_selector=get_example_selector(),
            example_prompt=example_prompt,
            prefix="Rephrase the provided input to ask an index that contains descriptions about the SQL tables and columns required to respond to the input",
            suffix="Input: {input}\nOutput:",
            input_variables=["input"],
        )
        llm_chain = LLMChain(
            llm=self.llm, prompt=rephrase_question_prompt, verbose=self.verbose, output_key=self.output_key
        )
        res = llm_chain(inputs[self.input_key])
        if run_manager:
            run_manager.on_text(f"Rephrased input: {res[self.output_key]}", color="blue", verbose=self.verbose)
        return {self.output_key: res[self.output_key].strip()}
