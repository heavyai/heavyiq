import re
from typing import Any, Optional

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.schema import BaseRetriever
from langchain.callbacks.manager import CallbackManagerForChainRun, AsyncCallbackManagerForChainRun
from pydantic import Field

from heavyiq.langchain.llms import get_llm
from heavyiq.langchain.chains import FileCallbackHandlerForChainMixin

question_prompt_template = """Use the following description of a SQL table.
If relevant to the question, return the text verbatim.

"{context}"

Question: {question}
Relevant text, if any:"""
QUESTION_PROMPT = PromptTemplate(
    template=question_prompt_template,
    input_variables=["context", "question"],
)

EXAMPLE_PROMPT = PromptTemplate(
    template="Content: {page_content}\nTable: {source}",
    input_variables=["page_content", "source"],
)

SQLSchemaQuestionChainNoResultsAnswer = "None relevant."
combine_prompt_template = """Given the following extracted parts of a long document describing a SQL database and a question that can be solved using data in the database, create a final answer with table names ("SOURCES").
If you don't know the answer, just say that you don't know. Don't try to make up an answer. If none of the tables are relevant, just say "{no_results_answer}".
ALWAYS return a "SOURCES" part in your answer.

QUESTION: What are the categories of the point of interest?
=========
Content: The movie_actors table primarily serves to store and organize data related to actors who have participated in various movies. It contains critical information such as movie_id (referring to the specific movie), actor_name, actor_gender, actor_order_num (the ordinal ranking of the actor's prominence in the film), num_total_actors (total number of actors involved in the movie), and movie_title. The table uses TEXT ENCODING DICT(32) for textual columns and DAYS(32) encoding for date columns to optimize data storage. The schema of the movie_actors table is:
Table: movie_actors
Content: - us_pois_safegraph.brands: The associated brand name(s).\n- us_pois_safegraph.top_category: A high-level category describing the industry of the point of interest.\n- us_pois_safegraph.sub_category: A more specific category describing the industry of the point of interest.\n- us_pois_safegraph.naics_code: A numerical code assigned to the point of interest based on the North American Industry Classification System (NAICS).\n- us_pois_safegraph.latitude: The latitude coordinate of the point of interest.\n- us_pois_safegraph.longitude: The longitude coordinate of the point of interest.\n- us_pois_safegraph.street_address: The street address of the point of interest.\n- us_pois_safegraph.primary_number: The primary street number associated with the point of interest.\n- us_pois_safegraph.street_predirection: The directional indicator that precedes a street name.\n- us_pois_safegraph.street_name: The name of the street on which the point of interest is located.\n- us_pois_safegraph.street_postdirection: The directional indicator that follows a street name.
Table: us_pois_safegraph
Content: Description: The us_pois_safegraph table contains information about Points of Interest (POIs) in the United States. Each record represents a POI with specific attributes including location_name, street_address, city, region, postal_code, geographic coordinates (latitude and longitude), categories (top_category, sub_category, and category_tags), business contact information (phone_number), and additional information such as operating hours, polygon information, and whether the POI includes a parking lot. The primary functionality of this table is to provide a comprehensive dataset of POIs across the United States for various applications.\nApplications: Potential use cases and applications for the us_pois_safegraph table include geographic data analysis and visualization, business directory services, mapping applications and navigation systems, location-based recommendations, market research, infrastructure planning, and emergency response planning.
Table: us_pois_safegraph
=========
FINAL ANSWER: The us_pois_safegraph table contains two category columns: top_category and sub_category.
SOURCES: us_pois_safegraph

QUESTION: Which country has cell towers with the highest signal strength?
=========
Content: Description: The cell_towers_us table is designed to store information about various cell towers in the United States. The primary functionality of this table is to manage and retrieve spatial details and operational characteristics for each tower, including radio technology, network operator, location, coverage range, signal strength, and timestamps for creation and modification. The secondary functionality is to enable analysis and visualization of different telecom networks' performances and properties.\nApplications: Potential use cases of this table might include querying cell towers' locations and capabilities for specific telecom network operators. Network engineers could perform coverage analysis and optimization, while researchers might analyze signal strengths, coverage area, and potential network bottlenecks to propose possible improvements in the telecom infrastructure. This table may also act as a vital resource for tools that need location-based information about cell towers.\nKeywords: cell towers, network operators, spatial details, coverage analysis, signal strength, location, radio technology, telecommunications, infrastructure, United States.
Table: cell_towers_us
Content:  - us_upstream_production.api_no_10: Unique well identifier provided by the American Petroleum Institute\n- us_upstream_production.current_operator_name_full: Full name of the current operator of the well\n- us_upstream_production.well_no: Number assigned to the well within a specific field\n- us_upstream_production.well_name: Name assigned to the well\n- us_upstream_production.state_name: State where the well is located\n- us_upstream_production.county_name: County where the well is located\n- us_upstream_production.basin_play_name: Name of the basin or play where the well is located\n- us_upstream_production.field_name: Name of the oil or gas field where the well is located\n- us_upstream_production.well_type: Classification of the well based on whether it produces oil, gas, or both\n- us_upstream_production.well_status: Classification of the well based on its current operational status\n- us_upstream_production.surface_lat: Latitude coordinate of the wellhead\n- us_upstream_production.surface_lon: Longitude coordinate of the wellhead\n- us_upstream_production.depth_tvd: Total vertical depth of the well in feet
Table: us_upstream_production
Content: Description: The table "firewall_in" is designed to store information about network traffic being filtered by a firewall. It contains details regarding connection attempts and traffic, along with geographical information and IP addresses. The primary function of this table is to record and monitor incoming or denied traffic through the firewall. The secondary function is to provide the necessary information for analysis and troubleshooting purposes.\nApplications: The data in this table can be used for a variety of purposes. Some potential use cases may include network security and monitoring, intrusion detection systems, understanding the nature of incoming traffic to a network, logging unapproved connections or threats, and providing valuable data for improving firewall rules and configurations. Additionally, the geographic data can be particularly useful in identifying potential threats from specific regions or countries.\nKeywords: Firewall, Network security, Traffic filtering, Geolocation, Intrusion detection, IP address, Data analysis, Connection monitoring, Protocol, Timestamp.
Table: firewall_in
Content: - cell_towers_world.radio: The type of radio technology used by the cell tower (UMTS, LTE, GSM, CDMA)\n- cell_towers_world.mcc: The Mobile Country Code identifying the country where the cell tower is located (310, 262, 250, 724, 208)\n- cell_towers_world.net: The Numeric Identifier for a network within a country (1, 2, 3, 10, 410)\n- cell_towers_world.area: The area code for the cell tower (1, 65534, 2, 5, 10)\n- cell_towers_world.cell: The cell identifier or cell ID of the cell tower (0, 4112, 65535, 50594049, 62)\n- cell_towers_world.unit: The signal measurement unit (0, -1, 1, 9, 8)\n- cell_towers_world.lon: The longitude of the location of the cell tower (in decimal degrees)\n- cell_towers_world.lat: The latitude of the location of the cell tower (in decimal degrees)\n- cell_towers_world.changeable: A flag specifying if the cell tower details can be changed (1 for yes, 0 for no)\n- cell_towers_world.created: The timestamp of when the cell tower was added to the database (in UTC)\n- cell_towers_world.updated: The timestamp of the last update of the cell tower details (in UTC)\n- cell_towers_world.average_signal: The average signal strength of the cell tower (in dBm)
Table: cell_towers_world
=========
FINAL ANSWER: Both the cell_towers_us and cell_towers_world tables contain information about signal strength.
SOURCES: cell_towers_us, cell_towers_world

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""
COMBINE_PROMPT = PromptTemplate(
    template=combine_prompt_template,
    input_variables=["summaries", "question"],
    partial_variables={"no_results_answer": SQLSchemaQuestionChainNoResultsAnswer},
)


class AskHeavyDBMetadataIndexChain(FileCallbackHandlerForChainMixin, RetrievalQAWithSourcesChain):
    """
    Takes a question and returns information about the tables that contain the data.

    If the question is about the table schema/columns, the response will be the answer to that question.
    If the question is about data in the database, the response will be what tables/data sources contain that data.

    Performs best when the question is about the schema of the tables, or other metadata contained in the metadata index.
    If the result at `answer_key` is the same as `no_results_answer`, then the question is unanswered.
    If the question that failed to get an answer is about data in the database (and not the structure of the database),
        then the question can be rephrased to try again (See: `SQLMetadataQuestionTransformerChain`),
        or the index can be queried directly. (See: `HeavyDBMetadataIndex`)

    If you need to restrict the tables that can be searched,
    """

    no_results_answer = SQLSchemaQuestionChainNoResultsAnswer

    retriever: BaseRetriever = Field(exclude=True)
    """Index to connect to. Ensure the retriever provided is filtered to the tables you want to search.
    Retrieving documents from a table the session does not have access to will result in an error."""

    @property
    def tables_answer_key(self) -> str:
        return self.sources_answer_key

    @classmethod
    def create(
        cls: type["AskHeavyDBMetadataIndexChain"], retriever: BaseRetriever, **kwargs
    ) -> "AskHeavyDBMetadataIndexChain":
        # ChatOpenAI has a hard time formatting proper response
        # Unfortunate because this is more expensive ($0.08 vs $0.008)
        llm = get_llm(temperature=0)
        llm_question_chain = LLMChain(llm=llm, prompt=QUESTION_PROMPT)
        llm_combine_chain = LLMChain(llm=llm, prompt=COMBINE_PROMPT)
        combine_results_chain = StuffDocumentsChain(
            llm_chain=llm_combine_chain,
            document_prompt=EXAMPLE_PROMPT,
            document_variable_name="summaries",
        )
        combine_document_chain = MapReduceDocumentsChain(
            llm_chain=llm_question_chain,
            combine_document_chain=combine_results_chain,
            document_variable_name="context",
        )
        return cls(
            combine_documents_chain=combine_document_chain,
            sources_answer_key="tables",
            retriever=retriever,
            **kwargs,
        )

    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        docs = self._get_docs(inputs, run_manager=_run_manager)
        answer = self.combine_documents_chain.run(input_documents=docs, **inputs)
        if re.search(r"SOURCES:\s", answer):
            answer, tables = re.split(r"SOURCES:\s", answer)
        else:
            tables = ""
        result: dict[str, Any] = {
            self.answer_key: answer.strip(),
            self.tables_answer_key: [tn.strip() for tn in tables.split(",")],
        }
        if self.return_source_documents:
            result["source_documents"] = docs
        # TODO?: fallback options if no results (rephrase / simple search)
        return result

    async def _acall(
        self, inputs: dict[str, Any], run_manager: Optional[AsyncCallbackManagerForChainRun] = None
    ) -> dict[str, Any]:
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        docs = await self._aget_docs(inputs, run_manager=_run_manager)
        answer = await self.combine_documents_chain.arun(input_documents=docs, **inputs)
        if re.search(r"SOURCES:\s", answer):
            answer, tables = re.split(r"SOURCES:\s", answer)
        else:
            tables = ""
        result: dict[str, Any] = {
            self.answer_key: answer.strip(),
            self.tables_answer_key: [tn.strip() for tn in tables.split(",")],
        }
        if self.return_source_documents:
            result["source_documents"] = docs
        return result
