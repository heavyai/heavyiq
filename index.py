from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field, validator

from modules.langchain.logging import log_chain_call
from modules.langchain.heavydb import HeavyDB
from modules.langchain.heavydb_index import heavydb_index

INDEX_QUERY = """
Query: {query}
"""


class IndexChainResponse(BaseModel):
    answer: str = Field(
        description="A useful answer that describes how, if possible, the data can be retrieved from the SQL database."
    )
    table_names: str = Field(description="The names of the sources that are relevant to the query. Comma-separated")

    @validator("table_names")
    def validate_table_names(cls, field):
        print(f"Validating table names: {field}")
        return field


parser = PydanticOutputParser(pydantic_object=IndexChainResponse)

prompt = PromptTemplate(
    template=INDEX_QUERY,
    input_variables=["query"],
)

question_prompt_template = """Use the following portion of a long document to see if any of the text is relevant to using a SQL database to answer the question.
Return any relevant text verbatim. Schemas are particularly relevant.
{context}
Question: {question}
Relevant text, if any:"""
QUESTION_PROMPT = PromptTemplate(template=question_prompt_template, input_variables=["context", "question"])

EXAMPLE_PROMPT = PromptTemplate(
    template="Content: {page_content}\nTable: {source}",
    input_variables=["page_content", "source"],
)

combine_prompt_template = """Given the following extracted parts of a long document describing a SQL database and a question that can be solved using data in the SQL database, create a final answer with table_names ("SOURCES").
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.

QUESTION: What is the average land value for parcels in each state in the United States?
=========
Content: The movie_actors table primarily serves to store and organize data related to actors who have participated in various movies. It contains critical information such as movie_id (referring to the specific movie), actor_name, actor_gender, actor_order_num (the ordinal ranking of the actor's prominence in the film), num_total_actors (total number of actors involved in the movie), and movie_title. The table uses TEXT ENCODING DICT(32) for textual columns and DAYS(32) encoding for date columns to optimize data storage. The schema of the movie_actors table is:
Source: movie_actors
Content: The florida_parcels_2020 table is a comprehensive dataset containing information on property parcels throughout Florida. It includes details such as county, parcel ID, property usage, owner information, property physical address, legal description, key valuations, land usage classification, and geographical data. The table utilizes various encoding options to optimize storage and performance. The schema includes columns for county name, link, parcel ID, parcel number, DORUC, PAUC, property usage description, SPASS_CD, improvement value, land value, JV, JV_CHNG, JV_HMSTD,
Source: florida_parcels_2020
Content: The airports table provides comprehensive information about various airports worldwide, including their identification codes, physical characteristics, and geographical location details. Each row represents a unique airport and includes information such as type, name, latitude, longitude, elevation, continent, country, region, municipality, scheduled service availability, and related codes. Additionally, data about external resources like home links, Wikipedia links, and keywords are stored. The schema of the airports table includes columns for id, ident, type, name, latitude_deg, longitude_deg, elevation_ft, continent, iso_country, iso_region, municipality, scheduled_service, gps_code, iata_code, local_code
Source: airports
=========
FINAL ANSWER: The florida_parcels_2020 table is available but only contains information about Florida parcels.
SOURCES: florida_parcels_2020

QUESTION: How many distinct firewall hosts are there in the log data?
=========
Content: The zillow_recent_sales_full_peninsula table stores key information about recent home sales transactions in the San Francisco Peninsula area. It includes columns tracking essential property attributes such as the identifier (zpid), address, price, sold date, and property type. Additionally, it provides details about the listing representation, including image URLs, 3D models, videos, and street view links. Some unique features include columns for rent and price zestimates provided by Zillow, the property coordinates (geo_lon, geo_lat), relevant property listing statuse
Source: zillow_recent_sales_full_peninsula
Content: encoding for the text columns to optimize query performance, and spatial data representation via the geometry data type 'location', which is encoded using the compressed format with the spatial reference system (SRID) of 4326. The schema of the cell_towers_us table is:\nCREATE TABLE cell_towers_us (\n  radio TEXT ENCODING DICT(8),\n  mcc TEXT ENCODING DICT(8),\n  net TEXT ENCODING DICT(16),\n  area TEXT ENCODING DICT(16),\n  cell TEXT ENCODING DICT(32),\n  unit TEXT ENCODING DICT(16),
Source: cell_towers_us
Content:  None,None,None,None,None,None,None,93.35.88.0/24,3173435,5,EU,Europe,Italy,Milan,Italy,None,3175395,None,0,0,20123,9.188899993896484,45.470699310302734,GeoIP2-City-CSV_20190611,93-35-88-19.ip54.fastwebnet.it,Y,U,N\n2018-02-14 08:01:46,fatcontroller.mapd.com,filterlog:,5,None,None,1000000103,igb0,match,block,in,4,0,None,54,2877,0,DF,6,tcp,None,None,None,40,167.114.222.188,10.0.0.5,22,27581,0,None,R,3399261706,None,0,None,None,167.114.216.0/21,6077243,1000,NA,North America,Canada,Montreal,Canada,None,6251999,None,0,0,H3A,-73.57939910888672,45.50630187988281,GeoIP2-City-CSV_20190611,None,Y,U,N\nSample values for text columns:\nfire_wall_host: fatcontroller.mapd.com, gateway\nlog_name: filterlog:\ninterface_name: igb0, igb1.1, igb1_vlan1, lo0, ovpns2
Source: firewall_in
Content: CREATE TABLE nyt_covid_counties_v2 (\n  reporting_date DATE ENCODING DAYS(32),\n  county TEXT ENCODING DICT(32),\n  state TEXT ENCODING DICT(32),\n  fips TEXT ENCODING DICT(32),\n  cases INTEGER,\n  deaths INTEGER,\n  total_case_pct_per_pop FLOAT,\n  total_mortality_pct_per_pop FLOAT,\n  new_cases_1_day_lag INTEGER,\n  new_cases_7_day_lag INTEGER,\n  new_deaths_1_day_lag INTEGER,\n  new_deaths_7_day_lag INTEGER,\n  daily_cases_pct_pop_infected_1_day_avg FLOAT,\n  daily_cases_pct_pop_infected_7_day_avg FLOAT,\n  daily_death
Source: nyt_covid_counties_v2
=========
FINAL ANSWER: The firewall_in table should be investigated for this query.
SOURCES: firewall_in

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""
COMBINE_PROMPT = PromptTemplate(template=combine_prompt_template, input_variables=["summaries", "question"])


def main():
    model_name = "gpt-3.5-turbo"
    llm = ChatOpenAI(model_name=model_name, temperature=0.3, client=None)
    retriever = heavydb_index.vectorstore.as_retriever()
    qa_chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        question_prompt=QUESTION_PROMPT,
        document_prompt=EXAMPLE_PROMPT,
        combine_prompt=COMBINE_PROMPT,
        retriever=retriever,
    )

    output = log_chain_call(
        qa_chain,
        "What are the top 3 network matches with the highest total data_length in log entries?",
        model_name,
        "index_test_chain",
    )
    print(output)


if __name__ == "__main__":
    main()
