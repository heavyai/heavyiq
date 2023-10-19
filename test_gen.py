import asyncio
import aiofiles
from aiofiles.ospath import exists as aexists
import aiocsv
from functools import partial
import os
import re
from typing import Optional

from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from heavyiq.config import get_config
from heavyiq.langchain import HeavyDB
from heavyiq.langchain.index import aget_heavydb_index

os.environ["OPENAI_API_KEY"] = get_config().openai_api_key


prompt = """Create questions that require querying and analyzing a SQL database, provided information about a table within the SQL Database

    1. Analyze the provided table schema: a. Identify the columns, and unique keys

    2. Examine sample rows and the most common values in each row: a. Determine important quantitative and qualitative patterns, trends, or anomalies. b. Identify potential associations between distinct variables or attributes.

    3. Framing questions to challenge the agentic LLM: a. Simple queries: Questions that involve one entity or attribute, such as count or basic retrieval operations. b. Intermediate queries: Questions requiring filtering, sorting, or aggregating the data, such as calculating averages, calculating the sum or the maximum and minimum values. c. Complex queries: Questions necessitating joining multiple tables, constructing multiple sub-queries, or conditionally retrieving data based on specific criteria.

    4. Contextualize the questions in real-world scenarios: a. Align the questions with actionable insights for a potential stakeholder or decision-maker. b. Test the agentic LLM's ability to generate clear, concise, and well-formulated queries directly addressing the given scenario.

Here are some general examples of questions to ask an agentic LLM for each level of complexity:

    Simple queries:
        How many [value] are there in the [table description]?
        What is the most common value in the [column_name] column?
    Intermediate queries:
        What is the average value of the [column_name] column for rows meeting certain criteria?
        What are the top 5 most frequent values in the [column_name] column, and how many times do they occur?
    Complex queries:
        What proportion of [Entity_A] has a relationship with [Entity_B] based on a specific parameter or threshold?
        For rows that meet certain criteria, which [Entity_A] has the highest occurrence of [Entity_B], and what is the associated value in the [column_name] column?
        [Geographic question]?

Keep in mind these examples are generic, and you will need to adapt them to the specific table schema, sample rows, and common values provided to you, in order to generate appropriate questions that effectively test the agentic LLM.
Only output questions for the agent in an unnumbered list. Do not include any other text or instructions in the output file. Do not include the table name in the question unless necessary. Do not include the column name directly. You may reference column names semantically.
Emulate questions from users that domain-specific knowledge but have no SQL knowledge."""


async def extract_list_items(texts: list[str]) -> list[str]:
    """Extract list items from a list of text strings.

    This function processes each line in the input texts, looking for lines
    that represent list items. The lines may start with a hyphen, a number followed
    by a period, or an asterisk, followed by optional whitespace. The function strips
    any surrounding single or double quotes from each item."""
    items = []
    pattern = re.compile(r"^\s*(?:-\s*|\*\s*|\d+\.\s*)(.+)", re.MULTILINE)

    for text in texts:
        matches = pattern.finditer(text)
        for match in matches:
            item = match.group(1)
            item = item.strip("'\"")  # Strip surrounding quotes
            items.append(item)

    return items


async def generate_questions_for_joined_tables(
    llm: ChatOpenAI, heavydb: HeavyDB, tables: tuple[str, str], messages: list
) -> list[str]:
    table, join_table = tables
    print(f"Generating questions for joined tables for {table}")
    tables_summary = await heavydb.aget_table_info([join_table])
    messages.append(
        HumanMessage(
            content=f"Here is a table that can be joined on {table}. Please generate another series of questions that would require a join or subquery. Do not make up tables that don't exist. Output the unnumbered list and nothing else. Output nothing if you can't think of any questions. Avoid using table names\n{tables_summary}"
        )
    )
    try:
        token_count = llm.get_num_tokens_from_messages(messages)
        if token_count > 8192:
            print(f"Error generating questions for joined tables for {table} due to token count {token_count}")
            return []
        resp = await llm.apredict_messages(messages)
    except Exception as e:
        print(f"Error generating questions for joined tables for {table}")
        print(str(e))
        return []
    return await extract_list_items([resp.content])


async def get_questions_for_table(
    heavydb: HeavyDB, tables: tuple[str, Optional[str]]
) -> list[tuple[str, bool, str, str]]:
    table, join_table = tables
    print(f"Processing {table}")
    messages: list = [SystemMessage(content=prompt)]
    table_info = await heavydb.aget_table_info([table])
    heavydb_index = await aget_heavydb_index()
    table_summary = await heavydb_index.aquery(f"What is the summary of the {table} table?")
    llm = ChatOpenAI(model=get_config().openai_gpt_model, temperature=0.7, client=None)
    messages.append(HumanMessage(content=f"{table_summary}\n\n{table_info}"))
    resp = await llm.apredict_messages(messages)
    single_table_questions = await extract_list_items([resp.content])
    single_table_questions_formatted: list[tuple[str, bool, str, str]] = [
        (table, False, "", question) for question in single_table_questions
    ]
    messages.append(AIMessage(content=resp.content))
    if join_table is not None and join_table != "":
        multi_table_questions = await generate_questions_for_joined_tables(llm, heavydb, (table, join_table), messages)
        multi_table_questions_formatted = [(table, True, join_table, question) for question in multi_table_questions]
    else:
        multi_table_questions_formatted = []

    return single_table_questions_formatted + multi_table_questions_formatted


async def main():
    """Creates a CSV of questions that should be able to be answered by querying the SQL Database"""
    heavydb = HeavyDB.from_env()
    process_func = partial(get_questions_for_table, heavydb)
    tables_list = [
        ("sp500_companies", "sp500_2018_2020_minute"),
        ("airports", "flights_2008"),
        ("cell_towers_us", "cell_towers_world"),
        ("movie_actors", "movie_details"),
        ("sfmta_bus_stops", "sfmta_buses"),
        ("us_pois_safegraph", "usa_states"),
        ("zillow_recent_sales_full_peninsula", "florida_parcels_2020"),
        ("nyt_covid_counties_v2", "usa_states"),
        ("world_ships", "aishub_2021_06_16"),
        ("us_upstream_production", "usa_states"),
        ("craigslist_vehicles", "usa_states"),
        ("usa_states", "cell_towers_us"),
    ]
    with get_openai_callback() as cb:
        tasks = []
        for tables in tables_list:
            tasks.append(asyncio.create_task(process_func(tables)))
        questions = await asyncio.gather(*tasks)
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Successful Requests: {cb.successful_requests}")
        print(f"Total Cost (USD): ${cb.total_cost}")

    file_name = "output_questions.csv"
    write_headers = not await aexists(file_name)
    async with aiofiles.open("output_questions.csv", "a", newline="", encoding="utf-8") as csvfile:
        writer = aiocsv.writers.AsyncWriter(csvfile)
        if write_headers:
            await writer.writerow(["primary_table", "is_multi_table", "secondary_table", "question"])
        for question in questions:
            await writer.writerows(question)


if __name__ == "__main__":
    asyncio.run(main())
