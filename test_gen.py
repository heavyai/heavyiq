import re

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from modules.langchain.heavydb import HeavyDB
from modules.langchain.tools.heavydb.tool import QueryHeavyDBSchemaTool

load_dotenv()

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

table_prompt = """{table_summary}
{table_info}
"""


def get_table_summary(table_info: str):
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.2, client=None)
    messages = [
        SystemMessage(
            content="With respect to the SQL table schema and sample rows provided, please create a comprehensive response encompassing the following aspects: \n\nTitle: Choose a concise and descriptive title reflecting the table's purpose.\nDescription: Write a brief yet informative summary of the table's purpose and primary functionality. Discuss any noteworthy constraints or unique features that set it apart.\nKeywords: Identify a set of essential keywords, including but not limited to column names or relationships that facilitate search, retrieval, and document indexing.\n\nUpon completion, evaluate the coherence, accuracy, and relevance of the generated response to ensure that it adheres to the requirements outlined above, maximizing its value and usefulness for search and retrieval tasks."
        ),
        HumanMessage(content=table_info),
    ]
    resp = llm(messages)
    return resp.content


def extract_list_items(texts: list[str]) -> list[str]:
    items = []
    pattern = re.compile(r"^\s*-\s*(.+)", re.MULTILINE)

    for text in texts:
        matches = pattern.finditer(text)
        for match in matches:
            items.append(match.group(1))

    return items


def generate_questions_for_joined_tables(
    llm: ChatOpenAI, table: str, index_tool: QueryHeavyDBSchemaTool, messages: list
) -> list[str]:
    res = index_tool.run(
        f"please return up to two tables that have columns that can be joined on {table} that are not {table}"
    )
    if res == "No information found. Please revise your prompt if you wish to try again.":
        return []

    messages.append(
        HumanMessage(
            content=f"Here are some tables that can be joined on {table}. Please generate another series of questions that would require a join or subquery. Do not make up tables that don't exist. Output the unnumbered list and nothing else. Output nothing if you can't think of any questions. Avoid using table names\n{res}"
        )
    )
    resp = llm(messages)
    return extract_list_items([resp.content])


def main():
    db = HeavyDB.from_env()
    index_tool = QueryHeavyDBSchemaTool(db=db)
    messages = [SystemMessage(content=prompt)]
    table_names = list(db.get_usable_table_names())
    table = table_names[0]
    table_info = db.get_table_info([table])
    table_summary = get_table_summary(table_info)
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.3, client=None)
    messages.append(HumanMessage(content=f"{table_summary}\n\n{table_info}"))
    resp = llm(messages)
    messages.append(AIMessage(content=resp.content))
    question_responses = extract_list_items([resp.content])
    question_responses.extend(generate_questions_for_joined_tables(llm, table, index_tool, messages))

    print(question_responses)


if __name__ == "__main__":
    main()
