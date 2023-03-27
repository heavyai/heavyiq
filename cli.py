import sys

from dotenv import load_dotenv
from langchain.llms import OpenAI

from modules.langchain.heavydb import HeavyDB
from modules.langchain.agents.agent_toolkits.heavydb.base import create_heavydb_agent
from modules.langchain.agents.agent_toolkits.heavydb.toolkit import HeavyDBToolkit

load_dotenv()

if __name__ == "__main__":
    llm = OpenAI(model_name="text-davinci-003", client=None)
    db = HeavyDB.from_env(
        ignore_tables=[
            "multilinestring_test2",
            "Multiline_import",
            "Multiline",
            "linestrings",
            "Multiline_7_0_restore",
            "Multiline_append",
        ]
    )
    toolkit = HeavyDBToolkit(db=db)
    agent = create_heavydb_agent(llm, toolkit, verbose=True)

    question = sys.argv[1]
    print("asking question: ", question)
    print(agent(question))
