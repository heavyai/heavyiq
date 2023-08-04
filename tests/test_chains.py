import unittest

from heavyiq.langchain.chains import NLtoSQLChain, NLtoAnswerChain
from heavyiq.langchain.llms import get_llm
from heavyiq.langchain.logging import log_chain_call
from heavyiq.langchain import HeavyDB


class TestChains(unittest.TestCase):
    def test_nl_to_sql(self):
        heavydb = HeavyDB.from_env(include_tables=["usa_states"])
        llm = get_llm(["unit_test", "chain", "nl_to_sql_chain"], temperature=0.0, client=None)
        chain = NLtoSQLChain(database=heavydb, llm=llm)
        res = log_chain_call(chain, "How many states begin with the letter A? What are they?", "")
        self.assertEqual("query" in res, True)
        self.assertEqual("sql" in res, True)

    def test_nl_to_answer(self):
        heavydb = HeavyDB.from_env(include_tables=["usa_states"])
        llm = get_llm(["unit_test", "chain", "nl_to_sql_chain"], temperature=0.0, client=None)
        chain = NLtoAnswerChain(database=heavydb, llm=llm)
        res = log_chain_call(chain, "How many states begin with the letter A? What are they?", "")
        self.assertEqual("query" in res, True)
        self.assertEqual("sql" in res, True)
        self.assertEqual("results" in res, True)
        self.assertEqual("answer" in res, True)
