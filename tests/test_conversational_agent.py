import unittest

from langchain.agents import AgentExecutor

from heavynl.langchain.agents.convo_agent import create_conversational_agent


class TestConversationalAgent(unittest.TestCase):
    def test_create_conversational_agent(self):
        agent = create_conversational_agent()

        self.assertIsInstance(agent, AgentExecutor)


if __name__ == "__main__":
    unittest.main()
