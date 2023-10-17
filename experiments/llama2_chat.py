# Before running this script, make sure to start an llama2 server by following the steps given on ./run_llm_server folder.
import sys, os

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from heavyiq.langchain.chat_models import Llama2Chat
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.chains import LLMChain
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferMemory


def main():
    template_messages = [
        SystemMessage(content="You are a helpful assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
    prompt_template = ChatPromptTemplate.from_messages(template_messages)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    model = Llama2Chat(openai_api_key="nothing", openai_api_base="http://localhost:5000/v1", max_tokens=500)
    chain = LLMChain(llm=model, prompt=prompt_template, memory=memory)
    # What can I see in Vienna? Propose a few locations. Names only, no details.
    print(prompt_template.format(chat_history=[], text=""))
    while True:
        input_txt = input(">>> ")
        if input_txt in ["\\q", "quit", "exit"]:
            print("Bye")
            break
        print(chain.run(text=input_txt))


if __name__ == "__main__":
    main()
