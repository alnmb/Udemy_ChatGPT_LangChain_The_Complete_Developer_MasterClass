from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory #,FileChatMessageHistory
from dotenv import load_dotenv
from pathlib import Path

dotenv_path = Path("/.env")
load_dotenv(dotenv_path=dotenv_path)


chat = ChatOpenAI()

memory = ConversationSummaryMemory(
    llm=chat,
    memory_key="messages", 
    return_messages=True
    #chat_memory=FileChatMessageHistory("messages.json")
    )

prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = LLMChain(
    llm = chat,
    prompt=prompt,
    memory=memory,
    verbose=True
)


while True:
    content = input('>> ')
    
    print(f'You entered: {content}')

    result = chain({'content':content})
    print(f">> {result['text']}")