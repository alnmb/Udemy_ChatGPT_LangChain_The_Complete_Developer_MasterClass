from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferMemory

from tools.sql import run_query_tool, list_tables, describe_tables_tool
from tools.report import write_report_tool

load_dotenv()

chat = ChatOpenAI()

tables = list_tables()

prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(content=(
            f'You are an AI that has access to a SQlite database \n'
            f'The database has tables of: {tables}\n'
            'Do not make any assumptions about what tables exist'
            'or what columns exist. instead, use the describe_tables function'
            ))
            ,
        MessagesPlaceholder(variable_name='chat_history'),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
)

memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

tools=[run_query_tool, 
       describe_tables_tool,
        write_report_tool]

agent = OpenAIFunctionsAgent(
    llm=chat,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(
    agent=agent,
    verbose=True,
    tools=tools,
    memory=memory
)

agent_executor(
    "How many orders are there? write the result to an html report"
)

agent_executor(
    "Repeat the exact same process for users"
)

#agent_executor('summarize the top 5 most popular prodcuts, write the results to a report file')
#agent_executor('How many users have provided a shipping address?')