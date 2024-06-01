#7a2ac8b150a140e19e406b9de97c99b6
import boto3
import json
from langchain_aws import ChatBedrock
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from IPython.display import Markdown


bedrock = boto3.client(service_name="bedrock-runtime",region_name="us-east-1")
model_id="anthropic.claude-3-sonnet-20240229-v1:0"
model_kwargs = {
    "max_tokens": 2048,
    "temperature": 0.1,
    "top_k": 250,
    "top_p": 1,
    "stop_sequences": ["\n\nHuman"],
}

# Create the LangChain Chat object
llm = ChatBedrock(
    client=bedrock,
    model_id=model_id,
    model_kwargs=model_kwargs,
)


researcher = Agent(
 role='Senior Research Analyst',
 goal='Uncover cutting-edge developments in AI and data science',
 backstory="You work at a leading tech think tank…",
 verbose=True,
 allow_delegation=False,
 llm=llm,
 tool=[SerperDevTool()]
)
check_advancements_task = Task(
 description="Conduct a comprehensive analysis of the latest advancements in AI in 2024…",
 expected_output="Full analysis report in bullet points",
 agent=researcher
)
crew = Crew(
 agents=[researcher],
 tasks=[check_advancements_task],
 verbose=2,
)


result = crew.kickoff()
#m(Markdown(result))

with open('output.md', 'w', encoding='utf-8') as f:
    f.write(result)
