"""
This script is an example inferencing system that uses cgm.py
and provides a simple chat interface in the terminal.
"""

from cgm import ChatGeoMagi
from config import SystemConfig

config = SystemConfig()
llm, embeddings, retriever = ChatGeoMagi.get_shared_resources(config)

model = ChatGeoMagi(config)
model.initialize(llm, embeddings, retriever)
model.visualize_agent()

while (prompt := input(">>> ")) != "q":
    output = model.stream_invoke(prompt)
    
    for step in output:
        step["messages"][-1].pretty_print()