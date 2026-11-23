from config import SystemConfig
from cgmengine import ChatGeoMagiEngine
from printer import cprint

def main():
    config = SystemConfig()
    llm, index = ChatGeoMagiEngine.get_shared_resources(config)
    engine = ChatGeoMagiEngine(config)
    engine.initialize_session(llm, index)

    while (prompt := input(">>> ")) != "q":
        traced_response = engine.invoke(prompt)
        print(traced_response)
        
        cprint(str(traced_response.llama_response), color="blue")

if __name__ == "__main__":
    main()