from typing import Any, Dict, Type

from src.agent.base import BaseAgent

# get os default path


# TODO: Refactor this file to use the new agent system
class AgentFactory:
    CLASS_MAP: Dict[str, Type[BaseAgent]] = {
        "sql": BaseAgent,
        # Add other agent types here
    }

    @classmethod
    def load_agent(cls, agent_type: str, **kwargs: Any) -> BaseAgent:
        """
        Load an agent by its type.
        :param agent_type: The type of the agent to load.
        :param kwargs: Additional arguments to pass to the agent's constructor.
        :return: An instance of the requested agent type.
        """
        if agent_type not in cls.CLASS_MAP:
            raise ValueError(f"Agent type '{agent_type}' is not supported.")
        client_class = cls.CLASS_MAP[agent_type]
        return client_class(**kwargs)
