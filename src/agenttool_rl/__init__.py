"""AgentTool-RL public package."""

from .contracts import TOOL_CONTRACTS, ToolContract, ToolParameter
from .environment import EpisodeFinishedError, IncidentEnvironment
from .evaluation import AgentReport, BenchmarkReport, EpisodeMetrics, evaluate_agents
from .generator import GENERATOR_VERSION, ScenarioGenerator
from .agents import AgentAction, BaseLLMAgent, RandomAgent, RuleBasedAgent
from .llm import ChatBackendError, ChatMessage, OpenAICompatibleBackend
from .models import StepResult, ToolError, TrajectoryStep
from .reward import RewardBreakdown, RewardConfig, score_trajectory

__all__ = [
    "GENERATOR_VERSION",
    "TOOL_CONTRACTS",
    "AgentAction",
    "AgentReport",
    "BaseLLMAgent",
    "BenchmarkReport",
    "ChatBackendError",
    "ChatMessage",
    "EpisodeFinishedError",
    "EpisodeMetrics",
    "IncidentEnvironment",
    "OpenAICompatibleBackend",
    "RandomAgent",
    "RewardBreakdown",
    "RewardConfig",
    "ScenarioGenerator",
    "RuleBasedAgent",
    "StepResult",
    "ToolContract",
    "ToolError",
    "ToolParameter",
    "TrajectoryStep",
    "score_trajectory",
    "evaluate_agents",
]
__version__ = "0.3.0"
