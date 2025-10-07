"""Common" variables."""
from configs.prompts import *

class CommonConfig:
    def __init__(self, ):
        self.__dict__ = {
            "OPENAI_CONFIG": { 
                "url": "<base_url>",
                "authorization": "<token>",
            },
            "DEEPSEEK_CONFIG": { 
                "url": "<base_url>",
                "authorization": "<token>",
            },
            "O3-MINI_CONFIG": { 
                "model_name": "o3-mini",
                "url": "<base_url>",
                "authorization": "<token>",
                "max_tokens": 4096,
            },
            "SANDBOX": {
                "tool_link": "<address>:30008"
            },
            "SOLVER_PROMPT": {
                "user_prompt": SolverPrompt_User_Template,
                "assistant_prefix": SolverPrompt_Assistant_Template
            },
            "CRITIC_PROMPT": {
                "user_prompt": CriticPrompt_User_Template,
                "assistant_prefix": CriticPrompt_Assistant_Template
            },
            "CRITIC_WITH_SUGGESTION_PROMPT": {
                "user_prompt": CriticWithSuggestionPrompt_User_Template,
            },
            "REFINE_PROMPT": {
                "user_prompt": RefinePrompt_User_Template,
                "assistant_prefix": RefinePrompt_Assistant_Template
            },
            "QUALITY_PROMPT": {
                "user_prompt": QualityPrompt_User_Template,
            },
            "SELECTOR_PROMPT": {
                "user_prompt": SelectPrompt_User_Template,
                "assistant_prefix": SelectPrompt_Assistant_Template
            },
        }

    def __getitem__(self, key):
        return self.__dict__.get(key, None)
