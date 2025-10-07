import os

cur_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(cur_dir, "solver_user_rag.txt"), "r", encoding="utf8") as f:
    SolverPrompt_User_Template = "".join(f.readlines())

with open(os.path.join(cur_dir, "solver_prefix_rag.txt"), "r", encoding="utf8") as f:
    SolverPrompt_Assistant_Template = "".join(f.readlines())

with open(os.path.join(cur_dir, "critic_user_rag.txt"), "r", encoding="utf8") as f:
    CriticPrompt_User_Template = "".join(f.readlines())

with open(os.path.join(cur_dir, "critic_prefix_rag.txt"), "r", encoding="utf8") as f:
    CriticPrompt_Assistant_Template = "".join(f.readlines())
    
with open(os.path.join(cur_dir, "critic_user_rag_with_suggestion.txt"), "r", encoding="utf8") as f:
    CriticWithSuggestionPrompt_User_Template = "".join(f.readlines())

with open(os.path.join(cur_dir, "refine_user_rag.txt"), "r", encoding="utf8") as f:
    RefinePrompt_User_Template = "".join(f.readlines())

with open(os.path.join(cur_dir, "refine_prefix_rag.txt"), "r", encoding="utf8") as f:
    RefinePrompt_Assistant_Template = "".join(f.readlines())

with open(os.path.join(cur_dir, "quality_user_rag.txt"), "r", encoding="utf8") as f:
    QualityPrompt_User_Template = "".join(f.readlines())

with open(os.path.join(cur_dir, "select_user_rag.txt"), "r", encoding="utf8") as f:
    SelectPrompt_User_Template = "".join(f.readlines())

with open(os.path.join(cur_dir, "select_prefix_rag.txt"), "r", encoding="utf8") as f:
    SelectPrompt_Assistant_Template = "".join(f.readlines())
