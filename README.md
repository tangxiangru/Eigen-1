# Eigen-1

🔗 Full paper: [https://arxiv.org/abs/2505.20286](https://arxiv.org/abs/2509.21193)

## Installation

```bash

conda env create -f environment.yaml
conda activate eigen1
pip install -e .

# install tool dependencies
cd mcp_sandbox/
pip install -r requirements.txt

```

---

## Configuration

1. Configure the tools following the instructions in `mcp_sandbox/README.md`.

2. Configure `configs/common_config.py` file, set the following parameters:

-   `DEEPSEEK_CONFIG`: **Deepseek v3.1** model, which is used as the main agent
-   `OPENAI_CONFIG`: for RAG
-   `O3-MINI_CONFIG`: for evaluation
-   `SANDBOX`: for MCP toolbox URL


---

## Running the Project

To run the agent on the HLE Bio/Chem dataset(`data/hle-bio.json`):

1.  **Run the Agent**
    ```bash
    python -m functions.eigen1_hle
    ```

2.  **Calculate Score**
    Once the agent has finished, run the scoring script:
    ```bash
    python utils/hle_score.py
    ```

## Acknowledgment

This repository benefit from [X-Master](https://github.com/sjtu-sai-agents/X-Master/tree/main).
