import requests
from typing import Dict, List, Any
from openai import OpenAI
import re
import json
from termcolor import colored
import os
import sys
import time
import multiprocessing
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from hipporag import HippoRAG
from copy import deepcopy



current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, ".."))
sys.path.append(current_dir)

from configs import CommonConfig
from llm_agent.utils import LLMConfig
from llm_agent.context import BaseContextManager
from llm_agent.tools.tool_manager import BaseToolManager


def strip_think_and_exec(text: str) -> str:
    """Keep only the visible answer part by removing </think> ... and </execution_results> tails."""
    if text is None:
        return ""
    out = text
    if "</think>" in out:
        out = out.split("</think>")[-1]
    if "</execution_results>" in out:
        out = out.split("</execution_results>")[-1]
    return out.strip()


def extract_code_block(language: str, text: str):
    """Extract the first code block of the specified language from the text."""
    text = text.strip()
    pattern = rf'```{language}\n(.*?)\n```'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def has_answer(text, finish_type='answer'):
    if finish_type == 'answer':
        matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    elif finish_type == 'select':
        matches = re.findall(r"<select>(.*?)</select>", text[-200:], flags=re.DOTALL)
    else:
        assert False, "finish_type must be 'answer' or 'select'"
    
    if not matches:  # 没有匹配到任何
        return False
    
    last_content = matches[-1].strip()  # 取最后一个并去掉首尾空格
    return len(last_content) > 0    


def search_local_documents(query: str) -> list[str]:
    """
    Searches the unified local knowledge base for relevant documents.
    If it fails to find results after 3 attempts, it returns an empty list.
    """
    result = ''
    try:
        # print("execution...")
        result = requests.post(
            f"{CommonConfig['SANDBOX']['tool_link']}/rag",
            headers={
                "Content-Type": "application/json"
            },
            json={
                "query": query
            },
            proxies={"http": None, "https": None}, # remove if error
        )

        return json.loads(result.json()['output'])
    
    except Exception as e:
        import traceback
        print(f"Error during local document search: {traceback.format_exc()}\n ----{result}----")
        return []


class BaseAgent:
    def __init__(self, llm_config:Dict[str, Any]):
                
        self.llm_config: LLMConfig = LLMConfig(llm_config)
        self.client = OpenAI(base_url=self.llm_config.base_url, api_key=self.llm_config.api_key)
        self.rag_monitor = OpenAI(base_url=self.llm_config.base_url, api_key=self.llm_config.api_key)
        self.rag_querier = OpenAI(base_url=self.llm_config.base_url, api_key=self.llm_config.api_key)
        self.rag_injector = OpenAI(base_url=self.llm_config.base_url, api_key=self.llm_config.api_key)
        self.rag_chunk = 512
        self.rag_overlapping = 128
        self.max_rag = 1 # max interrupt times every agent steps
        assert self.rag_chunk > self.rag_overlapping


    def check_condition(self, input_str: str):
        if not self.llm_config.stop_condition:
            return False
        matches = list(re.finditer(self.llm_config.stop_condition, input_str, re.DOTALL))
        detected_num = len(matches)
        
        if detected_num > 0:
            return True
        return False


    def check_rag(self, text: str):
        prompt_1 = f"""
Analyze the following text and determine if responding to it accurately requires retrieving information from an external source.
If you find any doubt or uncertainty about a concept or term in the text, consider it necessary to rag. You should tend to use rag because it helps with reasoning.

If retrieval is required, answer: yes
If no retrieval is required, answer: no

Text:
{text}

Judgment:
"""
        try:
            response = self.rag_monitor.chat.completions.create(
                model = 'gpt-4.1-mini',
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt_1}
                ],
                max_tokens=150,
                temperature=0
            )
            result = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error: {e}")
            return None

        if result.lower() != 'yes':
            return None

        prompt_2 = f"""
Your task is to generate a single, concise, and effective search query for retrieving the information required by the text below.

## Instructions
1. Return **only the search query** itself.
2. Do not include any explanations, punctuation, quotation marks, or other text.
3. The query should be direct and contain only the most essential keywords.

## Text
{text}

Search Query:
"""
        try:
            response = self.rag_querier.chat.completions.create(
                model = 'gpt-4.1-mini',
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt_2}
                ],
                max_tokens=150,
                temperature=0
            )
            result = response.choices[0].message.content.strip()
            return result
        except Exception as e:
            print(f"Error: {e}")
            return None

    
    def rag_search(self, query: str) -> str:
        docs = search_local_documents(query)
        if self.llm_config.is_debug:
            print(colored(f"[RAG results length {len(docs)}]", 'red', attrs=['bold']))
            print(colored(f"[RAG results: \n{json.dumps(docs, indent=4)}\n]", 'red', attrs=['bold']))
        return json.dumps(docs, indent=4)
    

    def add_rag(self, text: str) -> str:
        # if "<code>" in text and "</code>" not in text:
        if text.rfind("<code>") > text.rfind("</code>"):
            return text

        if self.llm_config.is_debug:
            print(colored(f"[Rag checking]", 'red', attrs=['bold']), end="", flush=True)

        rag_query = self.check_rag(text[-self.rag_chunk:])
        
        if rag_query is None:
            return text
        
        if self.llm_config.is_debug:
            print(colored(f"[Need to rag: {rag_query}]", 'red', attrs=['bold']))

        rag_result = self.rag_search(rag_query)
        prompt = f"""
## Role & Core Objective
You are an information integration specialist. Your sole task is to process the provided RAG (Retrieval-Augmented Generation) output. Your goal is to conduct an in-depth analysis of this output with the explicit objective of **maximizing the utilization** of all relevant information to substantively support the **reasoning, argumentation, or conclusions** presented in the main text. Your function is strictly limited to selecting, organizing, and polishing this content for inclusion; **any form of additional reasoning, interpretation, or conclusion generation falls outside your duty**.

## Content Integration Principles
-   **Comprehensive Extraction**: Prioritize extracting all valuable information from the RAG outputs. Focus specifically on identifying evidence, data, quotations, examples, and contextual details that can enhance the logical depth, robustness, and persuasiveness of the main text's arguments.
-   **Seamless Cohesion and Minimal Completion**: Ensure all integrated content maintains smooth contextual coherence and stylistic consistency with the main text. If the main text appears incomplete (e.g., ends mid-thought), you are permitted to perform the **minimal amount of completion necessary** *only* to facilitate a natural and logical transition into the supplemental RAG content. This completion must be neutral and based solely on the context of the main text.
-   **Neutral Representation**: Present all information from the RAG output in an objective and neutral manner. Do not evaluate, question, or express undue approval of the RAG content. Simply integrate it as factual support without subjective commentary.

## Output Specifications
-   You can use the following template for reference: "<main text completion if necessary>. Wait a minute, by searching information about <rag query>, I found that <rag result>. Now that I have more relevant information, let me continue my reasoning. "
-   It should be directly appendable to the end of the original main text without requiring any modifications to the main text itself.
-   Do not include any extraneous content such as summaries of your process, explanations, headings, bullet points, formatting markers (e.g., `-`, `*`, `#`), or labels like "Supplement:" or "Additional Information:".
-   Simply begin outputting the refined and relevant RAG content, ensuring it follows on naturally from the main text's final sentence.

## Instruction Recap
Your duty is only to **select, filter, organize, and polish** the information from the RAG output. **DO NOT** perform any external reasoning, draw new conclusions, or add information not present in the RAG output.

## Main Text  
{text}  

## RAG Query  
{rag_query}  

## RAG Result  
{rag_result}

Please start generating the following text following the main text, providing sufficient, helpful, and coherent RAG content.
"""
            
        response = self.rag_injector.chat.completions.create(
            model = 'gpt-4.1-mini',
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4096,
            temperature=0
        )
        rag_result = response.choices[0].message.content.strip()

        return text+rag_result
        

    def extract_tool_content(self, input_str: str):
        if not self.llm_config.tool_condition:
            return input_str, ''
        matches = list(re.finditer(self.llm_config.tool_condition, input_str, re.DOTALL))
        detected_num = len(matches)
        
        if detected_num > 0:
            match = matches[0]
            code_content = match.group(1)
            match_start_index = match.start()
            cut_text = input_str[:match_start_index]

            return cut_text, code_content

        return input_str, ''

    def call_api(self, prompt: str, enable_rag=True):
        try:
            messages = [{"role": "user", "content": prompt}] 

            with self.client.chat.completions.create(
                model=self.llm_config.model,
                messages=messages,
                stream=True,
                **self.llm_config.generation_config
            ) as stream:
                full_response = ""  
                rag_check_text = ""

                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content is not None:
                        if self.llm_config.is_debug:
                            print(content, end="", flush=True)
                        full_response += content
                        rag_check_text += content

                    if len(rag_check_text) >= self.rag_chunk and enable_rag:
                        rag_check_text = rag_check_text[-self.rag_overlapping:]
                        full_response_with_rag = self.add_rag(full_response)
                        if full_response_with_rag != full_response:
                            # interrupt the generation and return the rag-augmented response
                            return {
                                "content": full_response_with_rag.strip(),
                                "type": 'interrupted_by_rag',
                            }

                    stop_flag = self.check_condition(full_response)
                    if stop_flag:
                        return {
                            "content": full_response.strip(),
                            "type": 'interrupted_by_code',
                        }

        except KeyboardInterrupt:
            print("interrupt")
            
        except Exception as e:
            import traceback
            print(f"error: {traceback.format_exc()}, Model: {self.llm_config.model}.")
            
        return {
            "content": full_response.strip(),
            "type": 'full_text',
        }

    def step(self, input_prompt: str):
        api_call_count = 1
        step_response_content = ""
        while api_call_count <= (self.max_rag+1):
            step_response_dict = self.call_api(input_prompt, enable_rag=api_call_count<=self.max_rag)
            api_call_count += 1
            step_response_content += step_response_dict['content']

            step_response_type = step_response_dict['type']
            if step_response_type in ['full_text', 'interrupted_by_code']:
                break
            else:
                input_prompt += step_response_content
                if self.llm_config.is_debug:
                    print(colored(f"\n\n[Continue generation]\n{input_prompt}", 'cyan', attrs=['bold']))

        agent_response, tool_call_content = self.extract_tool_content(step_response_content)

        return {
            "step_response": agent_response,
            "tool_call_content": tool_call_content
        }



class LLMAgent:

    def __init__(self, model_url: str, model_key, toolbox_url: str, max_tokens: int = 64000, temperature: float = 0.5):
        self.llm_config: Dict[str, Any] = {
            'model': 'deepseek-reasoner',
            'base_url': model_url,
            'api_key': model_key,
            'generation_config': {
                'max_tokens': max_tokens,
                'temperature': temperature,
            },
            'stop_condition': r'<code[^>]*>((?:(?!<code).)*?)</code>',
            'tool_condition': r'<code[^>]*>((?:(?!<code).)*?)</code>',
        }
        self.toolbox_url = toolbox_url

        template_path = os.path.join(current_dir, '../configs/templates/r1_tool.jinja')
        with open(template_path, 'r', encoding='utf-8') as f:
            self.chat_template = f.read()


    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=0.5, max=4),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def call_model(self, user_prompt: str, assistant_prefix: str, finish_type: str = 'answer') -> str:
        """Run one conversation (think-while-calling-tools loop) and return rendered output."""

        # Build fresh instances to ensure thread-safety
        base_agent = BaseAgent(llm_config=self.llm_config)
        context_manager = BaseContextManager(chat_template=self.chat_template)
        tool_manager = BaseToolManager(url=self.toolbox_url)

        # Seed logs
        context_manager.agent_logs = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_prefix},
        ]

        # Main loop
        max_agent_step = 150
        max_empty_response = 5
        cur_steps = 0
        empty_response_count = 0

        while cur_steps <= max_agent_step:

            if empty_response_count == max_empty_response or cur_steps == max_agent_step:
                if finish_type == 'answer':
                    context_manager.log_agent(
                        "Now let me combine all the information above and write the answer in <answer></answer> with \\boxed."
                    )
                elif finish_type == 'select':
                    context_manager.log_agent(
                        "Now let me combine all the information above and select the best solution in <select></select>."
                    )
                else:
                    assert False, "finish_type must be 'answer' or 'select'"

            prompt = context_manager.build_input_prompt()
                
            result = base_agent.step(prompt)
            step_response = result.get('step_response', '')

            context_manager.log_agent(step_response)
            print(f"Agent response: {step_response}")

            tool_call_content = result.get('tool_call_content')
            if not tool_call_content and step_response:
                break

            if tool_call_content:
                context_manager.log_tool_call(tool_call_content)
                tool_result = tool_manager.execute_tool(tool_call_content)
                context_manager.log_tool_call_result(tool_result)

                print(f"Tool call content: {tool_call_content}")
                print(f"\033[32mTool call result: {tool_result}\033[0m")

            elif not step_response:
                empty_response_count += 1
                print(f"Empty response count: {empty_response_count}/{max_empty_response}")
                if empty_response_count > max_empty_response:
                    print("Max empty responses reached, stopping.")
                    break

            cur_steps += 1

        # Render final response with template
        return context_manager.chat_template.render(tool_logs=context_manager.agent_logs)


class QualityEvaluator:
    def __init__(self, model_url: str, model_key, max_tokens: int = 64000, temperature: float = 0.5):
        self.llm_config: Dict[str, Any] = {
            'model': 'o3',
            'base_url': model_url,
            'api_key': model_key,
        }

        self.msg_history = [
        ]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=0.5, max=4),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def call_model(self, user_input) -> str:
        """Directly call the model with a prompt and return the output."""
        from openai import OpenAI
        client = OpenAI(api_key=self.llm_config['api_key'], base_url=self.llm_config['base_url'])
        self.msg_history.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model=self.llm_config['model'],
            messages=self.msg_history,
        )
        assistant_reply = response.choices[0].message.content
        self.msg_history.append({"role": "assistant", "content": assistant_reply})

        return assistant_reply


class Eigen1Agent:
    def __init__(self, debug: bool = True, log_dir: str = "debug_logs"):
        self.deepseek_api_url = CommonConfig['DEEPSEEK_CONFIG']['url']
        self.deepseek_api_key = CommonConfig['DEEPSEEK_CONFIG']['authorization']
        self.sandbox_url = CommonConfig['SANDBOX']['tool_link']
        self.chat_obj = LLMAgent(self.deepseek_api_url, self.deepseek_api_key, self.sandbox_url)
        self.debug = debug
        self.log_dir = log_dir

    # ---- Stage runners ----

    def _forward_solver(self, query: str) -> str:
        user_prompt = CommonConfig["SOLVER_PROMPT"]["user_prompt"].format(query=query)
        assistant_prefix = CommonConfig["SOLVER_PROMPT"]["assistant_prefix"]

        return self.chat_obj.call_model(user_prompt, assistant_prefix)

    def _forward_critic(self, query: str, student_solution: str, suggestion: str=None) -> str:
        # print(111)
        s_solution = strip_think_and_exec(student_solution)
        if suggestion is not None:
            user_prompt = CommonConfig["CRITIC_WITH_SUGGESTION_PROMPT"]["user_prompt"].format(query=query, s_solution=s_solution, suggestion=suggestion)
        else:
            user_prompt = CommonConfig["CRITIC_PROMPT"]["user_prompt"].format(query=query, s_solution=s_solution)
        # print(user_prompt)
        assistant_prefix = CommonConfig["CRITIC_PROMPT"]["assistant_prefix"]
        return self.chat_obj.call_model(user_prompt, assistant_prefix)

    def _forward_refine(self, query: str, anchor_solution: str, reference_solutions: List[str]) -> str:
        assert len(reference_solutions) == 4, "refine expects 1 anchor + 4 references"
        user_prompt = CommonConfig["REFINE_PROMPT"]["user_prompt"].format(
            query=query,
            anchor_solution=strip_think_and_exec(anchor_solution),
            reference_1=strip_think_and_exec(reference_solutions[0]),
            reference_2=strip_think_and_exec(reference_solutions[1]),
            reference_3=strip_think_and_exec(reference_solutions[2]),
            reference_4=strip_think_and_exec(reference_solutions[3]),
        )
        assistant_prefix = CommonConfig["REFINE_PROMPT"]["assistant_prefix"]
        return self.chat_obj.call_model(user_prompt, assistant_prefix)

    def _forward_selector(self, query: str, candidates: List[str]) -> str:
        assert len(candidates) == 5, "selector expects exactly 5 candidates"

        user_prompt = CommonConfig["SELECTOR_PROMPT"]["user_prompt"].format(query=query, 
                                                                        solution_1=strip_think_and_exec(candidates[0]),
                                                                        solution_2=strip_think_and_exec(candidates[1]),
                                                                        solution_3=strip_think_and_exec(candidates[2]),
                                                                        solution_4=strip_think_and_exec(candidates[3]),
                                                                        solution_5=strip_think_and_exec(candidates[4]))
        assistant_prefix = CommonConfig["SELECTOR_PROMPT"]["assistant_prefix"]

        selector_response = self.chat_obj.call_model(user_prompt, assistant_prefix, finish_type='select')
        m = re.search(r'<select>Response (\d+)</select>', selector_response)
        if not m:
            print("Warning: Could not parse selector's decision. Defaulting to Response 1.")
            idx = 0
        else:
            idx = max(0, min(4, int(m.group(1)) - 1))
        return candidates[idx], selector_response

    def _evaluate_quality(self, query: str, solution: str) -> dict:
        """多角度质量与不确定性评估"""
        user_prompt = CommonConfig["QUALITY_PROMPT"]["user_prompt"]
        user_prompt = user_prompt.replace("{query}", query).replace("{solution}", solution)

        llm_evaluator = QualityEvaluator(self.deepseek_api_url, self.deepseek_api_key)

        resp = llm_evaluator.call_model(user_prompt)

        max_try = 5
        current_try = 0
        while current_try < max_try:
            try:
                json_code = extract_code_block("json", resp)
                if json_code is None:
                    resp = json.loads(resp)
                else:
                    resp = json.loads(json_code)
                logic, answer, explanation = resp["quality_scores"]
                # conf = resp["confidence"]
                suggestion = resp["suggestion"]
                return resp

            except Exception as e:
                # print(resp)
                resp = llm_evaluator.call_model(f"Invalid JSON format. Please respond again with correct JSON.\n Error: {e}")
                current_try += 1
                continue

        raise ValueError("Failed to get valid JSON response after multiple attempts.")

    # ---- Full pipeline ----
    def forward(self, query: str, question_id: str) -> str:
        print("Step 1 | Solver : Generating 5 solutions in parallel...")
        solutions: List[str] = ["" for _ in range(5)]
        with ThreadPoolExecutor(max_workers=5) as ex:
            fut2idx = {ex.submit(self._forward_solver, query): i for i in range(5)}
            for fut in as_completed(fut2idx):
                solutions[fut2idx[fut]] = fut.result()

        print("Step 2 | Critic : Critiquing each solution in parallel...")
        critic_solutions: List[str] = ["" for _ in range(5)]
        with ThreadPoolExecutor(max_workers=5) as ex:
            fut2idx = {ex.submit(self._forward_critic, query, solutions[i]): i for i in range(5)}
            for fut in as_completed(fut2idx):
                critic_solutions[fut2idx[fut]] = fut.result()

        print("Step 3 | Refine : Targeted repair using reference solutions...")
        refined_solutions: List[str] = ["" for _ in range(5)]
        with ThreadPoolExecutor(max_workers=5) as ex:
            fut2idx = {}
            for i in range(5):
                anchor = critic_solutions[i]
                refs = [critic_solutions[j] for j in range(5) if j != i]
                fut2idx[ex.submit(self._forward_refine, query, anchor, refs)] = i

            for fut in as_completed(fut2idx):
                refined_solutions[fut2idx[fut]] = fut.result()

        pass_solutions = []
        failed_solutions = deepcopy(refined_solutions)
        all_failed_solutions = []

        max_try = 1 ###### 3
        current_try = 0

        all_eval_results = []

        while current_try < max_try:

            eval_results = []
            with ThreadPoolExecutor(max_workers=5) as ex:
                fut2idx = {ex.submit(self._evaluate_quality, query, failed_solutions[i]): i for i in range(len(failed_solutions))}
                for fut in as_completed(fut2idx):
                    eval_results.append((fut2idx[fut], fut.result()))
            
            all_eval_results.append(eval_results)

            # print(eval_results)
            failed_sol_idx = []
            suggestions = {}
            for sol_idx, res in eval_results:
                logic, answer, explanation = res["quality_scores"]
                q_score = logic * 0.2  + answer * 0.6 + explanation * 0.2
                if q_score >= 3:
                    pass_solutions.append(failed_solutions[sol_idx])
                else:
                    failed_sol_idx.append(sol_idx)
                    suggestions[sol_idx] = res["suggestion"]

            current_try += 1
            failed_solutions = [failed_solutions[i] for i in failed_sol_idx]
            suggestions = [suggestions[i] for i in failed_sol_idx]

            if len(failed_solutions) == 0:
                break

            critic_falied_solutions = []
            with ThreadPoolExecutor(max_workers=len(failed_solutions)) as ex:
                fut2idx = {ex.submit(self._forward_critic, query, failed_solutions[i], suggestions[i]): i for i in range(len(failed_solutions))}
                for fut in as_completed(fut2idx):
                    critic_falied_solutions.append(fut.result())

            all_failed_solutions.append({
                "try": current_try,
                "failed_solutions": deepcopy(failed_solutions),
                "critic_falied_solutions": deepcopy(critic_falied_solutions),
            })

            failed_solutions = critic_falied_solutions

        pass_solutions = pass_solutions + failed_solutions

        assert len(pass_solutions) == 5, "There should be exactly 5 solutions after refinement and filtering."

        print("Step 4 | Selector : Selecting the best solution...")
        final_solution, selector_response = self._forward_selector(query, pass_solutions)

        if self.debug:
            dump = {
                "question_id": question_id,
                "query": query,
                "solutions": solutions,
                "critic_solutions": critic_solutions,
                "refined_solution": refined_solutions,
                "all_failed_solutions": all_failed_solutions,
                "pass_solutions": pass_solutions,
                "all_eval_results": all_eval_results,
                "final_solution": final_solution,
                "selector_response": selector_response,
            }
            os.makedirs(self.log_dir, exist_ok=True)
            with open(f"{self.log_dir}/pipeline_{question_id}.json", "w", encoding="utf-8") as f:
                json.dump(dump, f, ensure_ascii=False, indent=2)

        return final_solution


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Eigen1Agent with a query.")
    parser.add_argument('--query', type=str, default="What is the largest order of a non-cyclic torsion subgroup of an elliptic curve over $\\mathbb{Q}(\\sqrt{-3})$?", help='The query to process.')
    args = parser.parse_args()
    
    agent = Eigen1Agent()
    out = agent.forward(args.query)
    print("Final solution:", out)