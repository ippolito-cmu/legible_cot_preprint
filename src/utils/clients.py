import os
import json
import requests
import logging
from tqdm import tqdm
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from google.generativeai import GenerativeModel
import google.generativeai as genai
load_dotenv()
logging.getLogger('google').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
os.environ['GRPC_VERBOSITY'] = 'NONE'
os.environ['GRPC_TRACE'] = ''
os.environ['GOOGLE_CLOUD_DISABLE_GRPC_FOR_REST'] = 'true'
class GatewayClient:
    def __init__(self, api_key: str, base_url: str, default_temperature: float = 0.2, max_tokens: int = 8192):
        self.api_key = api_key
        self.base_url = base_url
        self.chat_endpoint = self.base_url + "/v1/chat/completions"
        self.responses_endpoint = self.base_url + "/v1/responses"
        self.default_temperature = default_temperature
        self.default_max_tokens = max_tokens
        self.supported_models = {
            "claude_37_sonnet": "claude-3-7-sonnet-20250219-v1:0",
            "claude_opus_4": "claude-opus-4-20250514-v1:0",
            "claude_sonnet_4": "claude-sonnet-4-20250514-v1:0",
            "o1_mini": "o1-mini-2024-09-12",
            "gpt_5": "gpt-5"
        }
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    def anthropic_payload(self, model, messages, thinking_budget=None, temperature=None):
        thinking_budget = thinking_budget if thinking_budget is not None else self.default_max_tokens
        temperature = temperature if temperature is not None else self.default_temperature
        assert model in self.supported_models, f"Model {model} not supported. Available options: {json.dumps(self.supported_models, indent=2)}"
        return {
            "model": self.supported_models[model],
            "messages": messages,
            "max_completion_tokens": int(thinking_budget * 1.5),
            "thinking": {
                "type": "enabled",
                "budget_tokens": thinking_budget
            }
        }
    def o1_payload(self, model, messages, max_new_tokens=None, temperature=None):
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.default_max_tokens
        temperature = temperature if temperature is not None else self.default_temperature
        assert model in self.supported_models, f"Model {model} not supported. Available options: {json.dumps(self.supported_models, indent=2)}"
        return {
            "model": self.supported_models[model],
            "input": messages,
            "max_output_tokens": max_new_tokens,
            "reasoning": {
                "summary": 'detailed'
            },
            "truncation": "auto"
        }
    def gpt5_payload(self, model, messages, max_new_tokens=None, temperature=None, reasoning_effort="medium"):
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.default_max_tokens
        temperature = temperature if temperature is not None else self.default_temperature
        assert model in self.supported_models, f"Model {model} not supported. Available options: {json.dumps(self.supported_models, indent=2)}"
        return {
            "model": self.supported_models[model],
            "input": messages,
            "max_output_tokens": max_new_tokens,
            "reasoning": {
                "effort": reasoning_effort,
                "summary": 'detailed'
            },
            "truncation": "auto"
        }
    def prepare_payload(self, model: str, messages: list, **kwargs):
        """
        Prepares the appropriate payload and endpoint based on the model type.
        Args:
            model (str): The model name.
            messages (list): The list of messages for the chat completion.
            **kwargs: Additional parameters specific to the model type.
        Returns:
            tuple: (endpoint, headers, payload)
        """
        if model.startswith("claude"):
            return self.chat_endpoint, self.headers, self.anthropic_payload(model, messages, **kwargs)
        elif model.startswith("o1"):
            return self.chat_endpoint, self.headers, self.o1_payload(model, messages, **kwargs)
        elif model == "gpt_5":
            return self.responses_endpoint, self.headers, self.gpt5_payload(model, messages, **kwargs)
        else:
            raise ValueError(f"Model {model} not supported. Available options: {json.dumps(self.supported_models, indent=2)}")
    def _response_wrapper(self, endpoint, headers, payload):
        response = requests.post(endpoint, headers=headers, json=payload, timeout = 60 * 5)
        response.raise_for_status()
        return response.json()
    def send_request(self, model: str, messages: list, index: int, **kwargs):
        endpoint, headers, payload = self.prepare_payload(model, messages, **kwargs)
        try:
            return {"index": index, "response": self._response_wrapper(endpoint, headers, payload)}
        except Exception as e:
            logger.error(f"An error occurred processing index {index} for model {model}: {str(e.last_attempt.exception())}")
            return {"error": str(e.last_attempt.exception()), "index": index}
    def collect_all_responses(self, model: str, results: dict, messages: list, max_workers = 25, timeout_seconds = 360, **kwargs):
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(self.send_request, model, message, i, **kwargs) for i, message in enumerate(messages)]
                try:
                    for future in tqdm(as_completed(futures, timeout=timeout_seconds), total=len(futures), desc=f"Processing {model}"):
                        result = future.result()
                        if 'error' in result:
                            results[result['index']] = result
                        else:
                            results[result['index']] = result['response']
                except TimeoutError:
                    print(f"TimeoutError: One or more tasks for model {model} did not complete within {timeout_seconds} seconds.")
            return results
    def collect_sequential_responses(self, model: str, results: dict, messages: list, max_workers = 25, timeout_seconds = 360, **kwargs):
            for i, message in enumerate(tqdm(messages, desc=f"Processing {model} Sequentially")):
                result = self.send_request(model, message, i, **kwargs)
                if 'error' in result:
                    results[result['index']] = result
                else:
                    results[result['index']] = result['response']
            return results
    class GeminiClient:
        def __init__(self, api_key: str):
            self.api_key = api_key
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        def __call__(self, prompt, system_prompt, model_name="gemini-2.5-flash-lite"):
            """Calls Gemini API with retry logic for answer extraction."""
            model = GenerativeModel(model_name, system_instruction=system_prompt)
            response = model.generate_content(prompt)
            return response.text.strip()
        def configure(self):
            pass
    @retry(wait=wait_random_exponential(multiplier=1, min=4, max=120), stop=stop_after_attempt(5))
    def _process_prompt(self, prompt, system_prompt, model_name="gemini-2.5-flash-lite"):
        return self(prompt, system_prompt, model_name=model_name)
    def process_prompt(self, prompt, system_prompt, model_name="gemini-2.5-flash-lite"):
        """Wrap processor with exception handler."""
        try:
            return self._process_prompt(self, prompt, system_prompt, model_name=model_name)
        except Exception as e:
            logger.error(f"An error occurred processing {prompt} for model {model_name}: {str(e.last_attempt.exception())}")
            return {"error": str(e.last_attempt.exception())}
