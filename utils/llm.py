"""
Author: Yujian Gan
Description: 
    This script is for calling different LLMs.
"""

import openai
from openai import OpenAI, AzureOpenAI
import os, re, json, copy
from pathlib import Path
from utils.log import llm_log
import time
from datetime import datetime

import pickle
from collections import OrderedDict

import uuid
from urllib.parse import urlparse

class LLM:
    def __init__(self, cache = None) -> None:
        self.cache_path = cache
        self.cache = None
        self.cache_capacity = 100000
        if self.cache_path:
            self.session_id = str(uuid.uuid4())
            if os.path.exists(self.cache_path):
                with open(self.cache_path, 'rb') as file:
                    self.cache = pickle.load(file)
            else:
                self.cache = OrderedDict()


    def extract_json_string(self, input_string):
        def process_colons_string(input_string, colon_positions):
            def find_string_bounds(s, start_pos, next_pos=None):
                colon_pos = s.find(':', start_pos)
                quote_start = colon_pos + 1
                
                while quote_start < len(s) and s[quote_start] in ' \t\n':
                    quote_start += 1
                
                if s[quote_start] != '"':
                    return s
                
                string_start = quote_start
                
                if next_pos:
                    substring = s[string_start:next_pos]
                    quote_end = next_pos
                    escaped = 0
                    while quote_end > string_start:
                        quote_end -= 1
                        if s[quote_end] == '"' and escaped != 2:
                            escaped += 1
                        elif s[quote_end] == '"' and escaped == 2:
                            break
                else:
                    substring = s[string_start:]
                    quote_end = len(s)
                    while quote_end > string_start:
                        quote_end -= 1
                        if s[quote_end] == '"':
                            break
                        
                actual_string = s[string_start+1:quote_end]
               
                escaped_string = ""
                is_previous_backslash = False
            
                for char in actual_string:
                    if char == '"' and not is_previous_backslash:
                        escaped_string += '\\"'
                    else:
                        escaped_string += char
                    is_previous_backslash = (char == '\\')

                new_string = s[:string_start+1] + escaped_string + s[quote_end:]
                
                return new_string
            
            result_string = input_string
            length_change = 0
            for i, pos in enumerate(colon_positions):
                end_pos = colon_positions[i+1] if i+1 != len(colon_positions) else None
                if end_pos:
                    result_string = find_string_bounds(result_string, pos + length_change, end_pos+ length_change)
                else:
                    result_string = find_string_bounds(result_string, pos + length_change)
                length_change = len(result_string) - len(input_string)
            return result_string

        def find_colons(input_string):
            pattern = r'"\s*:\s*'
            matches = []
            for match in re.finditer(pattern, input_string):
                colon_pos = match.start() + match.group().find(':')
                matches.append(colon_pos)
            return matches
        
        def clean_json_comma(json_str):
            cleaned_str = re.sub(r',\s*(\}|\])', r'\1', json_str)
            return cleaned_str

        def clean_json_string(json_str):
            def fix_json_string(json_str):
                def replace_newlines(match):
                    content = match.group(1)
                    content_fixed = content.replace("\n", "\\n")
                    return f'"{content_fixed}"'
                
                fixed_str = re.sub(r'"(.*?)"', replace_newlines, json_str, flags=re.DOTALL)
                return fixed_str
            json_str = fix_json_string(json_str)
            json_str = clean_json_comma(json_str)
            json_str = re.sub(r"'(?!s\b)", '"', json_str)
            json_str = re.sub(r'\bTrue\b', 'true', json_str)
            json_str = re.sub(r'\bFalse\b', 'false', json_str)  
            json_str = json_str.replace("\n", " ")
            json_str = re.sub(r'\(\s*("(?:[^"\\]|\\.)*?"|\w+)\s*,\s*(\d+|null|None)\s*\)', r'[\1, \2]', json_str)
            json_str = re.sub(r'\bNone\b', 'null', json_str, flags=re.IGNORECASE)
            colon_positions = find_colons(json_str)
            json_str = process_colons_string(json_str, colon_positions)
            return json_str

        start = -1
        brace_count = 0

        try:
            json.loads(input_string)
            return input_string
        except json.JSONDecodeError as e:
            pass
        
        for i, char in enumerate(input_string):
            if char == '{':
                if brace_count == 0:
                    start = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start != -1:
                    json_str = input_string[start:i+1]
                    json_str = clean_json_string(json_str)
                    
                    try:
                        json.loads(json_str)
                        return json_str
                    except json.JSONDecodeError as e:
                        print(json_str)
                        pass
                    start = -1
        return ''

    def from_cache(self, message, json_format=False, skip_cache=False):
        if self.cache:
            message_str = str(message)
            if message_str not in self.cache and type(message) == list and len(message) == 1 and message[0]['role'] == 'user':
                message_str = message[0]['content']
            if message_str in self.cache:
                refresh = self.cache[message_str]
                self.cache.pop(message_str)
                if skip_cache or refresh is None:
                    return None
                self.cache[message_str] = refresh
                if json_format:
                    return self.extract_json_string(refresh)
                else:
                    return refresh
        return None
    
    def save_to_cache(self, message, response):
        if self.cache_path and response:
            message_str = str(message)
            if type(message) == list and len(message) == 1 and message[0]['role'] == 'user':
                self.cache[message[0]['content']] = response
            self.cache[message_str] = response
            with open(self.cache_path, 'wb') as file:
                pickle.dump(self.cache, file)
            if len(self.cache) > self.cache_capacity:
                self.cache.popitem(last=False)
    
    def cache_exchange(self, message_old, message_new):
        if self.cache:
            message_str = str(message_old)
            message_str_new = str(message_new)
            if message_str in self.cache:
                refresh = self.cache[message_str]
                self.cache.pop(message_str) 
                self.cache[message_str_new] = refresh
                with open(self.cache_path, 'wb') as file:
                    pickle.dump(self.cache, file)

    def cache_copy(self, message_old, message_new):
        if self.cache:
            message_str = str(message_old)
            message_str_new = str(message_new)
            if message_str in self.cache:
                self.cache[message_str_new] = self.cache[message_str]
                with open(self.cache_path, 'wb') as file:
                    pickle.dump(self.cache, file)

    def request(self, prompt, stop, **kwargs):
        return
    
    def log(self, input, output, **kwargs):
        if "usage" in kwargs and self.cache_path and kwargs['usage'] and isinstance(kwargs['usage'], (list, tuple)) and len(kwargs['usage']) >= 3:
            if self.session_id in self.cache:
                self.cache[self.session_id][0] += kwargs['usage'][0]
                self.cache[self.session_id][1] += kwargs['usage'][1]
                self.cache[self.session_id][2] += kwargs['usage'][2]
                self.cache[self.session_id][3] += 1
            else:
                self.cache[self.session_id] = kwargs['usage'] + [1]
        llm_log(input, output, **kwargs)
        pass


    def get_and_clear_usage(self):
        if self.cache and self.session_id in self.cache:
            usage = self.cache[self.session_id]
            self.cache.pop(self.session_id)
            return usage
        return None


class ChatGPT(LLM):
    def __init__(self, name, cache = None, **kwargs) -> None:
        super().__init__(cache)
        if '@@@' in name:
            url, name_ = name.split('@@@')
        else:
            url = None
            name_ = name.lower() 
        if name_ in ["gpt-4","gpt4"]:
            self.model_name = "gpt-4-0613"
        else:
            self.model_name = name

        if 'reasoning_model' in kwargs and kwargs['reasoning_model']:
            self.reasoning_model = kwargs['reasoning_model']
        else:
            self.reasoning_model = None # o3-mini-2025-01-31

        if self.model_name.startswith("gpt-5") or self.model_name.startswith("o"):
            self.force_reasoning = True
            if not self.reasoning_model:
                self.reasoning_model = self.model_name
        else:
            self.force_reasoning = False


        if "OPENAI_API_KEY" in os.environ:
            openai.api_key = os.environ["OPENAI_API_KEY"]
            if 'base_url' in kwargs and kwargs['base_url']:
                self.client = OpenAI(base_url=kwargs['base_url'])
            else:
                self.client = OpenAI()
        
        if 'reasoning_effort' in kwargs and kwargs['reasoning_effort']:
            self.reasoning_effort = kwargs['reasoning_effort']
        elif self.model_name.startswith("gpt-5"):
            self.reasoning_effort = "minimal"
        else:
            self.reasoning_effort = "medium"

        if 'max_tokens' in kwargs and kwargs['max_tokens']:
            self.max_tokens = kwargs['max_tokens']
        else:
            self.max_tokens = 2048

        self.use_cache = True


    def generate_message(self, prompt, **kwargs):
        if type(prompt) == list and 'role' in prompt[0]:
            message = prompt
        else:   
            message = [{
                        "role": "user",
                        "content": prompt
                    }]            
        if 'previous_message' in kwargs and kwargs['previous_message']:
            kwargs['previous_message'].extend(message)
            message = kwargs['previous_message']
        return message

        
    def request(self, prompt, stop, **kwargs):
        message = self.generate_message(prompt, **kwargs)
        
        reasoning_effort = kwargs['reasoning_effort'] if 'reasoning_effort' in kwargs and kwargs['reasoning_effort'] else self.reasoning_effort
        max_tokens = kwargs['max_tokens'] if 'max_tokens' in kwargs and kwargs['max_tokens'] else self.max_tokens
        model_name = self.reasoning_model if (self.reasoning_model and 'reasoning_effort' in kwargs and kwargs['reasoning_effort']) or self.force_reasoning else self.model_name
        json_format = True if 'json_format' in kwargs and kwargs['json_format'] and model_name != "deepseek-reasoner" else False
        temperature = kwargs['temperature'] if 'temperature' in kwargs else 0
        skip_cache = kwargs['skip_cache'] if 'skip_cache' in kwargs and kwargs['skip_cache'] else False
        
        if self.use_cache:
            response = self.from_cache(message, json_format, skip_cache)
            if response:
                message.append({"role": "assistant", "content": response})
                # return response, message
                return (self.extract_json_string(response), message) if 'json_format' in kwargs and kwargs['json_format'] else (response, message)

        # try:
        if self.force_reasoning:
            completion = self.client.chat.completions.create(
                model=model_name,
                reasoning_effort=reasoning_effort,
                messages=message,
                stop = stop,
                seed = 8848,
                max_completion_tokens = max_tokens,
                **({"response_format": {"type": "json_object"}} if json_format else {})
            )
        else:
            completion = self.client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                messages=message,
                stop = stop,
                seed = 8848,
                max_tokens = max_tokens,
            **({"response_format": {"type": "json_object"}} if json_format else {})
            )
        time.sleep(0.5)
        super().log(message, completion.choices[0].message.content, model=completion.model, system_fingerprint = completion.system_fingerprint, usage = [completion.usage.prompt_tokens, completion.usage.completion_tokens, completion.usage.total_tokens])
        self.save_to_cache(message, completion.choices[0].message.content)
        
        message.append({"role": "assistant", "content": completion.choices[0].message.content})
        
        return (self.extract_json_string(completion.choices[0].message.content), message) if 'json_format' in kwargs and kwargs['json_format'] else (completion.choices[0].message.content, message)

    def cache_exchange(self, old_prompt, new_prompt):
        message_old = self.generate_message(old_prompt)
        message_new = self.generate_message(new_prompt)
        super().cache_exchange(message_old, message_new)
            
    def cache_copy(self, old_prompt, new_prompt):
        message_old = self.generate_message(old_prompt)
        message_new = self.generate_message(new_prompt)
        super().cache_copy(message_old, message_new) 

class QianFan(LLM):
    def __init__(self, name, cache = None) -> None:
        import qianfan
        super().__init__(cache)
        self.chat_comp = qianfan.ChatCompletion(model=name)
        # self.chat_comp = qianfan.ChatCompletion(model="ERNIE-Bot-turbo")

    def request(self, prompt, stop, **kwargs):
        message = [{
                    "role": "user",
                    "content": prompt
                }]
        if 'previous_message' in kwargs and kwargs['previous_message']:
            kwargs['previous_message'].extend(message)
            message = kwargs['previous_message']

        response = self.from_cache(message)
        if response:
            message.append({"role": "assistant", "content": response})
            return response, message
        
        try:
            completion = self.chat_comp.do(messages=message, top_p=1, temperature=0.0000001, penalty_score=1.0)
            time.sleep(0.5)
        except:
            time.sleep(5)
            completion = self.chat_comp.do(messages=message, top_p=1, temperature=0.0000001, penalty_score=1.0)
            time.sleep(0.5)
        super().log(message, completion.body['result'], model=self.chat_comp._model, system_fingerprint = completion.body['id'], usage = [completion.body['usage']['prompt_tokens'], completion.body['usage']['completion_tokens'], completion.body['usage']['total_tokens']])
        self.save_to_cache(message, completion.body['result'])
        
        message.append({"role": "assistant", "content": completion.body['result']})

        return completion.body['result'], message




class AWSBedrockLLAMA(LLM):
    def __init__(self, name, cache = None) -> None:
        import boto3
        from botocore.exceptions import ClientError
        super().__init__(cache)
        # name = name.lower()
        if name.lower() == 'llama3.1-405b':
            self.model_name = "meta.llama3-1-405b-instruct-v1:0"
        else:
            self.model_name = name
        session = boto3.Session(
            aws_access_key_id=os.environ["aws_access_key_id"],
            aws_secret_access_key=os.environ["aws_secret_access_key"],
            region_name=os.environ["region_name"]
        )
        # Create a Bedrock Runtime client in the AWS Region you want to use.
        self.client = session.client("bedrock-runtime")

    def request(self, prompt, stop, **kwargs):
        message = [{
                    "role": "user",
                    "content": [{"text": prompt}],
                }]
        
        system_prompts = None
        message_for_cache = message

        if 'previous_message' in kwargs and kwargs['previous_message']:
            kwargs['previous_message'].extend(message)
            message = kwargs['previous_message']
            if message[0]['role'] == 'system':
                system_prompts = [{"text": message[0]['content']}]
                message_for_cache = copy.deepcopy(message)
                del message[0]
            for m in message:
                if type(m['content']) == str:
                    m['content'] = [{"text": m['content']}]
        if 'max_tokens' in kwargs and kwargs['max_tokens']:
            max_tokens = kwargs['max_tokens']
        else:
            max_tokens = 1000

        response = self.from_cache(message_for_cache)
        
        if response:
            message.append({"role": "assistant", "content": [{"text": response}]})
            return (self.extract_json_string(response), message) if 'json_format' in kwargs and kwargs['json_format'] else (response, message)
        
        temperature = 0.00001
        topP = 0.9999
        while True:
            try:
                # Send the message to the model, using a basic inference configuration.
                if system_prompts:
                    response = self.client.converse(
                        modelId=self.model_name,
                        messages=message,   system = system_prompts,
                        inferenceConfig={"maxTokens":max_tokens,"temperature":temperature,"topP":topP},
                        additionalModelRequestFields={}
                    )
                else:
                    response = self.client.converse(
                        modelId=self.model_name, messages=message,
                        inferenceConfig={"maxTokens":max_tokens,"temperature":temperature,"topP":topP},additionalModelRequestFields={}
                    )

                # Extract and print the response text.
            except (ClientError, Exception) as e:
                time.sleep(10)
                if system_prompts:
                    response = self.client.converse(
                        modelId=self.model_name,
                        messages=message, system = system_prompts,
                        inferenceConfig={"maxTokens":max_tokens,"temperature":temperature,"topP":topP},
                        additionalModelRequestFields={}
                )
                else:
                    response = self.client.converse(
                        modelId=self.model_name, messages=message,
                        inferenceConfig={"maxTokens":max_tokens,"temperature":temperature,"topP":topP},additionalModelRequestFields={}
                    )
                time.sleep(0.5)
            temperature += 0.333 
            response_text = response["output"]["message"]["content"][0]["text"].strip()
            super().log(message_for_cache, response_text, system_fingerprint = response['stopReason'], usage = [response['usage']['inputTokens'], response['usage']['outputTokens'], response['usage']['totalTokens']])
            
            if 'json_format' in kwargs and kwargs['json_format'] and not self.extract_json_string(response_text):
                assert temperature < 1
            else:
                break        
        self.save_to_cache(message_for_cache, response_text)
        
        message.append({"role": "assistant", "content": [{"text": response_text}]})
        
        return (self.extract_json_string(response_text), message) if 'json_format' in kwargs and kwargs['json_format'] else (response_text, message)




class ChatGPTBatch(ChatGPT):
    def __init__(self, name, cache = None, **kwargs) -> None:
        super().__init__(name, cache, **kwargs)
        batch_folder = 'log/gpt_batch'
        Path(batch_folder).mkdir(parents=True, exist_ok=True) # create `log/gpt_batch` dir if not exist
        batch_cache_path = f'{batch_folder}/gpt_cache.pkl'
        if os.path.exists(batch_cache_path):
            with open(batch_cache_path, 'rb') as file:
                batch_cache = pickle.load(file)
        else:
            batch_cache = OrderedDict()
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        batch_cache[current_time] = {'state':'create', 'cache_path':cache}
        with open(batch_cache_path, 'wb') as file:
            pickle.dump(batch_cache, file)
        self.count = 0
        self.save_jsonl_path = f"log/gpt_batch/{current_time}.jsonl"

    def request(self, prompt, stop, **kwargs):
        message = self.generate_message(prompt, **kwargs)
        reasoning_effort = kwargs['reasoning_effort'] if 'reasoning_effort' in kwargs and kwargs['reasoning_effort'] else self.reasoning_effort
        max_tokens = kwargs['max_tokens'] if 'max_tokens' in kwargs and kwargs['max_tokens'] else self.max_tokens
        model_name = self.reasoning_model if (self.reasoning_model and 'reasoning_effort' in kwargs and kwargs['reasoning_effort']) or self.force_reasoning else self.model_name
        json_format = True if 'json_format' in kwargs and kwargs['json_format'] and model_name != "deepseek-reasoner" else False
        temperature = kwargs['temperature'] if 'temperature' in kwargs else 0
        skip_cache = kwargs['skip_cache'] if 'skip_cache' in kwargs and kwargs['skip_cache'] else False
        response = self.from_cache(message, json_format, skip_cache)
        if response:
            message.append({"role": "assistant", "content": response})
            return response, message
        
        
        example = {}
        example["custom_id"] = "request-{}".format(self.count)
        example["method"] = "POST"
        example["url"] = "/v1/chat/completions"
        body = {}
        body["model"] = model_name
        body["messages"] = message
        if self.force_reasoning:
            body["reasoning_effort"] = reasoning_effort
            body["max_completion_tokens"] = max_tokens
        else:
            body["max_tokens"] = max_tokens
            body["temperature"] = temperature
        if json_format:
            body["response_format"] = {"type": "json_object"}
        example["body"] = body
        self.count += 1

        with open(self.save_jsonl_path, "a") as jsonl_file:
            jsonl_file.write(json.dumps(example) + "\n")

        if 'tag' in kwargs and kwargs['tag']:
            example["yg_tag"] = kwargs['tag']
            with open(self.save_jsonl_path[:-5] + "yg_tag.jsonl", "a") as jsonl_file:
                jsonl_file.write(json.dumps(example) + "\n")
        
        message.append({"role": "assistant", "content": ''})
        
        return '', message



class LlamaFactory(LLM):
    def __init__(self, name = "8000", cache = None) -> None:
        super().__init__(cache)
        
        if name.startswith("http"):
            self.client = OpenAI(api_key="0", base_url=name)
            self.model_name = 'unknown'
        else:
            if ":" in name:
                name, port = name.split(":")
            elif name.isdigit():
                port = name
                name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
            else:
                port = "8000"
            self.client = OpenAI(api_key="0", base_url=f"http://0.0.0.0:{port}/v1")
            self.model_name = name


    def request(self, prompt, stop, **kwargs):
        if type(prompt) == list and 'role' in prompt[0]:
            message = prompt
        else:   
            message = [{
                        "role": "user",
                        "content": prompt
                    }]
        if 'previous_message' in kwargs and kwargs['previous_message']:
            kwargs['previous_message'].extend(message)
            message = kwargs['previous_message']
        if 'max_tokens' in kwargs and kwargs['max_tokens']:
            max_tokens = kwargs['max_tokens']
        else:
            max_tokens = 2048
        json_format = True if 'json_format' in kwargs and kwargs['json_format'] else False
        temperature = kwargs['temperature'] if 'temperature' in kwargs else 0

        response = self.from_cache(message, json_format)
        if response:
            message.append({"role": "assistant", "content": response})
            return response, message
        
        completion = self.client.chat.completions.create(messages=message, model=self.model_name, temperature=temperature, max_tokens=max_tokens)
        super().log(message, completion.choices[0].message.content)
        self.save_to_cache(message, completion.choices[0].message.content)

        return (self.extract_json_string(completion.choices[0].message.content), message) if 'json_format' in kwargs and kwargs['json_format'] else (completion.choices[0].message.content, message)



class DeepSeek(ChatGPT):
    def __init__(self, name, cache = None, **kwargs) -> None:
        super().__init__(name, cache, **kwargs)
        # LLM.__init__(self, cache)
        name_ = name.lower()
        if name_ in ["v3"]:
            self.model_name = "deepseek-chat"
        else:
            self.model_name = name
        self.client = OpenAI(api_key=os.environ["DeepSeek_API_KEY"], base_url="https://api.deepseek.com")


class Requesty(ChatGPT):
    def __init__(self, name, cache = None, **kwargs) -> None:
        super().__init__(name, cache, **kwargs)
        self.model_name = name
        self.client = OpenAI(api_key=os.environ["Requesty_API_KEY"], base_url="https://router.requesty.ai/v1")




class AzureAI(ChatGPT):
    def __init__(self, name, cache = None, **kwargs) -> None:
        super().__init__(name, cache, **kwargs)
        self.model_name = name
        self.client = AzureOpenAI(
            azure_endpoint = "https://agcohn-eastus-instance.openai.azure.com/", 
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
            api_version="2024-02-01"
            )


class Gemini(LLM):
    def __init__(self, name, cache = None, **kwargs) -> None:
        super().__init__(cache)
        from google import genai
        self.model_name = name
        if 'reasoning_model' in kwargs and kwargs['reasoning_model']:
            self.reasoning_model = kwargs['reasoning_model']
        else:
            self.reasoning_model = None 
        
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.force_reasoning = False
        
    def request(self, prompt, stop, **kwargs):
        if type(prompt) == list and len(prompt) > 2:
            raise NotImplementedError("Gemini does not support conversation mode.")
        message = str(prompt)
        message_return = [{'role': 'user', 'content': message}]
        
        reasoning_effort = kwargs['reasoning_effort'] if 'reasoning_effort' in kwargs and kwargs['reasoning_effort'] else "medium"
        max_tokens = kwargs['max_tokens'] if 'max_tokens' in kwargs and kwargs['max_tokens'] else 2048
        model_name = self.reasoning_model if (self.reasoning_model and 'reasoning_effort' in kwargs and kwargs['reasoning_effort']) or self.force_reasoning else self.model_name
        json_format = True if 'json_format' in kwargs and kwargs['json_format'] and model_name != "deepseek-reasoner" else False
        temperature = kwargs['temperature'] if 'temperature' in kwargs else 0
        

        response = self.from_cache(message, json_format)
        if response:
            message_return.append({"role": "assistant", "content": response})
            # return response, message
            return (self.extract_json_string(response), message_return) if 'json_format' in kwargs and kwargs['json_format'] else (response, message_return)

        response = self.client.models.generate_content(
            model=model_name, contents=prompt
        )
        super().log(message_return, response.text, model=self.model_name, usage = [response.usage_metadata.prompt_token_count, (response.usage_metadata.thoughts_token_count or 0)+response.usage_metadata.candidates_token_count, response.usage_metadata.total_token_count])
        self.save_to_cache(message, response.text)
        
        message_return.append({"role": "assistant", "content": response.text})
        
        return (self.extract_json_string(response.text), message_return) if 'json_format' in kwargs and kwargs['json_format'] else (response.text, message_return)
        # return completion.choices[0].message.content, message


class Requesty_Gemini(Gemini):
    def __init__(self, name, cache = None, **kwargs) -> None:
        super().__init__(name, cache, **kwargs)
        name_ = name.lower()
        self.model_name = name
        self.client = OpenAI(api_key=os.environ["Requesty_API_KEY"], base_url="https://router.requesty.ai/v1")

