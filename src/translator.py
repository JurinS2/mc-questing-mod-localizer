from __future__ import annotations
from typing import Dict, List, Tuple

import re
import json
import asyncio
import logging
from abc import ABC, abstractmethod

from streamlit.delta_generator import DeltaGenerator
from flatten_json import flatten, unflatten_list
from tenacity import retry, stop_after_attempt, wait_exponential
from json_repair import repair_json

import googletrans
import deepl
import tiktoken
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI

from src.utils import get_session_id
from src.constants import MINECRAFT_TO_DEEPL, MINECRAFT_TO_GOOGLE

def escape_text(text: str) -> str:
    '''Escape special characters to prevent translation API from altering them.'''
    text = re.sub(r"(\\n)", r"<br>", text) # escape newline
    text = re.sub(r"(&[0-9a-z])", lambda x: f"<{x.group(0)[1:]}>", text) # escape color code
    return text

def unescape_text(text: str) -> str:
    '''Unescape special characters to restore original format.'''
    text = re.sub(r"(<[0-9a-zA-Z]>)", lambda x: f"&{x.group(0)[1:-1].lower()}", text) # restore color code
    text = re.sub(r"&(?=[^0-9a-z]|$)", r"\&", text) # escape single &
    text = re.sub(r"(<br>|<BR>)", r"\\n", text) # restore newline
    return text

def make_batches(lang_dict: Dict, max_tokens: int) -> List[Dict]:
    '''Create batches of key-value pairs from a dictionary without exceeding max token limit.'''
    logger = logging.getLogger(f"make_batches ({get_session_id()})")
    
    batches = []
    current_batch = {}
    current_tokens = 0
    enc = tiktoken.encoding_for_model("gpt-4")

    for key, value in lang_dict.items():
        pair_str = json.dumps({key: value}, ensure_ascii=False)
        tokens = len(enc.encode(pair_str)) # count tokens
        if current_tokens + tokens > max_tokens and current_batch:
            batches.append(current_batch) # append current batch
            current_batch = {}
            current_tokens = 0

        current_batch[key] = value # add to current batch if not exceeding max tokens
        current_tokens += tokens
    
    if current_batch: # append the last batch
        batches.append(current_batch)
        
    logger.info("Created %d batches", len(batches))
    return batches

def concat_batches(batches: List[Dict]) -> Dict:
    '''Concatenate a list of dictionaries into a single dictionary.'''
    logger = logging.getLogger(f"concat_batches ({get_session_id()})")
    
    output = {}
    for batch in batches:
        output.update(batch)
    
    logger.info("Concatenated %d batches", len(batches))
    return output

def split_batch(batch: Dict) -> Tuple[Dict, Dict]:
    '''Split a batch into two dictionaries: one to keep unchanged, one to translate.'''
    batch_keep = {}
    batch_translate = {}

    for key, value in batch.items():
        keep_cond = not isinstance(value, str) \
                    or (value.startswith("[") and value.endswith("]")) \
                    or (value.startswith("{") and value.endswith("}"))
        if keep_cond:
            batch_keep[key] = value
        else:
            batch_translate[key] = value

    return batch_keep, batch_translate

def validate_and_update(source_dict: Dict, target_dict: Dict, translated_dict: Dict) -> List[str]:
    '''Validate translated dictionary and update target dictionary. Return error log.'''
    logger = logging.getLogger(f"validate_and_update ({get_session_id()})")
    
    error_log = []
    for key in source_dict.keys():
        if translated_dict.get(key) is None: # Missing key
            error_log.append(f"Missing translation: {key}")
            continue
        if isinstance(source_dict[key], list): # Invalid list length
            if not isinstance(translated_dict[key], list) or len(source_dict[key]) != len(translated_dict[key]):
                error_log.append(f"Invalid translation: {key}")
                continue
        target_dict[key] = translated_dict[key] # Update target_lang_dict only for valid keys
    
    logger.info("Validation completed with %d errors", len(error_log))
    return error_log
    
class TranslationManager:
    def __init__(self, translator: BaseTranslator):
        self.translator = translator
        self.logger = logging.getLogger(f"{self.__class__.__qualname__} ({get_session_id()})")

    async def __call__(self, source_dict: Dict, target_dict: Dict, target_lang: str, status: DeltaGenerator):
        source_dict_flatten = flatten(source_dict, separator="|") # Flatten json
        batches = make_batches(source_dict_flatten, max_tokens=6000) # 6000 tokens max
        batches_out = await self.translator.translate(batches, target_lang, status)
        translated_dict = unflatten_list(concat_batches(batches_out), separator="|")
        error_log = validate_and_update(source_dict, target_dict, translated_dict)
        
        if error_log:
            status.write('**Error Log**')
            status.code('\n'.join(error_log), language=None, line_numbers=True, height=300) # display error log

class BaseTranslator(ABC):
    def __init__(self, auth_key: str = None):
        self.logger = logging.getLogger(f"{self.__class__.__qualname__} ({get_session_id()})")

    async def translate(self, batches: List[Dict], target_lang: str, status: DeltaGenerator) -> List[Dict]:
        semaphore = asyncio.Semaphore(4) # concurrency limit
        progress_bar = status.progress(0, "Translating...")
        
        async def wrap_translate(idx, total, batch):
            async with semaphore:
                await asyncio.sleep(5) # 5 sec delay

                try:
                    progress_bar.progress(idx / total, f"Translating... ({idx}/{total})")
                    self.logger.info("Translating batch (%d/%d)", idx, total)
                    return await self._translate(batch, target_lang)
                except Exception:
                    self.logger.error("Failed to translate batch (%d/%d)", idx, total, exc_info=True)
                    return {} # return empty dict on failure

        tasks = [wrap_translate(idx, len(batches), batch) for idx, batch in enumerate(batches, start=1)]
        batches_out = await asyncio.gather(*tasks)

        progress_bar.empty()
        self.logger.info("Translated %d batches", len(batches_out))
        
        return batches_out

    @abstractmethod
    async def _translate(self, batch: Dict, target_lang: str) -> Dict:
        pass

class GoogleTranslator(BaseTranslator):
    def __init__(self, auth_key: str = None):
        self.translator = googletrans.Translator()
        super().__init__()
    
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=4, max=64), reraise=True)
    async def _translate(self, batch: Dict, target_lang: str) -> Dict:
        batch_keep, batch_translate = split_batch(batch) # split dict
        batch_input = [escape_text(value) for value in batch_translate.values()] # values to translate
        batch_output = {}

        if batch_input:
            batch_output = await self.translator.translate(batch_input, dest=MINECRAFT_TO_GOOGLE[target_lang])
            batch_output = {key: unescape_text(value.text) for key, value in zip(batch_translate.keys(), batch_output)}
        return batch_keep | batch_output

class DeepLTranslator(BaseTranslator):
    def __init__(self, auth_key: str):
        self.translator = deepl.DeepLClient(auth_key)
        super().__init__()

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=4, max=64), reraise=True)
    async def _translate(self, batch: Dict, target_lang: str) -> Dict:
        batch_keep, batch_translate = split_batch(batch)
        batch_input = [escape_text(value) for value in batch_translate.values()] # values to translate
        batch_output = {}

        if batch_input:
            batch_output = await asyncio.to_thread(
                self.translator.translate_text,
                text=batch_input,
                target_lang=MINECRAFT_TO_DEEPL[target_lang],
                context="This is a Minecraft quest text, so please keep the color codes and formatting intact. Example of color codes: <a>, <b>, <1>, <2>, <l>, <r>. Example of formatting: <br>. Example Translation: <a>Hello <br><b>Minecraft! -> <a>안녕하세요 <br><b>마인크래프트!",
                preserve_formatting=True
            )
            batch_output = {key: unescape_text(value.text) for key, value in zip(batch_translate.keys(), batch_output)}
        return batch_keep | batch_output

class GeminiTranslator(BaseTranslator):
    def __init__(self, auth_key: str):
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=auth_key,
            temperature=0
        )
        content_extractor = RunnableLambda(lambda msg: getattr(msg, 'content', '') if isinstance(msg, AIMessage) else str(msg)) # extract content
        json_extractor = RunnableLambda(self.extract_json) # extract json string
        json_parser = JsonOutputParser() # parse json
        prompt = PromptTemplate(
            template="""You are a Minecraft modpack quest translation assistant.
            Your task is to translate the given JSON-formatted text, while keeping the original JSON structure.
            Be aware that what you are translating is a quest text for Minecraft modpack.
            The property names in the JSON must remain UNCHANGED and enclosed in DOUBLE QUOTES.
            You must keep the color codes INTACT. Example of color codes: &a, &b, &1, &2, &l, &r.
            You must keep the new line symbol (\\n) INTACT.
            Text enclosed in [] or {{}} must be kept UNCHANGED.
            If there are words that are difficult or ambiguous to translate, translate them PHONETICALLY. Also, translate proper nouns PHONETICALLY.
            Translation Examples (en_us -> ko_kr):
            - &aDiamond Pickaxe&r -> &a다이아몬드 곡괭이&r
            - {{@pagebreak}} -> {{@pagebreak}}
            - While the &aUpgrade Template&r is not needed to make the initial tool, it will save you a lot of &6Allthemodium Ingots&r! -> &a업그레이드 템플릿&r은 초기 도구를 만드는 데 필요하지 않지만, &6올더모듐 주괴&r를 많이 절약할 수 있습니다!
            Your output must follow these format instructions: {format_instructions}
            Translate the following JSON-formatted text to {target_lang}:
            ```json
            {query}
            ```""",
            input_variables=["target_lang", "query"],
            partial_variables={"format_instructions": json_parser.get_format_instructions()}
        )
        self.translator = prompt | llm | content_extractor | json_extractor | json_parser
        self.handler = LLMCallbackHandler(self.__class__.__qualname__)
        super().__init__()
    
    @staticmethod
    def extract_json(text: str) -> Dict:
        '''
        Copyright 2025 moonzoo
        
        https://mz-moonzoo.tistory.com/m/89
        '''
        if isinstance(text, str):
            # find json code block
            if text.strip().startswith("```json"):
                start_block = text.find("{")
                end_block = text.rfind("}")
                if start_block != -1 and end_block != -1 and start_block < end_block:
                    json_str = text[start_block:end_block+1]
                    return repair_json(json_str)

            # find the last potential json string
            start = text.rfind('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and start < end:
                json_str = text[start:end+1]
                return repair_json(json_str)
            elif text.strip() == '{}':
                return "{}"
            else:
                raise ValueError("Invalid JSON format.")
        else:
            raise ValueError("Input must be a string.")

    # Langchain automatically retries failed requests
    async def _translate(self, batch: Dict, target_lang: str) -> Dict:
        batch_output = await self.translator.ainvoke(
            {
                "target_lang": target_lang,
                "query": json.dumps(batch, ensure_ascii=False)
            },
            config={
                "callbacks": [self.handler],
                "verbose": True
            }
        )
        return batch_output

class LLMCallbackHandler(BaseCallbackHandler):
    def __init__(self, cls_name, *args, **kwargs):
        self.logger = logging.getLogger(f"{cls_name} ({get_session_id()})")
        super().__init__(*args, **kwargs)

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.logger.info("LLM started: %s", serialized)

    def on_llm_error(self, error, **kwargs):
        self.logger.error("LLM error: %s", error)

    def on_retry(self, retry_state, **kwargs):
        self.logger.warning("LLM retrying: %s", retry_state)