from __future__ import annotations
from typing import Tuple, List, Dict

import re
import os
import json
import logging
import ftb_snbt_lib as slib

from abc import ABC, abstractmethod
from io import BytesIO
from ftb_snbt_lib import tag
from src.utils import read_file, get_session_id

def escape_text(text: str) -> str:
    for match, seq in ((r'%', r'%%'), (r'"', r'\"')):
        text = text.replace(match, seq)
    return text

def filter_text(text: str) -> bool:
    if not text:
        return False
    if text.startswith("{") and text.endswith("}"):
        return False
    if text.startswith("[") and text.endswith("]"):
        return False
    return True

def safe_name(name: str) -> str:
    return re.compile(r'\W+').sub("", name.lower().replace(" ", "_"))

class ConversionManager:
    def __init__(self, converter: BaseQuestConverter):
        self.converter = converter
        self.logger = logging.getLogger(f"{self.__class__.__qualname__} ({get_session_id()})")
        
    def __call__(self, modpack_name: str, quest_files: List[BytesIO], lang_dict: Dict) -> Tuple[List, Dict]:
        quest_arr, lang_dict = self.converter.convert(modpack_name, quest_files, lang_dict)
        return quest_arr, lang_dict

class BaseQuestConverter(ABC):
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__qualname__} ({get_session_id()})")

    @abstractmethod
    def read(self, quest: BytesIO) -> Tuple[str, Dict]:
        pass

    def convert(self, modpack_name: str, quest_files: List[BytesIO], lang_dict: Dict) -> Tuple[List, Dict]:
        modpack_name = safe_name(modpack_name)
        quest_arr = [self.read(quest) for quest in quest_files]
        for quest_name, quest_data in quest_arr:
            quest_key = f"{modpack_name}.{quest_name}"
            self.logger.info("Converting quest (%s)", quest_name)
            self._convert(quest_key, quest_data, lang_dict)
        self.logger.info("Converted %s quests", len(quest_arr))
        return quest_arr, lang_dict

    @abstractmethod
    def _convert(self, quest_key: str, quest_data: Dict):
        pass

class FTBQuestConverter(BaseQuestConverter):
    def read(self, quest: BytesIO) -> Tuple[str, tag.Compound]:
        quest_name = safe_name(os.path.splitext(quest.name)[0])
        quest_data = slib.loads(read_file(quest))
        assert isinstance(quest_data, tag.Compound), "The quest data must be a Compound tag object"
        return quest_name, quest_data
    
    def _convert(self, quest_key: str, quest_data: tag.Compound, lang_dict: Dict):
        for element in filter(lambda x: quest_data[x], quest_data):
            if isinstance(quest_data[element], tag.Compound):
                self._convert(f"{quest_key}.{element}", quest_data[element])
            elif isinstance(quest_data[element],tag.List) and issubclass(quest_data[element].subtype, tag.Compound):
                for idx in range(len(quest_data[element])):
                    self._convert(f"{quest_key}.{element}{idx}", quest_data[element][idx])

            if element in ("title", "subtitle", "description"):
                if isinstance(quest_data[element], tag.String) and filter_text(quest_data[element]):
                    lang_dict[f"{quest_key}.{element}"] = escape_text(quest_data[element])
                    quest_data[element] = tag.String(f"{{{quest_key}.{element}}}")
                elif isinstance(quest_data[element], tag.List) and issubclass(quest_data[element].subtype, tag.String):
                    for lang_idx, data_idx in enumerate(
                        filter(
                            lambda x: filter_text(quest_data[element][x]),
                            range(len(quest_data[element]))
                        )):
                        lang_dict[f"{quest_key}.{element}{lang_idx}"] = escape_text(quest_data[element][data_idx])
                        quest_data[element][data_idx] = tag.String(f"{{{quest_key}.{element}{lang_idx}}}")

class BQMQuestConverter(BaseQuestConverter):
    def read(self, quest: BytesIO) -> Tuple[str, Dict]:
        quest_name = safe_name(os.path.splitext(quest.name)[0])
        quest_data = json.loads(read_file(quest))
        assert isinstance(quest_data, dict), "The quest data must be a dictionary"
        return quest_name, quest_data

    def infer_version(self, quest_data: Dict) -> int:
        if 'questDatabase:9' in quest_data:
            return 1
        if 'questDatabase' in quest_data:
            if 'properties' in quest_data['questDatabase'][0]:
                return 3
            else:
                return 2
        raise ValueError("The quest data is not a valid format")
    
    def _convert(self, quest_key: str, quest_data: Dict, lang_dict: Dict):
        quest_version = self.infer_version(quest_data)
        match quest_version:
            case 1:
                convert_func = self._convert_v1
            case 2:
                convert_func = self._convert_v2
            case 3:
                convert_func = self._convert_v3
            case _:
                raise ValueError("Unknown quest version")
        convert_func(quest_key, quest_data, lang_dict)
    
    def _convert_v1(self, quest_key: str, quest_data: Dict, lang_dict: Dict):        
        quest_db = quest_data['questDatabase:9']
        for quest in quest_db.values():
            idx = quest.get('questID:3')
            properties = quest['properties:10']['betterquesting:10']
            self._update(properties, lang_dict, f'{quest_key}.quests{idx}', 'name:8', 'desc:8')

        questline_db = quest_data['questLines:9']
        for questline in questline_db.values():
            idx = questline.get('lineID:3')
            properties = questline['properties:10']['betterquesting:10']
            self._update(properties, lang_dict, f'{quest_key}.questlines{idx}', 'name:8', 'desc:8')

    def _convert_v2(self, quest_key: str, quest_data: Dict, lang_dict: Dict):
        quest_db = quest_data['questDatabase']
        for quest in quest_db:
            idx = quest.get('questID')
            properties = quest
            self._update(properties, lang_dict, f'{quest_key}.quests{idx}', 'name', 'description')

        questline_db = quest_data['questLines']
        for idx, questline in enumerate(questline_db):
            properties = questline
            self._update(properties, lang_dict, f'{quest_key}.questlines{idx}', 'name', 'description')

    def _convert_v3(self, quest_key: str, quest_data: Dict, lang_dict: Dict):
        quest_db = quest_data['questDatabase']
        for quest in quest_db:
            idx = quest.get('questID')
            properties = quest['properties']['betterquesting']
            self._update(properties, lang_dict, f'{quest_key}.quests{idx}', 'name', 'desc')

        questline_db = quest_data['questLines']
        for questline in questline_db:
            idx = questline.get('lineID')
            properties = questline['properties']['betterquesting']
            self._update(properties, lang_dict, f'{quest_key}.questlines{idx}', 'name', 'desc')

    def _update(self, properties: Dict, lang_dict: Dict, quest_key: str, name_key: str, desc_key: str):
        name = properties.get(name_key) or 'No Name'
        desc = properties.get(desc_key) or 'No Description'

        lang_dict[f"{quest_key}.name"] = name
        lang_dict[f"{quest_key}.desc"] = desc
        properties[name_key] = f"{quest_key}.name"
        properties[desc_key] = f"{quest_key}.desc"

class BaseTypeConverter(ABC):
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__qualname__} ({get_session_id()})")

class SNBTConverter(BaseTypeConverter):
    def convert_snbt_to_json(self, tag: slib.Compound) -> Dict:
        output = json.loads(json.dumps(tag, ensure_ascii=False))
        self.logger.info("Converted SNBT to JSON")
        return output

    def convert_json_to_snbt(self, data: Dict) -> slib.Compound:
        output = slib.Compound()
        for key, value in data.items():
            if isinstance(value, str):
                output[key] = slib.String(value.replace("\n", "\\n"))
            elif isinstance(value, list):
                output[key]  = slib.List([slib.String('')] * len(value))
                for idx, val in enumerate(value):
                    if isinstance(val, str):
                        output[key][idx] = slib.String(val.replace("\n", "\\n"))
        self.logger.info("Converted JSON to SNBT")
        return output

class LANGConverter(BaseTypeConverter):
    def convert_lang_to_json(self, data: str) -> Dict:
        output = {}
        for line in data.splitlines():
            if line.startswith("#") or not line:
                continue
            key, value = re.compile('(.*)=(.*)').match(line).groups()
            output[key] = value.replace("%n", r"\n")
        self.logger.info("Converted LANG to JSON")
        return output

    def convert_json_to_lang(self, data: Dict) -> str:
        output = ""
        for key, value in data.items():
            value = value.replace(r"\n", "%n")
            output += f"{key}={value}\n"
        self.logger.info("Converted JSON to LANG")
        return output