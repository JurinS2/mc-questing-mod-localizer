from __future__ import annotations
from typing import List

import os
import hashlib
import asyncio
from io import StringIO, BytesIO
from zipfile import ZipFile

import streamlit as st
import ftb_snbt_lib as slib
from streamlit.runtime.scriptrunner import get_script_run_ctx

from src.constants import MESSAGES

@st.cache_data(ttl=3600)
def read_file(file: BytesIO) -> str:
    try:
        return StringIO(file.getvalue().decode('utf-8')).read()
    except UnicodeDecodeError:
        return StringIO(file.getvalue().decode('ISO-8859-1')).read()

def write_file(data: str) -> BytesIO:
    return BytesIO(data.encode('utf-8'))

def compress_quests(quest_arr: List, dir: str, filename: str) -> str:
    """Compresses a list of quests into a zip file and returns the zip file path."""
    zip_dir = os.path.join(dir, filename)
    with ZipFile(zip_dir, "w") as zip_file:
        for quest_name, quest_data in quest_arr:
            zip_file.writestr(f"{quest_name}.snbt", slib.dumps(quest_data))
    return zip_dir

def get_session_id() -> str:
    return get_script_run_ctx().session_id

def set_task(task: int) -> None:
    """Sets the task flags in session state based on the selected task."""
    match task:
        case 0:
            st.session_state.do_convert = True
            st.session_state.do_translate = True
        case 1:
            st.session_state.do_convert = True
            st.session_state.do_translate = False
        case 2:
            st.session_state.do_convert = False
            st.session_state.do_translate = True

def check_auth_key(translator_cls, auth_key: str) -> None:
    """Checks the validity of the provided API key for the translator service."""
    if translator_cls.check_auth_key(auth_key) == -1:
        Message("api_key_empty", stop=True).info()
    elif translator_cls.check_auth_key(auth_key) == 0:
        Message("api_key_invalid", stop=True).error()

def schedule_task(key, coro):
    """Schedules an async task and stores it with a unique key."""
    if key not in st.session_state.tasks:
        st.session_state.tasks[key] = st.session_state.loop.create_task(coro)

def process_tasks():
    """Process pending tasks on the event loop."""
    pending = [task for task in st.session_state.tasks.values() if not task.done()]
    if pending:
        st.session_state.loop.run_until_complete(asyncio.gather(*pending))

def generate_task_key(*args):
    """Generate a unique hash-based key for a task."""
    return hashlib.sha256("-".join(map(str, args)).encode()).hexdigest()

class Message:
    """A utility class for displaying localized messages in Streamlit with optional stopping behavior."""
    message: str
    stop: bool
    
    def __init__(self, key: str, stop: bool = False, st_container = st, **kwargs):
        self.message = MESSAGES[st.session_state.language][key].format(**kwargs)
        self.stop = stop
        self.st_container = st_container

    @property
    def text(self) -> str:
        return self.message
    
    def _stop(self) -> None:
        if self.stop:
            self.st_container.stop()
    
    def send(self) -> None:
        self.st_container.write(self.message)
        self._stop()

    def info(self) -> None:
        self.st_container.info(self.message)
        self._stop()

    def warning(self) -> None:
        self.st_container.warning(self.message)
        self._stop()

    def error(self) -> None:
        self.st_container.error(self.message)
        self._stop()

    def caption(self) -> None:
        self.st_container.caption(self.message)
        self._stop()

    def toast(self) -> None:
        self.st_container.toast(body=self.message)
        self._stop()

    def subheader(self) -> None:
        self.st_container.subheader(self.message)
        self._stop()

    def title(self) -> None:
        self.st_container.title(self.message)
        self._stop()