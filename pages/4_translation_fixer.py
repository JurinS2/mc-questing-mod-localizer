import time
import json

import pandas as pd
import streamlit as st
import ftb_snbt_lib as slib

from src.constants import MINECRAFT_LANGUAGES
from src.converter import SNBTConverter
from src.translator import TranslationManager, get_translator_cls
from src.utils import *

Message("translation_fixer_title").title()
st.page_link(
    page = "pages/0_home.py",
    label = Message("back_to_home").text,
    icon = "↩️"
)

Message("translation_fixer_readme").info()

with st.form("lang_form"):
    Message("lang_check_header").subheader()
    lang_type = st.radio(
        label = Message("lang_check_label_extension").text,
        options = ["json", "snbt"],
        key = "lang_type"
    )
    
    lang_submit = st.form_submit_button()

    if not lang_submit and not st.session_state.get("lang_submit"):
        st.stop()

    st.session_state.lang_submit = True

with st.container(border=True):
    Message("upload_lang_header").subheader()
    lang_file = st.file_uploader(
        label = Message("upload_lang_label_translation_fixer").text,
        type = [lang_type],
        accept_multiple_files=False
    )
    
if not lang_file:
    st.stop()

with st.container(border=True):
    Message("settings_header").subheader()
    
    translator_service = st.pills(
        label = Message("select_translator_label").text,
        options = ["Google", "DeepL", "Gemini"],
        default = "Google",
        key = "translator_service",
    )
    
    translator_cls, auth_key = get_translator_cls(translator_service)
    check_auth_key(translator_cls, auth_key)
    translator = translator_cls(auth_key)
    lang_list = translator.lang_list
    
    target_lang = st.selectbox(
        label = Message("select_target_lang_label").text,
        options = lang_list,
        index = lang_list.index("en_us"),
        format_func = lambda x: f"{x} ({MINECRAFT_LANGUAGES[x]})"
    )

with st.container(border=True):
    Message("select_rows_header").subheader()

    if lang_type == "json":
        data = json.loads(read_file(lang_file))
        df = pd.DataFrame(pd.Series(data), columns=['value'], dtype=str)
        df.value = df.value.astype(str)
        df_selection = st.dataframe(
            df,
            use_container_width=True,
            on_select="rerun",
            selection_mode="multi-row"
        )
    elif lang_type == "snbt":
        snbt_converter = SNBTConverter()
        data = snbt_converter.convert_snbt_to_json(slib.loads(read_file(lang_file)))
        df = pd.DataFrame(pd.Series(data), columns=['value'], dtype=str)
        df_selection = st.dataframe(
            df,
            use_container_width=True,
            on_select="rerun",
            selection_mode="multi-row"
        )
    
    selection = {key: data[key] for key in df.index[df_selection.selection.rows]}

button = st.button(
    label = Message("start_button_label").text,
    type = "primary",
    use_container_width = True,
    key = "running",
    disabled = st.session_state.get("running", False)
)

if button:
    with st.spinner("Loading...", show_time=True):
        time.sleep(3)
    
    status = st.status(
        label = Message("status_in_progress").text,
        expanded = True
    )
    
    try:
        manager = TranslationManager(translator)
        task_key = f"task-{generate_task_key(time.time())}"
        schedule_task(
            task_key,
            manager(selection, data, target_lang, status)
        )
        process_tasks()
    except Exception as e:
        status.update(
            label = Message("status_error").text,
            state = "error"
        )
        status.error(f"An error occurred while localizing: {e}")
        st.stop()
    finally:
        if task_key in st.session_state.tasks:
            del st.session_state.tasks[task_key]
    
    status.update(
        label = Message("status_done").text,
        state = "complete"
    )
    
    with st.container(border=True):
        Message("downloads_header").subheader()
        
        if lang_type == "json":
            target_lang_filename = f"{target_lang}.json"
            target_lang_download = st.download_button(
                label = target_lang_filename,
                data = json.dumps(data, indent=4, ensure_ascii=False),
                file_name = target_lang_filename,
                on_click = "ignore",
                mime = "application/json"
            )
        elif lang_type == "snbt":
            target_lang_filename = f"{target_lang}.snbt"
            target_lang_download = st.download_button(
                label = target_lang_filename,
                data = slib.dumps(snbt_converter.convert_json_to_snbt(data)),
                file_name = target_lang_filename,
                on_click = "ignore",
                mime = "text/plain"
            )