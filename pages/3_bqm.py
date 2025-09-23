import copy
import time
import json

import streamlit as st

from src.constants import MINECRAFT_LANGUAGES
from src.converter import ConversionManager, BQMQuestConverter, LANGConverter
from src.translator import TranslationManager, get_translator_cls
from src.utils import *

Message("bqm_title").title()
st.page_link(
    page = "pages/0_home.py",
    label = Message("back_to_home").text,
    icon = "↩️"
)

st.divider()

Message("bqm_readme").info()

with st.form("task_form"):
    Message("modpack_name_header").subheader()
    modpack_name = st.text_input(
        label = Message("modpack_name_label").text,
        max_chars = 16,
        placeholder = "atm9",
    )
    
    Message("select_task_header").subheader()
    task = st.radio(
        label = Message("select_task_label").text,
        options = [0, 1, 2],
        format_func = lambda x: {
            0: Message("select_task_convert_translate").text,
            1: Message("select_task_convert_no_translate").text,
            2: Message("select_task_translate_only").text,
        }[x],
        key = "task"
    )
    
    with st.expander(Message("select_task_expander_label").text):
        Message("select_task_expander_desc").send()
    
    Message("lang_check_header").subheader()
    lang_exists = st.radio(
        label = Message("lang_check_label").text,
        options = [True, False],
        format_func = lambda x: {
            True: "Yes",
            False: "No",
        }[x],
        key = "lang_exists",
    )
    
    task_submit = st.form_submit_button()

    if not task_submit and not st.session_state.get("task_submit"):
        st.stop()
        
    st.session_state.task_submit = True

if task == 2 and not lang_exists:
    Message("select_task_nothing", stop=True).error()

set_task(task)

with st.container(border=True):    
    if st.session_state.do_convert:
        Message("upload_quest_header").subheader()
        quest_files = st.file_uploader(
            label = Message("upload_quest_label_bqm").text,
            type = ["json"],
            accept_multiple_files=True
        )

    if st.session_state.lang_exists:
        Message("upload_lang_header").subheader()
        lang_file = st.file_uploader(
            label = Message("upload_lang_label_bqm").text,
            type = ["lang"],
            accept_multiple_files=False
        )

if st.session_state.do_convert and not quest_files:
    st.stop()
if st.session_state.do_convert and len(quest_files) > 1:
    Message("upload_quest_multiple_warning", stop=True).warning()
if st.session_state.lang_exists and not lang_file:
    st.stop()
    
with st.container(border=True):
    if st.session_state.do_translate:
        Message("settings_header").subheader()
        
        translator_service = st.pills(
            label = Message("select_translator_label").text,
            options = ["Google", "DeepL", "Gemini", "OpenAI"],
            default = "Google",
            key = "translator_service",
        )
        
        translator_cls, auth_key = get_translator_cls(translator_service)
        check_auth_key(translator_cls, auth_key)
        translator = translator_cls(auth_key)
        lang_list = translator.lang_list
    else:
        lang_list = list(MINECRAFT_LANGUAGES)

    source_lang = st.selectbox(
        label = Message("select_source_lang_label").text,
        options = lang_list,
        index = lang_list.index("en_us"),
        format_func = lambda x: f"{x} ({MINECRAFT_LANGUAGES[x]})"
    )
    
    if st.session_state.do_translate:
        target_lang = st.selectbox(
            label = Message("select_target_lang_label").text,
            options = lang_list,
            index = lang_list.index("en_us"),
            format_func = lambda x: f"{x} ({MINECRAFT_LANGUAGES[x]})"
        )
    
        if source_lang == target_lang:
            Message("select_same_lang", stop=True).warning()

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

    lang_converter = LANGConverter()
    source_lang_dict = lang_converter.convert_lang_to_json(read_file(lang_file)) if st.session_state.lang_exists else {}
    
    try:
        if st.session_state.do_convert:
            Message("status_step_1", st_container=status).send()
            converter = BQMQuestConverter()
            conversion_manager = ConversionManager(converter)
            converted_quest_arr, source_lang_dict = conversion_manager(modpack_name, quest_files, source_lang_dict)
            
        if st.session_state.do_translate:
            Message("status_step_2", st_container=status).send()
            translation_manager = TranslationManager(translator)
            target_lang_dict = copy.deepcopy(source_lang_dict)
            if source_lang_dict:
                task_key = f"task-{generate_task_key(time.time())}"
                schedule_task(
                    task_key,
                    translation_manager(source_lang_dict, target_lang_dict, target_lang, status)
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
        if st.session_state.do_translate and source_lang_dict and task_key in st.session_state.tasks:
            del st.session_state.tasks[task_key]

    status.update(
        label = Message("status_done").text,
        state = "complete"
    )
    
    with st.container(border=True):
        Message("downloads_header").subheader()
        
        if st.session_state.do_convert:
            quest_filename = "DefaultQuests.json"
            quest_download = st.download_button(
                label = quest_filename,
                data = json.dumps(converted_quest_arr[0], indent=4, ensure_ascii=False),
                file_name = quest_filename,
                on_click = "ignore",
                mime = "application/json"
            )
            
            source_lang_filename = f"{source_lang}.lang"
            source_lang_download = st.download_button(
                label = source_lang_filename,
                data = lang_converter.convert_json_to_lang(source_lang_dict),
                file_name = source_lang_filename,
                on_click = "ignore",
                mime = "text/plain"
            )
        
        if st.session_state.do_translate:
            target_lang_filename = f"{target_lang}.lang"
            target_lang_download = st.download_button(
                label = target_lang_filename,
                data = lang_converter.convert_json_to_lang(target_lang_dict),
                file_name = target_lang_filename,
                on_click = "ignore",
                mime = "text/plain"
            )

    with st.container(border=True):
        Message("user_guide_header").subheader()

        Message("user_guide_bqm_1").send()
        if st.session_state.do_translate:
            Message("user_guide_bqm_2", source_lang=source_lang, target_lang=target_lang).send()
        else:
            Message("user_guide_bqm_3", source_lang=source_lang).send()