import importlib
import os
import re

import streamlit as st

from base_mod import BaseStreamlitApp

found_modules = []
for pth in os.listdir('.'):
    if mtch := re.match(r'^(module_[a-z0-9_.-]+)\.py$', pth):
        if not os.path.isdir(pth):
            txt_mod = mtch.group(1)
            try:
                mod = importlib.import_module(txt_mod)
            except Exception:
                pass
            else:
                attrs = dir(mod)
                attrs = [a for a in attrs if not (a.startswith('__') or a.endswith('__') or a == 'BaseStreamlitApp')]
                for a in attrs:
                    cls = getattr(mod, a)
                    if getattr(cls, '__class__', None) and hasattr(cls, 'mro'):
                        if BaseStreamlitApp in cls.mro():
                            found_modules.append(cls)
found_modules.sort(key=lambda x: (x.PRIORITY, x.NAME))
print(found_modules)
all_apps = {nr: app.NAME for nr, app in enumerate(found_modules)}


def format_func(item):
    return all_apps[item]


selected_app = st.selectbox('Select application', options=list(all_apps.keys()), format_func=format_func)

if selected_app is not None:
    sel_mod = found_modules[selected_app]
    run_app = sel_mod()
    if run_app.SHOW_SIDEBAR:
        with st.sidebar:
            run_app.get_sidebar()
    run_app.get_main_content()
