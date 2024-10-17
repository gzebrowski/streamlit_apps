import functools

import streamlit as st

from helpers import is_numeric


class DynamicFilters:
    def __init__(self, df):
        self._df = df.copy()
        self._all_single_filters = []
        self._all_multiple_filters = []
        self._bool_keys = []
        self._numeric_keys, self._ranges = [], {}
        self._str_keys, self._str_col_values = [], {}
        self._prepare_filters()
        self._single_filter_widgets = {}
        self._multiple_filter_widgets = {}
        self._sel_columns = None
        self._sel_single_filters = None
        self._sel_multiple_filters = None
        self._apply_filters = []

    def _prepare_filters(self):
        for c in self._df.columns:
            dset = set(self._df[c].unique())
            if dset == {0, 1}:
                self._bool_keys.append(c)
            elif not sum([0 if is_numeric(v) else 1 for v in dset]):
                self._numeric_keys.append(c)
                all_values = [float(x) for x in dset if x is not None]
                self._ranges[c] = (min(all_values), max(all_values))
            else:
                self._str_keys.append(c)
                self._str_col_values[c] = sorted([str(v or '') for v in dset])

    def show_filters(self):

        self._sel_columns = st.multiselect('Wybierz kolumny do przeglądania', options=self._df.columns)
        self._sel_single_filters = st.multiselect('Wybierz filtry pojedynczego wyboru', options=self._df.columns)
        mp_cols = [c for c in self._str_keys if c not in self._sel_single_filters]
        self._sel_multiple_filters = st.multiselect('Wybierz filtry wielokrotnego wyboru', options=mp_cols)

        if self._sel_single_filters or self._sel_multiple_filters:
            st.header('Wybrane filtry')
        for f in self._sel_single_filters:
            if f in self._bool_keys:
                self._single_filter_widgets[f] = st.radio(label=f, options=['dowolny', 'tak', 'nie'], horizontal=True)
            elif f in self._numeric_keys:
                self._single_filter_widgets[f] = {
                    'min': st.slider(label=f, min_value=self._ranges[f][0], max_value=self._ranges[f][1],
                                     value=self._ranges[f][0]),
                    'max': st.slider(label=f, min_value=self._ranges[f][0], max_value=self._ranges[f][1],
                                     value=self._ranges[f][1]),
                }
            else:
                self._single_filter_widgets[f] = st.selectbox(label=f, options=self._str_col_values[f])

        for f in self._sel_multiple_filters:
            self._multiple_filter_widgets[f] = st.multiselect(label=f, options=self._str_col_values[f])

    def show_dataframe(self):
        df2 = self._df.copy()
        for f, val in self._single_filter_widgets.items():
            if f in self._bool_keys:
                val = {'tak': 1, 'nie': 0}.get(val)
                if val is not None:
                    self._apply_filters.append(df2[f] == val)
            elif f in self._numeric_keys:
                self._apply_filters.append((df2[f] <= val['max']) & (df2[f] >= val['min']))
            else:
                self._apply_filters.append(df2[f] == val)
        for f, vals in self._multiple_filter_widgets.items():
            self._apply_filters.append(df2[f].isin(vals))

        if self._apply_filters:
            all_filters = functools.reduce(lambda a, b: a & b, self._apply_filters)
            df2 = df2[all_filters]
        df3 = df2[self._sel_columns or self._df.columns]
        st.dataframe(df3, use_container_width=True)
