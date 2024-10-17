import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from pycaret.regression import load_model, predict_model

from base_mod import BaseStreamlitApp


@st.cache_resource
def get_dataframe(filename):
    _df = pd.read_csv(filename)
    _df.set_index(['country_name'], inplace=True)
    return _df


@st.cache_data
def get_data(_df):
    fields: list[str] = [c for c in _df.columns if c not in ['year', 'country_name', 'happiness_score']]
    aggr_data = _df[fields].agg(['mean', 'max', 'min']).to_dict()
    return fields, aggr_data


@st.cache_resource
def cache_load_regression_model():
    return load_model('world_happines_regression_pipeline')


class HappinesPredictorModule(BaseStreamlitApp):
    SHOW_SIDEBAR = True
    NAME = 'Happines predictor'
    SESSION_DEFAULTS = [('gathered_data', {}), ('audio_hashes', []), ('audio_story_hash', None), ('logs', []),
                        ('confirmed_poll', False)]

    def __init__(self):
        super().__init__()
        self.df = get_dataframe('world-happiness-report_2023.csv')
        self.fields, self.aggr_data = get_data(self.df)
        self.user_values = {}

    def get_sidebar(self):
        for field in self.fields:
            self.user_values[field] = st.slider(field.replace('_', ' ').capitalize(),
                                                min_value=self.aggr_data[field]['min'],
                                                max_value=self.aggr_data[field]['max'],
                                                value=self.aggr_data[field]['mean'])

    def get_main_content(self):
        world_happines_model = cache_load_regression_model()

        tab1, tab2 = st.tabs(['Predykcja', 'Eksploracja danych'])

        with tab1:
            ret_val = predict_model(world_happines_model, data=pd.DataFrame([self.user_values]))
            predicted_value = ret_val.prediction_label[0]

            st.write('Wynik:', predicted_value)

            col1, col2 = st.columns(2)
            with col1:
                st.write('Kraje mające niższy happiness_score')
                st.dataframe(self.df[self.df['happiness_score'] < predicted_value][['happiness_score']].sort_values(
                    'happiness_score', ascending=False), use_container_width=True)

            with col2:
                st.write('Kraje mające wyższy lub równy happiness_score')
                st.dataframe(self.df[self.df['happiness_score'] >= predicted_value][['happiness_score']].sort_values(
                    'happiness_score', ascending=True), use_container_width=True)

            fig, ax = plt.subplots()
            ax.hist(self.df['happiness_score'], bins=40)
            ymin, ymax = ax.get_ybound()
            ax.set_xlabel('Happines score')
            ax.set_ylabel('Count')
            ax.vlines(predicted_value, ymin, ymax, color='red')
            st.pyplot(fig)

        with tab2:
            st.dataframe(self.df)
