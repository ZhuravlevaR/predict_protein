# -*- coding: utf-8 -*-
"""
Created on Wed Mar 01 12:49:45 2021

@author: zhuravleva_ro

Быстрый вывод приложения для тестирования модели Сделаны две модели предсказание
потребности протеина на две и четыре недели вперед. Для быстрого тестирования
сделала веб приложение на Dash. Параметры модели обображаются в таблице,
которую можно править и изменять условия для предсказания. По нажадию кнопки
происходит предсказание двух величин, на 2 и 4 недели вперед. После этого
показывается обновленный график с предсказаниями. Для удобства сделан инструмент
загрузки параметров из отчетного экселя.

"""
import base64
import io
from sqlalchemy import create_engine
import datetime
import json
import pandas as pd
import dash
import dash_table

import dash_core_components as dcc
import dash_html_components as html
import plotly
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
import plotly.express as px
from config import connection_string,color,backgroundColor,col_all, col_linear, col_automl


from model.scoring_file_v_1_0_0 import *

engine = create_engine(connection_string)


#
def get_fig():
    # make figure
    # Беру последние 20 предсказаний и строю по ним график
    query = """SELECT TOP(20) * FROM [DEV052].[dbo].[preduction_protein]
      ORDER BY time DESC"""
    df_fig = pd.read_sql(query, con=engine)
    fig = px.scatter(df_fig, x="time_for_predict", y="Linear_predict_4_week", text='comment')
    fig.add_scatter(x=df_fig["time"] + datetime.timedelta(days=14), y=df_fig["Linear_predict_2_week"], mode='markers',
                    name='Linear_predict_2_week')
    fig.add_scatter(x=df_fig["time"], y=df_fig["protein"], mode='markers',
                    name='protein')
    fig.update_layout(clickmode='event+select')
    fig.update_traces(marker_size=10)
    return fig


def get_model(num=4):
    # Загрузка модели из файла на 2 и 4 недели вперед
    with open(rf".\model\regr_{num}_week.pkl", 'rb') as f:
        regr = pickle.load(f)
    return regr

def get_dic():
    # Загрузка сопоставления имен из базы
    query = """ SELECT * FROM [DEV052].[dbo].[rus_eng_name]
        """
    df_name = pd.read_sql(query, con=engine)
    dic = {rus: eng for rus, eng in zip(df_name.name_rus, df_name.name_eng)}
    return dic



def get_start_data(col_all):
    """
    Загрузка последних данных из базы для отображения начальной страницы
    """
    query = """
    SELECT TOP(1) *
      FROM [DEV052].[dbo].[preduction_protein]
      ORDER BY time DESC """
    df = pd.read_sql(query, con=engine)
    # print(df.time)
    df['time_for_predict'] = df['time_for_predict'].dt.strftime("%d-%m-%Y %H:%M")
    df['time'] = df['time'].dt.strftime("%d-%m-%Y %H:%M")
    dic = get_dic()
    table_ = [{'eng_name': dic[col_], 'name': col_, 'value': df[col_].values[0]} for col_ in col_all]
    return table_

"""
MAIN PART OF PAGE
"""
external_stylesheets = ['https://cdn.jsdelivr.net/npm/bootswatch@4.5.2/dist/cerulean/bootstrap.min.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
title = 'Предсказание недельной потребности на белок'
app.title = title

app.layout = html.Div([  # big block
    html.Div(  # small block upper most
        dash_table.DataTable(
            id='table',
            columns=[{"name": i, "id": i, 'editable': edite} for i, edite in
                     zip(get_start_data(col_all)[0].keys(), [False, False, True])],
            data=get_start_data(col_all)
            , style_data_conditional=[{'if': {'column_id': ['eng_name', 'name', 'value'], "row_index": num},
                                       'backgroundColor': color,
                                       'color': 'black'}
                                       for num in range(0, len(col_all)) if col_all[num] in col_linear]
            , style_header={
                            'backgroundColor': backgroundColor,
                            'fontWeight': 'bold'}),
        style={'width': '45%', 'display': 'inline-block'}),
    html.Div([
        html.Div(children=title),
        dcc.Graph(
            id='fig_linear',
            figure=get_fig()
        ),
        html.Button('ReCalculate', id='butt_calc', n_clicks=0, style ={'background-color':backgroundColor}),

        dcc.Upload(
            id='upload_data',
            children=html.Div([
                'FOR TEST Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '60%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            # Allow multiple files to be uploaded
            multiple=False
        ),
    ],
        style={'width': '45%', 'float': 'right', 'display': 'inline-block'}),
])


@app.callback(
    [Output('table', 'data'),
     Output('fig_linear', 'figure'),
     Output('upload_data', 'contents')],
    [Input('butt_calc', 'n_clicks'),
     Input('upload_data', 'contents')],
    [State('table', 'data'),
     State('upload_data', 'filename')],
)
def update_output(n_clicks, contents, data,filename):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        if 'xls' in filename:
            try:
                print(1)
                df_excel = pd.read_excel(io.BytesIO(decoded), sheet_name='Online Table Update')
                print(df_excel)
                df_ = pd.DataFrame(data)
                df_.index = df_['name']
                df_['value'] = np.NaN
                for col_ in col_automl:
                    if col_ in df_excel.name.values:
                        print(col_, df_excel[df_excel.name == col_]['value'].values[0])
                        df_.loc[col_, 'value'] = df_excel[df_excel.name == col_]['value'].values[0]

                dic = get_dic()
                df_ = df_[['value']].T
                table_ = [{'eng_name': dic[col_], 'name': col_, 'value': df_[col_].values[0]} for col_ in col_all]
                return table_, get_fig(), None

            except Exception as e:
                print(f'error {e}')
                return get_start_data(col_all), get_fig(), None


    if n_clicks > 0:

        df_ = pd.DataFrame(data)
        df_.index = df_['name']
        df_ = df_[['value']].T
        regr = get_model(4) # model for 4 weeks predict
        df_['Linear_predict_4_week'] = regr.predict(df_[col_linear]).round(2)

        regr = get_model(2) # model for 2 weeks predict
        df_['Linear_predict_2_week'] = regr.predict(df_[col_linear]).round(2)


        df_['time_for_predict'] = (datetime.datetime.now() + datetime.timedelta(days=28))
        df_['time'] = datetime.datetime.now()

        df_.to_sql('preduction_protein', con=engine, schema='[DEV052].[dbo]', if_exists='append', index=False)

    return get_start_data(col_all), get_fig(), None




if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8088)
