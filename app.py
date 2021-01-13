# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import re
import numpy as np
import plotly.express as px

sample_text = '''FA17	AE 100 A	2.0	A		
FA17	AE 199 CD2	2.0	A	>R	
FA17	AVI 101	3.0	TR		
PARKLAND: AVI 101
FA17	BADM 1-- 7	3.0	PS		
FA17	CHEM 102 GL1	3.0	C+		
FA17	CHEM 103 S52	1.0	B+		
FA17	ENG 100 AE2	0.0	S		
FA17	ENGL 1-- 4	3.0	PS		
FA17	FR 2-- 5	2.0	PS		
FA17	GEOG 101 8	3.0	PS		
FA17	MATH 220 1	5.0	PS		
FA17	MATH 231 EL1	3.0	B		
FA17	MATH 299 EL1	1.0	B		
FA17	RHET 1-- 3	3.0	PS		
FA17	RHET 105 1	4.0	PS		
SP18	AE 199 SDM	1.0	A+	>R	
SP18	AVI 120	3.0	TR		
PARKLAND: AVI 120
SP18	MATH 241 BL2	4.0	B+		
SP18	PHYS 211 A3	4.0	A-		
SP18	SPAN 122 D1	4.0	C+		
FA18	AVI 130	3.0	TR		
PARKLAND: AVI 129
FA18	MATH 285 D1	3.0	B		
FA18	MSE 280 A	3.0	B		
FA18	PHYS 212 A2	4.0	C-		
FA18	TAM 210 AL2	2.0	C+		
WI19	ECON 102 ONL	3.0	B		
SP19	AE 202 A	3.0	A		
SP19	LAS 291 SAK	0.0	S		
SP19	MATH 415 AL4	3.0	D+		
SP19	ME 200 AL2	3.0	B+		
SP19	TAM 212 AE2	3.0	B		
FA19	AE 311 A	3.0	A		
FA19	AE 321 A	3.0	B-		
FA19	AE 353 A	3.0	A		
FA19	IE 300 BL1	3.0	A		
FA19	JS 212 A1	3.0	B+		
SP20	AE 312 A	3.0	CR		
SP20	AE 323 A	3.0	CR		
SP20	AE 352 BL	3.0	CR		
SP20	AE 370 A	3.0	CR		
SP20	ECE 205 AL1	3.0	CR		
SU20	AE 402 AO	3.0	B+		
SU20	ECE 206 A1	1.0	B		
FA20	AE 433 A	3.0	B+		
FA20	AE 442 A1	3.0	B		
FA20	AE 460 AE1	2.0	A		
FA20	AE 483 AE1	2.0	A		
FA20	CS 125 AL1	4.0	A		
FA20	CS 196 25	1.0	A+		
SP21	AE 443 A1	3.0	IP	>I	
SP21	AE 461 AS1	2.0	IP	>I	
SP21	AE 484 A	3.0	IP	>I	
SP21	FAA 102 A	3.0	IP	>I	'''


def parse_input(text):
    """ takes raw input and returns dataframe of courses, including gpa and points columns """

    pattern = r"^(?P<semester>[A-Z]{2}[0-9]{2})\t(?P<department>[A-Z]{2,4}) (?P<number>[0-9\-]*).*(?P<hours>[0-9][.][" \
              r"0-9])\t(?P<grade>[ABCDF]([+|-]|\t))"

    lines = text.splitlines()

    df = pd.DataFrame(columns=['semester', 'department', 'number', 'hours', 'grade'])

    for line in lines:
        result = re.match(pattern, line)
        if result is not None:
            row = []
            for key, value in result.groupdict().items():
                if key == 'hours':
                    row.append(float(value.strip()))
                else:
                    row.append(value.strip())
            df.loc[len(df)] = row

    # grade to gpa equivalency
    grade_to_gpa = {
        'A+': 4.00,
        'A': 4.00,
        'A-': 3.67,
        'B+': 3.33,
        'B': 3.00,
        'B-': 2.67,
        'C+': 2.33,
        'C': 2.00,
        'C-': 1.67,
        'D+': 1.33,
        'D': 1.00,
        'D-': 0.67,
        'F': 0.00
    }

    # create empty columns for gpa and points
    df['gpa'] = np.nan
    df['points'] = np.nan

    # populate gpa column with converted gpa
    for index, row in df.iterrows():
        df.loc[index, 'gpa'] = grade_to_gpa[row['grade']]
        df.loc[index, 'points'] = df.loc[index, 'hours'] * df.loc[index, 'gpa']

    return df


# no need to sort semesters because input is already sorted
def sort_semesters(series):
    """ takes series of semesterly GPAs and sorts semesters by defined order """

    sort_order = [
        "FA17",
        "WI18",
        "SP18",
        "SU18",
        "FA18",
        "WI19",
        "SP19",
        "SU19",
        "FA19",
        "WI20",
        "SP20",
        "SU20",
        "FA20"
    ]

    # convert series to dict
    dictionary = series.to_dict()
    index_map = {v: i for i, v in enumerate(sort_order)}
    dictionary_sorted = dict(sorted(dictionary.items(), key=lambda pair: index_map[pair[0]]))

    series_sorted = pd.Series(dictionary_sorted)

    return series_sorted


def semesters_from_df(df):
    # print(df)

    # get series of sums of hours and points columns of df
    sums = df.groupby('semester', sort=False)[['hours', 'points']].sum()

    # print(sums)

    # get series of GPAs for each semester
    semesters = np.floor(sums['points'] / sums['hours'] * 100) / 100

    # print(semesters)

    # sort series in chronological order
    return semesters


def get_cumulative_GPA(df):
    sums = df.groupby('semester', sort=False)[['hours', 'points']].sum()
    sums['gpa'] = np.floor(sums['points'] / sums['hours'] * 100) / 100
    sums['cum_pts'] = np.nan
    sums['cum_hrs'] = np.nan
    sums['cumulative'] = np.nan

    cum_pts = 0
    cum_hrs = 0
    for index, row in sums.iterrows():
        row['cum_pts'] = cum_pts + row['points']
        row['cum_hrs'] = cum_hrs + row['hours']
        cum_pts += row['points']
        cum_hrs += row['hours']

        row['cumulative'] = np.floor(row['cum_pts'] / row['cum_hrs'] * 100) / 100

    sums = sums.drop(columns=['cum_pts', 'cum_hrs'])

    return sums


def main_chart(text):
    # parse raw text
    df = parse_input(text)

    cumulative = get_cumulative_GPA(df)

    # create bar chart
    fig = px.bar(
        df,
        x=semesters_from_df(df).index,
        y=semesters_from_df(df).values,
        text=semesters_from_df(df).values,
    )

    # add cumulative gpa line
    fig.add_scatter(
        x=cumulative.index,
        y=cumulative.cumulative,
        name="Cumulative GPA"
    )

    # add axis lables
    fig.update_layout(
        xaxis_title="Semester",
        yaxis_title="GPA",
        title={'text': "GPA over time"},
    )

    return fig


def pie_chart(text):
    df = parse_input(text)

    fig = px.pie(
        df,
        values=df['department'].value_counts().values,
        names=df['department'].value_counts().index
    )
    fig.update_layout(
        title={'text': "Department distribution"}
    )

    return fig


def bubble_chart(text):
    df = parse_input(text)
    cumulative = get_cumulative_GPA(df)

    fig = px.scatter(cumulative,
                     x=cumulative.index,
                     y=cumulative.gpa,
                     size="hours",
                     size_max=30)
    fig.update_yaxes(range=[0, 4])
    fig.update_layout(
        title={'text': "GPA with credit hours"}
    )

    return fig


bubble_chart(sample_text)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1("gpa-dash"),
    html.Div([
        html.B("Instructions:"),
        html.Ol(children=[
            html.Li(['Generate a degree audit through the ',
                     html.A(
                         "Degree Audi System",
                         href="https://uachieve.apps.uillinois.edu/uachieve_uiuc/audit/create.html",
                         target="_blank",
                         rel="noopener noreferrer"
                     ),
                     ]),
            html.Li('Scroll down to the "SUMMARY OF COURSES TAKEN" section'),
            html.Li("Copy all text in the shaded area and paste it in the box below"),
            html.Li('Hit "Submit" and enjoy')
        ])
    ]),
    dcc.Textarea(
        id="text-input",
        placeholder="enter course data here"
    ),
    html.Button(id='submit-button', n_clicks=0, children='Submit'),
    dcc.Graph(figure=main_chart(sample_text), id='main-graph'),
    dcc.Graph(figure=pie_chart(sample_text), id='pie-chart'),
    dcc.Graph(figure=bubble_chart(sample_text), id='bubble-chart')
])


@app.callback(
    Output('main-graph', 'figure'),
    Output('pie-chart', 'figure'),
    Output('bubble-chart', 'figure'),
    Input('submit-button', 'n_clicks'),
    State('text-input', 'value'))
def update_figures(n_clicks, text):
    if text is None:
        return main_chart(sample_text), pie_chart(sample_text), bubble_chart(sample_text)
    else:
        return main_chart(text), pie_chart(text), bubble_chart(text)


if __name__ == '__main__':
    app.run_server(debug=True)
