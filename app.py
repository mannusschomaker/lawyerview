import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import seaborn as sns

def load_data():
    results_df = pd.read_csv('results.csv')  # replace with path to your data
    return results_df

app = dash.Dash(__name__)
df = load_data()

# Set the seaborn color palette to pastel
sns.set_palette("pastel")


# CSS styles for banners and charts
styles = {
    'banner': {'background-color': sns.color_palette()[0], 'padding': '10px', 'color': 'blue', 'height': '60px'},
    'side-banner': {'background-color': sns.color_palette()[1], 'padding': '10px'},
    'chart': {'padding': '10px'}
}
app.layout = html.Div(children=[
    html.Div(children=[
        html.H1('Welcome to My Web App', style=styles['banner'])
    ], className='banner'),

    html.Div(children=[
        html.Div([

            dcc.Graph(id='radar-chart'),
            dcc.Dropdown(
            id='firm-dropdown',
            options=[{'label': i, 'value': i} for i in df['Firm'].unique()],
            value=df['Firm'].unique()[0]
            ),
            dcc.Graph(id='pie-chart')
        ], className='side-banner', style=styles['side-banner']),

        html.Div(children=[
            # dcc.Graph(id='pie-chart'),  # Pie Chart
            # dcc.Graph(id='radar-chart'),  # Radar Chart
        ], className='charts'),
    ], className='container'),

    dcc.Interval(id='interval-component', interval=2000)  # Placeholder interval component
])

@app.callback(
    [Output('radar-chart', 'figure'),
    Output('pie-chart', 'figure')],
    [Input('firm-dropdown', 'value')]
)
def update_graph(selected_firm):
    firm_data_df = df[df['Firm'] == selected_firm]
    if not firm_data_df.empty:
        firm_data = firm_data_df.iloc[0]
    else:
        # Handle no data case here
        firm_data = df.iloc[0]  # use the first row of df as a default
    
    # For spider plot
    all_rechtsgebieden = df.columns[6:] 
    rechtsgebieden_data = firm_data[all_rechtsgebieden]
    rechtsgebieden_data = rechtsgebieden_data[rechtsgebieden_data > 0]
    rechtsgebieden_data = rechtsgebieden_data.drop('Niet bekend', errors='ignore')
    rechtsgebieden_data = rechtsgebieden_data.astype(float)
    rechtsgebieden_data = rechtsgebieden_data.nlargest(10)

    # Radar chart
    labels = np.array(rechtsgebieden_data.index)
    stats = rechtsgebieden_data.values
    radar_chart = go.Scatterpolar(
        r=stats,
        theta=labels,
        fill='toself'
    )
    radar_fig = go.Figure(data=[radar_chart])

    # For pie chart
    beedigings_cats = ['Num_Beëdigingsdatum_Old_10', 'Num_Beëdigingsdatum_Old_6', 'Num_Beëdigingsdatum_Old_3', 'Num_Beëdigingsdatum_Young_3']
    beedigings_data = firm_data[beedigings_cats]
    beedigings_data['Num_Beëdigingsdatum_Old_6'] = beedigings_data['Num_Beëdigingsdatum_Old_6'] - beedigings_data['Num_Beëdigingsdatum_Old_10']
    beedigings_data['Num_Beëdigingsdatum_Old_3'] = beedigings_data['Num_Beëdigingsdatum_Old_3'] - beedigings_data['Num_Beëdigingsdatum_Old_6']

    pie_chart = go.Pie(labels=beedigings_cats, values=beedigings_data)
    pie_fig = go.Figure(data=[pie_chart])

    return radar_fig, pie_fig


if __name__ == '__main__':
    app.run_server(debug=True)