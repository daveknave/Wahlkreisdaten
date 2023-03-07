import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sb
from sklearn.preprocessing import MinMaxScaler

wahlergebnisse = {}
wahlergebnisse['erststimme'] = {}
wahlergebnisse['zweitstimme'] = {}
wahlergebnisse['bvv'] = {}

#%%


wahlergebnisse['erststimme']['2023'] = pd.read_excel('DL_BE_AGHBVV2023.xlsx', sheet_name="AGH_W1")
wahlergebnisse['zweitstimme']['2023'] = pd.read_excel('DL_BE_AGHBVV2023.xlsx', sheet_name="AGH_W2")
wahlergebnisse['bvv']['2023'] = pd.read_excel('DL_BE_AGHBVV2023.xlsx', sheet_name="BVV")

wahlergebnisse['erststimme']['2021'] = pd.read_excel('DL_BE_AGHBVV2021.xlsx', sheet_name="AGH_W1")
wahlergebnisse['zweitstimme']['2021'] = pd.read_excel('DL_BE_AGHBVV2021.xlsx', sheet_name="AGH_W2")
wahlergebnisse['bvv']['2021'] = pd.read_excel('DL_BE_AGHBVV2021.xlsx', sheet_name="BVV")

#%%
dfs = {}
dfs['erststimme'] = pd.DataFrame()
dfs['zweitstimme'] = pd.DataFrame()
dfs['bvv'] = pd.DataFrame()

for k1, v1 in wahlergebnisse.items():
    for k2, v2 in v1.items():
        v2 = v2.rename(columns={'Wähler' : 'Wählende', 'Die Linke.' : 'DIE LINKE', 'Waehler' : 'Wählende', 'Gueltig' : 'Gültige Stimmen', 'Unguelt' : 'Ungültige Stimmen', 'WBezArt' : 'Wahlbezirksart'})
        for p in ['SPD', 'CDU', 'GRÜNE', 'DIE LINKE', 'FDP', 'AfD', 'Die PARTEI', 'Tierschutzpartei']:
            if not p in v2.columns:
                v2[p] = 0
        tmp_df = v2[['Wählende', 'Gültige Stimmen', 'Ungültige Stimmen', 'SPD', 'CDU', 'GRÜNE', 'DIE LINKE', 'FDP', 'AfD', 'Die PARTEI', 'Tierschutzpartei', 'Bezirksname', 'Wahlbezirk', 'Wahlbezirksart']]
        tmp_df.loc[:, 'Jahr'] = int(k2)
        tmp_df = tmp_df.set_index(['Jahr', 'Wahlbezirk'])

        # tmp_df[['SPD', 'CDU', 'GRÜNE', 'DIE LINKE', 'FDP', 'AfD']] = tmp_df[['SPD', 'CDU', 'GRÜNE', 'DIE LINKE', 'FDP', 'AfD']].div(tmp_df['Gültige Stimmen'], axis=0)
        tmp_df[['SPD', 'CDU', 'GRÜNE', 'DIE LINKE', 'FDP', 'AfD', 'Die PARTEI', 'Tierschutzpartei']] = tmp_df[['SPD', 'CDU', 'GRÜNE', 'DIE LINKE', 'FDP', 'AfD', 'Die PARTEI', 'Tierschutzpartei']]
        dfs[k1] = dfs[k1].append(tmp_df[tmp_df['Bezirksname'] == 'Mitte'].reset_index())
        dfs[k1].loc[:, 'Wahlbezirk'] = dfs[k1]['Wahlbezirk'].astype(str)
        dfs[k1] = dfs[k1].fillna(0)
#%%
#%%
import plotly.express as px
import geopandas as gpd
from io import StringIO
from dash import Dash, dcc, html, Input, Output


# token = open(".mapbox_token").read() # you will need your own token

app = Dash(__name__)

app.layout = html.Div([
    html.H4('Polotical candidate voting pool analysis'),
    html.P("Select a candidate:"),
    dcc.RadioItems(
        id='candidate',
        options=["Joly", "Coderre", "Bergeron"],
        value="Coderre",
        inline=True
    ),
    dcc.Graph(id="graph"),
])

@app.callback(
    Output("graph", "figure"),
    Input("candidate", "value"))
def display_choropleth(candidate):
    shapedata = gpd.read_file('GeoData/RBS_OD_UWB_AH21/RBS_OD_UWB_AH21.shp')
    # shapedata = shapedata.set_index('UWB3')
    wk_json = shapedata.to_json()

    fig = px.choropleth_mapbox(dfs['erststimme'].loc[dfs['erststimme']['Jahr'] == 2023],
                               geojson=wk_json,
                               locations=dfs['erststimme'].loc[dfs['erststimme']['Jahr'] == 2023, 'Wahlbezirk'],
                               featureidkey="properties.UWB3",
                               color='SPD',
                               mapbox_style='open-street-map', center={"lat": 52.531626, "lon": 13.390017},

                               zoom=10
                               )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    return fig


app.run_server(debug=True)