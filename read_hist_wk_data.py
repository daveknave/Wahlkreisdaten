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

wahlergebnisse['erststimme']['2016'] = pd.read_excel('DL_BE_EE_WB_AH2016.xlsx', sheet_name="Erststimme")
wahlergebnisse['zweitstimme']['2016'] = pd.read_excel('DL_BE_EE_WB_AH2016.xlsx', sheet_name="Zweitstimme")
wahlergebnisse['bvv']['2016'] = pd.read_excel('DL_BE_EE_WB_AH2016.xlsx', sheet_name="BVV")

wahlergebnisse['erststimme']['2011'] = pd.read_excel('DL_BE_AB2011.xlsx', sheet_name="Erststimme")
wahlergebnisse['zweitstimme']['2011'] = pd.read_excel('DL_BE_AB2011.xlsx', sheet_name="Zweitstimme")
wahlergebnisse['bvv']['2011'] = pd.read_excel('DL_BE_AB2011.xlsx', sheet_name="BVV")


wahlergebnisse['erststimme']['2006'] = pd.read_excel('DL_BE_AB2006.xlsx', sheet_name="Erststimme")
wahlergebnisse['zweitstimme']['2006'] = pd.read_excel('DL_BE_AB2006.xlsx', sheet_name="Zweitstimme")
wahlergebnisse['bvv']['2006'] = pd.read_excel('DL_BE_AB2006.xlsx', sheet_name="BVV")

wahlergebnisse['erststimme']['2001'] = pd.read_excel('DL_BE_AB2001.xlsx', sheet_name="Erststimme")
wahlergebnisse['zweitstimme']['2001'] = pd.read_excel('DL_BE_AB2001.xlsx', sheet_name="Zweitstimme")
wahlergebnisse['bvv']['2001'] = pd.read_excel('DL_BE_AB2001.xlsx', sheet_name="BVV")
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
structure = {
    'Erststimme' : 'erststimme',
    'Zweitstimme' : 'zweitstimme',
    'BVV' : 'bvv',
}
for name, val in structure.items():

    parteien = ['SPD', 'CDU', 'GRÜNE', 'DIE LINKE', 'FDP', 'AfD', 'Die PARTEI', 'Tierschutzpartei']
    tmp_df = dfs[val][['Jahr', 'Wahlbezirk', 'Gültige Stimmen'] + parteien].set_index('Wahlbezirk')
    tmp_df[['%'+p for p in parteien]] =  tmp_df[parteien].div(tmp_df['Gültige Stimmen'], axis=0)*100
    diffs = tmp_df.loc[tmp_df['Jahr'] == 2023] - tmp_df.loc[tmp_df['Jahr'] == 2021]
    diffs['%Gültige Stimmen'] = 100* (tmp_df.loc[tmp_df['Jahr'] == 2023, 'Gültige Stimmen'] - tmp_df.loc[
    tmp_df['Jahr'] == 2021, 'Gültige Stimmen']) / tmp_df.loc[tmp_df['Jahr'] == 2021, 'Gültige Stimmen']

    wk_cols = [[c for c in diffs.index if c[0] == '1'],
                    [c for c in diffs.index if c[0] == '2'],
                    [c for c in diffs.index if c[0] == '3'],
                    [c for c in diffs.index if c[0] == '4'],
                    [c for c in diffs.index if c[0] == '5'],
                    [c for c in diffs.index if c[0] == '6'],
                    [c for c in diffs.index if c[0] == '7']]

    c = 0
    for cols in wk_cols:
        c+=1
        fig = plt.figure(figsize=(9,10))
        ax1 = plt.gca()
        sb.heatmap(diffs.loc[cols, ['%'+p for p in parteien]+['%Gültige Stimmen']],
                   cmap= sb.color_palette("icefire", as_cmap=True),
                   vmin=-12, vmax=12,
                   yticklabels=1,
                   ax=ax1,
                   annot=True,
                   )
        plt.title(name + ": Gewinne/Verluste in Prozentpunkten für Wahlkreis " + str(c))
        plt.savefig('heatmaps/' + str(c) + '_' + name + '_heatmap_relativ_diff.pdf', dpi=300)
        plt.show()

        fig = plt.figure(figsize=(9,10))
        ax1 = plt.gca()
        sb.heatmap(diffs.loc[cols, parteien+['Gültige Stimmen']],
                   fmt='d',
                   cmap= sb.color_palette("icefire", as_cmap=True),
                   vmin=-120, vmax=120,
                   yticklabels=1,
                   ax=ax1,
                   annot=True)

        plt.title(name + ": Absolute Gewinne/Verluste für Wahlkreis " + str(c))
        plt.savefig('heatmaps/' + str(c) + '_' + name + '_heatmap_absolute_diff.pdf', dpi=300)
        plt.show()


#%%
abt_1_sbzk = [
    '100',
    '101',
    '102',
    '103',
    '104',
    '105',
    '106',
    '110',
    '111',
    '112',
    '113',
    '114',
    '117',
    '118',
    '119',
    '120',
    '121',
    '122',
    '123',
    '125'
]


diffs['Abt. 1'] = 0
diffs['Abt. 1'].loc[abt_1_sbzk] = 1
corrs = diffs.loc[wk_cols[0]].corr()

from sklearn.ensemble import RandomForestClassifier

m = RandomForestClassifier()
X = diffs.loc[wk_cols[0],[c for c in diffs.columns if c[0] == '%']]
X_train = X.sample(frac=0.8)
X_test = X.drop(X_train.index)
y = diffs.loc[wk_cols[0], 'Abt. 1']
y_train = y.loc[X_train.index]
y_test = y.drop(y_train.index)
m.fit(X_train, y_train)
print(m.score(X_test, y_test))
print(m.feature_importances_)
#%%
from sklearn import tree

tree_count = 0

for t in m.estimators_:

    tree.plot_tree(t,
                       feature_names=X.columns,
                       class_names=['Abt. 1', 'nicht Abt. 1'],
                       filled=True)
    plt.savefig(str(tree_count) + '_decision_tree.pdf', dpi=300)

    tree_count += 1

#%%
### Output
for name, val in structure.items():

    parteien = ['SPD', 'CDU', 'GRÜNE', 'DIE LINKE', 'FDP', 'AfD', 'Die PARTEI', 'Tierschutzpartei']
    tmp_df = dfs[val][['Jahr', 'Wahlbezirk', 'Gültige Stimmen'] + parteien].set_index(['Jahr', 'Wahlbezirk'])
    tmp_df[['%'+p for p in parteien]] = tmp_df[parteien].div(tmp_df['Gültige Stimmen'], axis=0)*100
    diffs = tmp_df.loc[2023] - tmp_df.loc[2021]

    diffs['%Gültige Stimmen'] = 100* (tmp_df.loc[2023, 'Gültige Stimmen'] - tmp_df.loc[2021, 'Gültige Stimmen']) / tmp_df.loc[2021, 'Gültige Stimmen']
    diffs.to_csv('data/' + name + '_diffs.csv')
    tmp_df.loc[2023].reset_index().to_csv('data/' + name + '_ergebnis_2023.csv')
    tmp_df.loc[2021].to_csv('data/' + name + '_ergebnis_2021.csv')

#%%

abt_sbzk = pd.read_csv('Zuordnung_Abteilungen_Stimmbezirke.csv', sep=";")
abt_sbzk = abt_sbzk.set_index('Wahlbezirk')

structure = {
    'Erststimme' : 'erststimme',
    'Zweitstimme' : 'zweitstimme',
    'BVV' : 'bvv',
}

for name, val in structure.items():

    parteien = ['SPD', 'CDU', 'GRÜNE', 'DIE LINKE', 'FDP', 'AfD', 'Die PARTEI', 'Tierschutzpartei']
    tmp_df = abt_sbzk.join(dfs[val][['Jahr', 'Wahlbezirk', 'Gültige Stimmen'] + parteien].set_index('Wahlbezirk')).reset_index()

    tmp_df.loc[:, 'Wahlkreis'] = tmp_df['Wahlbezirk'].apply(lambda x: str(x)[0])

    sums_df = tmp_df[['Abteilung', 'Wahlkreis', 'Gültige Stimmen', 'Jahr', 'Anteil']].groupby(['Abteilung', 'Wahlkreis', 'Jahr'], as_index=False).apply(lambda x: x.iloc[0])

    sums_df[parteien+['Gültige Stimmen']] = tmp_df.groupby(['Abteilung', 'Wahlkreis', 'Jahr'], as_index=False).apply(lambda x: x.sum(axis=0))[parteien + ['Gültige Stimmen']]

    sums_df[['%'+p for p in parteien]] = sums_df[parteien].div(sums_df['Gültige Stimmen'], axis=0) * 100

    sums_df = sums_df.set_index(['Abteilung', 'Wahlkreis'])

    diffs = (sums_df[parteien +['%'+p for p in parteien] + ['Gültige Stimmen']].loc[sums_df['Jahr'] == 2023] - sums_df[parteien +['%'+p for p in parteien] + ['Gültige Stimmen']].loc[sums_df['Jahr'] == 2021])

    diffs['%Gültige Stimmen'] = 100 * (sums_df.loc[sums_df['Jahr'] == 2023, 'Gültige Stimmen'] - sums_df.loc[sums_df['Jahr'] == 2021, 'Gültige Stimmen']).div(sums_df.loc[sums_df['Jahr'] == 2021, 'Gültige Stimmen'], axis=0)

    diffs.to_csv('data/' + name + '_diffs.csv', float_format='${:,.2f}'.format)
    # diffs.to_csv('data/' + name + '_diffs.csv')

#%%
import plotly.express as px
import geopandas as gpd

px.choropleth_mapbox()
