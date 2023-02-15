import pandas as pd
from sklearn.preprocessing import MinMaxScaler
#%%
wahlergebnisse = {}
wahlergebnisse['erststimme'] = {}
wahlergebnisse['zweitstimme'] = {}
wahlergebnisse['bvv'] = {}

#%%

parteien = {
    'P01' :  'SPD',
    'P02' :  'CDU',
    'P03' :  'GRÜNE',
    'P04' :  'DIE LINKE',
    'P05' :  'AfD',
    'P06' :  'FDP'
}

wahlergebnisse['erststimme']['2023'] = pd.read_csv('Datenexport_AGH2023_Erststimme_W_BE.csv', sep=';')
wahlergebnisse['zweitstimme']['2023'] = pd.read_csv('Datenexport_AGH2023_Zweitstimme_W_BE.csv', sep=';')
wahlergebnisse['bvv']['2023'] = pd.read_csv('Datenexport_BVV2023_Stimme_W_BE.csv', sep=';')

wahlergebnisse['erststimme']['2023'] = wahlergebnisse['erststimme']['2023'].rename(columns=parteien)
wahlergebnisse['zweitstimme']['2023'] = wahlergebnisse['zweitstimme']['2023'].rename(columns=parteien)
wahlergebnisse['bvv']['2023'] = wahlergebnisse['bvv']['2023'].rename(columns=parteien)
#%%

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
print(wahlergebnisse['erststimme']['2023'])

#%%
dfs = {}
dfs['erststimme'] = pd.DataFrame()
dfs['zweitstimme'] = pd.DataFrame()
dfs['bvv'] = pd.DataFrame()

for k1, v1 in wahlergebnisse.items():
    for k2, v2 in v1.items():
        v2 = v2.rename(columns={'Wähler' : 'Wählende', 'Die Linke.' : 'DIE LINKE', 'Waehler' : 'Wählende', 'Gueltig' : 'Gültige Stimmen', 'Unguelt' : 'Ungültige Stimmen', 'WBezArt' : 'Wahlbezirksart'})
        for p in ['SPD', 'CDU', 'GRÜNE', 'DIE LINKE', 'FDP', 'AfD']:
            if not p in v2.columns:
                v2[p] = 0
        tmp_df = v2[['Wählende', 'Gültige Stimmen', 'Ungültige Stimmen', 'SPD', 'CDU', 'GRÜNE', 'DIE LINKE', 'FDP', 'AfD', 'Bezirksname', 'Wahlbezirk', 'Wahlbezirksart']]
        tmp_df['Jahr'] = int(k2)
        tmp_df = tmp_df.set_index(['Jahr', 'Wahlbezirk'])

        tmp_df[['SPD', 'CDU', 'GRÜNE', 'DIE LINKE', 'FDP', 'AfD']] = tmp_df[['SPD', 'CDU', 'GRÜNE', 'DIE LINKE', 'FDP', 'AfD']].div(tmp_df['Gültige Stimmen'], axis=0)
        dfs[k1] = dfs[k1].append(tmp_df[tmp_df['Bezirksname'] == 'Mitte'].reset_index())
        dfs[k1]['Wahlbezirk'] = dfs[k1]['Wahlbezirk'].astype(str)
        dfs[k1] = dfs[k1].fillna(0)
#%%
dfs['erststimme'].corr()
dfs['zweitstimme'].corr()
dfs['erststimme']
#%%
import numpy as np
years = [2001,2006,2011,2016,2021,2023]
val_cols = ['SPD', 'CDU', 'GRÜNE', 'DIE LINKE', 'FDP', 'AfD']
subs_erststimme = pd.DataFrame()
subs_zweitstimme = pd.DataFrame()
subs_bvv = pd.DataFrame()
subs_spd = pd.DataFrame()

for i in range(len(years[:-1])):
    y1 = years[i]
    y2 = years[i+1]
    subs_erststimme[str(y1)+' - '+str(y2)] = (dfs['erststimme'].set_index(['Jahr', 'Wahlbezirk']).loc[y2, val_cols] - dfs['erststimme'].set_index(['Jahr', 'Wahlbezirk']).loc[y1, val_cols]).pow(2).sum(axis=1).pow(1/2)
    subs_zweitstimme[str(y1)+' - '+str(y2)] = (dfs['zweitstimme'].set_index(['Jahr', 'Wahlbezirk']).loc[y2, val_cols] - dfs['zweitstimme'].set_index(['Jahr', 'Wahlbezirk']).loc[y1, val_cols]).pow(2).sum(axis=1).pow(1/2)
    subs_bvv[str(y1)+' - '+str(y2)] = (dfs['bvv'].set_index(['Jahr', 'Wahlbezirk']).loc[y2, val_cols] - dfs['zweitstimme'].set_index(['Jahr', 'Wahlbezirk']).loc[y1, val_cols]).pow(2).sum(axis=1).pow(1/2)


subs_erststimme = subs_erststimme.transpose()
subs_zweitstimme = subs_zweitstimme.transpose()
subs_bvv = subs_bvv.transpose()
#%%
wk_01_cols = [c for c in subs_erststimme.columns if c[0] == '1']
subs_erststimme = subs_erststimme[wk_01_cols]
subs_zweitstimme = subs_zweitstimme[wk_01_cols]
subs_bvv = subs_bvv[wk_01_cols]
print(subs_erststimme)
#%%
test = dfs['erststimme'][['Jahr', 'Wahlbezirk', 'SPD', 'CDU', 'GRÜNE', 'DIE LINKE', 'FDP', 'AfD']].set_index('Wahlbezirk')

new_df = pd.DataFrame()
years = [2001,2006,2011,2016,2021,2023]
for y in years:
    tmp_df = test.copy()
    tmp_df.columns = [c + str(y) for c in tmp_df.columns]
    new_df = pd.concat([new_df,  tmp_df[tmp_df['Jahr' + str(y)] == y].drop(columns='Jahr' + str(y))], axis=1)

new_df = new_df.fillna(0).loc[wk_01_cols, :]

#%%
from sklearn.cluster import KMeans, AgglomerativeClustering
m = KMeans(5)
# m = AgglomerativeClustering(8)
test_neu = new_df.copy()
test_neu['clust'] = m.fit_predict(test_neu)
test_neu
m.cluster_centers_
#%%
subs_zweitstimme.to_csv('euclid_wk_01.csv')
#%%
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10,7))
ax = plt.gca()
ax.plot(dfs['erststimme'].set_index(['Jahr', 'Wahlbezirk'])['SPD'].unstack())
ax.legend(wk_01_cols, )
#%%
