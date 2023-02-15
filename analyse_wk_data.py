import pandas as pd
from sklearn.preprocessing import MinMaxScaler
#%%

erststimme_df = pd.read_excel('DL_BE_AGHBVV2021.xlsx', sheet_name="AGH_W1")
zweitstimme_df = pd.read_excel('DL_BE_AGHBVV2021.xlsx', sheet_name="AGH_W2")
bvv_df = pd.read_excel('DL_BE_AGHBVV2021.xlsx', sheet_name="BVV")
#%%
erststimme_df_mitte = erststimme_df[erststimme_df['Bezirksname'] == 'Mitte']
zweitstimme_df_mitte = zweitstimme_df[zweitstimme_df['Bezirksname'] == 'Mitte']
bvv_df_mitte = bvv_df[bvv_df['Bezirksname'] == 'Mitte']
print(erststimme_df_mitte.columns)
print(zweitstimme_df_mitte.columns)
print(bvv_df_mitte.columns)
#%%
bvv_df_mitte.columns

es_cols = ['Wahlberechtigte insgesamt','Wählende', 'Gültige Stimmen', 'Ungültige Stimmen',
       'SPD', 'CDU', 'GRÜNE', 'DIE LINKE', 'AfD', 'FDP', 'Die PARTEI',
       'Tierschutzpartei', 'PIRATEN', 'Graue Panther', 'NPD',
       'Gesundheitsforschung', 'LKR', 'DKP', 'SGP', 'BüSo', 'MENSCHLICHE WELT',
       'B*', 'ÖDP', 'TIERSCHUTZ hier!', 'dieBasis', 'Bildet Berlin!', 'DL',
       'Deutsche Konservative', 'Die Grauen', 'Neue Demokraten', 'REP', 'du.',
       'BÜNDNIS21', 'DIE FRAUEN', 'FREIE WÄHLER', 'Klimaliste Berlin', 'LD',
       'MIETERPARTEI', 'Die Humanisten', 'Team Todenhöfer', 'Volt']

norm_es = MinMaxScaler()
normed_es = norm_es.fit_transform(erststimme_df_mitte[es_cols])

pd.DataFrame(normed_es, columns=es_cols).corr()

#%%
zs_cols = ['Wahlberechtigte insgesamt','Wählende', 'Gültige Stimmen', 'Ungültige Stimmen',
       'SPD', 'CDU', 'GRÜNE', 'DIE LINKE', 'AfD', 'FDP', 'Die PARTEI',
       'Tierschutzpartei', 'PIRATEN', 'Graue Panther', 'NPD',
       'Gesundheitsforschung', 'LKR', 'DKP', 'SGP', 'BüSo', 'MENSCHLICHE WELT',
       'B*', 'ÖDP', 'TIERSCHUTZ hier!', 'dieBasis', 'Bildet Berlin!',
       'Deutsche Konservative', 'Die Grauen', 'Neue Demokraten', 'REP', 'du.',
       'BÜNDNIS21', 'FREIE WÄHLER', 'Klimaliste Berlin', 'MIETERPARTEI',
       'Die Humanisten', 'Team Todenhöfer', 'Volt']

norm_zs = MinMaxScaler()
normed_zs = norm_zs.fit_transform(zweitstimme_df_mitte[zs_cols])

pd.DataFrame(normed_zs, columns=zs_cols).corr()
#%%
from sklearn.linear_model import LinearRegression
from statsmodels.api import OLS
normed_zs_df = pd.DataFrame(normed_zs, columns=zs_cols)
normed_zs_df['intercept'] = 1
normed_zs_df = normed_zs_df.drop(columns=['Wahlberechtigte insgesamt', 'Gültige Stimmen', 'Ungültige Stimmen'])
lin_mod = OLS(normed_zs_df['SPD'], normed_zs_df.drop(columns=['SPD']))
results = lin_mod.fit()

print(results.summary())


#%%
normed_es - normed_zs
#%%
from sklearn.cluster import AgglomerativeClustering

cl = AgglomerativeClustering(n_clusters=10)
zs_class = pd.concat([pd.DataFrame(normed_zs, columns=zs_cols)[['SPD', 'CDU', 'GRÜNE', 'DIE LINKE', 'AfD', 'FDP', 'Die PARTEI']], pd.Series(cl.fit_predict(normed_es), name='class')], axis=1)
zs_class
#%%
import matplotlib.pyplot as plt

fig = plt.figure()

def add_points(x, p):
    plt.plot(x[p[0]],x[p[1]], linestyle='', marker='p')

zs_class.groupby('class').apply(lambda l: add_points(l, ('SPD', 'CDU')))
plt.show()
#%%
fig = plt.figure()

def add_points(x, p):
    plt.plot(x[p[0]],x[p[1]], linestyle='', marker='p')

zs_class.groupby('class').apply(lambda l: add_points(l, ('SPD', 'GRÜNE')))
plt.show()
#%%
fig = plt.figure()

def add_points(x, p):
    plt.plot(x[p[0]],x[p[1]], linestyle='', marker='p')

zs_class.groupby('class').apply(lambda l: add_points(l, ('SPD', 'FDP')))
plt.show()
#%%
fig = plt.figure()

def add_points(x, p):
    plt.plot(x[p[0]],x[p[1]], linestyle='', marker='p')

zs_class.groupby('class').apply(lambda l: add_points(l, ('SPD', 'DIE LINKE')))
plt.show()
#%%
fig = plt.figure()

def add_points(x, p):
    plt.plot(x[p[0]],x[p[1]], linestyle='', marker='p')

zs_class.groupby('class').apply(lambda l: add_points(l, ('SPD', 'AfD')))
plt.show()
