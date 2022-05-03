import pipeline_dp

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

brazil=gpd.read_file('C:/Users/alyss/Desktop/bcim_2016_21_11_2018.gpkg', layer ="lim_unidade_federacao_a")
brazil.columns
brazil.rename({"sigla": "state"}, axis=1, inplace=True)
brazil["center"]=brazil["geometry"].centroid
brazil_pts=brazil.copy()
brazil_pts.set_geometry("center", inplace = True)


df = pd.read_csv('C:/Users/alyss/Desktop/accidents2019_rev.csv')
df.rename(inplace=True, columns={'data_inversa' : 'date', 'municipio' : 'state', 'uf' : 'state_code', 'id' : 'accident_id'})
rows = [index_row[1] for index_row in df.iterrows()]


states = ["AC", "AL", "AP", "AM", "BA", "CE", "ES", "GO", "MA", "MT", "MS", "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO", "DF"]
df.head()
backend=pipeline_dp.LocalBackend()
# In the following line, we are setting epsilon and delta, which will determine the amount of noise to be added. 
budget_accountant=pipeline_dp.NaiveBudgetAccountant(total_epsilon=1,total_delta=1e-6)
dp_engine=pipeline_dp.DPEngine(budget_accountant,backend)
data_extractors = pipeline_dp.DataExtractors(
  # Here is where we note that we seek to anonymize each accident as noted by the accident id. Additionally, we are defining the partitions to be by the state codes.
   partition_extractor=lambda row: row.state_code,
   privacy_id_extractor=lambda row: row.accident_id,
   value_extractor=lambda row: 1)

params = pipeline_dp.AggregateParams(
  # Noise is being calcaulated by the Laplace transform as noted in the write up and epsilon-definition of Differential Privacy. 
   noise_kind=pipeline_dp.NoiseKind.LAPLACE,
   # In this example, the query is: how many occurrences of accidents were in the various Brazilian States? We are leveraging count for the metric. 
   metrics=[pipeline_dp.Metrics.COUNT],
   # In the next 2 variables, we are limiting the number of times in which an accident will be accounted for in the total count. We observe that an accident is accounted for exactly once and is a unique event. 
   max_partitions_contributed=1, 
   max_contributions_per_partition=1,
   # We are partitioning or looking at aggregates by the States within Brazil
   public_partitions=states)



dp_result = dp_engine.aggregate(rows, params, data_extractors)


budget_accountant.compute_budgets()


dp_result = list(dp_result)
cpy=dp_result

records=[]
for row in cpy:
  records.append((row[0],int(row[1][0])))

df=pd.DataFrame.from_records(records, columns=['state', 'Count'])
BRASIL=brazil.merge(df, on="state", how='left')

BRASIL.plot(column = 'Count', cmap='BuPu', legend=True, figsize=(16,10),edgecolor = 'green')
plt.title("Accidents in 2019 by Brazilian State - Differentially Private")
texts =[]
for x, y, label in zip(brazil_pts.geometry.x, brazil_pts.geometry.y, brazil_pts["state"]):
  texts.append(plt.text(x,y, label, fontsize =8, color="crimson"))

non_dp_count = {}
for row in rows:
  if row['state_code'] in non_dp_count:
    non_dp_count[row['state_code']] += 1
  else:
    non_dp_count[row['state_code']] =1


df2=pd.DataFrame.from_records([(k,v) for k, v in non_dp_count.items()], columns=['state', 'Count'])
BRASIL2=brazil.merge(df2, on="state", how='left')
BRASIL2.plot(column = 'Count', cmap='BuPu', legend=True, figsize=(16,10),edgecolor = 'green')
plt.title("Accidents in 2019 by Brazilian State") 
texts =[]
for x, y, label in zip(brazil_pts.geometry.x, brazil_pts.geometry.y, brazil_pts["state"]):
  texts.append(plt.text(x,y, label, fontsize =8, color="crimson"))


dp_count = {}
for count_sum_per_state in dp_result:
  dp_count[count_sum_per_state[0]] = count_sum_per_state[1][0]

x = np.arange(len(states))

count_dp=[]
count_ndp=[]
difference=[]
for i in states:
  count_dp.append(dp_count[i])
  count_ndp.append(non_dp_count[i])
  err = abs(dp_count[i]-non_dp_count[i])/non_dp_count[i]
  difference.append(err)
  print(i + " & %.4f" %err)

print(difference)
#Bar Chart to compare impact of the anonymization mechanism
width = 0.35
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, count_ndp, width, label='non-DP')
rects2 = ax.bar(x + width/2, count_dp, width, label='DP')
ax.set_ylabel('Accident count')
ax.set_title('Count accident per state')
ax.set_xticks(x)
ax.set_xticklabels(states)
ax.legend()
fig.tight_layout()
plt.show()


#Error Analysis
annotations = states
X = difference
Y = count_ndp
plt.scatter(difference, count_ndp)
plt.xlabel("Relative Error")
plt.ylabel("Accident Count")
plt.title("Relative Error vs. Actual Count by State")
for i, label in enumerate(annotations):
    plt.annotate(label, (X[i], Y[i]))
plt.show()
  
  
  
  
