import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import plotly.express as px


data = pd.read_csv(r'C:\Users\admin\Documents\GitHub\Stepic_статистика\Основы статистики\states.csv')

data_crop = data[['white', 'hs_grad', 'poverty']]
data_crop.head()
white, hs_grad, poverty = [column for column in data_crop.values.T]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs=white, ys=poverty, zs=hs_grad)

ax.set_xlabel('White(%)')
ax.set_ylabel('Poverty(%)')
ax.set_zlabel('Higher education(%)')
plt.show()

# Строим график с помощью plotly.express, он будет открываться в окне браузера
df = data
print(df)
# Построим плоскость предсказания
lm = smf.ols(formula='poverty ~ white + hs_grad', data=df).fit()
mesh_size = 1.0
margin = 2.0
x_min, x_max = df.white.min()- margin, df.white.max() + margin
y_min, y_max = df.hs_grad.min()- margin, df.hs_grad.max() + margin
z_pred = lambda x, y: lm.params.white * x  + lm.params.hs_grad * y + lm.params.Intercept
x_range = np.arange(x_min, x_max, mesh_size)
y_range = np.arange(y_min, y_max, mesh_size)
z_range = np.array([[z_pred(x, y) for x in x_range] for y in y_range])

# какие значения выше предсказания, а какие ниже
df['poverty_pred'] = np.array([poverty >= z_pred(df.white[i], df.hs_grad[i]) for i, poverty in df.poverty.items()])
print(df)

# составим график
fig = px.scatter_3d(df, x='white', y='hs_grad', z='poverty',
                    color='poverty_pred', color_discrete_sequence=['red', 'green'],
                    title='зависимость процента белого населения и уровня образования на бедность населения')
fig.update_traces(marker=dict(size=3))
fig.add_traces(go.Surface(x=x_range,y=y_range, z=z_range, name='prediction', opacity=0.8))
fig.show()