import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats
from matplotlib.backends.backend_pdf import PdfPages
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns


#URL = 'https://stepik.org/media/attachments/lesson/8083/genetherapy.csv'

data = pd.read_csv('C:/Users/admin/Documents/Stepic_статистика/Основы статистики/genetherapy.csv')

# строим график Box Plot для визуализации распределений по группам в выборке
"""with PdfPages('Stepic_статистика/Основы статистики/BoxPlot.pdf') as pdf:
    b_plot = data.boxplot('expr', by = 'Therapy', figsize=(12,8), grid=True)
    pdf.savefig()"""

sns.pointplot(x='Therapy', y='expr', hue='Therapy', data=data)
plt.show()

# табличка
stat_data = data.groupby("Therapy").agg(["count", "mean","std"])
stat_data.columns=["N","Mx","SD"]
print(stat_data)

A = data[data["Therapy"] == "A"]["expr"]
B = data[data["Therapy"] == "B"]["expr"]
C = data[data["Therapy"] == "C"]["expr"]
D = data[data["Therapy"] == "D"]["expr"]

df = pd.DataFrame({'group_A':list(A),
                   'group_B':list(B),
                   'group_C':list(C),
                   'group_D':list(D)})
print(df)

print(df.describe())

# c помощью встроенной библиотеки 
print(stats.f_oneway(A, B, C, D))
# c df
F, p = stats.f_oneway(df['group_A'], df['group_B'], df['group_C'], df['group_D'])
print(F, p)

# c рассчетом df, SSB и SSM
moda = ols('expr ~ Therapy',data=data).fit()
anova_table = sm.stats.anova_lm(moda, type=2)

print(anova_table)