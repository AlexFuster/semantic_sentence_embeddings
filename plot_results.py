import plotly.express as px
import pandas as pd

anisotropy_results=pd.read_csv('anisotropy_results.csv',index_col=0)
anisotropy_results=anisotropy_results[anisotropy_results['N']==1000]
fig = px.line(anisotropy_results, x="layer", y="anisotropy", color='pooling',facet_row='dataset',facet_col='model')
fig.show()