import plotly.express as px
import pandas as pd
import json

def plotly_plot(config,anisotropy_results):
    query=[]
    for k,v in config['filters'].items():
        if v is not None:
            if type(v)==str:
                query.append(f"{k} == '{v}'")
            else:
                query.append(f"{k} == {v}")

    if len(query)>0:
        query=' & '.join(query)
        anisotropy_results=anisotropy_results.query(query)

    print(anisotropy_results.shape)
    fig = px.line(anisotropy_results,**config['plot'])
    fig.show()

if __name__=="__main__":
    with open('config_plot.json','r') as f:
        config=json.load(f)
    anisotropy_results=pd.read_csv('anisotropy_results.csv',index_col=0)
    plotly_plot(config,anisotropy_results)