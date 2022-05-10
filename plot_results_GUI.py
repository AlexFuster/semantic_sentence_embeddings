import tkinter as tk
from tkinter import PhotoImage, ttk
import pandas as pd
import json
from plot_results import plotly_plot

TASK='STS'
OUT_PATH=f'{TASK}_results.csv'

def run_plot():
    filters={}
    for k,field in filter_fields.items():
        field_val=field.get()
        if field_val=='':
            field_val=None
        else:
            if field_val.lower()=='true':
                field_val=True
            elif field_val.lower()=='false':
                field_val=False
            elif field_val.isdigit():
                field_val=int(field_val)
        filters[k]=field_val

    plot={}
    for k,field in plot_fields.items():
        field_val=field.get()
        if field_val=='':
            field_val=None
        plot[k]=field_val

    config={'filters':filters,'plot':plot}
    print(config)
    plotly_plot(config,results)

results=pd.read_csv(OUT_PATH,index_col=0)
with open('config_plot.json','r') as f:
    aux_config=json.load(f)

filter_columns=list(aux_config['filters'].keys())
plot_keys=list(aux_config['plot'].keys())

root = tk.Tk()
root.title("Plot maker")
root.tk.call('wm', 'iconphoto', root._w, tk.PhotoImage(file='thuglife1.png'))
root.geometry("500x400")
root.configure(background='black')

tk.Label(root, text="Filters", bg='black',foreground="light green").grid(row=0, column=1)
filter_fields={}
for i,col in enumerate(filter_columns):
    aux_label = tk.Label(root, text=col, bg="black",foreground="light green")
    aux_combo = ttk.Combobox(root,values=['']+results[col].unique().tolist())
    filter_fields[col]=aux_combo
    aux_label.grid(row=i+1, column=0)
    aux_combo.grid(row=i+1, column=1, ipadx="100")

for col in ["N","max_length","cased","no_stop","no_sub"]:
    filter_fields[col].current(1)

plot_fields={}
row_offset=len(filter_columns)+1
tk.Label(root, text="Plot", bg="black",foreground="light green").grid(row=row_offset, column=1)
all_cols=['']+list(results.columns)
defaults={
    'x':all_cols.index("layer"),
    'y':all_cols.index(TASK),
    'color':all_cols.index("pooling"),
    'facet_row':0,
    'facet_col':0
}
for i,plot_key in enumerate(plot_keys):
    aux_label = tk.Label(root, text=plot_key, bg="black",foreground="light green")
    aux_combo = ttk.Combobox(root,values=all_cols)
    aux_label.grid(row=i+row_offset+1, column=0)
    aux_combo.grid(row=i+row_offset+1, column=1, ipadx="100")
    aux_combo.current(defaults[plot_key])
    plot_fields[plot_key]=aux_combo

row_offset+=len(plot_keys)

submit = tk.Button(root, text="Submit", fg="Black",
                        bg="cyan", command=run_plot)
submit.grid(row=row_offset+1, column=1)
root.mainloop()