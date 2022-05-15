# %%
# PLSR
from scipy import stats
import pandas as pd
from pyls import pls_regression

def plsr(roi_models, n_components=2, 
     n_perm=5000, n_boot=5000,
     gene_path='./data/gene/expression.csv',
     out_path='./data/gene/expression_plsr.csv'):
    roi_es_dict = {}
    for k, v in roi_models.items():
        roi_es_dict[int(k)] = v

    roi_df = pd.DataFrame.from_dict(roi_es_dict, orient='index', columns=['es'])

    gene_df = pd.read_csv(gene_path, index_col=0)
    
    es_filtered = roi_df[roi_df.index.isin(gene_df.index)]
    gene_filtered = gene_df[gene_df.index.isin(es_filtered.index)]
    es_filtered = es_filtered.sort_index()
    gene_filtered = gene_filtered.sort_index()

    x = gene_filtered.values
    y = es_filtered.values

    plsr = pls_regression(x, y, n_components=n_components, n_perm=n_perm,
                          n_boot=n_boot)

    pls1 = plsr.x_weights.T[0]
    pls2 = plsr.x_weights.T[1]
    gene_name = list(gene_filtered.columns)
    d = {'gene_name':gene_name, 'pls1':pls1, 'pls2':pls2}
    df = pd.DataFrame(d)
    df.set_index('gene_name')
    df.to_csv(out_path)
    return plsr
