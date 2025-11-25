import numpy as np
import pandas as pd
import os

__all__ = ['index_sample_label', 'intersection_columns', 'union_columns', 'colsum_filter', 'map_combined_analysis', 'combined_exemplar_matrix']

def index_sample_label(X: pd.DataFrame, sample_id: str, sample_id_sep: str = '_'):
    """
    Append a sample label to the index of a DataFrame

    Parameters
    ----------
    X: pd.DataFrame
        The DataFrame of which the index will be updated
    sample_id: str
        The sample ID to append to the index
    sample_id_sep: str
        A character separating the index label and sample ID. This character shouldn't appear in either the index or the sample ID. 
    """
    X.index = X.index.astype(str) + sample_id_sep + sample_id
    return(X)


def intersection_columns(colnames): 
    """
    Returns the intersection of a list of column name sets

    Parameters
    ----------
    colnames:
        Nested list of strings, containing the column names for a list of DataFrames
    """
    #colnames = [list(p.columns[~p.columns.duplicated()]) for p in X]
    colnames_intersection = list(set.intersection(*map(set, colnames)))
    #X = [p.loc[:, ~p.columns.duplicated()][colnames_intersection] for p in X]
    return(colnames_intersection)

def union_columns(colnames): 
    """
    Returns the union of a list of column name sets

    Parameters
    ----------
    colnames:
        Nested list of strings, containing the column names for a list of DataFrames
    """
    #colnames = [list(p.columns[~p.columns.duplicated()]) for p in X]
    colnames_union = list(set.union(*map(set, colnames)))
    #X = [p.loc[:, ~p.columns.duplicated()][colnames_intersection] for p in X]
    return(colnames_union)

def colsum_filter(X: pd.DataFrame, b: float):
    """
    Returns the column names of columns with signal above threshold b

    Parameters
    ----------
    X: pd.DataFrame
        DataFrame to be thresholded
    b: float
        A threshold on the interval [0, 1]. Columns will be retained if the column sum is greater than (b)(X.shape[0]) i.e. (b)(number of rows in X)
    """
    cs = np.sum(X, 0)
    X = X.loc[:, cs > (b * X.shape[0])].columns
    return(X)


def combined_exemplar_matrix(top_level_path, sample_root_names):
    """
    Creates combined data matrix of CA transformed exemplar points in the samples given by sample_root_names
    
    Parameters
    ----------
    top_level_path
        path name of the top level directory containing the analysis, as defined by the function multi_sample_directory_structure
    sample_root_names
        The sample root names that are included in the analysis, as defined by the function multi_sample_directory_structure

    """

    datalist = []
    for i in range(0, len(sample_root_names)):
        data = pd.read_csv(os.path.join(top_level_path, sample_root_names[i], 'cells/cell_exemplar_transform.txt.gz'), index_col= 'ID')
        data = index_sample_label(data, str(i))
        datalist.append(data)

    
    datalist = pd.concat(datalist)
    return(datalist)


def map_combined_analysis(top_level_path, sample_root_names):
    """
    Maps combined analysis of exemplar points extracted from list of DataFrames back to the individual \"cells/membership_all.txt.gz\" files

    Parameters
    ----------
    top_level_path
        path name of the top level directory containing the analysis, as defined by the function multi_sample_directory_structure
    sample_root_names
        The sample root names that are included in the analysis, as defined by the function multi_sample_directory_structure

    Notes
    -----

    This function assumes the indexes of the input DataFrames have been labelled as str(i) for i in range(0, len(sample_root_names)), using function index_sample_label. Custom index sample labels are not supported at this time. 
    The updated memberships are saved in combined_output/cells/
    """

    cell_mem = pd.read_csv(os.path.join(top_level_path, 'combined_output/cells/membership_all.txt.gz'))
    cell_mem = pd.concat([cell_mem, pd.DataFrame([x.split('_') for x in cell_mem['ID']], columns = ['raw_ID', 'patient'])], axis = 1)

    for i in range(0, len(sample_root_names)):
        cell_mem_patient = pd.read_csv(os.path.join(top_level_path, sample_root_names[i], "cells/membership_all.txt.gz"))
        sub = pd.read_csv(os.path.join(top_level_path, sample_root_names[i], "cells/exemplars.txt.gz"))
        sub['ID'] = sub['ID'].astype(str)
        sub1 = pd.merge(cell_mem.loc[cell_mem['patient'] == str(i)], sub, left_on = 'raw_ID', right_on = 'ID')
        sub1 = sub1[['ID_x', 'subcluster_x', 'subcluster_y', 'graph_cluster_x', 'graph_cluster_y']]
        sub1 = sub1.rename({'ID_x': 'ID', 'subcluster_x': 'combined_subcluster', 'subcluster_y': 'subcluster', 'graph_cluster_x': 'combined_graph_cluster', 'graph_cluster_y': 'graph_cluster'}, axis = 1)
        cell_mem_patient = pd.merge(cell_mem_patient, sub1, left_on = 'subcluster', right_on = 'subcluster')
        cell_mem_patient = cell_mem_patient[['ID_x', 'subcluster', 'graph_cluster_x', 'local_density', 'combined_subcluster', 'combined_graph_cluster']]
        cell_mem_patient = cell_mem_patient.rename({'ID_x': 'ID', 'graph_cluster_x': 'graph_cluster'}, axis = 1)
        #cell_mem_patient.to_csv(os.path.join(top_level_path, 'combined_output/cells/', sample_root_names[i] + '_combined_membership_all.txt.gz'))
        cell_mem_patient.to_csv(os.path.join(top_level_path, sample_root_names[i], 'cells/combined_membership_all.txt.gz'))





