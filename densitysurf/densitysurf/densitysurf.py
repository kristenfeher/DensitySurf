
# to install required packages, at the bash command line within this directory, type this command:
# pip install -r requirements.txt
# comment

# update requirements:
# pip freeze > requirements.txt

# check paths
# import sys
# sys.path

# change paths
# sys.path.append("path")

import numpy as np
import pandas as pd
import umap.umap_ as umap
#import scanpy as sc
#import scipy
#from sklearn.datasets import load_digits
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.extmath import randomized_svd
from sknetwork.clustering import Louvain
import igraph as ig
import os
import pickle
import seaborn as sb
import matplotlib.pyplot as plt
from scipy.stats import binom

#__all__ = ['NN_density_cluster', 'reconstruct', 'directory_structure', 'MultiSampleConcat', 'Workflow', 'Transform', 'Cluster', 'SpecificityNetwork', 'NeighbourhoodFlow']
__all__ = ['NN_density_cluster', 'reconstruct', 'directory_structure', 'MultiSampleConcat', 'Transform', 'Cluster', 'SpecificityNetwork', 'NeighbourhoodFlow']


# def spherical_transform(X):
#     """Helper function to normlise a vector to length one"""
#     return(X/np.sqrt(np.sum(np.square(X))))

def spherical_transform(X):
    """Helper function to normlise a vector to length one"""
    return(np.transpose(np.transpose(X) * 1/np.sum(X**2, axis = 1)**0.5))

def ca_transform(input_dataframe: pd.DataFrame): #input_dataframe must be pd.DataFrame

    """

    Parameters
    ----------
    input_dataframe: pd.DataFrame
        Dataframe with non-negative real numbers. 
    
    Returns
    -------
    pd.DataFrame
        DataFrame transformed using correspondence analysis transformation
    """
    D1 = np.array(input_dataframe)
    D1 = D1/D1.sum()
    cm = np.sum(D1, axis = 0)
    col_keep = cm != 0
    rm = np.sum(D1, axis = 1)
    row_keep = rm != 0

    # # CA transform and randomised SVD
    rc = np.outer(rm[row_keep], cm[col_keep])
    P = (D1[row_keep, :][:, col_keep] - rc)/np.sqrt(rc)
    return(pd.DataFrame(P, index = input_dataframe.index[row_keep], columns = input_dataframe.columns[col_keep]))


def NPN(I, ind, dist, rho):
    """Helper function for NN_density_cluster"""
    j = rho[ind[I, 1:]] > rho[I]
    if np.sum(j) > 0:
        argmin = np.argmin(dist[I, 1:][j])
        return ind[I, 1:][j][argmin]
    else:
        return -1

def NN_density_cluster(coords: pd.DataFrame, K_nn: int, p: float = 2, metric : str = 'minkowski'):
    """Density gradient clustering using a nearest neighbours approximation

    Parameters
    ----------
    coords : pd.DataFrame
        DataFrame containing data to be clustered. The rows are the observations to be clustered. 
    
    K_nn : int
        The number of nearest neighbours used to construct the density approximation.
    p : float
        Parameter used to define the minkowski distance. Minkowski distance with p = 2 is equivalent to the familiar euclidean distance.
    metric : str
        Parameter used to choose the distance metric. Default is Minkowski. 'chebyshev' is equivalent to the limit of the minkowski distance as p tends to infinity. The list of acceptable metrics can be found in scipy.spatial.distance

    Returns
    -------
    list
        The first element of the list is the cluster membership of the input DataFrame rows.
        The second element of the list are the highest density points of each cluster.
    """

    NN = NearestNeighbors(n_neighbors=K_nn, n_jobs=-1, p = p, metric = metric)
    NN.fit(coords)
    dist, ind = NN.kneighbors(coords)
    mindist = np.min(dist[dist > 0])  # some cells have exactly the same profile thus sometimes the Kth NND can be zero. 
    dist[dist == 0] = mindist
    rho = 1/dist[:, K_nn-1]

    B = [NPN(I, ind, dist, rho) for I in range(0, ind.shape[0])]
    bigG = [] #path to densest node
    for n in range(0, len(B)):
        g = B[n]
        G = [g]
        while g != -1:
            g = B[g]
            G.append(g)
        bigG.append(G)

    bigG_mem = pd.DataFrame([bigG[k][-2] if len(bigG[k]) > 1 else k for k in range(0, len(bigG))], index = coords.index, columns=['membership']) 
    # # get coords corresponding to densest nodes
    exemplars = coords.iloc[np.unique(bigG_mem)]

    DF1 = bigG_mem.loc[exemplars.index] # column 'membership' is numerical index of densest points, dataframe index is the points' IDs
    DF1.columns = ['membership']
    DF2 = pd.DataFrame(exemplars.groupby(list(exemplars)).ngroup(), columns = ['duplicate']) # column 'duplicate' labels densest points according to whether they are identical to each other
    exemplars = DF1.merge(DF2, right_index=True, left_index = True).merge(exemplars, right_index = True, left_index = True) # merge the information into one dataframe

    subcluster_membership = bigG_mem.replace(to_replace=list(exemplars['membership']), value = list(exemplars['duplicate']))
    density = pd.DataFrame({"local_density": rho}, index = coords.index)

    return(list((subcluster_membership, exemplars, density)))

def veclen(x):
    """A helper function to obtain the length of a vector"""
    return(np.sqrt(np.sum(np.square(x))))

# def var_expl(P_svd, i): # variance explained over rows, columns
#     """A helper function to obtain vector lengths in a matrix decomposition"""
#     s0 = P_svd[0][:, 0:i]
#     s1 = P_svd[1][0:i]
#     s2 = P_svd[2][0:i, :]
#     P_reconstruct = np.matmul(np.matmul(s0, np.diag(s1)), s2)
#     #Lcol = np.apply_along_axis(veclen, 0, P_reconstruct)
#     #Lrow = np.apply_along_axis(veclen, 1, P_reconstruct)
#     Lcol = np.sum(P_reconstruct**2, axis = 0)**0.5
#     Lrow = np.sum(P_reconstruct**2, axis = 1)**0.5
#     return([Lcol, Lrow])

def var_expl(svd0, svd1, svd2, i): # variance explained over rows, columns
    """A helper function to obtain vector lengths in a matrix decomposition"""
    s0 = np.array(svd0.iloc[:, 0:i])
    s1 = np.transpose(np.array(svd1.iloc[0:i, ]))[0]
    s2 = np.transpose(np.array(svd2.iloc[:, 0:i]))
    P_reconstruct = np.matmul(np.matmul(s0, np.diag(s1)), s2)
    Lcol = np.sum(P_reconstruct**2, axis = 0)**0.5
    Lrow = np.sum(P_reconstruct**2, axis = 1)**0.5
    return([Lcol, Lrow])


def reconstruct(path_name, pickle = False): # give option to use pickled Transform object
    """A helper function to create matrix from saved matrix decomposition"""
    if not pickle:
        s0 = pd.read_csv(path_name + 'svd0.txt.gz')
        s1 = pd.read_csv(path_name + 'svd1.txt')
        s2 = pd.read_csv(path_name + 'svd2.txt.gz')
    else:
        file = open(path_name, 'rb')
        Y = pickle.load(file)
        file.close()
        s0 = Y.svd0
        s1 = Y.svd1
        s2 = Y.svd2
    s0.index = s0.iloc[:, 0]
    s0 = s0.drop(s0.columns[0], axis = 1)
    s2.index = s2.iloc[:, 0]
    s2 = s2.drop(s2.columns[0], axis = 1)
    P = np.matmul(np.matmul(np.array(s0), np.diag(np.transpose(np.array(s1))[0, ])), np.transpose(np.array(s2)))
    P = pd.DataFrame(P, index = s0.index, columns = s2.index)
    return(P)

def directory_structure(path):
    """A function to create an output directory structure"""
    if not os.path.isdir(path + 'output/'):
        os.mkdir(path + 'output/')
    else:
        print("The output directory already exists.")
    if not os.path.isdir(path + 'output/transform'):
        os.mkdir(path + 'output/transform')
    if not os.path.isdir(path + 'output/cells'):
        os.mkdir(path + 'output/cells')
    if not os.path.isdir(path + 'output/genes'):
        os.mkdir(path + 'output/genes')
    if not os.path.isdir(path + 'output/specificity_network'):
        os.mkdir(path + 'output/specificity_network')
    if not os.path.isdir(path + 'output/figures'):
        os.mkdir(path + 'output/figures')
    if not os.path.isdir(path + 'output/neighbourhood_flow'):
        os.mkdir(path + 'output/neighbourhood_flow')

    if (
        os.listdir(path + 'output/transform') != [] or
        os.listdir(path + 'output/cells') != [] or
        os.listdir(path + 'output/genes') != [] or
        os.listdir(path + 'output/specificity_network') != [] or
        os.listdir(path + 'output/figures') != [] or
        os.listdir(path + 'output/neighbourhood_flow') != [] 
        ):
        raise FileExistsError("Directories already contain results. \nYou will overwrite these results if you proceed. \nPlease consider creating a new parent directory")
      
def MultiSampleConcat(path_names, sample_ids = None): # give option to just use exemplar cells # give warning in docs that columns with duplicated names are deleted
    """A function to concatenate transformed samples.
    
    Parameters
    ----------
    path_names : list
        A list of strings with path names of matrix decomposition location
    sample_ids : list
        An optional list of strings with sample labels to include in DataFrame index

    Returns
    -------

    pd.DataFrame
        A DataFrame with concatenated samples

    """
    
    P = [reconstruct(p) for p in path_names]

    colnames = [list(p.columns[~p.columns.duplicated()]) for p in P]
    colnames_intersection = list(set.intersection(*map(set, colnames)))

    P = [p.loc[:, ~p.columns.duplicated()][colnames_intersection] for p in P]

    if sample_ids == None:
        sample_ids = range(0, len(P))

    for i in sample_ids:
        P[i].index = P[i].index + '_' + str(i)

    P = pd.concat(P)
    return(P)


def local_gof(membership_matrix, coordinate_matrix, ncomp:int = 3):
    """A function to calculate local goodness of fit within clusters
    
    Parameters
    ----------

    membership_matrix : output of ds.Cluster
        Gene/cell cluster membership

    coordinate_matrix : output of ds.Transform
        Spherical gene/cell coordinates
    
    ncomp : int
        Number of cluster-wise svd components to use
    
    """
    proj_dist_matrix = []
    sval_list = []

    for i in range(0, np.max(membership_matrix['subcluster']) + 1):
        clus_ID = membership_matrix.loc[membership_matrix['subcluster'] == i].index
        data1 = coordinate_matrix.loc[clus_ID]
        #data1 = data1.drop('ID', axis = 1)
        if data1.shape[0] > 5:
            svd_data1 = np.linalg.svd(data1, full_matrices=False)
            svd_proj = np.matmul(coordinate_matrix, np.transpose(svd_data1[2][range(0, ncomp), :]))
            proj_dist = pd.DataFrame(np.sum(np.array(svd_proj)**2, axis = 1)**0.5, index = svd_proj.index)
            svals = svd_data1[1]
        else:
            data_mean = data1.mean(axis = 0)
            data_mean = data_mean/np.sqrt(np.sum(data_mean**2))
            proj_dist = np.matmul(coordinate_matrix, data_mean)
            svals = None
        proj_dist_matrix.append(proj_dist)
        sval_list.append(svals)
        
    proj_dist_matrix = pd.concat(proj_dist_matrix, axis = 1)
    proj_max = proj_dist_matrix.max(axis = 1)
    proj_mean = proj_dist_matrix.mean(axis = 1)

    subcluster_proj = []

    for i in range(0, np.max(membership_matrix['subcluster']) + 1):
        clus_ID = membership_matrix.loc[membership_matrix['subcluster'] == i].index
        subcluster_proj.append(proj_dist_matrix.loc[clus_ID].iloc[:, i])

    subcluster_proj = pd.DataFrame({'clus': pd.concat(subcluster_proj)})
    gof_df = pd.DataFrame({'mean': proj_mean, 'max': proj_max})
    gof_df = gof_df.join(subcluster_proj)

    return([gof_df, sval_list])

##############################################
class Transform:
    """
    A class to transform and perform matrix decomposition

    Attributes
    ---------- 
    col_keep
        columns with sums != 0
    row_keep
        rows with sums != 0
    svd0, svd1, svd2 : nd.array
        left singular vectors, singular values, right singular vectors respectively
    gene_coord : pd.DataFrame
        Spherically transformed, dimension reduced gene coordinates
    cell_coord : pd.DataFrame
        Spherically transformed, dimension reduced cell coordinates
    gof_cell : pd.DataFrame
        Goodness of fit statistics
    gof_gene : pd.DataFrame
        Goodness of fit statistics
    cell_umap : pd.DataFrame
        umap of cell_coord
    gene_umap : pd.DataFrame
        umap of gene_coord

    Methods
    -------
    umap(cell = True, gene = True, ncomp_cell:int = None, ncomp_gene:int = None)
        Creates a umap from cell_coord and/or gene_coord
    save(path, svd = True, cell_coord = True, gene_coord = True, cell_umap = True, gene_umap = True, goodness_of_fit = True)
        Saves output in R-friendly format
    """

    def __init__(self, input_dataframe: pd.DataFrame, ncomps: int = 50, n_iter: int = 10, n_oversamples: int = 50, transform: bool = True): #input_dataframe must be pd.DataFrame
        """
        Parameters
        ----------
        input_dataframe : pd.DataFrame
            The input count matrix. Rows are cells (or sampling units) and columns are genes (or biomarkers/features). In case of multisample analysis, row IDs (index) shouldn't contain underscores '_' . 
        ncomps : int
            The number of singular value components to compute. Default is 50
        n_iter : int
            The number of iterations to use in the randomised singular value decomposition. Default is 10.
        n_oversamples: int 
            Parameter controlling sampling of input_dataframe in the randomised singular value decomposition. Tune this parameter before tuning n_iter. See scikit docs for full details. 
        transform : bool
            Boolean indicating whether to perform the Correspondence Analysis transformation before SVD. Default is True. 
        goodness_of_fit : bool
            Boolean indicating whether to create goodness of fit statistics. 
        """

        self.col_names = input_dataframe.columns
        self.row_names = input_dataframe.index
  
        # normalise, row and columns masses
        # don't actually need this if transform == False, but just keep it for consistency
        # D1 = input_dataframe/(input_dataframe.sum().sum())
        # cm = D1.sum(axis = 0)
        # self.col_keep = cm != 0
        # rm = D1.sum(axis = 1)
        # self.row_keep = rm != 0

        D1 = np.array(input_dataframe)
        D1 = D1/D1.sum()
        cm = np.sum(D1, axis = 0)
        self.col_keep = cm != 0
        rm = np.sum(D1, axis = 1)
        self.row_keep = rm != 0

        # # CA transform and randomised SVD
        if transform:
            rc = np.outer(rm[self.row_keep], cm[self.col_keep])
            P = (D1[self.row_keep, :][:, self.col_keep] - rc)/np.sqrt(rc)
        else:
            P = np.array(input_dataframe.loc[self.row_keep, self.col_keep])
        #P_svd = randomized_svd(P.to_numpy(), n_components=ncomps, n_iter=n_iter, random_state = 0) # or dask svd for large data
        P_svd = randomized_svd(P, n_components=ncomps, n_iter=n_iter, n_oversamples = n_oversamples)

        self.cell_total_length = np.sum(P**2, axis = 1)**0.5
        self.gene_total_length = np.sum(P**2, axis = 0)**0.5
        self.ncomps = ncomps

        if transform:
            self.cell_raw_stats = pd.DataFrame({'mean_count': np.mean(np.array(input_dataframe.loc[self.row_keep, self.col_keep]), axis = 1)})
            self.cell_raw_stats.insert(0, 'median_count', np.median(np.array(input_dataframe.loc[self.row_keep, self.col_keep]), axis = 1))
            self.cell_raw_stats.insert(0, 'sd_count', np.std(np.array(input_dataframe.loc[self.row_keep, self.col_keep]), axis = 1))
            self.cell_raw_stats.insert(0, 'total_count', np.sum(np.array(input_dataframe.loc[self.row_keep, self.col_keep]), axis = 1))
            self.gene_raw_stats = pd.DataFrame({'mean_count': np.mean(np.array(input_dataframe.loc[self.row_keep, self.col_keep]), axis = 0)})
            self.gene_raw_stats.insert(0, 'median_count', np.median(np.array(input_dataframe.loc[self.row_keep, self.col_keep]), axis = 0))
            self.gene_raw_stats.insert(0, 'sd_count', np.std(np.array(input_dataframe.loc[self.row_keep, self.col_keep]), axis = 0))
            self.gene_raw_stats.insert(0, 'total_count', np.sum(np.array(input_dataframe.loc[self.row_keep, self.col_keep]), axis = 0))

        def skew_quartile(X): 
            q1 = np.quantile(X, 1/4)
            q2 = np.quantile(X, 2/4)
            q3 = np.quantile(X, 3/4)
            return((q3 + q1 - 2*q2)/(q3 - q1))
        
        skew_cell = np.abs(np.apply_along_axis(skew_quartile, 0, P_svd[0]))
        skew_gene = np.abs(np.apply_along_axis(skew_quartile, 0, np.transpose(P_svd[2])))
        self.skew = pd.DataFrame({'skew_cell' : skew_cell, 'skew_gene': skew_gene}, index = range(1, ncomps + 1))

        cnames = ['comp' + str(a) for a in range(1, ncomps + 1)]
        idx = self.row_names[self.row_keep]
        self.svd0 = pd.DataFrame(P_svd[0], index = idx, columns = cnames)
            
        self.svd1 = pd.DataFrame({'singular_values': P_svd[1]})
            
        cnames = ['comp' + str(a) for a in range(1, ncomps + 1)]
        idx = self.col_names[self.col_keep]
        self.svd2 = pd.DataFrame(np.transpose(P_svd[2]), index = idx, columns = cnames)

        # spherical cell coordinates
        ca_comps = P_svd[0] * P_svd[1] # cells
        cnames = ['cell_coord' + str(a) for a in range(1, ncomps + 1)]
        idx = self.row_names[self.row_keep]
        cell_coord = spherical_transform(ca_comps)
        self.cell_coord = pd.DataFrame(cell_coord, index = idx, columns = cnames)

        # spherical gene coordinates
        ca_comps = np.transpose(P_svd[2]) * P_svd[1] # genes
        cnames = ['gene_coord' + str(a) for a in range(1, ncomps + 1)]
        idx = self.col_names[self.col_keep]
        gene_coord = spherical_transform(ca_comps)
        self.gene_coord = pd.DataFrame(gene_coord, index = idx, columns = cnames)

               
    def goodness_of_fit(self, N: int = 10, min_percent = sum([list(range(1, 20)), list(range(20, 100, 10)), [95, 99]], [])):
        """
        Parameters
        ----------
        min_percent
            List of values on the interval (0, 100) indicating minimum percentages of variance for reconstruction error plot
        """

        L = list(range(0, self.ncomps, int(np.floor((self.ncomps)/N))))
        if L[-1] != self.ncomps:
            L.append(self.ncomps)
        L.pop(0)
        VE = [var_expl(self.svd0, self.svd1, self.svd2, i) for i in [i for i in L]]
        self.gof_gene = np.transpose(pd.DataFrame([x[0] for x in VE]))
        self.gof_cell = np.transpose(pd.DataFrame([x[1] for x in VE]))
        cnames = ['ncomps' + str(i) for i in L]
        self.gof_gene.columns = cnames
        self.gof_cell.columns = cnames
        self.gof_gene.index = self.svd2.index
        self.gof_cell.index = self.svd0.index
        
        self.gof_cell.insert(0, 'leverage', np.sum(self.svd0**2, axis = 1))   
        self.gof_gene.insert(0, 'leverage', np.sum(self.svd2**2, axis = 1))
        
        def err_freq(gof, c, min_percent, total_length) :
            L = gof.shape[0]
            err = gof[c]**2/total_length**2 * 100
            return([sum(1 for x in err if x > y)/L*100 for y in min_percent])
        
        def gof_plot(gof, cols, min_percent, total_length) :
            gp = np.transpose(pd.DataFrame([err_freq(gof, x, min_percent, total_length) for x in cols]))
            gp.columns = cols
            gp.insert(0, 'min_percent', min_percent)
            gp = gp.melt('min_percent')
            gp.insert(0, 'ncomps', [int(x) for x in gp['variable'].replace('ncomps', '', regex = True)])
            return(gp)
        self.gp_cell = gof_plot(self.gof_cell, cnames, min_percent, self.cell_total_length)
        self.gp_gene = gof_plot(self.gof_gene, cnames, min_percent, self.gene_total_length)

    def umap(self, cell = True, gene = True, ncomp_cell : int = None, ncomp_gene : int = None) :
        """
        Peforms umap on cell_coord and /or gene_coord

        Parameters
        ----------
        cell : bool
            Boolean indicating whether to perform umap on cell_coord. Default is True
        gene : bool
            Boolean indicating whether to perform umap on gene_coord. Default is True
        ncomp_cell : int
            Integer less than number of rows in cell_coord, indicating to perform umap on top ncomp_cell components. 
            Default is None and corresponds to using all components. 
        ncomp_gene : int
            Integer less than number of rows in gene_coord, indicating to perform umap on top ncomp_gene components. 
            Default is None and corresponds to using all components. 

        """

        # gene and cell umap coordinates
        if cell:
            reducer = umap.UMAP()
            if ncomp_cell == None:
                cell_umap = reducer.fit_transform(self.cell_coord)
            elif ncomp_cell <= self.cell_coord.shape[1]:
                cell_umap = reducer.fit_transform(self.cell_coord.iloc[:, 0:ncomp_cell])
            else:
                raise IndexError("ncomp_cell is larger than the number of columns in cell_coord")
            cnames = ['umap1', 'umap2']
            idx = self.row_names[self.row_keep]
            self.cell_umap = pd.DataFrame(cell_umap, index = idx, columns = cnames)
        
        if gene:
            reducer = umap.UMAP()
            if ncomp_gene == None:
                gene_umap = reducer.fit_transform(self.gene_coord)
            elif ncomp_gene <= self.gene_coord.shape[1]:
                gene_umap = reducer.fit_transform(self.gene_coord.iloc[:, 0:ncomp_gene])
            else:
                raise IndexError("ncomp_gene is larger than the number of columns in gene_coord")
            cnames = ['umap1', 'umap2']
            idx = self.col_names[self.col_keep]
            self.gene_umap = pd.DataFrame(gene_umap, index = idx, columns = cnames)
             
    def save(self, 
             path, 
             svd = True, 
             cell_coord = True, 
             gene_coord = True,
             svd_components_skew = True, 
             cell_umap = True, 
             gene_umap = True, 
             goodness_of_fit = True,
             goodness_of_fit_figure = True, 
             cell_raw_stats = True,
             gene_raw_stats = True
             ):
        """
        Save R-friendly output

        Parameters
        ----------
        path : str
            String indicating path to save directory
        svd : bool
            Boolean indicating whether to save SVD output. Default is True
        cell_coord : bool
            Boolean indicating whether to save cell_coord output. Default is True
        gene_coord : bool
            Boolean indicating whether to save gene_coord output. Default is True
        svd_components_skew : bool
            Boolean indicating whether to save svd components skew output. Default is True
        cell_umap : bool
            Boolean indicating whether to save cell_umap output. Default is True
        gene_umap : bool
            Boolean indicating whether to save gene_umap output. Default is True
        goodness_of_fit : bool
            Boolean indicating whether to save goodnes of fit output. Default is True
        goodness_of_fit_figure : bool
            Boolean indicating whether to save goodness of fit figure. Default is True
        cell_raw_stats: bool
            Boolean indicating whether to save cell raw stats
        gene_raw_stats: bool
            Boolean indicating whether to save gene raw stats
        """

        pd.DataFrame({'row_keep':self.row_keep}, index = self.row_names).to_csv(path + 'row_keep.txt', index_label = 'ID')
        pd.DataFrame({'col_keep':self.col_keep}, index = self.col_names).to_csv(path + 'col_keep.txt', index_label = 'ID')
        
        if svd:             
            self.svd0.to_csv(path + 'svd0.txt.gz')
            self.svd1.to_csv(path + 'svd1.txt', index = False)            
            self.svd2.to_csv(path + 'svd2.txt.gz')
            
        if cell_coord:
            self.cell_coord.to_csv(path + 'cell_coord.txt.gz', index_label = 'ID')

        if gene_coord: 
            self.gene_coord.to_csv(path + 'gene_coord.txt.gz', index_label = 'ID')

        if svd_components_skew:
            self.skew.to_csv(path + 'svd_skew.txt.gz', index_label = 'component')
            def skew_figure(X, path):
                plot = sb.lineplot(x = range(1, len(X) + 1), y = X)
                plot.set(title = 'SVD component skew')
                plot = plot.get_figure()
                plot.savefig(path)
                plt.close()

            skew_figure(self.skew['skew_cell'], path + 'svd_skew_cell.png')
            skew_figure(self.skew['skew_gene'], path + 'svd_skew_gene.png')

        if cell_umap and hasattr(self, 'cell_umap'):
            self.cell_umap.to_csv(path + 'cell_umap.txt.gz', index_label = 'ID')

        if gene_umap and hasattr(self, 'gene_umap'):
            self.gene_umap.to_csv(path + 'gene_umap.txt.gz', index_label = 'ID')

        if goodness_of_fit and hasattr(self, 'gof_cell'):
            self.gof_cell.to_csv(path + 'gof_cell.txt.gz', index_label = 'ID')
            self.gof_gene.to_csv(path + 'gof_gene.txt.gz', index_label = 'ID')
            self.gp_cell.to_csv(path + 'reconstruct_err_cell.txt.gz')
            self.gp_gene.to_csv(path + 'reconstruct_err_gene.txt.gz')

        if cell_raw_stats and hasattr(self, 'cell_raw_stats'):
            self.cell_raw_stats.to_csv(path + 'cell_raw_stats.txt.gz', index_label = 'ID')
        
        if gene_raw_stats and hasattr(self, 'gene_raw_stats'):
            self.gene_raw_stats.to_csv(path + 'gene_raw_stats.txt.gz', index_label = 'ID')

        if goodness_of_fit_figure and hasattr(self, 'gof_cell'):
            def gof_figure(G, path): # G is self.gp_cell/self.gp_gene
                G.insert(0, 'log_value', np.log10(G['value'] + 0.001))
                #G(0, 'mp', pd.Categorical(T.gp_gene['min_percent']))
                plot = sb.lineplot(data = G, x = 'ncomps', y = 'log_value', hue = 'min_percent')
                plot.set(title = 'min_percent: ' + str(G['min_percent'].unique()))
                plot = plot.get_figure()
                plot.savefig(path, pad_inches = 0.5, bbox_inches = 'tight', dpi = 500)
                plt.close()
                G.drop('log_value', axis = 1)
            
            gof_figure(self.gp_cell, path + 'reconstruct_err_cell.png')
            gof_figure(self.gp_gene, path + 'reconstruct_err_gene.png')
    
#################################################
class Cluster:
    """
    A Class to cluster an instance of the Transform class

    Attributes
    ----------
    membership_all : pd.DataFrame
        DataFrame containing subcluster membership of each cell or gene
    subcluster_points : pd.DataFrame
        DataFrame containing 'exemplar points' i.e. the densest point of each cluster
    subcluster_similarity : pd.DataFrame
        DataFrame containing all pairwise similarities between subclusters

    Methods
    -------
    save
        Save R-friendly output
    """

    def __init__(self, coords: Transform, K_nn: int = 5, clus_steps = 5, mode = 'cells', similarity_threshold: float = 0.2, p: int = 2, metric : str = 'minkowski', ncomp_clus: int = 0, prune : str = 'threshold', prune_K_nn : int = 5): 
        """
        Parameters
        ----------
        coords : Transform
            an instance of the class Transform
        K_nn : int
            The number of nearest neighbours which is used to approximate local density
        clus_steps: int
            The number of steps to use in the walktrap clustering algorithm
        mode : str
            Mode is a choice of 'cells' or 'genes', indicating whether to cluster the cell_coords or gene_coords.  
        similarity_threshold : float
            The threshold used to create the gene/cell scaffold graph. Similarity values below the threshold are set to zero and edges are omitted. 
            Default is 0.2. Used for threshold pruning method, and as a first step for backbone pruning method. 
        p: float
            A parameter controlling the minkowski distance. The default p = 2 is equivalent to the most commonly used euclidean distance. p = 1 is equivalent to the Manhattan distance. 
        ncomp_clus: float
            A parameter controlling how many components of cell/gene coords to use when clustering. Default is the dummy value 0, which translates to using all the components contained in cell/gene coords. 
        metric : str
            Parameter used to choose the distance metric. Default is 'minkowski'. 'chebyshev' is equivalent to the limit of the minkowski distance as p tends to infinity. The list of acceptable metrics can be found in scipy.spatial.distance
        prune : str
            Parameter used to choose thte graph pruning method. Options are 'threshold' for a simple threshold using the parameter similarity_threshold, 'NN' to choose the nearest neighbours, and 'backbone' a node degree distribution preserving method. 
        prune_K_nn : int
            Parameter used to threshold using K_nn method. ignored if prune = 'threshold'
            
        """

        self.ncomp_clus = ncomp_clus
        self.p = p
        self.metric = metric

        K_nn = K_nn + 1
        if mode == 'cells':
            if ncomp_clus == 0:
                clus = NN_density_cluster(coords.cell_coord, K_nn = K_nn, p = p, metric = metric)
            elif ncomp_clus > 1 and ncomp_clus <= coords.cell_coord.shape[1]:
                ca_comps = np.transpose((np.transpose(np.array(coords.svd0)) * np.array(coords.svd1)))
                ca_comps = ca_comps[:, range(0, ncomp_clus)]
                cnames = ['cell_coord' + str(a) for a in range(1, ncomp_clus + 1)]
                idx = coords.row_names[coords.row_keep]
                cell_coord = spherical_transform(ca_comps)
                cell_coord = pd.DataFrame(cell_coord, index = idx, columns = cnames)
                clus = NN_density_cluster(cell_coord, K_nn = K_nn, p = p, metric=metric)
            else: 
                raise IndexError("ncomp_clus is larger than the number of columns in cell_coord")
        elif mode == 'genes':
            if ncomp_clus == 0:
                clus = NN_density_cluster(coords.gene_coord, K_nn = K_nn, p = p, metric = metric)
            elif ncomp_clus > 1 and ncomp_clus <= coords.gene_coord.shape[1]:
                ca_comps = np.transpose((np.transpose(np.array(coords.svd2)) * np.array(coords.svd1)))
                ca_comps = ca_comps[:, range(0, ncomp_clus)]
                cnames = ['gene_coord' + str(a) for a in range(1, ncomp_clus + 1)]
                idx = coords.col_names[coords.col_keep]
                gene_coord = spherical_transform(ca_comps)
                gene_coord = pd.DataFrame(gene_coord, index = idx, columns = cnames)
                clus = NN_density_cluster(gene_coord, K_nn = K_nn, p = p, metric = metric)
            else:
                raise IndexError("ncomp_clus is larger than the number of columns in gene_coord")
        else:
            raise ValueError('mode is either cells or genes')

        self.mode = mode

        # # get coords and drop duplicated points
        T = clus[1][~clus[1].duplicated('duplicate')]
        # # dot product of exemplars 
        dotprod = np.matmul(T.drop(columns = ['membership', 'duplicate']).to_numpy(), np.transpose(T.drop(columns = ['membership', 'duplicate']).to_numpy()))
        dotprod_index = T['duplicate']
        self.subcluster_similarity = pd.DataFrame(dotprod, index = dotprod_index, columns = dotprod_index).rename_axis(index = None, columns = None)#.sort_index().sort_index(axis = 1)
        
        if prune == 'threshold':
            dotprod_thres = dotprod.copy()
            dotprod_thres[dotprod_thres < similarity_threshold] = 0

        if prune == 'NN':
            def similarity_NN_threshold_symmetric(graph, K_nn):
                for u in range(0, graph.shape[1]): 
                    toprank = np.argsort(np.argsort(-graph[:, u])) > K_nn
                    graph[toprank, u] = 0
                graph = np.add(graph, np.transpose(graph))
                return(graph)
            
            dotprod_thres = similarity_NN_threshold_symmetric(dotprod, prune_K_nn)

        # if prune == 'backbone':
        #     def pval(i, j, ki, t, sim):
        #         p = ki[i]*ki[j]/(2*t**2)
        #         return(1 - binom.cdf(sim[i, j], t, p))
            
        #     def binom_backbone(sim, alpha):
        #         np.fill_diagonal(sim, 0)
        #         sim = np.round(sim, 2)*100
        #         ki = np.sum(sim, 0)
        #         t = sum(ki) * 0.5
        #         pval_mtx = [[pval(i, j, ki, t, sim) for j in range(0, len(ki))] for i in range(0, len(ki))]
        #         pval_mtx = pd.DataFrame(pval_mtx)
        #         pval_mtx[pval_mtx > alpha] = 1
        #         pval_mtx[pval_mtx <= alpha] = 2
        #         pval_mtx = pval_mtx - 1
        #         return(sim * pval_mtx)
        #     dotprod_thres = dotprod.copy()
        #     dotprod_thres[dotprod_thres < similarity_threshold] = 0
        #     dotprod_thres = binom_backbone(dotprod_thres, alpha)

        self.subcluster_similarity_prune = pd.DataFrame(dotprod_thres, index = dotprod_index, columns = dotprod_index).rename_axis(index = None, columns = None)
        ##################################
        # # create graph
        graph = ig.Graph().Weighted_Adjacency(dotprod_thres, mode = 'undirected')
        # # betweeness, coreness and create clusters
        BT = graph.betweenness(weights = 'weight')
        CN = graph.coreness()
        Strength = graph.strength(vertices = graph.get_vertex_dataframe().index, weights = 'weight')
        
        def median(X):
            if len(X) > 2:
                return np.median(X)
            else:
                return None
            
        def mad(X):
            if len(X) > 2:
                med = np.median(X)
                resid = X - med
                mad = np.median(np.abs(resid))
                return mad
            else:
                return None

        T.insert(0, 'BT', BT)
        T.insert(0, 'CN', CN)
        T.insert(0, 'Strength', Strength)
        GN = [graph.neighbors(i) for i in range(0, graph.vcount())]
        BT_median = [median([BT[i] for i in GN[j]]) for j in range(0, graph.vcount())]
        BT_mad = [mad([BT[i] for i in GN[j]]) for j in range(0, graph.vcount())]
        T.insert(0, 'BT_median', BT_median)
        T.insert(0, 'BT_mad', BT_mad)

        CN_median = [median([CN[i] for i in GN[j]]) for j in range(0, graph.vcount())]
        CN_mad = [mad([CN[i] for i in GN[j]]) for j in range(0, graph.vcount())]
        T.insert(0, 'CN_median', CN_median)
        T.insert(0, 'CN_mad', CN_mad)

        Strength_median = [median([Strength[i] for i in GN[j]]) for j in range(0, graph.vcount())]
        Strength_mad = [mad([Strength[i] for i in GN[j]]) for j in range(0, graph.vcount())]
        T.insert(0, 'Strength_median', Strength_median)
        T.insert(0, 'Strength_mad', Strength_mad)

        if graph.vcount() < 500: # in future, give more control over choice of graph clustering. 
            w = np.array(graph.es['weight']).argsort().argsort() + 1
            clusWT = ig.Graph.community_walktrap(graph, weights = (w**2)/100, steps = clus_steps).as_clustering().membership
        else:
            clusWT = ig.Graph.community_leiden(graph, resolution = 0.5).membership
        T.insert(0, 'clus', clusWT)

        graph_membership = clus[0].replace(to_replace = list(T['duplicate']), value = list(T['clus']))
        self.membership_all = pd.DataFrame({'subcluster': clus[0]['membership'], 'graph_cluster': graph_membership['membership'], 'local_density': clus[2]['local_density']})

        self.subcluster_points = T[[ 'CN', 'BT', 'Strength', 'CN_median', 'CN_mad', 'BT_median', 'BT_mad', 'Strength_median', 'Strength_mad', 'clus', 'duplicate']].rename(columns = {'duplicate': 'subcluster', 'clus': 'graph_cluster', 'CN': 'coreness', 'BT': 'betweenness', 'CN_median': 'coreness_median', 'CN_mad': 'coreness_mad', 'BT_median': 'betweeness_median', 'BT_mad': 'betweeness_mad'})

        A = self.membership_all['subcluster'].value_counts()
        B = A.index
        C = self.subcluster_points['subcluster'].replace(to_replace = B, value = A)
        self.subcluster_points.insert(0, 'subcluster_count', C)

    def local_goodness_of_fit(self, coords: Transform, ncomp : int = 3):
        if self.mode == 'cells':
            if self.ncomp_clus == 0:
                self.lgof = local_gof(self.membership_all, coords.cell_coord, ncomp)
            elif self.ncomp_clus > 1 and self.ncomp_clus <= coords.cell_coord.shape[1]:
                ca_comps = np.transpose((np.transpose(np.array(coords.svd0)) * np.array(coords.svd1)))
                ca_comps = ca_comps[:, range(0, self.ncomp_clus)]
                cnames = ['cell_coord' + str(a) for a in range(1, self.ncomp_clus + 1)]
                idx = coords.row_names[coords.row_keep]
                cell_coord = spherical_transform(ca_comps)
                cell_coord = pd.DataFrame(cell_coord, index = idx, columns = cnames)
                self.lgof = local_gof(self.membership_all, cell_coord, ncomp)
            else: 
                raise IndexError("ncomp_clus is larger than the number of columns in cell_coord")
                
        if self.mode == 'genes' :
            if self.ncomp_clus == 0:
                self.lgof = local_gof(self.membership_all, coords.gene_coord, ncomp)
            elif self.ncomp_clus > 1 and self.ncomp_clus <= coords.gene_coord.shape[1]:
                ca_comps = np.transpose((np.transpose(np.array(coords.svd2)) * np.array(coords.svd1)))
                ca_comps = ca_comps[:, range(0, self.ncomp_clus)]
                cnames = ['gene_coord' + str(a) for a in range(1, self.ncomp_clus + 1)]
                idx = coords.col_names[coords.col_keep]
                gene_coord = spherical_transform(ca_comps)
                gene_coord = pd.DataFrame(gene_coord, index = idx, columns = cnames)
                self.lgof = local_gof(self.membership_all, gene_coord, ncomp)
            else: 
                raise IndexError("ncomp_clus is larger than the number of columns in gene_coord")
           
    def save(self, 
             path, 
             membership_all = True, 
             subcluster_points = True,
             subcluster_similarity = True, 
             lgof = True
             ):
        """
        Save R-friendly output
        
        Parameters
        ----------
        path : str
            String indicating path to save directory
        membership_all : bool
            Boolean indicating whether to save membership_all output
        subcluster_points : bool
            Boolean indicating whether to save subcluster_points output
        subcluster_similarity : bool
            Boolean indicating whether to save subcluster_similarity output
        lgof : bool
            Boolean indicating whether to save local goodness of fit matrix
        """

        if membership_all:
            self.membership_all.to_csv(path + 'membership_all.txt.gz', index_label = 'ID')
       
        if subcluster_points:
            self.subcluster_points.to_csv(path + 'subcluster_points.txt.gz', index_label = 'ID')

        if subcluster_similarity:
            self.subcluster_similarity.to_csv(path + 'subcluster_similarity.txt.gz', index_label = 'ID')
            self.subcluster_similarity_prune.to_csv(path + 'subcluster_similarity_prune.txt.gz', index_label = 'ID')

        if lgof and hasattr(self, 'lgof'):
            self.lgof[0].to_csv(path + self.mode + '_lgof.txt.gz', index_label = 'ID')

########################################################
class SpecificityNetwork:
    """
    A class used to create a Specificity Network

    Attributes
    ----------
    bip_graph : pd.DataFrame
        A bipartite graph representing the association between gene subclusters and cell subcluster
    virtual_stain : pd.DataFrame
        Creates a representative value of each gene cluster for each cell

    Methods
    -------
    save
        Save R-friendly output
    """

    def __init__(self, coords: Transform, cells: Cluster, genes: Cluster, similarity_threshold = 0.2):
        """
        Parameters
        ----------
        coords : Transform
            An instance of the class Transform
        cells : Cluster
            An instance of the class Cluster, created from argument of coords
        genes : Cluster
            An instance of the class Cluster, created from the argument of coords
        similarity_threshold : float
            The threshold used to create the bipartite graph. Similarities below this threshold are set to zero and edges are omitted
            Default is 0.2

        """


# self.ncomp_clus = ncomp_clus
#         self.p = p
#         self.metric = metric

#         K_nn = K_nn + 1
#         if mode == 'cells':
#             if ncomp_clus == 0:
#                 clus = NN_density_cluster(coords.cell_coord, K_nn = K_nn, p = p, metric = metric)
#             elif ncomp_clus > 1 and ncomp_clus <= coords.cell_coord.shape[1]:
#                 ca_comps = np.transpose((np.transpose(np.array(coords.svd0)) * np.array(coords.svd1)))
#                 ca_comps = ca_comps[:, range(0, ncomp_clus)]
#                 cnames = ['cell_coord' + str(a) for a in range(1, ncomp_clus + 1)]
#                 idx = coords.row_names[coords.row_keep]
#                 cell_coord = spherical_transform(ca_comps)
#                 cell_coord = pd.DataFrame(cell_coord, index = idx, columns = cnames)
#                 clus = NN_density_cluster(cell_coord, K_nn = K_nn, p = p, metric=metric)
#             else: 
#                 raise IndexError("ncomp_clus is larger than the number of columns in cell_coord")
#         elif mode == 'genes':
#             if ncomp_clus == 0:
#                 clus = NN_density_cluster(coords.gene_coord, K_nn = K_nn, p = p, metric = metric)
#             elif ncomp_clus > 1 and ncomp_clus <= coords.gene_coord.shape[1]:
#                 ca_comps = np.transpose((np.transpose(np.array(coords.svd2)) * np.array(coords.svd1)))
#                 ca_comps = ca_comps[:, range(0, ncomp_clus)]
#                 cnames = ['gene_coord' + str(a) for a in range(1, ncomp_clus + 1)]
#                 idx = coords.col_names[coords.col_keep]
#                 gene_coord = spherical_transform(ca_comps)
#                 gene_coord = pd.DataFrame(gene_coord, index = idx, columns = cnames)
#                 clus = NN_density_cluster(gene_coord, K_nn = K_nn, p = p, metric = metric)
#             else:
#                 raise IndexError("ncomp_clus is larger than the number of columns in gene_coord")
#         else:
#             raise ValueError('mode is either cells or genes')



        if cells.ncomp_clus != genes.ncomp_clus:
            raise ValueError("Cell clustering and gene clustering not performed on same number of components (controlled by ncomp_clus argument of Cluster)")


        if cells.ncomp_clus == 0:
            cc = coords.cell_coord.loc[cells.subcluster_points.index]
            gc = coords.gene_coord.loc[genes.subcluster_points.index]
        elif cells.ncomp_clus > 0 and cells.ncomp_clus <= coords.cell_coord.shape[1]:
            # cells
            cc = np.transpose((np.transpose(np.array(coords.svd0)) * np.array(coords.svd1)))
            cc = cc[:, range(0, cells.ncomp_clus)]
            cnames = ['cell_coord' + str(a) for a in range(1, cells.ncomp_clus + 1)]
            idx = coords.row_names[coords.row_keep]
            cc = spherical_transform(cc)
            cc = pd.DataFrame(cc, index = idx, columns = cnames)

            # genes
            gc = np.transpose((np.transpose(np.array(coords.svd2)) * np.array(coords.svd1)))
            gc = gc[:, range(0, cells.ncomp_clus)]
            cnames = ['gene_coord' + str(a) for a in range(1, cells.ncomp_clus + 1)]
            idx = coords.col_names[coords.col_keep]
            gc = spherical_transform(gc)
            gc = pd.DataFrame(gc, index = idx, columns = cnames)

            #cc = cc.iloc[:, range(0, cells.ncomp_clus)]
            #gc = gc.iloc[:, range(0, genes.ncomp_clus)]
        else:
            raise IndexError("ncomp_clus is larger than the number of columns in gene_coord")
        cnames = ['c' + str(K) for K in cells.subcluster_points['subcluster']]
        gnames = ['g' + str(K) for K in genes.subcluster_points['subcluster']]

        core_dotprod = np.matmul(np.array(cc), np.transpose(np.array(gc)))

        s1 = core_dotprod.shape[0]
        bpg1 = np.vstack((np.zeros((s1, s1), dtype = int), np.transpose(core_dotprod)))
        s2 = core_dotprod.shape[1]
        bpg2 = np.vstack((core_dotprod, np.zeros((s2, s2), dtype = int)))
        self.bip_graph = pd.DataFrame(np.hstack((bpg1, bpg2)), index = np.concatenate([cnames, gnames]), columns = np.concatenate([cnames, gnames]))

        core_dotprod[core_dotprod < similarity_threshold] = 0
        VS = [cells.membership_all['subcluster'].replace(to_replace = list(cells.subcluster_points['subcluster']), value = list(core_dotprod[:, k])) for k in range(0, core_dotprod.shape[1])]
        self.virtual_stain = pd.DataFrame(np.transpose(np.vstack(VS)), index = cells.membership_all.index, columns = gnames)
        G = self.virtual_stain.sum(axis = 1) == 0
        self.virtual_stain.insert(0, column = 'no_stain', value = G)

        def countx(x):
                return(len(x[x > 0]))
        def medianx(x):
            if len(x[x > 0]) > 0:
                return(np.median(x[x > 0]))
            else:
                return(0)
        def meanx(x):
            if len(x[x > 0]) > 0:
                return(np.mean(x[x > 0]))
            else:
                return(0)
            
        G = self.virtual_stain[gnames].apply(countx, axis = 1)
        self.virtual_stain.insert(0, column = 'which_stain', value = G)
        G = self.virtual_stain[gnames].apply(meanx, axis = 1)
        self.virtual_stain.insert(0, column = 'mean_stain', value = G)
        G = self.virtual_stain[gnames].apply(medianx, axis = 1)
        self.virtual_stain.insert(0, column = 'median_stain', value = G)

    def save(self, 
        path, 
        bip_graph = True, 
        virtual_stain = True
        ):

        if bip_graph:
            self.bip_graph.to_csv(path + 'bip_graph.txt.gz', index_label = 'ID')

        if virtual_stain:
            self.virtual_stain.to_csv(path + 'virtual_stain.txt.gz', index_label = 'ID')

class NeighbourhoodFlow: 
    """
    Class to create a neighbourhood flow, when associated spatial data exists. 

    Attributes
    ----------
    neighbourhood_flow : pd.DataFrame
        A matrix encoding the probability that cell subcluster A will occur in the neighbourhood of cell subcluster B

    Methods
    -------
    save
        Save R-friendly output    
    """

    def __init__(self, cell_cluster: Cluster, XY: pd.DataFrame, K_nn: int, dist_thres: float, cell_join_id: str = None, edge_quantile_threshold : float = 0.99):
        """
        Parameters
        ----------
        cell_cluster : Cluster
            An instance of the Cluster class
        XY : pd.DataFrame
            spatial X-Y coordinates of each cell in cell_cluster. The columns must be specified as 'X' and 'Y' and the index labelling the cells should correspond to the index of the cells in the original input data (in Transform). 
            (default). Alternatively, the cell ID can be contained in a third column. 
        K_nn : int
            The number of spatial neighbours with which to create the neighbourhood_flow matrix
        dist_thres : float
            The maximum nearest neighbour distance that is included in the spatial neighbours
        cell_join_id: str
            The name of the column with cell labels in cell_cluster and XY. Default is None, corresponding to XY.index being used as the cell label. Otherwise the cell label should be supplied in a third column of XY. 
        edge_quantile_threshold: float
            The edge weight quantile at which to threshold the neighbourhood flow graph
        """

        XY = XY.join(cell_cluster.membership_all, on = cell_join_id, how = 'inner')

        NN = NearestNeighbors(n_neighbors = K_nn + 1, n_jobs = -1)
        NN.fit(XY[['X', 'Y']])
        dist, ind = NN.kneighbors(XY[['X', 'Y']])

        #L = len(np.unique(XY['subcluster']))
        L = np.max(XY['subcluster']) + 1
        n_mtx = np.zeros((L, L))

        def func(X):
            s = np.sum(X)
            if s == 0:
                return(X)
            else:
                return(X/s)

        #dist_thres = 300
        ncol = dist.shape[1]

        for i in range(0, dist.shape[0]):
            w = dist[i, range(1, ncol)] < dist_thres
            s1 = int(XY.iloc[i]['subcluster'])
            s2 = XY.iloc[ind[i, range(1, ncol)][w]]['subcluster']
            t = np.bincount(s2)
            t_names = np.nonzero(t)[0]
            t_count = t[t_names]
            for j in range(0, len(t_names)):
                n_mtx[s1, t_names[j]] = n_mtx[s1, t_names[j]] + t_count[j]
        n_mtx = np.apply_along_axis(func, 1, n_mtx)

        # why is this step done? wrong?
        # for i in range(0, ncol):
        #     for j in range(0, ncol):
        #         if i < j:
        #             if n_mtx[i, j] < n_mtx[j, i]:
        #                 n_mtx[i, j] = 0
        #             elif n_mtx[j, i] < n_mtx[i, j]:
        #                 n_mtx[j, i] = 0

        self.neighbourhood_flow = pd.DataFrame(n_mtx, index = np.unique(XY['subcluster']), columns = np.unique(XY['subcluster']))

        # def NN(x, k):
        #     y = x.copy()
        #     r = (-y).argsort().argsort()
        #     r = r > k - 1
        #     y[r] = 0
        #     return(y)
        #NF = np.apply_along_axis(row_norm, 1, self.neighbourhood_flow)
        #self.NF = np.apply_along_axis(NN, 1, NF, NN_k)
        self.NF = n_mtx.copy()
        Q = np.quantile(self.NF[self.NF > 0], edge_quantile_threshold)
        self.NF[self.NF < Q] = 0
        NF_graph = ig.Graph().Weighted_Adjacency(self.NF, mode = 'directed')
        w = np.array(NF_graph.es['weight']).argsort().argsort() + 1
        clus = ig.Graph.community_walktrap(NF_graph, weights = (w**2)/100, steps = 5).as_clustering().membership
        indeg = NF_graph.indegree()
        idx = self.neighbourhood_flow.index
        self.NF_clus = pd.DataFrame({'clus': clus, 'indegree': indeg}, index = idx)
        self.NF = pd.DataFrame(self.NF, index = idx, columns = idx)

    def save(self, path):
        """
        Save R-friendly output

        Parameters
        ----------
        path : str
            String indicating path to save directory
        """

        self.neighbourhood_flow.to_csv(path + 'neighbourhood_flow_raw_frequency.txt.gz', index_label = 'ID')
        self.NF.to_csv(path + 'neighbourhood_flow_graph.txt.gz', index_label = 'ID')
        self.NF_clus.to_csv(path + 'spatial_niche.txt.gz', index_label = 'ID')
