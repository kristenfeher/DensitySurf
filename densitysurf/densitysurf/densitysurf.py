
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
import scipy
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.utils.extmath import randomized_svd
from sknetwork.clustering import Louvain
import igraph as ig
import os
import pickle

__all__ = ['NN_density_cluster', 'reconstruct', 'directory_structure', 'MultiSampleConcat', 'Workflow', 'Transform', 'Cluster', 'SpecificityNetwork', 'NeighbourhoodFlow']

def spherical_transform(X):
    return(X/np.sqrt(np.sum(np.square(X))))

def NPN(I, ind, dist, rho):
    j = rho[ind[I, 1:]] > rho[I]
    if np.sum(j) > 0:
        argmin = np.argmin(dist[I, 1:][j])
        return ind[I, 1:][j][argmin]
    else:
        return -1

def NN_density_cluster(coords: pd.DataFrame, K_nn):
    K_nn = K_nn + 1
    NN = NearestNeighbors(n_neighbors=K_nn, n_jobs=-1)
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
    # get coords corresponding to densest nodes
    exemplars = coords.iloc[np.unique(bigG_mem)] 

    # # merge identical points with different labels
    DF1 = bigG_mem.loc[exemplars.index] # column 'membership' is numerical index of densest points, dataframe index is the points' IDs
    DF1.columns = ['membership']
    DF2 = pd.DataFrame(exemplars.groupby(list(exemplars)).ngroup(), columns = ['duplicate']) # column 'duplicate' labels densest points according to whether they are identical to each other
    exemplars = DF1.merge(DF2, right_index=True, left_index = True).merge(exemplars, right_index = True, left_index = True) # merge the information into one dataframe

    subcluster_membership = bigG_mem.replace(to_replace=list(exemplars['membership']), value = list(exemplars['duplicate']))

    return(list((subcluster_membership, exemplars)))

def veclen(x):
    return(np.sqrt(np.sum(np.square(x))))

def var_expl(P_svd, i): # variance explained over rows, columns
    s0 = P_svd[0][:, 0:i]
    s1 = P_svd[1][0:i]
    s2 = P_svd[2][0:i, :]
    P_reconstruct = np.matmul(np.matmul(s0, np.diag(s1)), s2)
    Lcol = np.apply_along_axis(veclen, 0, P_reconstruct)
    Lrow = np.apply_along_axis(veclen, 1, P_reconstruct)
    return([Lcol, Lrow])

def reconstruct(path_name, pickle = False): # give option to use pickled Transform object
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

def directory_structure(path, force = False):
    if not os.path.isdir(path + 'output/'):
        os.mkdir(path + 'output/')
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
        if force:
            print("Directories already contain results. \nYou will overwrite these results if you proceed. \nPlease consider creating a new parent directory")
        else:
            raise FileExistsError("Directories already contain results. \nYou will overwrite these results if you proceed. \nPlease create a new parent directory or use force = True")
      
def MultiSampleConcat(path_names, sample_ids = None): # give option to just use exemplar cells # give warning in docs that columns with duplicated names are deleted
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

def Workflow(                
        input_dataframe: pd.DataFrame, # input count matrix
        input_data_path: str, # location of input_dataframe
        output_parent_directory: str, # location of directory structure
        ncomps: int = 50, # input parameter for Transform
        n_iter: int = 100, # input parameter for Transform
        cell_K_nn: int = 10, # input parameter for Cluster
        cell_clus_steps: int = 1000,  # input parameter for Cluster
        gene_K_nn: int = 10,  # input parameter for Cluster
        gene_clus_steps: int = 1000,  # input parameter for Cluster
        similarity_threshold: float = 0.2,  # input parameter for SpecificityNetwork
        neighbourhood_flow_K_nn: int = 6,  # input parameter for NeighbourhoodFlow
        dist_thres: float = 10000, # input parameter for NeighbourhoodFlow
        cell_join_id: str = 'cell_id', # input parameter for NeighbourhoodFlow
        XY: pd.DataFrame = None, # input parameter for NeighbourhoodFlow 

        new_directory_structure: bool = True, # control arguments for Transform
        overwrite_directory_structure: bool = False,
        transform: bool = True,
        transform_pickle_path: str = None,
        umap: bool = True, 
        umap_ncomp_gene: int = None,
        umap_ncomp_cell: int = None,
        transform_goodness_of_fit: bool = True, 
        transform_overwrite: bool = False,
        transform_R_output: bool = True,
        transform_pickle_output: bool = True,

        cluster_cells: bool = True, # control arguments for Cluster (cells)
        cluster_cells_subdirectory: str = None,
        cluster_cells_overwrite: bool = False,
        cluster_cells_pickle_path: str = None,
        cluster_cells_R_output: bool = True, 
        cluster_cells_pickle_output: bool = True,

        cluster_genes: bool = True, # control arguments for Cluster (genes)
        cluster_genes_subdirectory: str = None,
        cluster_genes_overwrite: bool = False,
        cluster_genes_pickle_path: str = None, 
        cluster_genes_R_output: bool = True, 
        cluster_genes_pickle_output: bool = True,

        specificity_network: bool= True, # control arguments for Specificity Network
        specificity_network_subdirectory: str = None,
        specificity_network_overwrite: bool = False, 
        #specificity_network_pickle_path:str = None, # a potential future application might need to import spec network
        specificity_network_R_output: bool = True,
        specificity_network_pickle_output: bool = True, 

        neighbourhood_flow: bool = False, # control arguments for NeighbourhoodFlow
        neighbourhood_flow_subdirectory: str = None, 
        neighbourhood_flow_overwrite: bool = False, 
        neighbourhood_flow_R_output: bool = True, 
        neighbourhood_flow_pickle_output: bool = True
        ):

        if new_directory_structure: 
            directory_structure(output_parent_directory, overwrite_directory_structure)

        if transform:
            if os.listdir(output_parent_directory + '/output/transform/') != []:
                if not transform_overwrite:
                    raise FileExistsError("/output/transform/ already contain results. \nYou will overwrite these results if you proceed. \nPlease create a new parent directory or use transform_overwrite = True")            

            if transform_goodness_of_fit:
                Y = Transform(input_dataframe, ncomps = ncomps, n_iter = n_iter, transform = True, goodness_of_fit = True)
            else: 
                Y = Transform(input_dataframe, ncomps = ncomps, n_iter = n_iter, transform = True, goodness_of_fit = False)
            if umap:
                Y.umap(ncomp_cell = umap_ncomp_cell, ncomp_gene = umap_ncomp_gene)

            if transform_R_output:
                Y.save(path = output_parent_directory + '/output/transform/')
            if transform_pickle_output:
                file = open(output_parent_directory + '/output/transform/transform.pkl', 'wb')
                pickle.dump(Y, file)
                file.close()
            file = open(output_parent_directory + '/output/transform/info.txt', 'w')
            file.write("Raw expression matrix path: " + '"' + input_data_path + '"')
            file.write("\nncomps: " + str(ncomps))
            file.write("\nn_iter: " +  str(n_iter))
            file.close()
        
        elif transform_pickle_path != None:
            file = open(transform_pickle_path, 'rb')
            Y = pickle.load(file)
            file.close()

        else:
            raise ReferenceError("A Transform object is required for subsequent steps.")

        # clustering
        if cluster_cells:
            if (not transform and transform_pickle_path == None):
                raise ReferenceError("Cluster requires Transform output")
            cell_cluster = Cluster(Y, K_nn = cell_K_nn, clus_steps = cell_clus_steps)
            if cluster_cells_subdirectory == None:
                subdir = "/output/cells/"
            else:
                subdir = "/output/cells/" + cluster_cells_subdirectory
            if not os.path.isdir(output_parent_directory + subdir):
                os.mkdir(output_parent_directory + subdir)
            else:
                if os.listdir(output_parent_directory + subdir) != []:
                    if not cluster_cells_overwrite:
                        raise FileExistsError("/output/cells/cluster_cells_subdirectory already contain results. \nYou will overwrite these results if you proceed. \nPlease create a new directory or use cluster_cells_overwrite = True")            
            if cluster_cells_R_output:
                cell_cluster.save(output_parent_directory + subdir)
            if cluster_cells_pickle_output:
                file = open(output_parent_directory + subdir + 'cluster.pkl', 'wb')
                pickle.dump(cell_cluster, file)
                file.close()
            # clustering info
            file = open(output_parent_directory + subdir + 'info.txt', 'w')
            file.write("K_nn: " +  str(cell_K_nn) + '\n')
            file.write("clus_steps: " + str(cell_clus_steps) + '\n')
            if transform:
                file.write("Raw expression matrix path: " + '"' + input_data_path + '"')
                file.write("ncomps: " + str(ncomps) + '\n')
                file.write("n_iter: " + str(n_iter) + '\n')
            else:
                file.write("Pickled transform object: " + '"' + transform_pickle_path + '"' + '\n')
            file.close()

        elif not cluster_cells and cluster_cells_pickle_path != None:
            file = open(cluster_cells_pickle_path, 'rb')
            cell_cluster = pickle.load(file)
            file.close()

        if cluster_genes:
            if (not transform and transform_pickle_path == None):
                raise ReferenceError("Cluster requires Transform output")
            gene_cluster = Cluster(Y, K_nn = gene_K_nn, clus_steps = gene_clus_steps, mode = 'genes')
            if cluster_genes_subdirectory == None:
                subdir = "/output/genes/"
            else:
                subdir = "/output/genes/" + cluster_genes_subdirectory
            if not os.path.isdir(output_parent_directory + subdir):
                os.mkdir(output_parent_directory + subdir)
            else:
                if os.listdir(output_parent_directory + subdir) != []:
                    if not cluster_genes_overwrite:
                        raise FileExistsError("/output/genes/cluster_genes_subdirectory already contain results. \nYou will overwrite these results if you proceed. \nPlease create a new directory or use cluster_genes_overwrite = True")            
            if cluster_genes_R_output:
                gene_cluster.save(output_parent_directory + subdir)
            if cluster_genes_pickle_output:
                file = open(output_parent_directory + subdir + 'cluster.pkl', 'wb')
                pickle.dump(gene_cluster, file)
                file.close()
            # clustering info
            file = open(output_parent_directory + subdir + 'info.txt', 'w')
            file.write("K_nn: " + str(gene_K_nn) + '\n')
            file.write("clus_steps: " + str(gene_clus_steps) + '\n')
            if transform:
                file.write("Raw expression matrix path: " + '"' + input_data_path + '"' + '\n')
                file.write("ncomps: " + str(ncomps) + '\n')
                file.write("n_iter: " + str(n_iter) + '\n')
            else:
                file.write("Pickled transform object: " + '"' + transform_pickle_path + '"' + '\n')
            file.close()

        elif not cluster_genes and cluster_genes_pickle_path != None:
            file = open(cluster_genes_pickle_path, 'rb')
            gene_cluster = pickle.load(file)
            file.close()

        if specificity_network:

            if (not cluster_genes and cluster_genes_pickle_path == None) or (not cluster_cells and cluster_cells_pickle_path == None) or (not transform and transform_pickle_path == None):
                raise ReferenceError("SpecificityNetwork requires transform and clustering output")

            spec_network = SpecificityNetwork(Y, cell_cluster, gene_cluster, similarity_threshold)
            if specificity_network_subdirectory == None:
                subdir = "/output/specificity_network/"
            else:
                subdir = "/output/specificity_network/" + specificity_network_subdirectory
            if not os.path.isdir(output_parent_directory + subdir):
                os.mkdir(output_parent_directory + subdir)
            else:
                if os.listdir(output_parent_directory + subdir) != []:
                    if not specificity_network_overwrite:
                        raise FileExistsError("/output/specificity_network/specificity_network_subdirectory already contain results. \nYou will overwrite these results if you proceed. \nPlease create a new directory or use specificity_network_overwrite = True")            
            if specificity_network_R_output:
                spec_network.save(output_parent_directory + subdir)
            if specificity_network_pickle_output:
                file = open(output_parent_directory + subdir + 'specificity_network.pkl', 'wb')
                pickle.dump(spec_network, file)
                file.close()
            file = open(output_parent_directory + subdir + 'info.txt', 'w')
            file.write("similarity_threshold: " + str(similarity_threshold) + '\n')
            if transform:
                file.write("Raw expression matrix path: " + '"' + input_data_path + '"' + '\n')
                file.write("ncomps: " + str(ncomps) + '\n')
                file.write("n_iter: " + str(n_iter) + '\n')
            else:
                file.write("Pickled transform object: " + '"' + transform_pickle_path + '"' + '\n')
            if cluster_genes: 
                file.write("gene_K_nn: " + str(gene_K_nn) + '\n')
                file.write("gene_clus_steps: " + str(gene_clus_steps) + '\n')
            else:
                file.write('Pickled gene object: ' + cluster_genes_pickle_path + '\n')
            if cluster_cells: 
                file.write("cell_K_nn: " + str(cell_K_nn) + '\n')
                file.write("cell_clus_steps: " + str(cell_clus_steps) + '\n')
            else:
                file.write('Pickled cell object: ' + '"' + cluster_cells_pickle_path + '"' + '\n')
            file.close()            

        if neighbourhood_flow:

            if (not cluster_cells and cluster_cells_pickle_path == None) or XY == None:
                raise ReferenceError("NeighbourhoodFlow requires cell Cluster object and XY DataFrame")
            NF = NeighbourhoodFlow(cell_cluster = cell_cluster, XY = XY, K_nn = neighbourhood_flow_K_nn, dist_thres = dist_thres, cell_join_id = cell_join_id)
            if neighbourhood_flow_subdirectory == None:
                subdir = "/output/neighbourhood_flow/"
            else:
                subdir = "/output/neighbourhood_flow/" + specificity_network_subdirectory
            if not os.path.isdir(output_parent_directory + subdir):
                os.mkdir(output_parent_directory + subdir)
            else:
                if os.listdir(output_parent_directory + subdir) != []:
                    if not specificity_network_overwrite:
                        raise FileExistsError("/output/neighbourhood_flow/neighbourhood_flow_subdirectory already contain results. \nYou will overwrite these results if you proceed. \nPlease create a new directory or use neighbourhood_flow_overwrite = True")            
            if neighbourhood_flow_R_output:
                NF.save(output_parent_directory + subdir)
            if neighbourhood_flow_pickle_output:
                file = open(output_parent_directory + subdir + 'neighbourhood_flow.pkl', 'wb')
                pickle.dump(NF, file)
                file.close()

            file = open(output_parent_directory + subdir + 'info.txt', 'w')
            file.write("XY path: " + '"' + XY_path + '"' + '\n')
            if transform:
                file.write("Raw expression matrix path: " + '"' + input_data_path + '"' + '\n')
                file.write("ncomps: " + str(ncomps) + '\n')
                file.write("n_iter: " + str(n_iter) + '\n')
            elif transform_pickle_path != None:
                file.write("Pickled transform object: " + '"' + transform_pickle_path + '"' + '\n')
            else:
                file.write("Transform: check cell cluster directory for input data location")
            if cluster_cells: 
                file.write("cell_K_nn: " + str(cell_K_nn) + '\n')
                file.write("cell_clus_steps: " + str(cell_clus_steps) + '\n')
            else:
                file.write('Pickled cell object: ' + '"' + cluster_cells_pickle_path + '"' + '\n')
            file.close()


##############################################
class Transform:
    def __init__(self, input_dataframe: pd.DataFrame, ncomps: int = 50, n_iter: int = 100, transform: bool = True, goodness_of_fit: bool = True): #input_dataframe must be pd.DataFrame
        
        self.col_names = input_dataframe.columns
        self.row_names = input_dataframe.index
  
        # normalise, row and columns masses
        # don't actually need this if transform == False, but just keep it for consistency
        D1 = input_dataframe/(input_dataframe.sum().sum())
        cm = D1.sum(axis = 0)
        self.col_keep = cm != 0
        rm = D1.sum(axis = 1)
        self.row_keep = rm != 0

        # # CA transform and randomised SVD
        if transform:
            rc = np.outer(rm[self.row_keep], cm[self.col_keep])
            P = np.array((1/cm[self.col_keep]**0.5)) * (D1.loc[self.row_keep, self.col_keep] - rc) * np.array((1/rm[self.row_keep]**0.5)).reshape((-1,1))
        else:
            P = input_dataframe.loc[self.row_keep, self.col_keep]
        P_svd = randomized_svd(P.to_numpy(), n_components=ncomps, n_iter=n_iter, random_state = 0) # or dask svd for large data

        if goodness_of_fit:
            L = list(range(1, ncomps+1, int(np.floor((ncomps+1)/10))))
            if L[-1] != ncomps+1:
                L.append(ncomps+1)
            L.pop(0)
            VE = [var_expl(P_svd, i) for i in [i for i in L]]
            self.gof_gene = np.transpose(pd.DataFrame([x[0] for x in VE]))
            self.gof_cell = np.transpose(pd.DataFrame([x[1] for x in VE]))
            cnames = ['ncomps' + str(i) for i in L]
            self.gof_gene.columns = cnames
            self.gof_cell.columns = cnames
            self.gof_gene.index = input_dataframe.loc[self.row_keep, self.col_keep].columns
            self.gof_cell.index = input_dataframe.loc[self.row_keep, self.col_keep].index

            self.gof_cell.insert(0, 'total_length', np.apply_along_axis(veclen, 1, input_dataframe.loc[self.row_keep, self.col_keep]))
            self.gof_cell.insert(0, 'mean_count', np.apply_along_axis(np.mean, 1, input_dataframe.loc[self.row_keep, self.col_keep]))
            self.gof_cell.insert(0, 'median_count', np.apply_along_axis(np.median, 1, input_dataframe.loc[self.row_keep, self.col_keep]))
            self.gof_cell.insert(0, 'sd_count', np.apply_along_axis(np.std, 1, input_dataframe.loc[self.row_keep, self.col_keep]))
                       
            self.gof_gene.insert(0, 'total_length', np.apply_along_axis(veclen, 0, input_dataframe.loc[self.row_keep, self.col_keep]))
            self.gof_gene.insert(0, 'mean_count', np.apply_along_axis(np.mean, 0, input_dataframe.loc[self.row_keep, self.col_keep]))
            self.gof_gene.insert(0, 'median_count', np.apply_along_axis(np.median, 0, input_dataframe.loc[self.row_keep, self.col_keep]))
            self.gof_gene.insert(0, 'sd_count', np.apply_along_axis(np.std, 0, input_dataframe.loc[self.row_keep, self.col_keep]))

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
        cell_coord = np.apply_along_axis(spherical_transform, 1, ca_comps)
        self.cell_coord = pd.DataFrame(cell_coord, index = idx, columns = cnames)

        # spherical gene coordinates
        ca_comps = np.transpose(P_svd[2]) * P_svd[1] # genes
        cnames = ['gene_coord' + str(a) for a in range(1, ncomps + 1)]
        idx = self.col_names[self.col_keep]
        gene_coord = np.apply_along_axis(spherical_transform, 1, ca_comps)
        self.gene_coord = pd.DataFrame(gene_coord, index = idx, columns = cnames)

    def umap(self, cell = True, gene = True, ncomp_cell:int = None, ncomp_gene:int = None):
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
                gene_umap = reducer.fit_transform(self.gene_coord[:, 0:ncomp_gene])
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
             cell_umap = True, 
             gene_umap = True, 
             goodness_of_fit = True
             ):

        pd.DataFrame({'row_keep':self.row_keep}, index = self.row_names).to_csv(path + 'row_keep.txt')
        pd.DataFrame({'col_keep':self.col_keep}, index = self.col_names).to_csv(path + 'col_keep.txt')
        
        if svd:             
            self.svd0.to_csv(path + 'svd0.txt.gz')
            self.svd1.to_csv(path + 'svd1.txt', index = False)            
            self.svd2.to_csv(path + 'svd2.txt.gz')
            
        if cell_coord:
            self.cell_coord.to_csv(path + 'cell_coord.txt.gz')

        if gene_coord: 
            self.gene_coord.to_csv(path + 'gene_coord.txt.gz')

        if cell_umap and hasattr(self, 'cell_umap'):
            self.cell_umap.to_csv(path + 'cell_umap.txt.gz')

        if gene_umap and hasattr(self, 'gene_umap'):
            self.gene_umap.to_csv(path + 'gene_umap.txt.gz')

        if goodness_of_fit and hasattr(self, 'gof_cell'):
            self.gof_cell.to_csv(path + 'gof_cell.txt.gz')
            self.gof_gene.to_csv(path + 'gof_gene.txt.gz')


    
#################################################
class Cluster:
    def __init__(self, coords: Transform, K_nn = 20, clus_steps = 1000, mode = 'cells', similarity_threshold = 0.2): #coords is a pd.dataframe, norm of each row should be 1. Clus_steps should be high, K_nn should be low
        
        K_nn = K_nn + 1
        if mode == 'cells':
            clus = NN_density_cluster(coords.cell_coord, K_nn = K_nn)
        elif mode == 'genes':
            clus = NN_density_cluster(coords.gene_coord, K_nn = K_nn)
        else:
            print('mode is either cells or genes')

        self.mode = mode

        # # get coords and drop duplicated points
        T = clus[1][~clus[1].duplicated('duplicate')]
        # # dot product of exemplars 
        dotprod = np.matmul(T.drop(columns = ['membership', 'duplicate']).to_numpy(), np.transpose(T.drop(columns = ['membership', 'duplicate']).to_numpy()))
        dotprod_index = T['duplicate']
        self.subcluster_similarity = pd.DataFrame(dotprod, index = dotprod_index, columns = dotprod_index).rename_axis(index = None, columns = None)#.sort_index().sort_index(axis = 1)
        dotprod_thres = dotprod.copy()
        dotprod_thres[dotprod_thres < similarity_threshold] = 0
        
        ##################################
        # # create graph
        graph = ig.Graph().Weighted_Adjacency(dotprod_thres, mode = 'undirected')
        # # betweeness, coreness and create clusters
        BT = graph.betweenness()
        CN = graph.coreness()
        
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
        GN = [graph.neighbors(i) for i in range(0, graph.vcount())]
        BT_median = [median([BT[i] for i in GN[j]]) for j in range(0, graph.vcount())]
        BT_mad = [mad([BT[i] for i in GN[j]]) for j in range(0, graph.vcount())]
        T.insert(0, 'BT_median', BT_median)
        T.insert(0, 'BT_mad', BT_mad)

        CN_median = [median([CN[i] for i in GN[j]]) for j in range(0, graph.vcount())]
        CN_mad = [mad([CN[i] for i in GN[j]]) for j in range(0, graph.vcount())]
        T.insert(0, 'CN_median', CN_median)
        T.insert(0, 'CN_mad', CN_mad)

        if graph.vcount() < 500: # in future, give more control over choice of graph clustering. 
            clusWT = ig.Graph.community_walktrap(graph, weights = graph.es['weight'], steps = clus_steps).as_clustering().membership
        else:
            clusWT = ig.Graph.community_leiden(graph, resolution = 0.5).membership
        T.insert(0, 'clus', clusWT)

        graph_membership = clus[0].replace(to_replace = list(T['duplicate']), value = list(T['clus']))
        self.membership_all = pd.DataFrame({'subcluster': clus[0]['membership'], 'graph_cluster': graph_membership['membership']})

        # cluster wise metrics
        BT_DF = T[['BT', 'clus']]
        Q1 = BT_DF.groupby('clus').quantile(0.25)
        Q3 = BT_DF.groupby('clus').quantile(0.75)
        IQR = Q3 - Q1
        maxBT = Q3 + 1.5 * IQR

        CN_DF = T[['CN', 'clus']]
        Q1 = CN_DF.groupby('clus').quantile(0.25)
        Q3 = CN_DF.groupby('clus').quantile(0.75)
        IQR = Q3 - Q1
        minCN = Q1 - 1.5 * IQR

        # prune graph clusters
        clus_core = []
        for i in range(0, max(BT_DF['clus'] + 1)):
            cond1 = BT_DF['clus'] == i
            cond2 = BT_DF['BT'] <= maxBT[maxBT.index == i].iloc[0, 0]
            cond3 = CN_DF['CN'] >= minCN[minCN.index == i].iloc[0, 0]
            clus_core.append(BT_DF.loc[(cond1) & (cond2) & (cond3)].index)

        # self.clus_core = clus_core

        # update T to show whether subclusters are core members of graph clusters, based on BT and CN statistics above
        T.insert(0, 'core', False)
        for i in range(0, len(clus_core)):
            T.loc[clus_core[i], 'core'] = True

        self.subcluster_points = T[['core', 'CN', 'BT', 'CN_median', 'CN_mad', 'BT_median', 'BT_mad', 'clus', 'duplicate']].rename(columns = {'duplicate': 'subcluster', 'clus': 'graph_cluster', 'CN': 'coreness', 'BT': 'betweenness', 'CN_median': 'coreness_median', 'CN_mad': 'coreness_mad', 'BT_median': 'betweeness_median', 'BT_mad': 'betweeness_mad'})

        A = self.membership_all['subcluster'].value_counts()
        B = A.index
        C = self.subcluster_points['subcluster'].replace(to_replace = B, value = A)
        self.subcluster_points.insert(0, 'subcluster_count', C)
               
    def save(self, 
             path, 
             membership_all = True, 
             subcluster_points = True,
             subcluster_similarity = True
             ):

        if membership_all:
            self.membership_all.to_csv(path + 'membership_all.txt.gz')
       
        if subcluster_points:
            self.subcluster_points.to_csv(path + 'subcluster_points.txt.gz')

        if subcluster_similarity:
            self.subcluster_similarity.to_csv(path + 'subcluster_similarity.txt.gz')

########################################################
class SpecificityNetwork:
    def __init__(self, coords: Transform, cells: Cluster, genes: Cluster, similarity_threshold = 0.2):
        cc = coords.cell_coord.loc[cells.subcluster_points.index]
        gc = coords.gene_coord.loc[genes.subcluster_points.index]
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
            self.bip_graph.to_csv(path + 'bip_graph.txt.gz')

        if virtual_stain:
            self.virtual_stain.to_csv(path + 'virtual_stain.txt.gz')

class NeighbourhoodFlow: 
    def __init__(self, cell_cluster: Cluster, XY: pd.DataFrame, K_nn: int, dist_thres: float, cell_join_id: str):
        XY = XY.join(cell_cluster.membership_all, on = cell_join_id, how = 'inner')

        NN = NearestNeighbors(n_neighbors = K_nn + 1, n_jobs = -1)
        NN.fit(XY[['X', 'Y']])
        dist, ind = NN.kneighbors(XY[['X', 'Y']])

        L = len(np.unique(XY['subcluster']))
        n_mtx = np.zeros((L, L))

        def func(X):
            return(X/np.sum(X))

        dist_thres = 300
        ncol = dist.shape[1]

        for i in range(0, dist.shape[0]):
            w = dist[i, range(1, ncol)] < dist_thres
            s1 = XY.iloc[i]['subcluster']
            s2 = XY.iloc[ind[i, range(1, ncol)][w]]['subcluster']
            t = np.bincount(s2)
            t_names = np.nonzero(t)[0]
            t_count = t[t_names]
            for j in range(0, len(t_names)):
                n_mtx[s1, t_names[j]] = n_mtx[s1, t_names[j]] + t_count[j]
        n_mtx = np.apply_along_axis(func, 1, n_mtx)

        for i in range(0, ncol):
            for j in range(0, ncol):
                if i < j:
                    if n_mtx[i, j] < n_mtx[j, i]:
                        n_mtx[i, j] = 0
                    elif n_mtx[j, i] < n_mtx[i, j]:
                        n_mtx[j, i] = 0

        self.neighbourhood_flow = pd.DataFrame(n_mtx, index = np.unique(XY['subcluster']), columns = np.unique(XY['subcluster']))

    def save(self, path):
        self.neighbourhood_flow.to_csv(path + 'neighbourhood_flow.txt.gz')


