########
# Some example code for importing and plotting the output of DensitySurf
########


library(data.table)
library(igraph)

# my personal colour palette for plotting clusterings, it has good distinguishability. 
# For large numbers of clusters, colours are repeated (there is a limit to how many distinct colours the human eye can keep track of anyway)
colvec <- c(brewer.pal(name = 'Set1', n = 15), brewer.pal(name = 'Dark2', n = 15), brewer.pal(name = 'Pastel1', n = 15))
colvec[14] <- 'cyan'
colvec <- rep(colvec, 20)


#############
# helper function
# This will eventually be a companion package with more helper functions for plotting
############

#' loads similarity network or specificity network output from DensitySurf_python
#' @title load graph output
#' @param path path to similarity network output file
#' @param similarity_threshold A value between -1 and 1. Default is -1 which doesn't threshold.
#' @param mode A string in c('directed' or 'undirected'). Default is 'undirected'. See igraph::graph_from_adjacency_matrix
#' @param diag Boolean indicating whether to include diagonal. Default is FALSE. See igraph::graph_from_adjacency_matrix
#' @param weighted Boolean indicating whether to include edge weights. Default is TRUE. See igraph::graph_from_adjacency_matrix
#' @description
#' This file loads the similarity network output created in the Cluster step of DensitySurf_python.
#' The input matrix contains all pairwise comparisons between all gene subcluster, or all cell subclusters. 
#' The similarities between subclusters are quantified with a value between -1 and 1. A value of 1 means the two subclusters are identical. A value of -1 means the subclusters are perfectly anticorrelated. A value of zero means the subclusters are orthogonal to each other. 
#' Values below similarity_threshold are set to zero. A sensible value is something a little bit bigger than zero (orthogonality) and the default is 0.2
#' @return an undirected, weighted igraph graph 
#' 
get_graph <- function(path, similarity_threshold = -1, mode = 'undirected', diag = FALSE, weighted = TRUE) {
  mygraph <- fread(path, header = TRUE)
  rn <- mygraph[[1]]
  mygraph <- as.matrix(mygraph[, -1])
  if (is.integer(rn)) {
    rn <- as.character(rn)
  }
  rownames(mygraph) <- rn
  colnames(mygraph) <- rn
  if (similarity_threshold > -1) {
    mygraph[mygraph < similarity_threshold] <- 0
  }
  mygraph <- igraph::graph_from_adjacency_matrix(mygraph, mode = mode, diag = diag, weighted = weighted)
  return(mygraph)
}



####################
# Analysis
####################

# specify path to output directory

path <- 'path/to/output/'

# importing data

# umap
cell_umap <- fread(paste0(path, 'transform/cell_umap.txt.gz'))
gene_umap <- fread(paste0(path, 'transform/gene_umap.txt.gz'))

# clustering membership
cell_mem <-  fread(paste0(path, 'cells/membership_all.txt.gz'))
gene_mem <-  fread(paste0(path, 'genes/membership_all.txt.gz'))

# using data.table to merge umap coordinates with clustering membership, and plot the umap
# python indexing commences from zero, thus need to add one to the 'subcluster' column

cell_mem[, plot(umap1, umap2, pch = 16, cex = 0.5, col = colvec[subcluster + 1])]
gene_mem[, plot(umap1, umap2, pch = 16, cex = 1, col = colvec[subcluster + 1])]

# importing data about each subcluster
cell_points <- fread(paste0(path, 'cells/subcluster_points.txt.gz'))
gene_points <- fread(paste0(path, 'genes/subcluster_points.txt.gz'))

# importing the 'cell scaffold' - a graph representing similarity between  clusters
cell_scaffold <- get_graph(paste0(path, 'cells/subcluster_similarity_prune.txt.gz'))

# checking for connected components
cell_comps <- components(cell_scaffold)
w <- which.max(cell_comps$csize)
cell_subgraph <-subgraph(cell_scaffold, cell_comps$membership == W)

# setting up the plot
ew <- (1:5)[cut(E(cell_subgraph)$weight, 5, include.lowest = TRUE)]
vs <- (1:5)[cut(cell_points[cell_comps$membership == w, log10(subcluster_count)], 5, include.lowest = TRUE)]
# Colour using the graph cluster column (adding one because starts from zero) 
col <- cell_points[cell_comps$membership == w, colvec[graph_cluster + 1]]
set.seed(42)
plot(cell_subgraph, vertex.size = vs^2/3, edge.width = ew^2/5, vertex.color = col)

# Repeat the above procedure to plot a gene scaffold
gene_scaffold <- get_graph(paste0(path, 'genes/subcluster_similarity_prune.txt.gz'))

# checking for connected components
gene_comps <- components(gene_scaffold)
w <- which.max(gene_comps$csize)
gene_subgraph <-subgraph(gene_scaffold, gene_comps$membership == W)

# setting up the plot
ew <- (1:5)[cut(E(gene_subgraph)$weight, 5, include.lowest = TRUE)]
vs <- (1:5)[cut(gene_points[gene_comps$membership == w, log10(subcluster_count)], 5, include.lowest = TRUE)]
# Colour using the graph cluster column (adding one because starts from zero) 
col <- gene_points[gene_comps$membership == w, colvec[graph_cluster + 1]]
set.seed(42)
plot(gene_subgraph, vertex.size = vs^2/3, edge.width = ew^2/5, vertex.color = col)

# Plotting the biomarker specificity network
SN <- get_graph(paste0(path, 'specificity_network/SN_graph.txt.gz'), 0.2)
# Create data.table to help with plotting
vertex_df <- lapply(1:length(SN), function(i) data.table(name = names(V(SN)), colour = rgb(0.5, 0, 0, 0.8), size = 4))
lapply(1:length(SN), function(i) vertex_df[grepl('c', name), colour := rgb(1, 0, 0, 0.5)])
lapply(1:length(SN), function(i) vertex_df[grepl('c', name), size := 2])
ew <- (1:5)[cut(E(SN)$weight, 5, include.lowest = TRUE)]
set.seed(42)
plot(SN, vertex.size = vertex_df$size, edge.width = ew^2/5, vertex.color = vertex_df$colour, edge.color = 'black')

# import XY coords (for spatial data)
XY <- fread("path/to/XYcoords.txt.gz")

# merge XY coordinates with cell cluster membership for plotting
XY[cell_mem, on = 'V1 == ID'][, plot(X, Y, pch = 16, cex = 0.8, col = colvec[graph_cluster + 1])]
XY[cell_mem, on = 'V1 == ID'][, plot(X, Y, pch = 16, cex = 0.5, col = colvec[subcluster + 1])]

# importing the neighbourhood flow graph and spatial niche clustering
NF <- get_graph(paste0(path, 'neighbourhood_flow/neighbourhood_flow_graph.txt.gz', mode = 'directed', diag = TRUE))
spatial_niche <-  fread(paste0(path, 'neighbourhood_flow/spatial_niche.txt.gz'))

# plotting the neighbourhood flow graph with the spatial niches
ew <- (1:5)[cut(E(NF)$weight, 5, include.lowest = TRUE)]
col <- spatial_niche[, colvec[clus + 1]]
set.seed(42)
plot(NF, vertex.size = 5, edge.arrow.size = 1*0.2, edge.arrow.width = 1*0.2, edge.width = ew^2/5, vertex.color = col)
# plotting the neighbourhood flow graph with the cell scaffold graph clusters
col = cell_points[NF_clus, on = 'subcluster == ID'][, colvec[graph_cluster + 1]]
set.seed(42)
plot(NF, vertex.size = 5, edge.arrow.size = 1*0.2, edge.arrow.width = 1*0.2, edge.width = ew^2/5, vertex.color = col)

# plotting the spatial niches in the spatial domain
NF_clus[XY[cell_mem, on = 'V1 == ID'], on = 'ID == subcluster'][, plot(X, Y, pch = 16, cex = 2, col = colvec[factor(clus)])]



























