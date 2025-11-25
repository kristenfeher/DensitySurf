########
# Some example code for importing and plotting the output of DensitySurf
########


library(data.table)
library(igraph)
library(RColorBrewer)
library(rhdf5)
library(pheatmap)

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

#############
# helper functions for gene annotation

enrich_results <- function(gset) {
  IDS <- bitr(gset, fromType = 'SYMBOL', toType = 'ENTREZID', OrgDb = 'org.Hs.eg.db')
  genrich <- enrichGO(gene = IDS$ENTREZID, OrgDb = 'org.Hs.eg.db', ont = 'BP')
  kenrich <- enrichKEGG(gene = IDS$ENTREZID, organism = "hsa")
  return(list(GO = genrich, KEGG = kenrich))
}

GO_display <- function(enrich, k = 10) {
  if (!is.null(enrich[[1]])) {
    tmp <- enrich$GO@result$Description[enrich$GO@result$p.adjust < 0.05]
    if (length(tmp) > k) {
      #print("top GO terms")
      #print(tmp[1:k])
      return(tmp[1:k])
    } else {
      #print("top GO terms")
      #print(tmp)
      return(tmp)
    }
    # } else {
    #   #print('Top GO terms')
    #   #print("No enrichment")
    #   return('No enrichment')
  }
}

KEGG_display <- function(enrich, k = 10) {
  if (!is.null(enrich[[2]])) {
    tmp <- enrich$KEGG@result$Description[enrich$KEGG@result$p.adjust < 0.05]
    if (length(tmp) > k) {
      #print("top KEGG terms")
      #print(tmp[1:k])
      return(tmp[1:k])
    } else {
      #print("top KEGG terms")
      #print(tmp)
      return(tmp)
    }
    # } else {
    #   print('Top KEGG terms')
    #   pnt('No enrichment')
  }
  
}


#########################
#########################
# Basic plots - single sample
#########################

root_path <- '/path_to_parent_directory/'
path <- paste0(root_path, 'output/stomics_top_level/D04319C6/')
#path <- paste0(root_path, 'output/stomics_top_level/C04597D6/')

gene_mem <- fread(paste0(path, 'genes/membership_all.txt.gz'))

cell_umap <- fread(paste0(path, 'transform/cell_umap.txt.gz'))
gene_umap <- fread(paste0(path, 'transform/gene_umap.txt.gz'))

# clustering membership
cell_mem <-  fread(paste0(path, 'cells/membership_all.txt.gz'))
gene_mem <-  fread(paste0(path, 'genes/membership_all.txt.gz'))

cell_mem[, subcluster_factor := factor(subcluster)]

# using data.table to merge umap coordinates with clustering membership, and plot the umap
# python indexing commences from zero, thus need to add one to the 'subcluster' column

cell_mem <- cell_mem[cell_umap, on = 'ID == ID']
gene_mem <- gene_mem[gene_umap, on = 'ID == ID']

png(paste0(path, 'figures/cell_exemplar_umap.png'), height = 1000, width = 1000)
par(mfrow = c(1, 1))
cell_mem[sample(.N), plot(umap1, umap2, pch = 16, cex = 1, col = colvec[graph_cluster + 1], main = 'cell exemplar subcluster')]
dev.off()

# importing the 'cell scaffold' - a graph representing similarity between  clusters

cell_scaffold <- get_graph(paste0(path, 'cells/exemplar_similarity_matrix_prune.txt.gz'))
cell_points <- fread(paste0(path, 'cells/exemplars.txt.gz'))
# checking for connected components
cell_comps <- components(cell_scaffold)
w <- which.max(cell_comps$csize)
cell_subgraph <-subgraph(cell_scaffold, cell_comps$membership == w)

# setting up the plot
ew <- (1:5)[cut(E(cell_subgraph)$weight, 5, include.lowest = TRUE)]
vs <- (1:5)[cut(cell_points[cell_comps$membership == w, log10(subcluster_count)], 5, include.lowest = TRUE)]
fac <- factor(paste0(cell_points[, as.numeric(Strength - Strength_median > -1)], '_', cell_points[, as.numeric(betweenness - betweeness_median > -1)]))
# Colour using the graph cluster column (adding one because starts from zero) 
col <- cell_points[cell_comps$membership == w, colvec[graph_cluster + 1]]

png(paste0(path, 'figures/cell_scaffold.png'), height = 2000, width = 2000)
par(mfrow = c(1, 1))
set.seed(42)
plot(cell_subgraph, vertex.size = vs^2/3, edge.width = ew^2/2, vertex.color = col, edge.color = rgb(0, 0, 0, 0.2))
dev.off()

# importing the 'gene scaffold' - a graph representing similarity between  clusters
gene_scaffold <- get_graph(paste0(path, 'genes/exemplar_similarity_matrix_prune.txt.gz'), similarity_threshold = 0.2)
E(gene_scaffold)$weight <- E(gene_scaffold)$weight^2/100
gene_points <- fread(paste0(path, 'genes/exemplars.txt.gz'))
# checking for connected components
gene_comps <- components(gene_scaffold)
w <- which.max(gene_comps$csize)
gene_subgraph <-subgraph(gene_scaffold, gene_comps$membership == w)

# setting up the plot
ew <- (1:5)[cut(E(gene_subgraph)$weight, 5, include.lowest = TRUE)]
vs <- (1:5)[cut(gene_points[gene_comps$membership == w, log10(subcluster_count)], 5, include.lowest = TRUE)]
# Colour using the graph cluster column (adding one because starts from zero) 
col <- gene_points[gene_comps$membership == w, colvec[graph_cluster + 1]]

png(paste0(path, 'figures/gene_scaffold.png'), height = 1000, width = 1000)
set.seed(42)
plot(gene_subgraph, vertex.size = vs^2/3, edge.width = ew^2/2, vertex.color = col, edge.color = rgb(0, 0, 0, 0.2))
dev.off()

##### 
# biomarker specifity network

fn <-  paste0(path, 'specificity_network/biomarker_specificity_network.txt.gz')
SN <- fread(fn)

SN <- get_graph(path = fn, 0.4)
#SN <- fread(fn)
# Create data.table to help with plotting
vertex_df <-  data.table(name = names(V(SN)), colour = 'lightgrey', size = 4)
vertex_df[grepl('c', name), colour := rgb(0, 0.5, 0, 0.5)]
vertex_df[grepl('c', name), size := 2]

vertex_df[, which := gsub('_[0-9]+', '', name)]
vertex_df[, subcluster := as.numeric(gsub('[a-z]_', '', name))]

vertex_df[, graph_colour := colour]
for (i in 0:max(cell_mem$graph_cluster)) {
  w <- cell_mem[graph_cluster == i, unique(subcluster)]
  vertex_df[which == 'c' & subcluster %in% w, graph_colour := colvec[i+1]] # tumour subclusters
}

ew <- (1:5)[cut(E(SN)$weight, 5, include.lowest = TRUE)]

png(paste0(path, 'figures/spec_network.png'), height = 2000, width = 2000)
set.seed(42)
plot(SN, vertex.size = vertex_df$size, edge.width = ew^2/2, vertex.color = vertex_df$graph_colour, edge.color = rgb(0, 0, 0, 0.2))
dev.off()

###############
# virtual staining (exemplar genes): XY

col_fn <- function(x) {
  breaks <- unique(c(min(x), 0, 0.1, quantile(x[x > 0], seq(0.01, 0.99, by = 0.01)), max(x)))
  col <- c(rgb(0, 0, 0, 0.2), colorRampPalette(c('skyblue3', 'magenta2'))(length(breaks) - 1))
  col <- col[cut(x, breaks, include.lowest = TRUE)]
  return(col)
}

cell_points <- fread(paste0(path, 'cells/exemplars.txt.gz'))

VS <- fread(paste0(path, 'specificity_network/virtual_stain.txt.gz'))
VS[cell_points, on = 'ID == ID', subcluster := i.subcluster]
cn <- colnames(VS)
cn <- cn[grep('g_', cn)]
o <- as.numeric(gsub('g_', '', cn))
cnames <- cn[order(o)]

# columns of XY data.table should include 'X', 'Y', and an ID column to identify cells/bins (the IDs should be the same as those contained in the gene expression matrix)
XY_path <- c('/path_to_XY_coords/XY_D04319C6.txt.gz')

XY <- fread(paste0(XY_path[i]))
XY <- VS[XY, on = 'ID == V1'] #replace 'ID' with the name of the ID column in XY
png(paste0(path, 'figures/VS_all_scale_.png'), height = 7000, width = 8000)
par(mfrow = c(7,8)) # set up grid according to number of gene groups in cnames
for (k in 1:length(cnames)) {
  XY[, plot(X, Y, pch = 16, cex = 0.5, col = rgb(0, 0, 0, 0.1), main = cnames[k], cex.main = 4)]
  XY[!is.na(get(cnames[k])), points(X, Y, pch = 16, cex = 0.7, col = col_fn(get(cnames[k])))]
}
dev.off()

# correlation between virtual stains of exemplar genes
png(paste0(path, 'figures/VS_all_cor.png'), height = 1000, width = 1000)
VS[, pheatmap(cor(.SD)), .SDcols = cnames]
dev.off()

###############
# virtual staining (all genes): heatmap

gene_mem <-  fread(paste0(path, 'genes/membership_all.txt.gz'))
cell_points <- fread(paste0(path, 'cells/exemplars.txt.gz'))

cell_coord <- fread(paste0(path, 'transform/cell_coord.txt.gz'))
cell_ID <- cell_coord$ID
K <- 50 # how many SVD components were chosen?
cell_coord <- t(apply(cell_coord[, 2:(K+1)], 1, function(x) x/sqrt(sum(x^2))))

gene_coord <- fread(paste0(path, 'transform/gene_coord.txt.gz'))
gene_ID <- gene_coord$ID
gene_coord <- t(apply(gene_coord[, 2:(K+1)], 1, function(x) x/sqrt(sum(x^2))))

VS <- cell_coord[cell_ID %in% cell_points$ID, ] %*% t(gene_coord)

# creating an ordering for the heatmap
VS1 <- fread(paste0(path, 'specificity_network/virtual_stain.txt.gz'))
cn <- colnames(VS1)
cn <- cn[grep('g_', cn)]
o <- as.numeric(gsub('g_', '', cn))
cn <- cn[order(o)]
cnames <- cn
C <- VS1[, cor(.SD), .SDcols = cnames]
H <- hclust(dist(C))

exemplar_rescale <- 
  lapply((0:gene_mem[, max(subcluster)])[H$order], function(K) {
    if (gene_mem[subcluster == K, .N] > 1) {
      S <- VS[, gene_mem$subcluster == K]
      w <- gene_mem[subcluster == K, local_density]
      S <- S[, order(w)]
    }
  }
  )
gaps_col <- mapply(ncol, exemplar_rescale)
exemplar_rescale <- do.call(cbind, exemplar_rescale)
gaps_col <- cumsum(unlist(gaps_col))

breaks <- c(min(exemplar_rescale), 0, 0.2, quantile(exemplar_rescale[exemplar_rescale > 0.2], seq(0.001, 0.999, by = 0.001)), max(exemplar_rescale))
col <- c('gray20', colorRampPalette(c('skyblue3', 'magenta2'))(length(breaks) - 1))

png(paste0(path, 'figures/virtual_staining_all_genes_heatmap.png'), height = 1000, width = 5000)
pheatmap(exemplar_rescale[order(exemplar$graph_cluster), ], cluster_rows = FALSE, cluster_cols = FALSE, breaks = breaks, color = col, gaps_col = gaps_col)
dev.off()


# Choose a gene group containing a particular gene (e.g. TP63) and plot a heatmap of all exemplar cells for one gene group
g <- gene_mem[subcluster == gene_mem[ID == 'TP63', subcluster], ID]
ld <- gene_mem[subcluster == gene_mem[ID == 'TP63', subcluster], local_density]
VS <- cell_coord[cell_ID %in% cell_points$ID, ] %*% t(gene_coord[gene_ID %in% g,])
colnames(VS) <- g
breaks <- c(min(VS), 0, 0.2, quantile(VS[VS > 0.2], seq(0.001, 0.999, by = 0.001)), max(VS))
col <- c('gray20', colorRampPalette(c('skyblue3', 'magenta2'))(length(breaks) - 1))
png(paste0(path, 'figures/heatmap_dot_product_TP63.png'), height = 1000, width = 2000)
pheatmap(VS[order(cell_points$graph_cluster), order(ld)], cluster_rows = FALSE, cluster_cols = FALSE, breaks = breaks, color = col)#, 
dev.off()


######
##### 
# Gene set annotations
# GO/KEGG enrichment
# Panglao marker genes

er <- 
  lapply(0:max(gene_mem$subcluster), function(j) {
    g <- gene_mem[abs(max_gof - self_gof) < .Machine$double.eps^0.5][subcluster == j, ID]
    e <- enrich_results(g)
    return(e)
  }
  )

subcluster_GO <-
  lapply(1:length(er), function(k) cbind(paste0('subcluster ', k), GO_display(er[[k]])))
subcluster_GO <- do.call(rbind, subcluster_GO[mapply(ncol, subcluster_GO)==2])
fwrite(subcluster_GO, file = paste0(path, 'figures/enriched_GO_terms.csv'))

subcluster_KEGG <-
  lapply(1:length(er), function(k) cbind(paste0('subcluster ', k), KEGG_display(er[[k]])))
subcluster_KEGG <- do.call(rbind, subcluster_KEGG[mapply(ncol, subcluster_KEGG)==2])
fwrite(subcluster_KEGG, file = paste0(path, 'figures/enriched_KEGG_terms.csv'))

# download the Panglao marker gene file and import
cell_types <- fread("../../PanglaoDB_markers_27_Mar_2020.tsv")
types <- names(cell_types[, table(`cell type`)])

present_genes <- sapply(1:length(types), function(i) {
  types_genes <- cell_types[`cell type` == types[[i]], `official gene symbol`]
  gene_mem$ID %in% types_genes
}
)
colnames(present_genes) <- types

clus_tab <- as.vector(gene_mem[, table(factor(subcluster))])
m <- gene_mem[, max(subcluster)]

cell_type_enrich <- lapply(1:ncol(present_genes), function(i){
  if (nrow(gene_mem[present_genes[, i]]) > 0) {
    clus_tab_subset1 <- as.vector(gene_mem[present_genes[, i], table(factor(subcluster, levels = 0:..m))])
    clus_tab_subset <- clus_tab_subset1/sum(clus_tab_subset1)
    w <- clus_tab_subset > 0
    tmp <- clus_tab[w]/sum(clus_tab)
    
    enrich <- log2(clus_tab_subset[w]/tmp)
    #names(enrich) <- which(w) - 1
    return(data.table(enrich_score = enrich, raw_freq = clus_tab_subset1[w], subcluster = which(w) - 1, cell_type = types[i]))
  } else {
    return(NULL)
  }
}
)

w <- !mapply(is.null, cell_type_enrich)
cell_type_enrich <- cell_type_enrich[w]
cell_type_enrich <- do.call(rbind, cell_type_enrich)
fwrite(cell_type_enrich, paste0(path, 'figures/panglao_cell_type.csv'))



#######################################
#######################################
#######################################
#######################################
# Example two sample analysis
# Under development - it will become more streamlined soon with a compact python workflow


# sample 1
path <- paste0(root_path, 'output/stomics_top_level/C04597D6/')

gof_gene <- fread(paste0(path, 'transform/gof_gene.txt.gz'))
gene_mem <- fread(paste0(path, 'genes/membership_all.txt.gz'))
cell_mem <- fread(paste0(path, 'cells/membership_all.txt.gz'))
cell_points <- fread(paste0(path, 'cells/exemplars.txt.gz'))
cell_coord <- fread(paste0(path, 'transform/cell_coord.txt.gz'))
cell_ID <- cell_coord$ID
cell_coord <- t(apply(cell_coord[, 2:51], 1, function(x) x/sqrt(sum(x^2))))
gene_coord <- fread(paste0(path, 'transform/gene_coord.txt.gz'))
gene_ID <- gene_coord$ID
gene_coord <- t(apply(gene_coord[, 2:51], 1, function(x) x/sqrt(sum(x^2))))
VS <- cell_coord[cell_ID %in% cell_points$ID, ] %*% t(gene_coord)

bundle_C <- list(gene_mem = gene_mem, cell_mem = cell_mem, cell_points = cell_points,cell_ID = cell_ID, gene_ID = gene_ID, VS = VS)

# sample 2
path <- paste0(root_path, 'output/stomics_top_level/D04319C6/')

gof_gene <- fread(paste0(path, 'transform/gof_gene.txt.gz'))
gene_mem <- fread(paste0(path, 'genes/membership_all.txt.gz'))
cell_mem <- fread(paste0(path, 'cells/membership_all.txt.gz'))
cell_points <- fread(paste0(path, 'cells/exemplars.txt.gz'))
cell_coord <- fread(paste0(path, 'transform/cell_coord.txt.gz'))
cell_ID <- cell_coord$ID
cell_coord <- t(apply(cell_coord[, 2:51], 1, function(x) x/sqrt(sum(x^2))))
gene_coord <- fread(paste0(path, 'transform/gene_coord.txt.gz'))
gene_ID <- gene_coord$ID
gene_coord <- t(apply(gene_coord[, 2:51], 1, function(x) x/sqrt(sum(x^2))))
VS <- cell_coord[cell_ID %in% cell_points$ID, ] %*% t(gene_coord)

bundle_D <- list(gene_mem = gene_mem, cell_mem = cell_mem, cell_points = cell_points,cell_ID = cell_ID, gene_ID = gene_ID, VS = VS)

# gathering virtual stains of exemplar cells of both samples and combining, create similarity matrix
VS_both <- rbind(bundle_C$VS, bundle_D$VS)
VS_both[VS_both < 0.2] <- 0
VS_both <- t(apply(VS_both, 1, function(x) x/sqrt(sum(x^2))))
VS_sim <- VS_both %*% t(VS_both)
VS_sim[VS_sim < 0.2] <- 0

# create similarity network, louvain clustering
VS_gr <- graph_from_adjacency_matrix(VS_sim, mode = 'undirected', weighted = TRUE, diag = FALSE)
v_col <- c('red', 'blue')[rep(1:2, c(nrow(bundle_C$cell_points), nrow(bundle_D$cell_points)))]
v_size <- log10(c(bundle_C$cell_points$subcluster_count, bundle_D$cell_points$subcluster_count))
v_size <- (1:5)[cut(v_size, quantile(v_size, (0:5)/5), include.lowest = TRUE)]

VS_clus <- cluster_louvain(VS_gr, weight = E(VS_gr)$weight^2, resolution = 1)
comps <- components(VS_gr)
w <- comps$membership == which.max(comps$csize)
ew <- (1:5)[cut(E(subgraph(VS_gr, w))$weight, 5, include.lowest = TRUE)]

# plot the similarity network, nodes coloured by patients
png(paste0(root_path, 'output/stomics_top_level/cell_scaffold_combined_sample.png'), height = 1500, width = 1500)
set.seed(42)
plot(subgraph(VS_gr, w), vertex.size = (v_size^2/10)[w], vertex.color = v_col[w], edge.width = ew^2/20, edge.color = rgb(0, 0, 0, 0.8), vertex.label = NA)
dev.off()

# similarity network, nodes coloured by louvain clustering
set.seed(43)
col <- sample(colvec[1:15])
png(paste0(root_path, 'output/stomics_top_level/cell_scaffold_combined_cluster.png'), height = 1500, width = 1500)
set.seed(42)
plot(subgraph(VS_gr, w), vertex.size = (v_size^2/10)[w], vertex.color = col[VS_clus$membership][w], edge.width = ew^2/20, edge.color = rgb(0, 0, 0, 0.8), vertex.label = NA)
dev.off()

# transferring the louvain clustering to the XY domain
Lc <- nrow(bundle_C$cell_points)
Ld <- nrow(bundle_D$cell_points)
clus_dt <- rbind(
  data.table(ID = bundle_C$cell_points$ID, subcluster = bundle_C$cell_points$subcluster, sample = 'C', clus = VS_clus$membership[1:Lc]),
  data.table(ID = bundle_D$cell_points$ID, subcluster = bundle_D$cell_points$subcluster, sample = 'D', clus = VS_clus$membership[(Lc+1):(Lc+Ld)])
)

XY_path <- c( '/XY_path/XY_C04597D6.txt.gz', '/XY_path/XY_D04319C6.txt.gz')

i = 1
XY <- fread(XY_path[i])
png(paste0(root_path, 'output/stomics_top_level/C04597D6/figures/XY_combined.png'), height = 2000, width = 2000)
XY[bundle_C$cell_mem[clus_dt[sample == 'C'], on = 'subcluster == subcluster'], on = 'V1 == ID'][, plot(X, Y, pch = 16, cex = 1.25, col = col[clus])]
dev.off()

i = 2
XY <- fread(XY_path[i])
png(paste0(root_path, 'output/stomics_top_level/D04319C6/figures/XY_combined.png'), height = 2000, width = 2000)
XY[bundle_D$cell_mem[clus_dt[sample == 'D'], on = 'subcluster == subcluster'], on = 'V1 == ID'][, plot(X, Y, pch = 16, cex = 1.25, col = col[clus])]
dev.off()

# transferring louvain clustering to biomarker specificity network

SN <- get_graph(path = fn, 0.4)
#SN <- fread(fn)
# Create data.table to help with plotting
vertex_df <-  data.table(name = names(V(SN)), colour = 'lightgrey', size = 4)
vertex_df[grepl('c', name), colour := rgb(0, 0.5, 0, 0.5)]
vertex_df[grepl('c', name), size := 3]

vertex_df[, which := gsub('_[0-9]+', '', name)]
vertex_df[, subcluster := as.numeric(gsub('[a-z]_', '', name))]

set.seed(43)
col <- sample(colvec[1:15])
vertex_df[, clus_both := -1]
vertex_df[which == 'c', colour := col[clus_dt[sample == 'C', clus]]]

ew <- (1:5)[cut(E(SN)$weight, 5, include.lowest = TRUE)]
png(paste0(path, 'figures/spec_network.png'), height = 1000, width = 1000)
set.seed(42)
plot(SN, vertex.size = vertex_df$size, edge.width = ew^2/2, vertex.color = vertex_df$colour, edge.color = rgb(0, 0, 0, 0.2), vertex.label = NA)
dev.off()


























