def Workflow(                   
        input_dataframe: pd.DataFrame, # input count matrix
        input_data_path: str, # location of input_dataframe
        output_parent_directory: str, # location of directory structure
        ncomps: int = 50, # input parameter for Transform
        n_iter: int = 10, # input parameter for Transform
        n_oversamples: int = 50, # input parameter for Transform
        transform_input: bool = True,
        cell_K_nn: int = 5, # input parameter for Cluster
        cell_clus_steps: int = 1000,  # input parameter for Cluster
        gene_K_nn: int = 5,  # input parameter for Cluster
        gene_clus_steps: int = 1000,  # input parameter for Cluster
        similarity_threshold_clus_cell: float = 0.2,  # input parameter for Cluster (cell)
        similarity_threshold_clus_gene: float = 0.2, # input parameter for Cluster (gene)
        p_cell: int = 2, # Minkowski distance parameter
        p_gene: int = 2, # Minkowski distance parameter
        ncomp_clus: int = 0, # number of components used for clustering
        metric: str = 'minkowski', # clustering metric
        similarity_threshold_specificity: float = 0.2, # input parameter for SpecificityNetwork
        neighbourhood_flow_K_nn: int = 6,  # input parameter for NeighbourhoodFlow
        dist_thres: float = 10000, # input parameter for NeighbourhoodFlow
        cell_join_id: str = 'cell_id', # input parameter for NeighbourhoodFlow
        XY: pd.DataFrame = None, # input parameter for NeighbourhoodFlow 
        XY_path: str = '',

        new_directory_structure: bool = True, # control arguments for Transform
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
        """A workflow function to automate sample processing. 

        Use the defaults to create all the outputs in a fresh directory. 
        There are many parameter options for finer-grained control but consider these experimental at this stage. 
        
        Parameters
        ----------
        input_dataframe : pd.DataFrame
            input count DataFrame
        input_data_path : str
            location of input_dataframe, to include in output documentation
        output_parent_directory : str
            location of directory structure, where to save output
        ncomps : int 
            input parameter for Transform. Default is 50
        n_iter : int
            input parameter for Transform. Default is 10
        n_oversamples : int
            input parameter for Transform. Default is 50
        transform_input : bool
            Boolean indicating whether to perform transformation of input data. Default is True. 
        cell_K_nn : int 
            input parameter for Cluster. Default is 10
        cell_clus_steps : int
            input parameter for Cluster. Default is 1000
        gene_K_nn : int
            input parameter for Cluster. Default is 10
        gene_clus_steps : int
            input parameter for Cluster. Default is 1000
        similarity_threshold_clus_cell : float
            input parameter for Cluster (mode = 'cell'). Default is 0.2
        similarity_threshold_clus_gene : float
            input parameter for Cluster (mode = 'gene'). Default is 0.2
        p_cell: float 
            Input parameter for Cluster (mode = 'cell') to use with metric = 'minkowski'. Default is 2
        p_gene: float 
            Input parameter for Cluster (mode = 'gene') to use with metric = 'minkowski'. Default is 2
        ncomp_clus : int
            Number of components used for clustering. Default is dummy value 0 to use all available components. 
        metric : str
            Metric to use to calculate nearest neighbours. Default is 'minkowski'. 
        similarity_threshold_clus_specificity : float
            input parameter for SpecificityNetwork. Default is 0.2
        neighbourhood_flow_K_nn : int
            input parameter for NeighbourhoodFlow. Default is 5
        dist_thres : float
            input parameter for NeighbourhoodFlow. Default is 10000
        cell_join_id : str
            input parameter for NeighbourhoodFlow. Default is 'cell_id'
        XY : pd.DataFrame = None
            input parameter for NeighbourhoodFlow.  

        new_directory_structure : bool
            Boolean indicating whether to create new output directory structure. Default is True
        transform : bool = True
            Boolean indicating whether to create new instance of Transform class. Default is True. 
        transform_pickle_path : str 
            String indicating path of previously saved instance of Transform class. Default is None. 
        umap : bool 
            Boolean indicating whether to create umaps in instance of Transform class. Default is True. 
        umap_ncomp_gene : int
            Integer indicating whether to create umap on subset of components. Default is None and all components are used. 
        umap_ncomp_cell : int
            Integer indicating whether to create umap on subset of components. Default is None and all components are used. 
        transform_goodness_of_fit : bool = True
            Boolean indicating whether to create goodness of fit statistics. Default is True. 
        transform_overwrite : bool = False
            Boolean indicating whether to overwrite existing results. Used as a safeguard against accidents. Default is False. 
        transform_R_output : bool
            Boolean indicating whether to save R-friendly output. Default is True. 
        transform_pickle_output : bool = True
            Boolean indicating whether to pickle the Transform class instance. Default is True. 

        cluster_cells : bool
            Boolean indicating whether to cluster the Transform class instance. Default is True. 
        cluster_cells_subdirectory : str = None,
            String indicating whether to create a subdirectory in the standard output directory structure. 
            Used to create clustering with multiple parameter settings. Default is None. 
        cluster_cells_overwrite : bool 
            Boolean indicating whether to overwrite existing results. Used as a safeguard against accidents. Default is False.
        cluster_cells_pickle_path : str
            String indicating path of previously saved instance of Cluster class. Default is None. 
        cluster_cells_R_output : bool
            Boolean indicating whether to save R-friendly output. Default is True.
        cluster_cells_pickle_output : bool
            Boolean indicating whether to pickle the Cluster class instance. Default is True.

        cluster_genes : bool
            Boolean indicating whether to cluster the Transform class instance. Default is True.
        cluster_genes_subdirectory : str
            String indicating whether to create a subdirectory in the standard output directory structure. 
            Used to create clustering with multiple parameter settings. Default is None. 
        cluster_genes_overwrite : bool
            Boolean indicating whether to overwrite existing results. Used as a safeguard against accidents. Default is False.
        cluster_genes_pickle_path : str
            String indicating path of previously saved instance of Transform class. Default is None. 
        cluster_genes_R_output : bool
            Boolean indicating whether to save R-friendly output. Default is True.
        cluster_genes_pickle_output : bool
            Boolean indicating whether to pickle the Cluster class instance. Default is True.

        specificity_network : bool= True
            Boolean indicating whether to create a specificity network. Default is True.
        specificity_network_subdirectory : str
            String indicating whether to create a subdirectory in the standard output directory structure. 
            Used to create specificity networks with multiple parameter settings. Default is None. 
        specificity_network_overwrite : bool
            Boolean indicating whether to overwrite existing results. Used as a safeguard against accidents. Default is False.
        specificity_network_R_output : bool
            Boolean indicating whether to save R-friendly output. Default is True.
        specificity_network_pickle_output : bool
            Boolean indicating whether to pickle the SpecificityNetwork class instance. Default is True.

        neighbourhood_flow : bool = False, # control arguments for NeighbourhoodFlow
        neighbourhood_flow_subdirectory : str 
            String indicating whether to create a subdirectory in the standard output directory structure. 
            Used to create neighbourhood flows with multiple parameter settings. Default is None. 
        neighbourhood_flow_overwrite : bool
            Boolean indicating whether to overwrite existing results. Used as a safeguard against accidents. Default is False.
        neighbourhood_flow_R_output : bool
            Boolean indicating whether to save R-friendly output. Default is True.
        neighbourhood_flow_pickle_output : bool
            Boolean indicating whether to pickle the NeighbourhoodFlow class instance. Default is True.

        """

        if new_directory_structure: 
            directory_structure(output_parent_directory)

        if transform:
            if os.listdir(output_parent_directory + 'output/transform/') != []:
                if not transform_overwrite:
                    raise FileExistsError("/output/transform/ already contain results. \nYou will overwrite these results if you proceed. \nPlease create a new parent directory or use transform_overwrite = True")            
            Y = Transform(input_dataframe, ncomps = ncomps, n_iter = n_iter, n_oversamples=n_oversamples, transform = transform_input, goodness_of_fit = transform_goodness_of_fit)    
            # if transform_goodness_of_fit:
            #     if transform_input:
            #         Y = Transform(input_dataframe, ncomps = ncomps, n_iter = n_iter, transform = True, goodness_of_fit = True)
            #     else:
            #         Y = Transform(input_dataframe, ncomps = ncomps, n_iter = n_iter, transform = False, goodness_of_fit = True)
            # else: 
            #     if transform_input:
            #         Y = Transform(input_dataframe, ncomps = ncomps, n_iter = n_iter, transform = True, goodness_of_fit = False)
            #     else:
            #         Y = Transform(input_dataframe, ncomps = ncomps, n_iter = n_iter, transform = False, goodness_of_fit = False)
            if umap:
                Y.umap(ncomp_cell = umap_ncomp_cell, ncomp_gene = umap_ncomp_gene)

            if transform_R_output:
                Y.save(path = output_parent_directory + 'output/transform/')
            if transform_pickle_output:
                file = open(output_parent_directory + 'output/transform/transform.pkl', 'wb')
                pickle.dump(Y, file)
                file.close()
            file = open(output_parent_directory + 'output/transform/info.txt', 'w')
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
            cell_cluster = Cluster(Y, K_nn = cell_K_nn, clus_steps = cell_clus_steps, similarity_threshold = similarity_threshold_clus_cell, p = p_cell, ncomp_clus = ncomp_clus, metric = metric)
            if cluster_cells_subdirectory == None:
                subdir = "output/cells/"
            else:
                subdir = "output/cells/" + cluster_cells_subdirectory
            if not os.path.isdir(output_parent_directory + subdir):
                os.mkdir(output_parent_directory + subdir)
            else:
                if os.listdir(output_parent_directory + subdir) != []:
                    if not cluster_cells_overwrite:
                        raise FileExistsError("output/cells/cluster_cells_subdirectory already contain results. \nYou will overwrite these results if you proceed. \nPlease create a new directory or use cluster_cells_overwrite = True")            
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
            file.write("similarity_threshold: " + str(similarity_threshold_clus_cell) + '\n')
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
            gene_cluster = Cluster(Y, K_nn = gene_K_nn, clus_steps = gene_clus_steps, mode = 'genes', similarity_threshold = similarity_threshold_clus_gene, p = p_gene, ncomp_clus = ncomp_clus, metric = metric)
            if cluster_genes_subdirectory == None:
                subdir = "output/genes/"
            else:
                subdir = "output/genes/" + cluster_genes_subdirectory
            if not os.path.isdir(output_parent_directory + subdir):
                os.mkdir(output_parent_directory + subdir)
            else:
                if os.listdir(output_parent_directory + subdir) != []:
                    if not cluster_genes_overwrite:
                        raise FileExistsError("output/genes/cluster_genes_subdirectory already contain results. \nYou will overwrite these results if you proceed. \nPlease create a new directory or use cluster_genes_overwrite = True")            
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
            file.write("similarity_threshold: " + str(similarity_threshold_clus_gene) + '\n')
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

            spec_network = SpecificityNetwork(Y, cell_cluster, gene_cluster, similarity_threshold_specificity)
            if specificity_network_subdirectory == None:
                subdir = "output/specificity_network/"
            else:
                subdir = "output/specificity_network/" + specificity_network_subdirectory
            if not os.path.isdir(output_parent_directory + subdir):
                os.mkdir(output_parent_directory + subdir)
            else:
                if os.listdir(output_parent_directory + subdir) != []:
                    if not specificity_network_overwrite:
                        raise FileExistsError("output/specificity_network/specificity_network_subdirectory already contain results. \nYou will overwrite these results if you proceed. \nPlease create a new directory or use specificity_network_overwrite = True")            
            if specificity_network_R_output:
                spec_network.save(output_parent_directory + subdir)
            if specificity_network_pickle_output:
                file = open(output_parent_directory + subdir + 'specificity_network.pkl', 'wb')
                pickle.dump(spec_network, file)
                file.close()
            file = open(output_parent_directory + subdir + 'info.txt', 'w')
            file.write("similarity_threshold: " + str(similarity_threshold_specificity) + '\n')
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

            if (not cluster_cells and cluster_cells_pickle_path == None) or XY is None:
                raise ReferenceError("NeighbourhoodFlow requires cell Cluster object and XY DataFrame")
            NF = NeighbourhoodFlow(cell_cluster = cell_cluster, XY = XY, K_nn = neighbourhood_flow_K_nn, dist_thres = dist_thres, cell_join_id = cell_join_id)
            if neighbourhood_flow_subdirectory == None:
                subdir = "output/neighbourhood_flow/"
            else:
                subdir = "output/neighbourhood_flow/" + neighbourhood_flow_subdirectory
            if not os.path.isdir(output_parent_directory + subdir):
                os.mkdir(output_parent_directory + subdir)
            else:
                if os.listdir(output_parent_directory + subdir) != []:
                    if not neighbourhood_flow_overwrite:
                        raise FileExistsError("output/neighbourhood_flow/neighbourhood_flow_subdirectory already contain results. \nYou will overwrite these results if you proceed. \nPlease create a new directory or use neighbourhood_flow_overwrite = True")            
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

