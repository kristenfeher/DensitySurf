
import os

__all__ = ['directory_structure', 'multi_sample_directory_structure']

import os

def directory_structure(path):
    """A function to create an output directory structure"""
    # if not os.path.isdir(path + 'output/'):
    #     os.mkdir(path + 'output/')
    # else:
    #     print("The output directory already exists.")
    if not os.path.isdir(os.path.join(path , 'transform')):
        os.mkdir(os.path.join(path , 'transform'))
    if not os.path.isdir(os.path.join(path , 'cells')):
        os.mkdir(os.path.join(path , 'cells'))
    if not os.path.isdir(os.path.join(path , 'genes')):
        os.mkdir(os.path.join(path , 'genes'))
    if not os.path.isdir(os.path.join(path , 'specificity_network')):
        os.mkdir(os.path.join(path , 'specificity_network'))
    if not os.path.isdir(os.path.join(path , 'figures')):
        os.mkdir(os.path.join(path , 'figures'))
    if not os.path.isdir(os.path.join(path , 'neighbourhood_flow')):
        os.mkdir(os.path.join(path , 'neighbourhood_flow'))

    if (
        os.listdir(os.path.join(path , 'transform')) != [] or
        os.listdir(os.path.join(path , 'cells')) != [] or
        os.listdir(os.path.join(path , 'genes')) != [] or
        os.listdir(os.path.join(path , 'specificity_network')) != [] or
        os.listdir(os.path.join(path , 'figures')) != [] or
        os.listdir(os.path.join(path , 'neighbourhood_flow')) != [] 
        ):
        raise FileExistsError("Directories already contain results. \nYou will overwrite these results if you proceed. \nPlease consider creating a new parent directory")
 

def multi_sample_directory_structure(top_level_path, sample_root_names):

    if not isinstance(top_level_path, str):
        raise TypeError("top_level_path argument should be a string containing the path to the top level directory")

    if isinstance(sample_root_names, list):
        if not all([isinstance(item, str) for item in sample_root_names]):
        #if not all(isinstance(sample_root_names, str)):
            raise TypeError("sample_root_names argument must be a list of strings corresponding to sample labels")

    if not os.path.isdir(top_level_path):
        os.mkdir(top_level_path)

    for i in range(0, len(sample_root_names)):
        file_name = os.path.join(top_level_path, sample_root_names[i])
        if not os.path.isdir(file_name):
            os.mkdir(file_name)
        try:
            directory_structure(file_name)
        except FileExistsError as e:
            print("WARNING: A directory in \"" + file_name +  "\" exists and contains results. \nYou may overwrite results if you proceed.")

    file_name = os.path.join(top_level_path, "combined_output")
    if not os.path.isdir(file_name):
        os.mkdir(file_name)
    try:
        directory_structure(file_name)
    except FileExistsError as e:
        print("WARNING: A directory in \"" + file_name +  "\" exists and contains results. \nYou may overwrite results if you proceed.")