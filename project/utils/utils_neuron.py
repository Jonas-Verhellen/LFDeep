import numpy as np
import pickle

def bin2dict(bin_spikes_matrix):
    """
    Convert a binary spikes matrix to a dictionary of spike times for each row.

    This function takes a binary spikes matrix as input and converts it into a dictionary where the keys
    represent row indices, and the values are lists of spike times for each row.

    Args:
        bin_spikes_matrix (numpy.ndarray): A binary spikes matrix where each row represents a neuron and each
            column represents a time step. Non-zero values indicate spike occurrences.

    Returns:
        dict: A dictionary where keys are row indices and values are lists of spike times.

    """
    spike_row_inds, spike_times = np.nonzero(bin_spikes_matrix)
    row_inds_spike_times_map = {}
    for row_ind, syn_time in zip(spike_row_inds,spike_times):
        if row_ind in row_inds_spike_times_map.keys():
            row_inds_spike_times_map[row_ind].append(syn_time)
        else:
            row_inds_spike_times_map[row_ind] = [syn_time]
    return row_inds_spike_times_map

def dict2bin(row_inds_spike_times_map, num_segments, sim_duration_ms):
    """
    Convert a dictionary of spike times to a binary spikes matrix.

    This function takes a dictionary of spike times for each row and converts it into a binary spikes matrix
    where each row represents a neuron, and each column represents a time step. Non-zero values in the matrix
    indicate spike occurrences.

    Args:
        row_inds_spike_times_map (dict): A dictionary where keys are row indices, and values are lists of spike times.
        num_segments (int): The number of rows in the binary spikes matrix.
        sim_duration_ms (int): The duration in milliseconds represented by the binary spikes matrix.

    Returns:
        numpy.ndarray: A binary spikes matrix with the specified number of segments and duration.

    """
    bin_spikes_matrix = np.zeros((num_segments, sim_duration_ms), dtype='bool')
    for row_ind in row_inds_spike_times_map.keys():
        for spike_time in row_inds_spike_times_map[row_ind]:
            bin_spikes_matrix[row_ind,spike_time] = 1.0
    return bin_spikes_matrix

def parse_sim_experiment_file(sim_experiment_file):        
    """
    Parse a simulation experiment file and extract relevant data.

    This function loads a simulation experiment file using pickle, and then extracts various data from it,
    including inputs, spike times, soma voltages, dendritic voltages, and nexus voltages.

    Args:
        sim_experiment_file (str): The file path of the simulation experiment file.

    Returns:
        tuple: A tuple containing the following data:
        - X (numpy.ndarray): Binary spike data for all synapses, shape (num_synapses, sim_duration_ms, num_simulations).
        - y_spike (numpy.ndarray): Spike times data, shape (sim_duration_ms, num_simulations).
        - y_soma (numpy.ndarray): Soma voltage data, shape (sim_duration_ms, num_simulations).
        - y_DVTs (numpy.ndarray): Dendritic voltage data, shape (num_segments, sim_duration_ms, num_simulations).
        - y_nexus (numpy.ndarray): Nexus voltage data, shape (sim_duration_ms, num_simulations).

    """
    experiment_dict = pickle.load(open(sim_experiment_file, "rb" ),encoding='latin1')
    
    # gather params
    num_simulations = len(experiment_dict['Results']['listOfSingleSimulationDicts'])
    num_segments    = len(experiment_dict['Params']['allSegmentsType'])
    sim_duration_ms = experiment_dict['Params']['totalSimDurationInSec'] * 1000
    num_ex_synapses  = num_segments
    num_inh_synapses = num_segments
    num_synapses = num_ex_synapses + num_inh_synapses
    
    # collect X, y_spike, y_soma
    X = np.zeros((num_synapses,sim_duration_ms,num_simulations), dtype='bool')
    y_spike = np.zeros((sim_duration_ms,num_simulations))
    y_soma = np.zeros((sim_duration_ms,num_simulations))
    y_nexus = np.zeros((sim_duration_ms,num_simulations))

    # if we recive PCA model of DVTs, then output the projection on that model, else return the full DVTs
    y_DVTs = np.zeros((num_segments,sim_duration_ms,num_simulations), dtype=np.float16)
    
    # go over all simulations in the experiment and collect their results
    for k, sim_dict in enumerate(experiment_dict['Results']['listOfSingleSimulationDicts']):
        X_ex  = dict2bin(sim_dict['exInputSpikeTimes'] , num_segments, sim_duration_ms)
        X_inh = dict2bin(sim_dict['inhInputSpikeTimes'], num_segments, sim_duration_ms)
        X[:,:,k] = np.vstack((X_ex,X_inh))
        spike_times = (sim_dict['outputSpikeTimes'].astype(float) - 0.5).astype(int)
        y_spike[spike_times,k] = 1.0
        y_soma[:,k] = sim_dict['somaVoltageLowRes'] # the whole low res worries me a bit, they seem to be doing osme add hoc regularisation their code - are there issues there? Can we subsample correctly from the high res one? 
        y_DVTs[:,:,k] = sim_dict['dendriticVoltagesLowRes']
        y_nexus[:,:,k]  = sim_dict['nexusVoltageLowRes']
    return X, y_spike, y_soma, y_DVTs, y_nexus

def parse_multiple_sim_experiment_files(sim_experiment_files):
    """
    Parse multiple simulation experiment files and aggregate the results.

    This function takes a list of simulation experiment files, parses each file using the `parse_sim_experiment_file`
    function, and aggregates the results into arrays for X (binary spike data), y_spike (spike times data),
    y_soma (soma voltage data), y_DVT (dendritic voltage data), and y_nexus (nexus voltage data).

    Args:
        sim_experiment_files (list of str): A list of file paths to the simulation experiment files.

    Returns:
        tuple: A tuple containing the following data:
        - X (numpy.ndarray): Binary spike data for all synapses, shape (num_synapses, sim_duration_ms, num_simulations).
        - y_spike (numpy.ndarray): Spike times data, shape (sim_duration_ms, num_simulations).
        - y_soma (numpy.ndarray): Soma voltage data, shape (sim_duration_ms, num_simulations).
        - y_DVT (numpy.ndarray): Dendritic voltage data, shape (num_segments, sim_duration_ms, num_simulations).
        - y_nexus (numpy.ndarray): Nexus voltage data, shape (sim_duration_ms, num_simulations).

    """
    for k, sim_experiment_file in enumerate(sim_experiment_files):
        X_curr, y_spike_curr, y_soma_curr, y_DVT_curr, y_nexus_curr = parse_sim_experiment_file(sim_experiment_file)
        if k == 0:
            X       = X_curr
            y_spike = y_spike_curr
            y_soma  = y_soma_curr
            y_DVT   = y_DVT_curr
            y_nexus = y_nexus_curr
        else:
            X       = np.dstack((X,X_curr))
            y_spike = np.hstack((y_spike,y_spike_curr))
            y_soma  = np.hstack((y_soma,y_soma_curr))
            y_DVT   = np.dstack((y_DVT,y_DVT_curr))
            y_nexus = np.dstack((y_nexus,y_nexus_curr))
    return X, y_spike, y_soma, y_DVT, y_nexus

