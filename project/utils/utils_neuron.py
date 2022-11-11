import numpy as np
import pickle

def bin2dict(bin_spikes_matrix):
    spike_row_inds, spike_times = np.nonzero(bin_spikes_matrix)
    row_inds_spike_times_map = {}
    for row_ind, syn_time in zip(spike_row_inds,spike_times):
        if row_ind in row_inds_spike_times_map.keys():
            row_inds_spike_times_map[row_ind].append(syn_time)
        else:
            row_inds_spike_times_map[row_ind] = [syn_time]
    return row_inds_spike_times_map

def dict2bin(row_inds_spike_times_map, num_segments, sim_duration_ms):
    bin_spikes_matrix = np.zeros((num_segments, sim_duration_ms), dtype='bool')
    for row_ind in row_inds_spike_times_map.keys():
        for spike_time in row_inds_spike_times_map[row_ind]:
            bin_spikes_matrix[row_ind,spike_time] = 1.0
    return bin_spikes_matrix

def parse_sim_experiment_file(sim_experiment_file):        
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

