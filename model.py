import numpy as np
import itertools as it
import pickle
import datetime
import plot_func

import matplotlib.pyplot as plt

from tqdm import tqdm
from copy import deepcopy

import os

import model_io

_model_groups = {}
_model_inputs = {}
_model_connections = {}

model_experiment_name = ""
model_timesteps = 0
model_dt = 0.2# in frame/msd
model_learn = True
model_snapshots = 0
model_snapshot_timesteps = 0


model_dpi = 700

def init(experiment_name, T, dt, learn, snapshots=0, snapshot_timesteps=0):
    global model_experiment_name
    global model_timesteps
    global model_dt
    global model_learn
    global model_snapshots
    global model_snapshot_timesteps

    model_experiment_name = experiment_name
    model_timesteps = T
    model_dt = dt
    model_learn = learn
    model_snapshots = snapshots
    model_snapshot_timesteps = snapshot_timesteps

    if not os.path.exists("experiments"):
        os.mkdir(f"experiments")

    if not os.path.exists(f"experiments/{experiment_name}"):
        os.mkdir(f"experiments/{experiment_name}")

    sorted_trial_paths = sorted(os.listdir(f"experiments/{experiment_name}"))
    if len(sorted_trial_paths) != 0:
        print(f"loading experiments/{experiment_name}/{sorted_trial_paths[-1]}/model\n")
        load(f"experiments/{experiment_name}/{sorted_trial_paths[-1]}/model")

def load(model_path):
    if not os.path.exists(model_path):
        return

    model_groups_copy, model_connections_copy, model_dt = pickle.load(open(model_path, "rb"))

    #TODO Currently only copying weights, but want a better file structure for this...A
    for group_identifier in model_groups_copy:
        _model_groups[group_identifier].T = model_groups_copy[group_identifier].T

    for connection_identifier in model_connections_copy:
        _model_connections[connection_identifier].weights = model_connections_copy[connection_identifier].weights

def save(model_path):
    model_groups_copy = deepcopy(_model_groups)
    model_inputs_copy = deepcopy(_model_inputs)
    model_connections_copy = deepcopy(_model_connections)


    #Plotting performance, for windowlength see plot_func.py
    if plot_func.plot_decoding_performance(model_inputs_copy.values()):
        plt.savefig(f"{model_path}/decoding_performance.svg", dpi=model_dpi)

    for group in model_groups_copy.values():
        group.save(model_path)
        group.clear_log()
    plt.close('all')

    for inp in model_inputs_copy.values():
        inp.save(model_path)
        inp.clear_log()
    plt.close('all')

    for connection in model_connections_copy.values():
        connection.save(model_path)
        connection.clear_log()
    plt.close('all')

    print(f"\nSaving {model_path}\n")

    pickle.dump( (model_groups_copy, model_connections_copy, model_dt),
                 open(model_path + "model", "wb"))


def run(time_0 = 0, print_info=True):
    model_timestamp = time_0
    timesteps = model_timesteps
    learn = model_learn

    timestamp_string = f"{datetime.datetime.now():%d:%m:%Y-%H:%M:%S}"
    path = f"experiments/{model_experiment_name}/{timestamp_string}/"

    os.mkdir(path)

    if print_info:
        print(f"Running experiment: \"{model_experiment_name}\"\n")

        for group in _model_groups.values():
            print(f"{group} with sources:")
            for source in group.source_identifiers:
                print(f"\t {source}")

    snapshot_timepoints= [ int((timesteps - time_0)/model_snapshots*(i+1)) -1 for i in range(model_snapshots) ]

    for t in tqdm(np.arange(0, timesteps) + time_0):
        _model_update((_model_groups, _model_inputs, _model_connections), t, learn)
        model_timestamp = t

        if t in snapshot_timepoints:
            run_snapshot(model_snapshot_timesteps, path, int(np.floor(t*model_snapshots/(timesteps-time_0))))

    _model_cleanup((_model_groups, _model_inputs, _model_connections))

    save(path)

def run_snapshot(timesteps, path, snapshot):
    global _model_groups
    global _model_inputs
    global _model_connections

    groups = deepcopy(_model_groups)
    inputs = deepcopy(_model_inputs)
    connections = deepcopy(_model_connections)

    for inp in _model_inputs.values():
        inp.log= True
        inp.input_func = inp.snapshot_func
        inp.clear_log()

    for connection in _model_connections.values():
        connection.log =True
        connection.clear_log()

    for group in _model_groups.values():
        group.log=True
        group.clear_log()

    print(f"Running snapshot {snapshot}")

    for t in np.arange(0,timesteps):
        _model_update((_model_groups, _model_inputs, _model_connections), t, False)
        model_timestamp = t

    _model_cleanup((_model_groups, _model_inputs, _model_connections))


    model_io.save_snapshot((_model_groups, _model_inputs, _model_connections), path, snapshot)

    _model_groups = groups
    _model_inputs = inputs
    _model_connections = connections

def _model_update(model, timestamp, learn = False):
    groups, inputs, connections = model

    #Postsynaptic activity update

    for group in groups.values():
        group.update_post_synaptic()

    for inp in inputs.values():
        inp.update(timestamp)

    for connection in connections.values():
        connection.update_post_synaptic()

    #Presynaptic membrane potential and spikes
    for group in groups.values():
        group.update_pre_synaptic()
        group.update_spikes()

    if learn:
        for group in groups.values():
            group.update_weights()

        for connection in connections.values():
            connection.update_weights()

    for group in groups.values():
        group.update_log(learn)

    for inp in inputs.values():
        if inp.log:
            inp.update_log()

def _model_cleanup(model):
    groups, inputs, connections = model

    for group in groups.values():
        group.cleanup()

    for inp in inputs.values():
        inp.cleanup()

    for connection in connections.values():
        connection.cleanup()

class Group():

    def __repr__(self):
        return f'Neuron group {self.identifier}'

    def __init__(self, group_shape, identifier=None, log=False):
        if identifier == None:
            self.identifier = len(_model_groups)
        else:
            self.identifier = identifier

        _model_groups[self.identifier] = self

        self.shape = np.atleast_1d(group_shape)
        self.group_size = np.atleast_1d(group_shape)[0] # The first index is the size of of the group

        self.log = log

        #Presynaptic activity
        self.u = np.zeros(self.shape) #Membrane potential
        self.u_log = []

        #Postsynaptic activity
        self.s = np.zeros(self.group_size) #Spikes
        self.s_log = []
        self.z = np.zeros(self.group_size) #Spike trace
        self.z_log = []

        #Source/target connections
        self.source_identifiers = []
        self.target_identifiers = []

    def update_post_synaptic(self):
        print(f"post synaptic dynamics have not been defined for {self}")

    def update_pre_synaptic(self):
        print(f"pre synaptic dynamics have not been defined for {self}")

    def update_spikes(self):
        print(f"Spike dynamics have not been defined for {self}")

    def update_weights(self):
        print(f"Weight update has not been defined for {self}")

    def update_log(self, learn=False):
        if self.log:
            self.u_log.append(self.u)
            self.s_log.append(self.s)
            self.z_log.append(self.z)

    def cleanup(self):
        self.u_log = np.array(self.u_log).T
        self.s_log = np.array(self.s_log).T
        self.z_log = np.array(self.z_log).T

    def clear_log(self):
        self.u_log = []
        self.s_log = []
        self.z_log = []

    def save(self, path):
        if len(self.u_log) != 0:
            plot_func.plot_membrane_potential(self.u_log, group_name=self.identifier)
            plt.savefig(f"{path}/{self.identifier.replace(' ','_')}_membrane_potential.svg", dpi=model_dpi)
        if len(self.z_log) != 0:
            plot_func.plot_activity(self.z_log, group_name=self.identifier)
            plt.savefig(f"{path}/{self.identifier.replace(' ','_')}_activity.svg", dpi=model_dpi)
        if len(self.s_log) != 0:
            plot_func.plot_spike_trace(self.s_log, name=self.identifier, ylabel="s")
            plt.savefig(f"{path}/{self.identifier.replace(' ','_')}_spike_trace.svg", dpi=model_dpi)
            plot_func.plot_rate(self.s_log, name=self.identifier, dt= model_dt)
            plt.savefig(f"{path}/{self.identifier.replace(' ','_')}_rate.svg", dpi=model_dpi)

class Input():
    def __repr__(self):
        return f'Input {self.identifier}'

    def __init__(self, input_shape, func=None, identifier=None, log=False, snapshot_func=None):
        if identifier == None:
            self.identifier = len(_model_groups)
        else:
            self.identifier = identifier

        _model_inputs[self.identifier] = self

        self.shape = np.atleast_1d(input_shape)
        self.x = np.zeros(input_shape)
        self.x_log = []

        self.log = log

        if callable(func):
            self._use_func = True
            self.input_func = func
        else:
            self._use_func = False

        if callable(func):
            self.snapshot_func = snapshot_func

        #Target connections
        self.target_identifiers = []

    def update(self, timestamp):
        if self._use_func:
            self.x = self.input_func(timestamp, model_dt)

    def update_log(self):
        self.x_log.append(self.x)

    def set(self, value):
        self.x = value

    def cleanup(self):
        self.x_log = np.array(self.x_log).T

    def clear_log(self):
        self.x_log = []

    def save(self, path):
        pass

    def plot_decoding_performance(self):
        pass


class Connection():
    def __repr__(self):
        return f'Connection {self.identifier}'

    def __init__(self, source, target, identifier=None, active=True):
        self.active = active

        if identifier == None:
            self.identifier = len(_model_connections)
        else:
            self.identifier = identifier

        _model_connections[self.identifier] = self

        if isinstance(source, Group):
            self.source_identifier = source.identifier
            self.source_type = 'group'
        elif isinstance(source, Input):
            self.source_identifier = source.identifier
            self.source_type = 'input'
        else:
            raise Exception("Not a valid source")

        if not isinstance(target, Group):
            raise Exception("Not a valid target")

        self.target_identifier = target.identifier

        #Add this connection as a source to the target
        target.source_identifiers.append(self.identifier)

        #Add this connection as a target to the source
        source.target_identifiers.append(self.identifier)

        #To be initialized
        self.weights = None
        self.value = None

    def update_post_synaptic(self):
        print(f"Dynamics have not been defined for {self}")

    def update_weights(self):
        print(f"Weight update has not been defined for {self}")

    def cleanup(self):
        pass

    def clear_log(self):
        pass

    def save(self, path):
        plot_func.plot_weights(self.weights, name=self.identifier)
        plt.savefig(f"{path}/{self.identifier.replace(' ','_')}_weights.svg", dpi=model_dpi)


def get_group(group_identifier):
    return _model_groups[group_identifier]
def get_input(input_identifier):
    return _model_inputs[input_identifier]
def get_connection(connection_identifier):
    return _model_connections[connection_identifier]


