import numpy as np
import plot_func
import matplotlib.pyplot as plt

import model
import evaluate

from copy import copy

def init(experiment_name, T, dt, learn, snapshots=0, snapshot_timesteps=0):
    model.init(experiment_name, T, dt, learn, snapshots, snapshot_timesteps)

def run(time_0 = 0):
    model.run(time_0)

class Group(model.Group):

    def __init__(self, group_size, identifier=None, **kwargs):
        #We assert that we have no dendritic compartments
        assert len(np.shape(np.atleast_1d(group_size))) == 1

        super().__init__(group_size, identifier)

        self.dt = model.model_dt

        self.T = kwargs.get("T", np.ones(group_size)*0.01) #Spiking threshhold
        self.T_learning_rate = self.dt*kwargs.get("T_learning_rate", 1e-2)

        self.T_log = []

        self.deltaU = kwargs.get("deltaU", 1e-2) #Spread of the spiking
        self.deltaUinv = 1/self.deltaU

        self.spike_rate = kwargs.get("spike_rate", 10)*1e-3 #rho, default is 10 spikes/sec

        self.timescale = kwargs.get("timescale",20) #tau/timescale

        self._decay_const = np.exp(-self.dt/self.timescale) #decay for spiking dynamics


    def update_post_synaptic(self):
        self.z = self.z*self._decay_const + self.s

    def update_pre_synaptic(self):
        self.u = np.sum([ model.get_connection(source).value for source in self.source_identifiers ],axis=0).flatten()
        p = np.random.rand(self.group_size)

    def update_spikes(self):
        p = np.random.rand(self.group_size)
        self.s = ( 1/(1+np.exp(-(self.u - self.T)*self.deltaUinv)) > p )*1

    def update_weights(self):
        self.T += self.T_learning_rate*(self.s - self.spike_rate * self.dt)

    def update_log(self, learn):
        super().update_log(learn)
        if learn:
            self.T_log.append(copy(self.T))

    def cleanup(self):
        super().cleanup()
        self.T_log = np.array(self.T_log).T

    def clear_log(self):
        super().clear_log()
        self.T_log = []

    def save(self, path):
        super().save(path)
        if len(self.T_log) != 0:
            plot_func.plot_threshholds(self.T_log, self.dt, self.identifier)
            plt.savefig(f"{path}/{self.identifier.replace(' ','_')}_threshholds.svg", dpi=model.model_dpi)

class Input(model.Input):
    def __init__(self, input_shape, func=None, identifier=None, snapshot_func=None):
        super().__init__(input_shape, func, identifier, snapshot_func=snapshot_func)

        self.dt = model.model_dt

        self.decoding = np.zeros(input_shape)
        self.decoding_log = []


    def update(self, timestamp):
        super().update(timestamp)

        self.decoding = np.sum([ model.get_connection(target).weights.T.dot(
                                 model.get_group(model.get_connection(target).target_identifier).z)
                                 for target in self.target_identifiers ], axis =0)

    def update_log(self):
        super().update_log()
        self.decoding_log.append(self.decoding)

    def cleanup(self):
        super().cleanup()
        self.decoding_log = np.array(self.decoding_log).T

    def clear_log(self):
        super().clear_log()
        self.decoding_log = []

    def save(self, path):
        super().save(path)
        if len(self.x_log) != 0:
            plot_func.plot_decoding(self.x_log, self.decoding_log, name=self.identifier, dt = self.dt)
            plt.savefig(f"{path}/{self.identifier.replace(' ','_')}_decoding.svg", dpi=model.model_dpi)


    def plot_decoding_performance(self):
        super().plot_decoding_performance()

        if len(self.decoding_log) != 0:
            plot_func.plot_decoding_performance(self)



class fw_Connection(model.Connection):
    def __init__(self, source, target, identifier=None, active=True, update_weights=True, learning_rate=1e-3):
        super().__init__(source, target, identifier,  active)
        self.dt = model.model_dt

        self.weights = np.ones((target.group_size, *source.shape))*0.01
        self.value = np.ones(target.group_size)*0

        self.learning_rate = self.dt * learning_rate

    def update_post_synaptic(self):
        if self.active:
            if self.source_type == 'group':
                self.value = self.weights.dot(model.get_group(self.source_identifier).z)
            elif self.source_type == 'input':
                self.value = self.weights.dot(model.get_input(self.source_identifier).x)

    def update_weights(self):
        if self.update_weights:
            #TODO
            source = model.get_input(self.source_identifier)
            target = model.get_group(self.target_identifier)

            self.weights += self.learning_rate*np.outer(target.z, source.x - source.decoding)

class re_Connection(model.Connection):
    def __init__(self, source, target, identifier, active=True):
        super().__init__(source, target, identifier, active)
        self.dt = model.model_dt

        self.weights = np.ones((target.group_size, *source.shape))
        self.value = np.ones(target.group_size)*0

        self.connection_pairs = []

        #Graph theory for matching the encoder/decoder pairs
        for decoder_identifier in source.source_identifiers:
            decoder = model.get_connection(decoder_identifier)

            if isinstance(decoder, fw_Connection):
                for encoder_identifier in model.get_input(decoder.source_identifier).target_identifiers:

                    encoder = model.get_connection(encoder_identifier)
                    if isinstance(encoder, fw_Connection) and \
                       encoder.active and \
                       encoder.target_identifier == self.target_identifier:

                           self.connection_pairs.append((decoder_identifier, encoder_identifier))


    def update_post_synaptic(self):
        if self.source_type == 'group':
            self.value = self.weights.dot(model.get_group(self.source_identifier).z)
        elif self.source_type == 'input':
            self.value = self.weights.dot(model.get_input(self.source_identifier).x)

    def update_weights(self):
        if self.update_weights:
            self.weights = - np.sum([ model.get_connection(pair[1]).weights.dot(model.get_connection(pair[0]).weights.T)
                                     for pair in self.connection_pairs ], axis=0)


