import numpy as np
from .activations import *


class LIFrate:
    
    def __init__(self,
        size,           # number of neurons
        v_rest=0,       # membrane resting potential
        v_thres=1,      # membrane threshold potential
        tau_v=0.1,      # membrane potential time constant
        tau_s=1,        # synapse time constant
        R=5,            # membrane resistance
        activation=f5,
        gamma=0.5,
        theta=0.5
    ):  
        self.size = size
        self.ntype = 'LIFrate'
        
        # parameters
        self.v_rest = v_rest
        self.v_thres = v_thres
        self.tau_v = tau_v
        self.tau_s = tau_s
        self.R = R
        self.activation = activation
        self.gamma = gamma
        self.theta = theta
        
        # variables
        self.current = np.zeros(shape=(self.size), dtype=np.float32)
        self.voltage = self.v_rest * np.ones(shape=(self.size), dtype=np.float32)
        self.outputs = np.zeros(shape=self.size, dtype=np.float32)
        
        self.inputs = None
        self.synapses = None
        
        self.compiled = False
        self.train = False
        self.stim = None
        
    def step(self, dt=0.001):
        self.current = self.synapses @ self.inputs
        if self.stim is not None:
            self.current += self.stim
        
        if self.train:
            ltp = self.gamma * self.activation(self.gamma * (self.voltage - self.v_thres), deriv=True)
            ltd = self.theta * self.activation(self.theta * (self.v_rest - self.voltage), deriv=True)
            self.synapses += np.outer(ltp - ltd, self.inputs) / self.tau_s * dt
        
        self.voltage += (self.v_rest - self.voltage + self.R * self.current) / self.tau_v * dt
        self.outputs = self.activation(self.gamma * (self.voltage - self.v_thres))
        
    def __repr__(self):
        if self.compiled:
            return f'LIFrateBundle(size={self.size}, inputs={self.inputs.shape})'
        else:
            return f'LIFrateBundle(size={self.size}, inputs={self.inputs})'


class LIF:
    
    def __init__(self,
        size,           # number of neurons
        v_rest=-70,     # membrane resting potential
        v_thres=-55,    # membrane threshold potential
        v_reset=-65,    # membrane reset potential
        R_v=100,        # membrane resistance
        R_d=50,         # dendritic resistance
        R_r=10,         # refractory current resistance
        tau_v=0.1,      # membrane potential time constant
        tau_d=0.02,     # dedritic current time constant
        tau_r=0.01,     # refractory current time constant
        tau_s=1         # synapstic plasticity time constant
    ):  
        self.size = size
        self.ntype = 'LIF'
        
        # parameters
        self.v_rest = v_rest
        self.v_thres = v_thres
        self.v_reset = v_reset
        self.R_v = R_v
        self.R_d = R_d
        self.R_r = R_r
        self.tau_v = tau_v
        self.tau_d = tau_d
        self.tau_r = tau_r
        self.tau_s = tau_s
        self.theta = 0.0
        
        # variables
        self.refractory = np.zeros(shape=self.size, dtype=np.float32)
        self.current = np.zeros(shape=self.size, dtype=np.float32)
        self.voltage = self.v_rest * np.ones(shape=self.size, dtype=np.float32)
        self.outputs = np.zeros(shape=self.size, dtype=np.float32)
        
        self.inputs = None
        self.synapses = None
        self.dendrites = None
        self.stim = None
        
        self.compiled = False
        self.train = False
        
    def step(self, dt):
        '''Implements the LIF nonlinearity.'''
        
        dd = (-self.dendrites + self.R_d * (self.synapses * self.inputs)) / self.tau_d * dt
        dr = (-self.refractory - self.R_r * self.outputs) / self.tau_r * dt
        self.current = np.sum(self.dendrites, axis=1) + self.refractory
        if self.stim is not None:
            self.current += self.stim
        dv = (self.v_rest - self.voltage + self.R_v * self.current) / self.tau_v * dt
        
        if self.train:
            ltp = np.abs(self.dendrites.T * self.outputs).T
            ltd = np.outer(self.refractory - self.theta, self.inputs)
            self.synapses += (ltp + ltd) / self.tau_s * dt
                              
        self.dendrites += dd
        self.refractory += dr
        self.voltage = np.where(self.outputs, self.v_reset, self.voltage + dv)
        self.outputs = np.where(self.voltage > self.v_thres, 1.0, 0.0)
        
    def __repr__(self):
        if self.compiled:
            return f'LIFBundle(size={self.size}, inputs={self.inputs.shape})'
        else:
            return f'LIFBundle(size={self.size}, inputs={self.inputs})'

        
class ExpIF:
    
    def __init__(self,
        size,           # number of neurons
        v_rest=-70,     # membrane resting potential
        v_thres=-55,    # membrane threshold potential
        v_reset=-65,    # membrane reset potential
        v_spike=30,
        R_d=10,         # membrane resistance
        R_r=20,         # membrane resistance
        R_v=50,         # membrane resistance
        D=5,           # membrane exponential delta
        tau_v=0.1,      # membrane potential time constant
        tau_d=0.02,     # dedritic current time constant
        tau_r=0.01,     # refractory current time constant
        tau_s=1         # synapstic plasticity time constant
    ):  
        self.size = size
        self.ntype = 'ExpIF'
        
        # parameters for membrane potential v
        self.v_rest = v_rest
        self.v_thres = v_thres
        self.v_reset = v_reset
        self.v_spike = v_spike
        self.R_v = R_v
        self.R_d = R_d
        self.R_r = R_r
        self.D = D
        self.tau_v = tau_v
        self.tau_d = tau_d
        self.tau_r = tau_r
        self.tau_s = tau_s
        self.theta = 0.0
        
        # variables
        self.refractory = np.zeros(shape=self.size, dtype=np.float32)
        self.current = np.zeros(shape=self.size, dtype=np.float32)
        self.voltage = self.v_rest * np.ones(shape=self.size, dtype=np.float32)
        self.outputs = np.zeros(shape=self.size, dtype=np.float32)
        
        self.inputs = None
        self.synapses = None
        self.dendrites = None
        self.stim = None
        
        self.compiled = False
        self.train = False
        
    def step(self, dt):
        '''Implements the exponential integrate and fire nonlinearity.'''
        
        dd = (-self.dendrites + self.R_d * (self.synapses * self.inputs)) / self.tau_d * dt
        dr = (-self.refractory - self.R_r * self.outputs) / self.tau_r * dt
        self.current = np.sum(self.dendrites, axis=1) + self.refractory
        if self.stim is not None:
            self.current += self.stim
        dv = np.nan_to_num(
            (self.v_rest - self.voltage + self.D * np.exp((self.voltage - self.v_thres) / self.D) + self.R_v * self.current) / self.tau_v * dt,
            nan=self.v_spike - self.v_thres,
            posinf=self.v_spike - self.v_thres
        )
        if self.train:
            ltp = np.abs(self.dendrites.T * self.outputs).T
            ltd = np.outer(self.refractory - self.theta, self.inputs)
            self.synapses += (ltp + ltd) / self.tau_s * dt
        
        self.dendrites += dd
        self.refractory += dr
        self.voltage = np.where(self.outputs, self.v_reset, self.voltage + dv)
        self.outputs = np.where(self.voltage > self.v_spike, 1.0, 0.0)
        
        
    def __repr__(self):
        if self.compiled:
            return f'ExpIFBundle(size={self.size}, inputs={self.inputs.shape})'
        else:
            return f'ExpIFBundle(size={self.size}, inputs={self.inputs})'


class adEx:
    
    def __init__(self,
        size,           # number of neurons
        v_rest=-70,     # membrane resting potential
        v_thres=-55,    # membrane threshold potential
        v_reset=-65,    # membrane reset potential
        v_spike=30,
        R_d=10,         # membrane resistance
        R_r=20,         # membrane resistance
        R_v=50,         # membrane resistance
        D=5,            # membrane exponential delta
        tau_v=0.1,      # membrane potential time constant
        tau_w=0.1,      # adaptation time constant
        tau_d=0.02,     # dedritic current time constant
        tau_r=0.01,     # refractory current time constant
        tau_s=1         # synapstic plasticity time constant
    ):  
        self.size = size
        self.ntype = 'ExpIF'
        
        # neuron parameters
        self.v_rest = v_rest
        self.v_thres = v_thres
        self.v_reset = v_reset
        self.v_spike = v_spike
        self.R_v = R_v
        self.R_d = R_d
        self.R_r = R_r
        self.D = D
        self.tau_v = tau_v
        self.tau_w = tau_w
        self.tau_d = tau_d
        self.tau_r = tau_r
        self.tau_s = tau_s
        self.theta = 0.0
        self.a = 0.01
        self.b = 1
        
        # variables
        self.refractory = np.zeros(shape=self.size, dtype=np.float32)
        self.current = np.zeros(shape=self.size, dtype=np.float32)
        self.w = np.zeros(shape=self.size, dtype=np.float32)
        self.voltage = self.v_rest * np.ones(shape=self.size, dtype=np.float32)
        self.outputs = np.zeros(shape=self.size, dtype=np.float32)
        
        self.inputs = None
        self.synapses = None
        self.dendrites = None
        self.stim = None
        
        self.compiled = False
        self.train = False
        
    def step(self, dt):
        '''Implements the adEx (adaptive exponential) nonlinearity.'''
        
        dd = (-self.dendrites + self.R_d * (self.synapses * self.inputs)) / self.tau_d * dt
        dr = (-self.refractory - self.R_r * self.outputs) / self.tau_r * dt
        self.current = np.sum(self.dendrites, axis=1) + self.refractory
        if self.stim is not None:
            self.current += self.stim
        dw = (self.a * (self.voltage - self.v_rest) - self.w) / self.tau_w * dt
        dv = np.nan_to_num(
            (self.v_rest - self.voltage + self.D * np.exp((self.voltage - self.v_thres) / self.D) + self.R_v * (self.current - self.w)) / self.tau_v * dt,
            nan=self.v_spike - self.v_thres,
            posinf=self.v_spike - self.v_thres
        )
        if self.train:
            ltp = np.abs(self.dendrites.T * self.outputs).T
            ltd = np.outer(self.refractory - self.theta, self.inputs)
            self.synapses += (ltp + ltd) / self.tau_s * dt
        
        self.dendrites += dd
        self.refractory += dr
        self.w += np.where(self.outputs, self.b, 0.0) + dw
        self.voltage = np.where(self.outputs, self.v_reset, self.voltage + dv)
        self.outputs = np.where(self.voltage > self.v_spike, 1.0, 0.0)
        
    def __repr__(self):
        if self.compiled:
            return f'adExBundle(size={self.size}, inputs={self.inputs.shape})'
        else:
            return f'adExBundle(size={self.size}, inputs={self.inputs})'
    
    
class Sensory:
    
    def __init__(self, size):
        self.size = size
        self.ntype = 'Sensory'
        self.outputs = np.zeros(self.size, dtype=np.float32)
        
    def __repr__(self):
        return f'SensoryBundle(size={self.size})'
