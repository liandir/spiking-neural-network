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
        activation=f6,
        gamma=1,
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
        
        self.inputs = None
        self.synapses = None
        self.outputs = None
        
        self.compiled = False
        self.train = False
        self.stim = None
    
    @property
    def outputs(self):
        return 
        
    def step(self, dt=0.001):
        self.current = self.synapses @ self.inputs # + self.biases
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
        self.alpha = 1
        
        # variables
        self.refractory = np.zeros(shape=(self.size), dtype=np.float32)
        self.current = np.zeros(shape=(self.size), dtype=np.float32)
        self.voltage = self.v_rest * np.ones(shape=(self.size), dtype=np.float32)
        
        self.inputs = None
        self.synapses = None
        self.dendrites = None
        self.stim = None
        
        self.compiled = False
        self.train = False
    
    @property
    def outputs(self):
        '''Get output/spikes.'''
        return (self.voltage > self.v_thres).astype(np.float32)
        
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
        self.voltage = np.where(self.outputs, self.v_reset, self.voltage)
        self.voltage += dv
        
    def __repr__(self):
        if self.compiled:
            return f'LIFBundle(size={self.size}, inputs={self.inputs.shape})'
        else:
            return f'LIFBundle(size={self.size}, inputs={self.inputs})'
    
    
class adEx:
    
    def __init__(self,
        size,           # number of neurons
        v_rest=-70,     # membrane resting potential
        v_thres=-55,    # membrane threshold potential
        v_reset=-60,    # membrane reset potential
        R=1,          # membrane resistance
        D=5,            # membrane exponential delta
        tau=0.1,        # membrane potential time constant
        tau_c=0.5,      # current time constant
        a=1.1,          # adaptation linearity coefficient
        b=1,            # adaptation reset
        tau_w=0.1       # adaptation time constant
    ):  
        self.size = size
        self.ntype = 'adEx'
        
        # parameters for membrane potential v
        self.v_rest = v_rest
        self.v_thres = v_thres
        self.v_reset = v_reset
        self.R = R
        self.D = D
        self.tau = tau
        self.tau_c = tau_c
        
        # parameters for adaptation w
        self.a = a
        self.b = b
        self.tau_w = tau_w
        
        # variables
        self.current = np.zeros(shape=(self.size), dtype=np.float32)
        self.voltage = self.v_rest * np.ones(shape=(self.size), dtype=np.float32)
        self.w = np.zeros(shape=(self.size), dtype=np.float32)
        self.inputs = None
        self.synapses = None
        
        self.compiled = False
        self.train = False
    
    @property
    def outputs(self):
        '''Get output/spikes.'''
        return (self.voltage > 30).astype(np.float32)
        
    def step(self, dt):
        '''Implements the adEx (adaptive exponential) nonlinearity.'''
        self.current += (self.synapses @ self.inputs - self.current) / self.tau_c * dt
        
        spike = self.outputs
        self.w += np.where(spike, self.b, 0)
        self.voltage = np.where(spike, self.v_reset, self.voltage)
        
        self.w += (self.a * (self.voltage - self.v_rest) - self.w) / self.tau_w * dt
        self.voltage += (-(self.voltage - self.v_rest) + self.D * np.exp((self.voltage - self.v_thres) / self.D) + (self.current - self.w) / self.R) / self.tau * dt
        
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
