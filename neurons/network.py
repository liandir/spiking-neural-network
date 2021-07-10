import numpy as np
from collections import OrderedDict



class History:
    
    def __init__(self):
        self._current = []
        self._voltage = []
        self._outputs = []
    
    @property
    def current(self):
        return np.stack(self._current)
    
    @property
    def voltage(self):
        return np.stack(self._voltage)
    
    @property
    def outputs(self):
        return np.stack(self._outputs)
    
    def reset(self):
        self._current = []
        self._voltage = []
        self._outputs = []
    
    
class Network:
    
    def __init__(self):
        self.bundles = OrderedDict()
        self.connections = OrderedDict()
        
        self.compiled = False
        self._train = False
        self._history = True
        
    def add(self, ntype, size, name, **kwargs):
        if name not in self.bundles:
            self.bundles[name] = ntype(size, **kwargs)
        else:
            print(f"bundle with name '{name}' already exists.")
            
    def remove(self, name):
        if self.compiled:
            print("model has already been compiled.")
        elif name in self.bundles:
            self.bundles.pop(name)
        else:
            print(f"bundle with name '{name}' does not exist.")
        
    def connect(self, bundle_in, bundle_out):
        if bundle_in in self.bundles and bundle_out in self.bundles:
            if bundle_out in self.connections:
                self.connections[bundle_out] |= {bundle_in}
            else:
                self.connections[bundle_out] = {bundle_in}
        else:
            if bundle_in not in self.bundles and bundle_out not in self.bundles:
                print(f"'{bundle_in}' and '{bundle_out}' do not exist.")
            elif bundle_in not in self.bundles:
                print(f"'{bundle_in}' does not exist.")
            elif bundle_out not in self.bundles:
                print(f"'{bundle_out}' does not exist.")
            
    def disconnect(self, bundle_in, bundle_out):
        if self.compiled:
            print("model has already been compiled.")
        elif bundle_in in self.bundles and bundle_out in self.bundles:
            self.connections[bundle_out] -= {bundle_in}
        else:
            if bundle_in not in self.bundles and bundle_out not in self.bundles:
                print(f"'{bundle_in}' and '{bundle_out}' do not exist.")
            elif bundle_in not in self.bundles:
                print(f"'{bundle_in}' does not exist.")
            elif bundle_out not in self.bundles:
                print(f"'{bundle_out}' does not exist.")
    
    @property
    def inputs(self):
        if 'input' in self.bundles:
            return self.bundles['input'].copy()
        else:
            print("'input' bundle not found.")
            return None
        
    @inputs.setter
    def inputs(self, inputs):
        if 'input' in self.bundles:
            self.bundles['input'].outputs = inputs
        else:
            print("'input' bundle not found.")
    
    @property
    def outputs(self):
        if 'output' in self.bundles:
            return self.bundles['output'].copy()
        else:
            print("'output' bundle not found.")
            return None
    
    @property
    def train(self):
        return self._train
    
    @train.setter
    def train(self, active):
        for bundle in self.bundles:
            self.bundles[bundle].train = active
        self._train = active
            
    @property
    def record_history(self):
        return self._history
    
    @record_history.setter
    def record_history(self, active):
        self._history = active
      
    def reset_history(self):
        for name in self.bundles:
            self.history[name].reset()
    
    def compile(self):
        '''initialize synapses and inputs. also check for coherence of model architecture.'''
        
        if not self.compiled:
            print('initializing neurons...', end=' ')
            for head in self.connections:
                n_inputs = np.sum([self.bundles[tail].size for tail in sorted(self.connections[head])])
                self.bundles[head].inputs = np.zeros(shape=n_inputs, dtype=np.float32)
                self.bundles[head].synapses = np.random.normal(0, 1 / np.sqrt(n_inputs), size=(self.bundles[head].size, n_inputs)).astype(np.float32)
                self.bundles[head].dendrites = np.zeros(shape=(self.bundles[head].size, n_inputs), dtype=np.float32)
                self.bundles[head].compiled = True
            print('done.')
            
            print('checking if graph is connected...', end=' ')
            self.compiled = True
            for name in self.bundles:
                if self.bundles[name].ntype != 'Sensory':
                    if self.bundles[name].inputs is None:
                        self.compiled = False
            print('done.')
            
            print('initializing history...', end=' ')
            self.history = {}
            for name in self.bundles:
                self.history[name] = History()
            print('done.')
            
            if self.compiled:
                print('model successfully compiled.\n')
                # print(self)
            else:
                print('model was not compiled. some bundles are not connected.')
        else:
            print('model has already been compiled.')
    
    def step(self, dt=0.001):
        '''single time step. requires model to be compiled.'''
        
        # update inputs of bundles
        for head in self.connections:
            inputs = np.concatenate([self.bundles[tail].outputs for tail in sorted(self.connections[head])])
            self.bundles[head].inputs = inputs
        
        # update neuron states
        for name in self.bundles:
            if self.bundles[name].ntype != 'Sensory':
                self.bundles[name].step(dt=dt)
                if self.record_history:
                    self.history[name]._current += [self.bundles[name].current.copy()]
                    self.history[name]._voltage += [self.bundles[name].voltage.copy()]
                    self.history[name]._outputs += [self.bundles[name].outputs.copy()]
    
    def __repr__(self):
        out = f"Network Summary\n"
        out += '='*64 + '\n'
        out += 'Bundles:\n'
        out += '_'*64 + '\n'
        out += 'name\t\tntype\t\tunits\n'
        out += '-'*64 + '\n'
        for name in self.bundles:
            out += f'{name}\t\t{self.bundles[name].ntype}\t\t{self.bundles[name].size}\n'
        if len(self.connections) > 0:
            out += '='*64 + '\n'
            out += 'Connections:\n'
            out += '_'*64 + '\n'
            for name in self.connections:
                for inc in self.connections[name]:
                    out += f'{inc} -> {name}\n'
        out += '='*64 + '\n'
        out += f'model compiled: {self.compiled}'
        return out
    
    
