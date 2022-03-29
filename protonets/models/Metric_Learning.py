import torch
from torch import nn

from qtensor_ai import HybridModule, DefaultOptimizer
from qtensor_ai import ParallelComposer

'''Circuit for learnable quantum embeddings. Lloyd et. al., arXiv:2001.03622'''
class MetricLearningComposer(ParallelComposer):

    def __init__(self, n_qubits, n_layers):
        self.n_layers = n_layers
        super().__init__(n_qubits)
    
    def zz_layer(self, zz_params):
        for i in range(self.n_qubits//2):
            control_qubit = self.qubits[2*i]
            target_qubit = self.qubits[2*i+1]
            self.apply_gate(self.operators.ZZ, control_qubit, target_qubit, alpha=zz_params[:,control_qubit])
        for i in range((self.n_qubits+1)//2-1):
            control_qubit = self.qubits[2*i+1]
            target_qubit = self.qubits[2*i+2]
            self.apply_gate(self.operators.ZZ, control_qubit, target_qubit, alpha=zz_params[:,control_qubit])
    
    # A single layer of rotation gates depending on trainable parameters
    def variational_layer(self, gate, layer_params):
        for i in range(self.n_qubits):
            qubit = self.qubits[i]
            self.apply_gate(gate, qubit, alpha=layer_params[:, i])
    
    # Building circuit that needs to be measured#
    def forward_circuit(self, inputs, zz_params, y_params):

        """
        Parameters
        ----------
        inputs: torch.Tensor
                Has dimension (n_batch, n_qubits). It contains data to be encoded.

        zz_params: torch.Tensor
                Has dimension (n_batch, n_qubits-1, n_layers). It stores ZZ angles.

        y_params: torch.Tensor
                Has dimension (n_batch, n_qubits, n_layers). It stores RY angles.
        """

        for layer in range(self.n_layers):
            self.variational_layer(self.operators.XPhase, inputs)
            layer_zz_params = zz_params[:, :, layer]
            self.zz_layer(layer_zz_params)
            layer_y_params = y_params[:, :, layer]
            self.variational_layer(self.operators.YPhase, layer_y_params)
        self.variational_layer(self.operators.XPhase, inputs)



    '''This function MUST be written for all custom circuit Composers.
    Building circuit whose first amplitude is the inner product'''
    def updated_full_circuit(self, **parameters):
        inputs1, inputs2, zz_params, y_params = parameters['inputs1'], parameters['inputs2'], parameters['zz_params'], parameters['y_params']
        self.builder.reset()
        self.device = inputs1.device
        self.forward_circuit(inputs1, zz_params, y_params)
        first_part = self.builder.circuit
        self.builder.reset()
        self.forward_circuit(inputs2, zz_params, y_params)
        self.builder.inverse()
        second_part = self.builder.circuit
        self.builder.reset()
        '''There is no cost operator unlike in the previous circuit
        because we just want the overlap between embeddings'''
        return first_part + second_part

    '''This function MUST be written for all custom circuit Composers.
    Returns the name of the circuit composer'''
    def name(self):
        return 'MetricLearning'


'''Module for evaluating the inner product between quantum embeddings
learned by circuits proposed by Lloyd et. al. arXiv:2001.03622.'''
class MetricLearning(HybridModule):
    
    def __init__(self, n_qubits, n_layers, optimizer=DefaultOptimizer(), entanglement=2):
                
        '''Initializing module parameters'''
        circuit_name = 'n_{}_l_{}'.format(n_qubits, n_layers)
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        '''Define the circuit composer and initialize the hybrid module'''
        composer = MetricLearningComposer(n_qubits, n_layers)
        super(MetricLearning, self).__init__(circuit_name=circuit_name, composer=composer, optimizer=optimizer)

        '''Initializing quantum circuit parameters (Not input to be embedded)'''
        '''self.weight are model weights. Weights must be defined after super().__init__()'''
        self.zz_params = nn.Parameter(torch.rand(1, n_qubits-1, n_layers, dtype=torch.float32)*entanglement)
        self.y_params = nn.Parameter(torch.rand(1, n_qubits, n_layers, dtype=torch.float32)*entanglement)

    def forward(self, inputs1, inputs2):

        """
        Parameters
        ----------
        inputs1: torch.tensor
                Classical rotation angles for the circuit encoding Rx gates for the first embedding.

        inputs2: torch.tensor
                Classical rotation angles for the circuit encoding Rx gates for the second embedding.
        """

        n_batch = inputs1.shape[0]
        zz_params = self.zz_params.expand(n_batch, -1, -1) # (n_batch, n_qubits-1, n_layers)
        y_params = self.y_params.expand(n_batch, -1, -1) # (n_batch, n_qubits, n_layers)
        '''The actual simulation must be run by calling the parent_forward method of the parent class. 
        The parameters should be the same parameters as those accepted by the circuit composer'''
        out = self.parent_forward(inputs1=inputs1, inputs2=inputs2, zz_params=zz_params, y_params=y_params)
        return out