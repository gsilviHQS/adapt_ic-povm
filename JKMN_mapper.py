from qiskit_nature.mappers.second_quantization.fermionic_mapper import FermionicMapper
from qiskit_nature.mappers.second_quantization.qubit_mapper import QubitMapper

from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info.operators import Pauli

import numpy as np


"""The JKMN Mapper. Originally developed by Matteo A.C. Rossi. Adapted by Giorgio Silvi """

class JKMNMapper(FermionicMapper): 
    def __init__(self):
        """The JKMN fermion-to-qubit mapping."""
        super().__init__(allows_two_qubit_reduction=False)


    def _nodeindex(self,p,l):
        """Returns index of a node of depth l on the path p on the ternary tree of height h.
        Ternary tree is just a string of natural numbers starting from 0 ordered in a tree where
        each node has three children. For image and formula see arXiv:1910.10746, Eq. (3).
        Args:
            p (list): list of strings of integers 0,1 or 2 of length h
            l (int): depth of tree to calculate the index on, l <= h
        Returns:
            int: index of a node corresponding to a qubit number
        """
        h = len(p)
        assert l <= h, "should be l <= h, where h = len(p)"
        for idx in p:
            assert (idx == '0') or (idx == '1') or (idx == '2'), "indices should be 0,1 or 2"
        
        prefactor = (3**l - 1)/2 
        sumfactors = [3**(l-1-j)*int(p[j]) for j in range(l)]
        return int(prefactor + sum(sumfactors))

    def _hfromnum_qubits(self,num_qubits):
        """Calculates the needed height of the tree from number of qubits.
        Args:
            num_qubits (int): number of qubits
        
        Returns:
            height (int): required height of the ternary tree.
        """
        height = np.ceil(np.log(2*num_qubits+1)/np.log(3)) # base 3 logarithm
        return int(height) # integererize the output
        
    def _xyzstring(self,h):
        """Generate a list of repeating 'X','Y','Z' pattern to fill the ternary tree.
        Args:
            h (int): Height of the ternary tree
        Returns:
            list: List of strings ['null', 'X','Y','Z','X','Y,'Z','X',...]
        """
        num_idxs = int((3**h - 1) // 2) # number of indices (qubits) to add
        num_idxs_triplets = int(num_idxs / 3) # number of triplets 
        output = ['null'] # add an index for the 0th qubit
        for _ in range(num_idxs_triplets):
            output += ['X','Y','Z']
        return output 

    def _paulipaths_full(self,h):
        """Generate all Pauli paths from a tree of height h.
        
        
        Args:
            h (int): height of the ternary tree
        Returns:
            list: List of Pauli strings
        """
        xyzs = self._xyzstring(h+1) # generate the xyz string for tree height + 1
        num_qubits = int((3**(h) - 1) // 2) # get number of qubits from the height
        # generate all paths by looping over the number of paths in ternary base 
        paths = []
        for i in range(3**(h)):
            paths += [np.base_repr(i,base=3).rjust(h,'0')]
        
        # generate the Pauli strings from paths by substituting I with appropriate Pauli gate
        paulistrings = []
        for path in paths:
            pstring = ['I']*num_qubits # initialize a string with I's
            # for each depth, get the index at which the IIII.. path should be substituted and 
            # idx2 at which the substitution Pauli is located
            for depth in range(h):
                idx = self._nodeindex(path,depth)
                idx2 = self._nodeindex(path,depth+1)
                pstring[idx] = xyzs[idx2]
            # add the resulting string to the list, converted from string to list
            paulistrings += ["".join(pstring)]
        return paulistrings

    def _paulipaths(self,num_qubits: int):
        # Step 1: Introduce a complete ternary tree with height h
        h = self.hfromnum_qubits(num_qubits) # get tree max height
        
        # generate two full trees, the larger one should 
        # accommodate all qubits
        pphm1 = self._paulipaths_full(h-1)
        pph = self._paulipaths_full(h)
        
        # get number of qubits in the smaller tree 
        # and the number of extra qubits
        n_qubits_hm1 = len(pphm1[0])
        n_extraqubits = num_qubits - n_qubits_hm1
        paulistrings = []
        # loop over each gate in the larger tree for extra qubits 
        # to add the extra paths, also truncate the added paths 
        # up to the real number of qubits,
        # then add the path, converted to a Pauli object
        for i in range(n_extraqubits * 3):
            path = pph[i][:len(pphm1[0]) + n_extraqubits]
            # print(path)
            paulistrings += [Pauli(path)]
            
            # paulistrings += path
        # loop over each gate in the smaller tree, skipping over 
        # the first gates of extra qubits, also add extra 'I' gates,
        # then add the path, converted to a Pauli object
        for i in range(n_extraqubits, len(pphm1)):
            path = ''.join([pphm1[i]] + ['I'] * n_extraqubits)
            paulistrings += [Pauli(path)]
            # paulistrings += path
        return paulistrings

    def map(self, second_q_op: FermionicOp) -> PauliSumOp:
        nmodes = second_q_op.register_length
        # reshape the pauli strings from a list into a paired list, skipping the very last one
        allpaths = self._paulipaths(nmodes)
        paths = [(allpaths[2*k],allpaths[2*k + 1]) for k in range(int((len(allpaths)-1)//2))] 
        return QubitMapper.mode_based_mapping(second_q_op, paths)