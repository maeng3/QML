from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import PauliEvolutionGate, QFT, UnitaryGate, RYGate
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit.synthesis import LieTrotter
import numpy as np
from scipy.linalg import expm


class HHL:
    def __init__(self, t0, matA, bvec, bnum, cnum, C=1.0):
        self.t0=t0
        
        self.matA = matA
        self.bvec = bvec
        self.bnum = bnum
        self.cnum = cnum
        self.C=C
        
        self.areg = QuantumRegister(1, "Ancilla")
        self.creg = QuantumRegister(self.cnum, "RegC")
        self.breg = QuantumRegister(self.bnum, "RegB")
        self.qc = QuantumCircuit(self.areg, self.creg, self.breg)
        
        self.sparse_pauli_A = None


    def init(self):    
        norm = np.linalg.norm(self.bvec)
        normalized_vector = self.bvec / norm
        
        # encoding을 통해 |b> 만들기
        # initialize함수를 이용해서 자동으로 진폭 encoding
        self.qc.initialize(normalized_vector, self.breg[:])
        
        self.qc.barrier()
        
        
        
        
    def sparsePauli(self):
        # 행렬 A를 pauli 연산자로 분해
        self.sparse_pauli_A = SparsePauliOp.from_operator(self.matA)
        
        
        

    def QPE(self):
        for i in range(self.cnum):
            self.qc.h(self.creg[i])
            
        #self.sparsePauli()

        for i in range(self.cnum):
            t = self.t0 * (2**i)
            
            # e^iAt 계산 후 unitary gate로 변환하여 U_A gate 만들기
            U_matrix = expm(1j * self.matA * t) 
            U_gate = UnitaryGate(U_matrix)
            
            C_U_gate = U_gate.control(1)
            
            # 제어: creg[i], 타겟: breg
            self.qc.append(C_U_gate, [self.creg[i]] + self.breg[:])

        iqft=QFT(num_qubits=self.cnum, inverse=True).to_gate()
        
        self.qc.append(iqft, self.creg[:])
        
        self.qc.barrier()
        
        
        
        
    def AQE(self):
    # |1> 이 나올 확률 : (C / eigen_value)^2
    # eigen_value_min <= C
    
        # theta = 2*np.arcsin(self.C/lam)
        # for k in reversed(range(self.cnum)):
        #     ry_gate=RYGate((2**k)*np.pi/2**self.cnum)
        #     c_ry_gate=ry_gate.control(1)
        #     self.qc.append(c_ry_gate, [self.creg[k]] + [self.areg[0]])
        
        
        for k in range(1, 2**self.cnum):
        # regC가 가질 수 있는 값에 따른 고유값 계산
        # k : 큐비트에 저장된 고유값에 대한 정보를 가진 값
        # k = (2^n * eigen_value * t) / (2 * pi)
        # k값을 역계산하여 eigen_value값을 알아내는 것이 목적
        
            phase = k / (2 ** self.cnum)
            eigen_value = (2 * np.pi * phase) / self.t0
            if self.C <= eigen_value:
                # 회전 각도 계산: 2 * arcsin(C/eigen_value)
                theta = 2 * np.arcsin(self.C / eigen_value)
                
                # k를 cnum자리수의 이진수로 변환
                bin_k = format(k, f'0{self.cnum}b')
                
                # ctrl_state를 통해 RegC가 bin_k일 때만 Ancilla를 회전시킴
                # 비용 너무 큼..
                cont_ry = RYGate(theta).control(self.cnum, ctrl_state=bin_k)
                
                self.qc.append(cont_ry, self.creg[:] + [self.areg[0]])
                
            else :
                print(f"C is bigger than eigen_value (eigen_value : {eigen_value}, k : {k}, C : {self.C})")

        self.qc.barrier()
        
        
        


    def inverse_QPE(self):
        qft = QFT(num_qubits=self.cnum, inverse=False).to_gate()
        self.qc.append(qft, self.creg[:])

        for i in reversed(range(self.cnum)):
            t = self.t0 * (2**i)
            
            U_matrix = expm(1j * self.matA * -t) 

            U_gate = UnitaryGate(U_matrix)

            C_U_gate = U_gate.control(1)
            
            self.qc.append(C_U_gate, [self.creg[i]] + self.breg[:])


        for i in range(self.cnum):
            self.qc.h(self.creg[i])

        self.qc.barrier()

        
        
        
        
    def measurement(self):
        self.c_ancilla = ClassicalRegister(1, "c_ancilla")
        self.c_result = ClassicalRegister(self.bnum, "c_result")
        self.qc.add_register(self.c_ancilla, self.c_result)

        self.qc.measure(self.breg, self.c_result)
        self.qc.measure(self.areg[0], self.c_ancilla[0])

        


    def HHL_Algorithm(self):
        self.init()
        self.QPE()
        self.AQE()
        self.inverse_QPE()
        self.measurement()
        
        
    
    def get_circuit(self):
        return self.qc
        