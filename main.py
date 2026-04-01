from General_HHL import HHL
from qiskit import transpile
from qiskit_aer import AerSimulator
import numpy as np
import argparse

if __name__ == '__main__' :
    cnum = 4
    # cnum : eigenvalue 표현력 결정
    # ex. cnum=2; 0, 0.25, 0.5, 0.75로 eigenvalue를 이진소수화 할 수  있음
    # cnum이 작으면 eigenvalue가 세밀할때 제대로 표현 못하고 근사해버림
    
    t0 = np.pi/(2**(cnum-1))
    # (eigenvalue_max * t0 / 2 * pi) <= 1
    
    matA = np.array([[5, 2, 0, 0],
              [2, 5, 0, 0],
              [0, 0, 3, 1],
              [0, 0, 1, 3]])
    bvec = [0.5, 0.3, 0.1, 0.5]
    # n이 2의 거듭제곱이어야 함. 아니라면 패딩 추가 필요.
    
    bnum = 2
    # bnum : vec b의 차원에 따라 결정. log(dim.b)

    shots=8192
    C=1.0
    
    
    
    hhl = HHL(t0=t0, matA=matA, bvec=bvec, bnum=bnum, cnum=cnum, C=C)
    hhl.HHL_Algorithm()
    
    simulator = AerSimulator()
    compiled_circuit = transpile(hhl.qc, simulator)
    result = simulator.run(compiled_circuit, shots=shots).result()
    counts = result.get_counts()
    print(f"counts:{counts}")
    


    pure_counts = {}
    for bitstring, count in counts.items():
        # 공백으로 구분된 비트 문자열 분리
        res_bits, ancilla_bit = bitstring.split()
        
        # 보조 큐비트가 1인 결과만!
        if ancilla_bit == '1':
            pure_counts[res_bits] = count
    print(f"pure counts: {pure_counts}")
    

    total_valid_shots = sum(pure_counts.values())
    quantum_solution = np.zeros(2**bnum)
    
    for bitstring, count in pure_counts.items():
        idx = int(bitstring, 2)
        # 각각의 측정횟수를 전체 측정횟수(ancilla가 1인 valid shot만)로 나눠서 제곱근씌움
        # 확률 진폭을 이용해 측정횟수로 해를 찾아내므로 부호를 알기 어려움
        # 음수 나타내려면 측정 다르게 해야함 (파울리 기댓값 계산, Implementation of the HHL)
        quantum_solution[idx] = np.sqrt(count / total_valid_shots)

    print("\n[HHL 알고리즘 결과]")
    print(f"n : {len(matA)}")
    print(f"# of b register: {bnum}")
    print(f"# of C register : {cnum}\n")
    
    
    print(f"ancilla가 |1>일 확률 : {total_valid_shots/shots*100:.2f}%")
    print(f"HHL 계산 : {quantum_solution}")

    # 클래식 정답과 비교
    # x = A^-1 * b
    inv_A = np.linalg.inv(matA)
    classical_solution = inv_A @ bvec
    classical_solution_norm = classical_solution / np.linalg.norm(classical_solution)

    print(f"클래식 계산 : {classical_solution_norm}\n")
    
    
    
    # --print : draw circuit
    parser = argparse.ArgumentParser()
    parser.add_argument('--print', action='store_true')
    args = parser.parse_args()
    if args.print:
        print(hhl.get_circuit().draw(output='text'))