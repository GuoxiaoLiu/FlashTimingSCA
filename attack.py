import pandas as pd
import numpy as np
import time
from numba import njit, prange

# ==========================================
# 0. AES Constants & Helpers
# ==========================================
SBOX = np.array([
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
], dtype=np.uint8)

@njit(fastmath=True)
def gmul2(x): return ((x << 1) ^ 0x1B) & 0xFF if (x & 0x80) else (x << 1) & 0xFF
@njit(fastmath=True)
def gmul3(x): return gmul2(x) ^ x

MUL2 = np.array([gmul2(x) for x in range(256)], dtype=np.uint8)
MUL3 = np.array([gmul3(x) for x in range(256)], dtype=np.uint8)
T_MUL2 = MUL2[SBOX]
T_MUL3 = MUL3[SBOX]
T_SBOX = SBOX

# ==========================================
# 1. Parallel Kernels
# ==========================================

@njit(parallel=True, fastmath=True)
def parallel_phase1_search(p0, p5, p10, p15, cycles, h5, h10, h15):
    n = len(cycles)
    results = np.zeros((256, 6), dtype=np.float64)
    
    for k0 in prange(256):
        h0 = k0 >> 4
        t0_arr = np.zeros(n, dtype=np.uint8)
        for i in range(n): t0_arr[i] = T_MUL2[p0[i] ^ k0]
        
        best_local_score = -1.0
        best_cfg = np.zeros(5, dtype=np.uint8) 
        
        for l5 in range(16):
            k5 = ((h0 ^ h5) << 4) | l5
            t01 = np.zeros(n, dtype=np.uint8)
            for i in range(n): t01[i] = t0_arr[i] ^ T_MUL3[p5[i] ^ k5]
            
            for l10 in range(16):
                k10 = ((h0 ^ h10) << 4) | l10
                t012 = np.zeros(n, dtype=np.uint8)
                for i in range(n): t012[i] = t01[i] ^ T_SBOX[p10[i] ^ k10]
                
                for l15 in range(16):
                    k15 = ((h0 ^ h15) << 4) | l15
                    cnts = np.zeros(16, dtype=np.int32)
                    sums = np.zeros(16, dtype=np.float64)
                    lhs_offset = k15 >> 4
                    total_sum = 0.0
                    
                    for i in range(n):
                        t3 = T_SBOX[p15[i] ^ k15]
                        pp0 = t012[i] ^ t3
                        lhs = (p15[i] >> 4) ^ lhs_offset
                        sk = (pp0 >> 4) ^ h0 ^ lhs 
                        cnts[sk] += 1
                        sums[sk] += cycles[i]
                        total_sum += cycles[i]
                    
                    for sk in range(16):
                        c = cnts[sk]
                        if c > 20 and c < n-20: 
                            m1 = sums[sk] / c
                            m2 = (total_sum - sums[sk]) / (n - c)
                            s = abs(m1 - m2)
                            if s > best_local_score:
                                best_local_score = s
                                best_cfg[0] = l5
                                best_cfg[1] = l10
                                best_cfg[2] = l15
                                best_cfg[3] = sk
                                best_cfg[4] = 0
        
        results[k0, 0] = best_local_score
        results[k0, 1] = k0
        results[k0, 2] = best_cfg[0]
        results[k0, 3] = best_cfg[1]
        results[k0, 4] = best_cfg[2]
        results[k0, 5] = best_cfg[3]
    return results

@njit(parallel=True, fastmath=True)
def parallel_batch_solver(cycles, p_cols0, p_cols1, p_cols2, p_cols3, 
                          cands0, cands1, cands2, cands3,
                          lhs_vals, kp_base):
    n = len(cycles)
    num_c0 = len(cands0)
    results = np.zeros((num_c0, 5), dtype=np.float64)
    
    for i0 in prange(num_c0):
        k0 = cands0[i0]
        kp_h = (k0 >> 4) ^ kp_base
        local_best_s = -1.0
        local_idxs = np.zeros(4, dtype=np.int32)
        local_idxs[0] = i0
        
        t0 = np.zeros(n, dtype=np.uint8)
        for i in range(n): t0[i] = T_MUL2[p_cols0[i] ^ k0]
        
        for i1 in range(len(cands1)):
            k1 = cands1[i1]
            t01 = np.zeros(n, dtype=np.uint8)
            for i in range(n): t01[i] = t0[i] ^ T_MUL3[p_cols1[i] ^ k1]
            
            for i2 in range(len(cands2)):
                k2 = cands2[i2]
                t012 = np.zeros(n, dtype=np.uint8)
                for i in range(n): t012[i] = t01[i] ^ T_SBOX[p_cols2[i] ^ k2]
                
                for i3 in range(len(cands3)):
                    k3 = cands3[i3]
                    cnt = 0
                    s_sum = 0.0
                    total_sum = 0.0
                    for i in range(n):
                        t3 = T_SBOX[p_cols3[i] ^ k3]
                        pp = t012[i] ^ t3
                        target = (pp >> 4) ^ kp_h
                        if lhs_vals[i] == target:
                            cnt += 1
                            s_sum += cycles[i]
                        total_sum += cycles[i]
                    
                    if cnt > 10 and cnt < n-10:
                        m1 = s_sum / cnt
                        m2 = (total_sum - s_sum) / (n - cnt)
                        s = abs(m1 - m2)
                        if s > local_best_s:
                            local_best_s = s
                            local_idxs[1] = i1
                            local_idxs[2] = i2
                            local_idxs[3] = i3
        
        results[i0, 0] = local_best_s
        results[i0, 1] = local_idxs[0]
        results[i0, 2] = local_idxs[1]
        results[i0, 3] = local_idxs[2]
        results[i0, 4] = local_idxs[3]

    return results

# ==========================================
# 2. Main Attack Class
# ==========================================

class AESCorrectAttack:
    def __init__(self, filename):
        self.filename = filename
        self.P = None
        self.cycles = None
        self.rel_high = np.zeros(16, dtype=int)
        self.found_keys = {}
        self.kp_high_truth = {} 
        self.sk_high_constraints = {} 
        self.start_time = 0

    def print_status(self, step_name, details):
        elapsed = time.time() - self.start_time
        print(f"[{elapsed:6.2f}s] [{step_name}] {details}")

    def load_data(self):
        print(f"[*] Loading data from {self.filename}...")
        try:
            df = pd.read_csv(self.filename)
            cols = [f'P{i}' for i in range(16)]
            self.P = df[cols].values.astype(np.uint8)
            self.cycles = df['Cycles'].values.astype(np.float64)
            print(f"    Total traces loaded: {len(self.cycles)}")
        except Exception:
            print("[!] CSV not found. Gen dummy.")
            self.P = np.random.randint(0, 256, (10000, 16), dtype=np.uint8)
            self.cycles = np.random.randn(10000)

    def diff_of_means(self, mask):
        c = np.sum(mask)
        if c < 20 or c > len(self.cycles)-20: return 0.0
        return abs(np.mean(self.cycles[mask]) - np.mean(self.cycles[~mask]))

    def step1_adjacency(self):
        print("\n" + "="*60 + "\nSTEP 1: Adjacency\n" + "="*60)
        diffs = np.zeros(15, dtype=int)
        p_high = self.P >> 4
        for i in range(15):
            d_arr = p_high[:, i] ^ p_high[:, i+1]
            best_s = -1; best_d = 0
            counts = np.bincount(d_arr, minlength=16)
            sums = np.bincount(d_arr, weights=self.cycles, minlength=16)
            total = np.sum(self.cycles)
            total_n = len(self.cycles)
            for d in range(16):
                if counts[d] > 50:
                    m1 = sums[d] / counts[d]
                    m2 = (total - sums[d]) / (total_n - counts[d])
                    s = abs(m1 - m2)
                    if s > best_s: best_s = s; best_d = d
            diffs[i] = best_d
            print(f"k[{i}]^k[{i+1}] | {best_d:X}     | {best_s:.2f}")
        
        curr = 0
        self.rel_high[0] = 0
        for i in range(15):
            curr ^= diffs[i]
            self.rel_high[i+1] = curr
        self.print_status("DONE", "Relative High Nibbles Map Built.")

    def step2_anchor(self):
        print("\n" + "="*60 + "\nSTEP 2: Anchor\n" + "="*60)
        self.print_status("START", "Parallel Search...")
        res_matrix = parallel_phase1_search(
            self.P[:,0], self.P[:,5], self.P[:,10], self.P[:,15], self.cycles,
            self.rel_high[5], self.rel_high[10], self.rel_high[15]
        )
        best_res = res_matrix[np.argmax(res_matrix[:, 0])]
        
        k0 = int(best_res[1])
        l5, l10, l15, sk13_h = int(best_res[2]), int(best_res[3]), int(best_res[4]), int(best_res[5])
        
        h0 = k0 >> 4
        self.found_keys[0] = k0
        self.found_keys[5] = ((h0 ^ self.rel_high[5]) << 4) | l5
        self.found_keys[10] = ((h0 ^ self.rel_high[10]) << 4) | l10
        self.found_keys[15] = ((h0 ^ self.rel_high[15]) << 4) | l15
        
        self.kp_high_truth[0] = h0 ^ sk13_h
        self.sk_high_constraints[13] = sk13_h 
        
        print(f">>> Found k0: 0x{k0:02X}, Score: {best_res[0]:.4f}")
        print(f">>> High4(S(k13)): 0x{sk13_h:X}")

    def _get_pp(self, col, row):
        idx = [[0, 5, 10, 15], [4, 9, 14, 3], [8, 13, 2, 7], [12, 1, 6, 11]][col]
        p = [self.P[:, i] for i in idx]
        k = [self.found_keys[i] for i in idx]
        v = [SBOX[p[i]^k[i]] for i in range(4)]
        if row == 0: return MUL2[v[0]]^MUL3[v[1]]^v[2]^v[3]
        if row == 1: return v[0]^MUL2[v[1]]^MUL3[v[2]]^v[3]
        if row == 2: return v[0]^v[1]^MUL2[v[2]]^MUL3[v[3]]
        if row == 3: return MUL3[v[0]]^v[1]^v[2]^MUL2[v[3]]

    def step3_correct_flow(self):
        print("\n" + "="*60 + "\nSTEP 3: Chain & Deterministic Batch\n" + "="*60)
        
        # --- 1. Chain Col 0 to Find Constraints ---
        self.print_status("CHAIN", "Analyzing Col 0 Chain...")
        pp0 = self._get_pp(0, 0)
        
        # k'[1]
        pp1 = self._get_pp(0, 1)
        lhs = (pp0 >> 4) ^ self.kp_high_truth[0]
        self.kp_high_truth[1] = self._solve_chain_nibble(lhs, pp1)
        k1_h = (self.found_keys[0] >> 4) ^ self.rel_high[1]
        self.sk_high_constraints[14] = k1_h ^ self.kp_high_truth[1]
        print(f"  [Info] S(k14)>>4 = 0x{self.sk_high_constraints[14]:X}")
        
        # k'[2]
        pp2 = self._get_pp(0, 2)
        lhs = (pp1 >> 4) ^ self.kp_high_truth[1]
        self.kp_high_truth[2] = self._solve_chain_nibble(lhs, pp2)
        
        # k'[3]
        pp3 = self._get_pp(0, 3)
        lhs = (pp2 >> 4) ^ self.kp_high_truth[2]
        self.kp_high_truth[3] = self._solve_chain_nibble(lhs, pp3)
        k3_h = (self.found_keys[0] >> 4) ^ self.rel_high[3]
        self.sk_high_constraints[12] = k3_h ^ self.kp_high_truth[3]
        print(f"  [Info] S(k12)>>4 = 0x{self.sk_high_constraints[12]:X}")

        # --- 2. Batch Solving ---
        
        # Batch 1 (Col 1)
        self.print_status("BATCH", "Solving Col 1 (Target: k4, k9, k14, k3)...")
        self._solve_batch_deterministic(1, [4, 9, 14, 3], self.kp_high_truth[3], self.kp_high_truth[0])
        self._calc_kp_high_algebraic(1, 4, 7) # Fills kp_high_truth 4..7
        
        # Batch 2 (Col 2)
        self.print_status("BATCH", "Solving Col 2 (Target: k8, k13, k2, k7)...")
        k7_h = self.kp_high_truth[7] ^ self.kp_high_truth[3]
        print(f"  [Constraint] Forcing k7 High Nibble: 0x{k7_h:X}")
        self._solve_batch_deterministic(2, [8, 13, 2, 7], self.kp_high_truth[7], self.kp_high_truth[4], 
                                        override_highs={3: k7_h}) 
        self._calc_kp_high_algebraic(2, 8, 11) 
        
        # Batch 3 (Col 3)
        self.print_status("BATCH", "Solving Col 3 (Target: k12, k1, k6, k11)...")
        k11_h = self.kp_high_truth[11] ^ self.kp_high_truth[7]
        print(f"  [Constraint] Forcing k11 High Nibble: 0x{k11_h:X}")
        self._solve_batch_deterministic(3, [12, 1, 6, 11], self.kp_high_truth[11], self.kp_high_truth[8],
                                        override_highs={3: k11_h})
        
        print("\n" + "="*60)
        print("RECOVERY COMPLETE")
        k_str = " ".join([f"{self.found_keys.get(i,0):02X}" for i in range(16)])
        print(f"Master Key: {k_str}")

    def _solve_chain_nibble(self, lhs, pp_next):
        best_s = -1; best_h = 0
        for h in range(16):
            s = self.diff_of_means(lhs == ((pp_next >> 4) ^ h))
            if s > best_s: best_s = s; best_h = h
        return best_h

    def _solve_batch_deterministic(self, col, indices, prev_kp_h, base_kp_h, override_highs=None):
        if override_highs is None: override_highs = {}
        cands = []
        for i, kid in enumerate(indices):
            if kid in self.sk_high_constraints:
                target = self.sk_high_constraints[kid]
                c = np.array([x for x in range(256) if (SBOX[x]>>4) == target], dtype=np.uint8)
                cands.append(c)
            elif i in override_highs:
                h = override_highs[i]
                cands.append(np.array([(h<<4)|l for l in range(16)], dtype=np.uint8))
            else:
                h = (self.found_keys[0] >> 4) ^ self.rel_high[kid]
                cands.append(np.array([(h<<4)|l for l in range(16)], dtype=np.uint8))
        
        dims = [len(c) for c in cands]
        print(f"  -> Search Space: {dims[0]}x{dims[1]}x{dims[2]}x{dims[3]} = {np.prod(dims)}")
        
        prev_idx_abs = col * 4 - 1
        pp_prev = self._get_pp((prev_idx_abs)//4, prev_idx_abs%4)
        lhs = (pp_prev >> 4) ^ prev_kp_h
        p_cols = [self.P[:, i] for i in indices]
        
        res_matrix = parallel_batch_solver(self.cycles, 
                                           p_cols[0], p_cols[1], p_cols[2], p_cols[3],
                                           cands[0], cands[1], cands[2], cands[3],
                                           lhs, base_kp_h)
        
        best_vals = res_matrix[np.argmax(res_matrix[:, 0])]
        print(f"  -> Score: {best_vals[0]:.4f}")
        
        final_keys = [cands[0][int(best_vals[1])], cands[1][int(best_vals[2])], 
                      cands[2][int(best_vals[3])], cands[3][int(best_vals[4])]]
        
        print(f"  -> Recovered: {[hex(int(x)) for x in final_keys]}")
        for i, k in enumerate(final_keys):
            self.found_keys[indices[i]] = int(k)

    def _calc_kp_high_algebraic(self, col, start, end):
        # Calculate k'[start] from newly recovered k[start] and k'[start-4]
        # REPAIR: Use rel_high to get k[start] High nibble! Do not rely on found_keys being full.
        
        # 1. Calculate k'[start]
        # k'[start] = k[start] ^ k'[start-4]
        # High(k[start]) comes from Adjacency (rel_high)
        k_start_h = (self.found_keys[0] >> 4) ^ self.rel_high[start]
        self.kp_high_truth[start] = k_start_h ^ self.kp_high_truth[start-4]
        
        # 2. Propagate for rest
        for i in range(start, end):
            next_idx = i + 1
            # k'[next] = k[next] ^ k'[next-4]
            k_next_h = (self.found_keys[0] >> 4) ^ self.rel_high[next_idx]
            self.kp_high_truth[next_idx] = k_next_h ^ self.kp_high_truth[next_idx-4]
            
        print(f"  [Calc] Derived High(k') for {start}..{end}: {[hex(self.kp_high_truth[x]) for x in range(start, end+1)]}")

if __name__ == "__main__":
    attacker = AESCorrectAttack("index_time_plaintext.csv")
    attacker.start_time = time.time()
    attacker.load_data()
    attacker.step1_adjacency()
    attacker.step2_anchor()
    attacker.step3_correct_flow()
    print(f"Total time: {time.time()-attacker.start_time:.2f}s")