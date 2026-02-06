import re
import sys
import pandas as pd
from collections import defaultdict, Counter

# =================RP2350 XIP Cache =================
CACHE_SIZE = 16 * 1024   # 16KB
LINE_SIZE = 8            # 8 Bytes
WAYS = 2                 # 2-way
SETS = CACHE_SIZE // (LINE_SIZE * WAYS) # 1024 Sets

def get_set_index(addr):
    return (addr >> 3) & (SETS - 1)

def get_tag(addr):
    return addr >> 13

class CacheDetective:
    def __init__(self, dump_file):
        self.dump_file = dump_file
        self.tables = {}      
        self.inst_cache_map = defaultdict(set)

    def load_dump(self):
        print(f"[*] Parsing {self.dump_file}...")
        

        re_sym = re.compile(r"^\s*([0-9a-fA-F]+)\s+.*?\s+([0-9a-fA-F]+)\s+(Te[0-3]|sbox)$")
        
        re_func = re.compile(r"^[0-9a-fA-F]+\s+<([^>]+)>:$")
        re_inst = re.compile(r"^\s*([0-9a-fA-F]+):\s+([0-9a-fA-F ]+)(\s+.*)?$")

        current_func = "unknown_startup"
        
        try:
            with open(self.dump_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue

                    if m_sym := re_sym.match(line):
                        # 提取名字，Te0/Te1... 或 sbox
                        name = m_sym.group(3)
                        self.tables[name] = (int(m_sym.group(1), 16), int(m_sym.group(2), 16))
                        continue

                    if m_func := re_func.search(line):
                        current_func = m_func.group(1)
                        continue
                        
                    if m_inst := re_inst.match(line):
                        if len(m_inst.group(1)) < 4: continue
                        addr = int(m_inst.group(1), 16)
                        if ".word" in line or ".short" in line: continue
                        
                        s_idx = get_set_index(addr)
                        tag = get_tag(addr)
                        self.inst_cache_map[s_idx].add((tag, current_func))

        except FileNotFoundError:
            print("File not found.")
            sys.exit(1)
            
        print(f"    Found Objects: {list(self.tables.keys())}")

    def analyze(self):
        self.load_dump()
        print("\n" + "="*60)
        print("   CACHE CONFLICT DETECTIVE REPORT (RP2350 2-Way)")
        print("="*60)
        

        targets = ["Te0", "Te1", "Te2", "Te3", "sbox"]
        
        for name in targets:
            if name not in self.tables: 
                print(f"[{name}] Not found in dump (maybe stripped?).")
                continue
            
            start, size = self.tables[name]
            end = start + size
            total_lines = size // LINE_SIZE
            
            offenders = Counter()
            conflict_lines = 0
            
            for t_addr in range(start, end, LINE_SIZE):
                s_idx = get_set_index(t_addr)
                t_tag = get_tag(t_addr)
                
                inst_info = self.inst_cache_map.get(s_idx, set())
                unique_inst_tags = set(tag for tag, func in inst_info)
                all_tags = unique_inst_tags.union({t_tag})
                
                if len(all_tags) >= 3:
                    conflict_lines += 1
                    for _, func in inst_info:
                        offenders[func] += 1
            
            print(f"\nTarget: [{name}]")
            print(f"  Size: {size} Bytes ({total_lines} Cache Lines)")
            print(f"  Lines with contention: {conflict_lines} / {total_lines}")
            
            if offenders:
                print(f"  Top 3 Functions causing eviction in {name}:")
                for func, count in offenders.most_common(3):
                    print(f"    - {func:<30} (overlaps {count} lines)")
            else:
                print("    (No conflicts found - Safe)")

if __name__ == "__main__":
    CacheDetective("dump.txt").analyze()