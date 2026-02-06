# Flash Timing Attack Demo

This repository contains two AES implementations on different MCUs and scripts/data for timing side-channel experiments:

- **STM32F407**: AES implemented with a classic **S-box lookup**
- **RP2350**: AES implemented with **T-tables** (**Te0–Te3**)
- **`attack.py` (repo root)**: end-to-end timing attack script **targeting the STM32F407 S-box implementation**
- **`RP2350/dump.txt`**: a “mapping dump” (symbols + disassembly) used to map **table objects / instructions** into **RP2350 XIP cache sets** for conflict analysis

> Reference paper: *Const Is Not Constant: Flash Timing Side-Channel Attacks on MCUs*


---

## Repository Layout

```text
.
├── attack.py                         # Timing attack against STM32F407 (S-box AES)
├── index_time_plaintext.csv          # Default traces for attack.py (P0..P15, Cycles)
├── README.md
├── RP2350/
│   ├── main.c                        # RP2350: AES T-table implementation (Te0..Te3)
│   ├── check.py                      # Parses dump.txt and reports XIP cache set conflicts
│   ├── dump.txt                      # Generated: symbols + disassembly mapping (see below)
│   └── aes.elf                       # The ELF (used to generate dump.txt)
├── STM32F407/
│   └── main.c                        # STM32F407: AES S-box implementation
└── Traces/
    ├── index_time_plaintext_stm32.csv
    ├── index_time_plaintext_rp2350_sbox.zip
    ├── index_time_plaintext_rp2350_ttable.zip
    ├── distribution_stability_2x3.pdf
    └── plot.py                       # Plot distribution + stability figures
````

---

## What’s What

### STM32F407 (S-box AES)

* Source: `STM32F407/main.c`
* Targeted by: `attack.py` (root)
* The traces are expected to include:

  * `P0 ... P15`: 16 plaintext bytes
  * `Cycles`: measured encryption latency (clock cycles or equivalent timing counter)

### RP2350 (T-table AES)

* Source: `RP2350/main.c`
* Uses T-tables: `Te0`, `Te1`, `Te2`, `Te3` (and optionally `sbox`)
* `RP2350/check.py` analyzes **XIP cache set contention** by parsing `dump.txt`

---

## Dependencies

### For `attack.py`

* Python 3
* `numpy`, `pandas`, `numba`

Install:

```bash
pip install numpy pandas numba
```

### For plotting (`Traces/plot.py`)

* `matplotlib`, `seaborn`

Install:

```bash
pip install matplotlib seaborn
```

---

## Run the STM32 Attack

`attack.py` loads `index_time_plaintext.csv` by default.

```bash
python3 attack.py
```

### Example Output (Key Recovery)

```text
RECOVERY COMPLETE
Master Key: 2B 7E 15 16 28 AE D2 A6 AB F7 15 88 09 CF 4F 3C
Total time: 16.10s
```

---

## Plot Trace Distribution & Stability

From the `Traces/` directory:

```bash
cd Traces
python3 plot.py
```

This generates:

* `distribution_stability_2x3.pdf`

---

## RP2350: Generating `dump.txt` (Symbols + Disassembly Mapping)

`RP2350/check.py` expects a `dump.txt` that contains:

1. **Symbol table entries** for `Te0–Te3` / `sbox` including **address and size**
2. **Disassembly** including function labels and instruction addresses

You said you will add the corresponding **ELF** into `RP2350/` (e.g., `RP2350/firmware.elf`).
After placing the ELF, generate `dump.txt` with one of the following options.

### GNU `arm-none-eabi-objdump`

```bash
cd RP2350
arm-none-eabi-objdump -t -d -w aes.elf > dump.txt
```

Notes:

* `-t` prints the symbol table (includes object sizes like `Te0`, etc.)
* `-d` prints disassembly (function labels + instruction addresses)
* `-w` disables line wrapping (helps parsers)



If your `check.py` relies on raw instruction bytes, remove `--no-show-raw-insn`.

### Verifying the Dump

After generation, `dump.txt` should contain lines resembling:

* Symbol entries like:

  * `<addr> ... <size> Te0` / `Te1` / `Te2` / `Te3` / `sbox`
* Function labels like:

  * `00001234 <aes_encrypt>:`
* Instruction lines like:

  * `  00001238:  ....  <mnemonic ...>`

---

## RP2350 Cache Conflict Analysis

Once `dump.txt` exists:

```bash
cd RP2350
python3 check.py
```

The script reports, for each target table (`Te0..Te3`, `sbox`):

* table size and number of cache lines
* how many lines experience **2-way contention**
* the top functions whose instructions map to the same cache sets (likely evictors)

---


