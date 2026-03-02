# Validation & Verification for Aerospace Applications — Fluent RANS Assignment (Tyrrell-26)

This repository contains my coursework for **Validation and Verification for Aerospace Applications**.
It is a comparative CFD study (ANSYS Fluent) of **RANS turbulence models** and **Riemann solvers** for the 2D **Tyrrell-26** wing/airfoil section, including validation against experimental pressure coefficient data and uncertainty quantification.

---

## 1. Assignment overview

The coursework covers:

- **Task 1**: Flux type (Roe-FDS vs AUSM) and spatial discretization sensitivity  
- **Task 2**: Iterative uncertainty (CL/CD/Cp convergence + tail statistics)  
- **Task 3**: Cp vs experimental data (+ optional adaptive refinement)  
- **Task 4**: Turbulence-model uncertainty (≥3 models via collaboration)  
- **Task 5**: Grid convergence + Richardson extrapolation + GCI

Optional:
- **Case 3 (ground effect)**: AoA = 3.6°, h/c = 0.224

---

## 2. Repository layout

```
Assignment/
  data_process/
    data/                  # raw outputs exported from Fluent (CL/CD/Cp)
      kw/taskX/...         # X = 1..5 (Standard k–ω with curvature correction)
      sa/taskX/...         # X = 1..5 (Spalart–Allmaras)
    result/                # post-processed figures + summary CSVs
    process_task1.py
    process_task2_task4.py
    process_task3.py
    process_task5.py
    process_case3_task4.py
  jou/                     # Fluent journals (case2/case3 + refine/ama)
  report/                  # report.pdf
  mesh/                    # meshes (case2/case3)
```

### Notes
- Each Python post-processing script (`process_*.py`) includes a **run example at the top of the file**.
- Fluent **journal files** may be recorded (GUI/TUI) or written/edited manually; they automate repeated solver steps.

---

## 3. Mesh files

```
mesh/tyrrell.msh                         # case2 mesh
mesh/tyrrell_3_6_degree_hc_0_224_ge.msh   # case3 (ground effect) mesh
```

---

## 4. Fluent journals (jou/)

### case2 (baseline / refine / adaptive)
```
jou/tyrell_sa.jou
jou/tyrell_sa_refine1.jou
jou/tyrell_sa_refine2.jou
jou/tyrell_sa_ama.jou

jou/tyrrell_kw.jou
jou/tyrrell_kw_refine1.jou
jou/tyrrell_kw_refine2.jou
jou/tyrrell_kw_ama.jou
```

### case3 (ground effect)
```
jou/tyrrell_3_6_degree_sa.jou
jou/tyrrell_3_6_degree_kw.jou
```

### Boundary-condition parameters (Named Expressions TSV)
```
jou/param_case2_fluent2d.tsv
```

**Before running a journal**, update the mesh path and TSV path if your working directory differs.

---

## 5. Raw data conventions (data_process/data)

Two top-level folders correspond to the two assigned turbulence models:

- `kw/` → Standard k–ω (with curvature correction), abbreviated as **kw**
- `sa/` → Spalart–Allmaras , abbreviated as **SA**

Each contains `taskX/` (X = 1..5).

### 5.1 Common file types
- `cd_*.out` → drag coefficient **C_D**
- `cl_*.out` → lift coefficient **C_L**
- `sa_*.txt` / `kw_*.txt` → pressure coefficient distribution **C_p(x/c)**

### 5.2 Task 1 encoding (x/y/z)

**SA** (`sa_x_y_*`):
- `x`: Flow discretization (1 = First Order Upwind, 2 = Second Order Upwind)
- `y`: Modified Turbulent Viscosity discretization (1 = First, 2 = Second)
- Flux type: `AUSM` or `Roe`

**kw** (`kw_x_y_z_*`):
- `x`: Flow (1 = First, 2 = Second)
- `y`: Turbulent Kinetic Energy k (1 = First, 2 = Second)
- `z`: Specific Dissipation Rate ω (1 = First, 2 = Second)
- Flux type: `AUSM` or `Roe`

### 5.3 Task 3 adaptive refinement encoding
Suffix `ama_300` means:
- `ama`: adaptive mesh refinement around the airfoil surface/boundary layer
- `300`: first run 300 iterations to obtain an initial flow field, then apply adaptive refinement

### 5.4 Task 5 refinement encoding
Three-mesh sequence:
- baseline: `*_AUSM.*`
- refine1: `*_refine1.*`
- refine2: `*_refine2.*`

Refinement method: Fluent Solution → Cell Registers → Region → Refine (twice)

---

## 6. Task-to-script mapping (post-processing)

- `data_process/process_task1.py`  
  Task 1: Flux type comparison (Roe vs AUSM) + discretization sensitivity

- `data_process/process_task2_task4.py`  
  Task 2: Iterative uncertainty  
  Task 4: Model uncertainty (SA vs kw + third model)

- `data_process/process_task3.py`  
  Task 3: Cp vs experimental data (+ adaptive refinement comparison)

- `data_process/process_task5.py`  
  Task 5: Grid convergence, Richardson extrapolation, and GCI

- `data_process/process_case3_task4.py`  
  Case 3: ground-effect case processed in the same style as Task 4

Outputs are written to:
- `data_process/result/` (figures + summary CSVs)

---

## 7. Reproducibility workflow (recommended)

1) **Run Fluent journals** to generate raw data into `data_process/data/...`  
2) **Run Python post-processing** scripts to generate `data_process/result/...`  

---

## 8. License

This repository is for coursework submission and reproducibility. If you plan to reuse code/data, please cite appropriately and respect any course or third‑party constraints.
