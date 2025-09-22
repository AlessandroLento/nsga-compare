# NSGA-II vs NSGA-III comparison (pymoo)

Repository scaffold to compare NSGA-II and NSGA-III (pymoo) on standard multi-objective test problems.

## Structure
```
repo/
│
├── src/
│   ├── opt.py
│   └── utils.py
│
├── test/
│   ├── problem/
│   │   └── problems.py
│   └── payload/
│       ├── config1.json
│       └── config2.json
│       ├── config3.json
│       └── config4.json
│
├── result/
│
├── main_test.py
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

## How to use

1. Create a virtual environment and install dependencies (using poetry or pip).
   Example (pip):
   ```
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
   Or with poetry:
   ```
   poetry install
   ```
2. Run the main test script:
   ```
   python main_test.py
   ```
   This will run the configured experiments and save outputs into `result/`.

## Notes
- The scripts expect `pymoo` to be installed (tested with pymoo>=0.5).
- Config files are JSON in `test/payload/`. You can add more configs or problems.
- After running, results (populations, front scatterplots, metrics) are stored under `result/`.


## References
(Blank and Deb 2020)

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-xie2018" class="csl-entry">

J. Blank and K. Deb, *pymoo: Multi-Objective Optimization in Python*. 
In IEEE Access, vol. 8, pp. 89497-89509, 2020, doi: 10.1109/ACCESS.2020.2990567

</div>

</div>
