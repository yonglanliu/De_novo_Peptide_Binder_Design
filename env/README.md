## Three conda environments need to be set up.
1. [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion)
2. peptide_design environment

---

### 1. RFdiffusion Set up

Using RFdiffusion to generate the backbone of peptides, giving hotspot residues 

```bash
conda env create -f env/SE3nv.yml

conda activate SE3nv
cd env/SE3Transformer
pip install --no-cache-dir -r requirements.txt
python setup.py install
cd ../.. # change into the root directory of the repository
pip install -e . # install the rfdiffusion module from the root of the repository
```

---

### 2. Peptide Design Environment


* This environment is used for:

* ProteinMPNN sequence design

* Sequence prioritization based on score

* Rosetta / PyRosetta FastRelax

* Interface scoring and refinement

* Binder energy scoring and refinement

* Alphafold2 binder conformation scoring

* Alphafold-multimer interface scoring

```bash
conda env create -f env/peptide_design.yml
conda activate peptide_design
pip install -e .
```

--- 



## Final Environment Summary

After setup, you should have **three conda environments**:

### SE3nv

* RFdiffusion
* Backbone generation of peptide binders with hotspot constraints

### peptide_design

* ProteinMPNN
* PyRosetta / FastRelax
* Sequence design and structural refinement
* Modified Alphafold2
* Final structural validation of peptide–protein complexes

