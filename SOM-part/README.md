# *HSF1* Tandem Repeat Analysis

This repository contains scripts for analyzing *HSF1* tandem repeat sequences using Self-Organizing Maps (SOM) with configurable k-mer features.

### SOM Classification
**`pure_som_classifier_v3_kmer_en.py`** - SOM-based sequence clustering with configurable k-mer length

**Features:**
- Configurable k-mer lengths (4-mer, 5-mer, 6-mer)
- Excel-derived statistical features integration
- 2D visualization with patient annotations and population information

**Usage:**
```bash
python3 pure_som_classifier_v3_kmer_en.py \
    --neighborhood-factor 3.0 \
    --merge-threshold 0.8 \
    --max-iter 5000 \
    --learning-rate 0.02 \
    --width 6 \
    --height 6 \
    --all -v
```

**Outputs:**
- `HSF1_Xmer_excel_som_model.pkl` - Trained SOM models
- `HSF1_Xmer_excel_som_visualization.png` - 6-panel SOM visualization
- `HSF1_Xmer_excel_som_2d_patient.png` - 2D scatter plot with patient annotations and population info
- `HSF1_Xmer_excel_som_detailed.txt` - Detailed classification report
- `HSF1_Xmer_excel_cluster_composition.csv` - Cluster composition data

### Required:
- `characte of seq.xlsx` - Excel file containing sample metadata and statistical features
- `seqs_700bp.txt` - Sequence file (one sequence per line)
- `ethics.txt` - Population information for control samples (format: `SampleID\tPopulation`)

## Key Features

### Population Information
Control samples are annotated with population codes:
- **AFR** - African
- **EAS** - East Asian
- **EUR** - European
- **SAS** - South Asian
- **AMR** - American

## Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib seaborn scipy openpyxl
```

## Output Directory

All results are saved to `pure_som_kmer_models/` directory.

## Citation

If you use these scripts, please cite the corresponding publication.
