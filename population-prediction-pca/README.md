# 1000 Genomes PCA Analysis & Ancestry Inference

A comprehensive toolkit for Principal Component Analysis (PCA) of genomic data and ancestry inference based on the 1000 Genomes Project.

## 📊 Data Acquisition

### 1. 1000 Genomes Reference Data

**VCF Files (hg38/GRCh38)**:
- Download from: https://hgdownload.cse.ucsc.edu/gbdb/hg38/1000Genomes/
- Files are split by chromosome: `ALL.chr1.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf.gz`, etc.

**Population Information**:
- Download from: ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.panel
- This file contains sample-to-population mapping

### 2. Study Samples (VCF Sources)

The samples used in this study come from three sources:

**Part 1: 1000g-ONT Project**
- URL: https://s3.amazonaws.com/1000g-ont/index.html?prefix=PROCESSED_DATA/ALIGNED_TO_HG38/CLAIR3/PHASED_VCF/
- Long-read ONT sequencing data aligned to hg38

**Part 2: Human Pangenomics Consortium**
- URL: https://s3-us-west-2.amazonaws.com/human-pangenomics/index.html?prefix=submissions/759B21AD-0ED8-4640-A433-7C92A57EA3D3--UW_EEE_SV_Calls/GRCh38/
- High-quality phased assemblies

**Part 3: SRA Data**
- Download raw FASTQ files from NCBI SRA
- Process through standard variant calling pipeline (alignment → variant calling → phasing)

## 🚀 Quick Start

### Installation

```bash
# Install required packages
pip install numpy pandas scikit-learn scikit-allel matplotlib plotly tqdm

# Or use requirements.txt
pip install -r requirements.txt
```

### Basic Usage

#### Step 1: Train PCA Model

```bash
python3 1000genomes_pca_ultimate.py \
    all.1kg.phase3.GRCh38.vcf.gz \
    integrated_call_samples_v3.20130502.ALL.panel \
    --output 1kg_pca \
    --n-components 3 \
    --n-jobs 8 \
    --save-snps selected_snps.npz \
    --save-model pca_model.pkl
```

#### Step 2: Project New Samples

**Single VCF**:
```bash
python3 pca_projection.py \
    --snps selected_snps.npz \
    --model pca_model.pkl \
    --reference 1kg_pca_results.csv \
    --query-vcf your_sample.vcf.gz \
    --output-dir results
```

**Batch Processing** (Multiple VCFs):
```bash
python3 pca_projection.py \
    --snps selected_snps.npz \
    --model pca_model.pkl \
    --reference 1kg_pca_results.csv \
    --query-vcf /path/to/vcf_folder/ \
    --output-dir batch_results \
    --n-jobs 8
```

## 📦 Pre-trained Models

Pre-trained models based on 1000 Genomes Project data are included in this repository.

### Model Files (in project directory)

| File | Description | 
|------|-------------|
| `reference_model.pkl` | PCA model trained on 1000 Genomes |
| `reference_snps.npz` | SNP positions used for PCA |
| `allgenome_results.csv` | Reference PCA coordinates for all samples |
| `allgenome_pc1_pc2.png` | 2D PCA visualization (PC1 vs PC2) |
| `allgenome_3d.png` | 3D PCA visualization |

### Model Specifications

- **Samples**: 2,548 individuals from 5 super populations (AFR, AMR, EAS, EUR, SAS)
- **Reference Genome**: GRCh38/hg38
- **Variant Filtering**:
  - Minor Allele Frequency (MAF) ≥ 0.01
  - Missing rate ≤ 10%
  - Spatial thinning: max 20 SNPs per 100kb window
  - LD pruning: r² < 0.2 in 500kb windows
- **Final SNP Count**: ~72,000 independent SNPs
- **PC Components**: 3 (explaining ~15% of variance)

### Usage with Pre-trained Models

```bash
# Project your samples using provided models
python3 pca_projection.py \
    --snps reference_snps.npz \
    --model reference_model.pkl \
    --reference allgenome_results.csv \
    --query-vcf your_cohort/ \
    --output-dir ancestry_results \
    --n-jobs 8
```

## 📊 Output Files

### Projection Output Structure
```
output_dir/
├── all_samples_predictions.txt      # Summary: sample → population
├── sample1/                          # Per-sample folder
│   ├── sample1_predictions.txt      # Sample prediction
│   ├── detailed_results.csv         # Full PCA coords + probabilities
│   ├── ancestry_report.txt          # Detailed text report
│   ├── pca_2d.png                   # 2D visualization
│   ├── pca_3d.png                   # 3D visualization
│   └── pca_3d_interactive.html      # Interactive plot
└── sample2/
    └── ...
```

### Population Codes

| Code | Population | Description |
|------|------------|-------------|
| AFR | African | African ancestry |
| AMR | Ad Mixed American | Native American + European + African |
| EAS | East Asian | East Asian ancestry |
| EUR | European | European ancestry |
| SAS | South Asian | South Asian ancestry |

## 📝 Citation

If you use this tool in your research, please cite:

```
To be continued
```

## 📧 Contact

For questions or issues, please open an issue on GitHub or via email(xiaosheng@zju.edu.cn).
