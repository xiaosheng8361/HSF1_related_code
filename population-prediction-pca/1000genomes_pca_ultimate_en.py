#!/usr/bin/env python3
"""
1000 Genomes PCA Analysis - Ultimate Optimized Version
Memory-optimized version for large-scale whole-genome analysis

Key Features:
1. Chromosome-wise processing - prevents memory overflow
2. Automatic merging - seamlessly combines all chromosome results
3. Resume capability - automatically skips processed chromosomes
4. LD pruning pre-filtering - spatial sampling reduces variants by 10x
"""

import numpy as np
import pandas as pd
import allel
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import gc
import pickle
from pathlib import Path

# Try to import plotly
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Set font for plots
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# Cache directory
CACHE_DIR = Path(".pca_cache")

# Standard chromosome list
CHROMOSOMES = [str(i) for i in range(1, 23)] + ['X']


def spatial_thinning(allele_counts, positions, bin_size=100000, max_per_bin=20):
    """
    空间稀疏化：在LD修剪前先做空间筛选
    
    策略：每100kb范围内，如果超过max_per_bin sites，随机抽样
    
    参数:
        allele_counts: 等位基因矩阵 (samples x variants)
        positions: 位点位置
        bin_size: 区间大小（bp）
        max_per_bin: 每个区间max 保留的位点数
    
    返回:
        筛选后的数据
    """
    if positions is None or len(positions) == 0:
        return allele_counts, positions
    
    print(f"\nSpatial thinning pre-processing...")
    print(f"  Parameters: bin={bin_size//1000}kb, max{max_per_bin} sites")
    print(f"  Input: {allele_counts.shape[1]}  variant sites")
    
    # Calculate bin for each site
    bins = positions // bin_size
    unique_bins = np.unique(bins)
    
    selected_indices = []
    
    for bin_id in unique_bins:
        # Find all sites in this bin
        bin_mask = bins == bin_id
        bin_indices = np.where(bin_mask)[0]
        
        # If exceeds max count, randomly sample
        if len(bin_indices) > max_per_bin:
            np.random.seed(42)  # Ensure reproducibility
            selected = np.random.choice(bin_indices, max_per_bin, replace=False)
            selected_indices.extend(selected)
        else:
            selected_indices.extend(bin_indices)
    
    # Sort indices
    selected_indices = sorted(selected_indices)
    
    # Apply filtering
    allele_counts_thinned = allele_counts[:, selected_indices]
    positions_thinned = positions[selected_indices]
    
    print(f"  Output: {len(selected_indices)}  sites")
    print(f"  Reduced: {allele_counts.shape[1] - len(selected_indices)}  sites " +
          f"({(1 - len(selected_indices)/allele_counts.shape[1])*100:.1f}%)")
    
    return allele_counts_thinned, positions_thinned


def generate_chr_cache_filename(vcf_file, chromosome, maf_threshold):
    """Generate cache filename for single chromosome"""
    cache_name = f"chr_{chromosome}_maf{maf_threshold:.3f}.pkl.gz"
    print(f"chr_{chromosome}_maf{maf_threshold:.3f}.pkl.gz")
    return CACHE_DIR / cache_name


def generate_merged_cache_filename(maf_threshold):
    """生成合并后的全基因组缓存文件名"""
    cache_name = f"merged_all_chr_maf{maf_threshold:.3f}.pkl.gz"
    return CACHE_DIR / cache_name


def save_chromosome_data(allele_counts, samples, positions, cache_file):
    """保存Single chromosome的数据"""
    CACHE_DIR.mkdir(exist_ok=True)
    
    cache_data = {
        'allele_counts': allele_counts,
        'samples': samples,
        'positions': positions,
        'shape': allele_counts.shape,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    file_size_mb = cache_file.stat().st_size / (1024 * 1024)
    return file_size_mb


def load_chromosome_data(cache_file):
    """加载Single chromosome的数据"""
    with open(cache_file, 'rb') as f:
        cache_data = pickle.load(f)
    
    # 确保positions是int64类型（兼容旧缓存）
    positions = cache_data['positions']
    if positions is not None and positions.dtype != np.int64:
        positions = positions.astype(np.int64)
    
    return cache_data['allele_counts'], cache_data['samples'], positions


def read_and_filter_chromosome(vcf_file, chromosome, maf_threshold=0.01, 
                               missing_threshold=0.1, use_cache=True):
    """
    读取和过滤Single chromosome的数据
    支持缓存功能
    """
    cache_file = generate_chr_cache_filename(vcf_file, chromosome, maf_threshold)
    
    # 检查缓存
    if use_cache and cache_file.exists():
        print(f"\n  📂 Using cache: {cache_file.name}")
        try:
            allele_counts, samples, positions = load_chromosome_data(cache_file)
            print(f"     Shape: {allele_counts.shape}, cache loaded successfully ✓")
            return allele_counts, samples, positions
        except Exception as e:
            print(f"     Cache loading failed: {e}，re-reading")
    
    # 读取VCF
    print(f"\n  🔍 Reading chromosome {chromosome}...")
    try:
        callset = allel.read_vcf(vcf_file, region=chromosome)
    except Exception as e:
        print(f"     Reading failed: {e}")
        return None, None, None
    
    if callset is None or 'calldata/GT' not in callset:
        print(f"     Chromosome {chromosome} no data, skipping")
        return None, None, None
    
    # 转换数据
    genotypes = allel.GenotypeArray(callset['calldata/GT'])
    gn = genotypes.to_n_alt()
    allele_counts = gn.T  # samples x variants
    # 确保positions是int64类型，避免后续加offset时溢出
    positions = callset['variants/POS'].astype(np.int64) if 'variants/POS' in callset else None
    samples = callset.get('samples', None)
    
    if samples is None:
        raise ValueError(f"Chromosome {chromosome} 无samples信息")
    
    # 确保samples是字符串
    if isinstance(samples[0], bytes):
        samples = np.array([s.decode('utf-8') for s in samples])
    else:
        samples = np.array([str(s) for s in samples])
    
    print(f"     Original: {allele_counts.shape[0]} samples, {allele_counts.shape[1]} variants")
    
    # 过滤
    n_samples = allele_counts.shape[0]
    
    # 1. 缺失率过滤
    missing_count = np.sum(allele_counts == -1, axis=0)
    missing_rate = missing_count / n_samples
    mask = missing_rate <= missing_threshold
    allele_counts = allele_counts[:, mask]
    if positions is not None:
        positions = positions[mask]
    
    # 2. MAF过滤
    valid_mask = allele_counts != -1
    allele_sum = np.sum(np.where(valid_mask, allele_counts, 0), axis=0)
    valid_count = np.sum(valid_mask, axis=0) * 2
    
    with np.errstate(divide='ignore', invalid='ignore'):
        af = allele_sum / valid_count
        af[valid_count == 0] = 0
    
    maf = np.minimum(af, 1 - af)
    mask = maf >= maf_threshold
    allele_counts = allele_counts[:, mask]
    if positions is not None:
        positions = positions[mask]
    
    # 3. 单态位点过滤
    std_vals = np.std(allele_counts, axis=0)
    mask = std_vals > 0
    allele_counts = allele_counts[:, mask]
    if positions is not None:
        positions = positions[mask]
    
    print(f"     After filtering: {allele_counts.shape[1]} variants")
    
    # 保存缓存
    if use_cache and allele_counts.shape[1] > 0:
        file_size = save_chromosome_data(allele_counts, samples, positions, cache_file)
        print(f"     💾 Cache saved: {file_size:.1f} MB")
    
    # 释放内存
    del genotypes, gn, callset
    gc.collect()
    
    return allele_counts, samples, positions


def process_all_chromosomes(vcf_file, maf_threshold=0.01, chromosomes=None,
                            use_cache=True):
    """
    处理所有染色体并合并
    """
    if chromosomes is None:
        chromosomes = CHROMOSOMES[:22]  # 默认1-22号染色体
    
    print("=" * 80)
    print(f"Chromosome-wise processing: will process {len(chromosomes)}  chromosomes")
    print("=" * 80)
    
    all_allele_counts = []
    all_positions = []
    common_samples = None
    
    for i, chrom in enumerate(chromosomes, 1):
        print(f"\n[{i}/{len(chromosomes)}] Chromosome {chrom}")
        print("-" * 80)
        
        allele_counts, samples, positions = read_and_filter_chromosome(
            vcf_file, chrom, maf_threshold=maf_threshold, use_cache=use_cache
        )
        
        if allele_counts is None or allele_counts.shape[1] == 0:
            print(f"     ⚠️  跳过Chromosome {chrom}")
            continue
        
        # 检查样本一致性
        if common_samples is None:
            common_samples = samples
        elif not np.array_equal(common_samples, samples):
            print(f"     ⚠️  警告：Chromosome {chrom}  has inconsistent samples, skipping")
            continue
        
        # 添加染色体信息到positions
        if positions is not None:
            # 为每 chromosomes的position添加偏移，避免overlapping
            # positions已经在读取时转换为int64，这里直接计算offset
            chrom_offset = int(chrom) * 1_000_000_000 if chrom.isdigit() else 0
            positions_offset = positions + chrom_offset
            all_positions.append(positions_offset)
        
        all_allele_counts.append(allele_counts)
        
        # 显示进度
        total_variants = sum(ac.shape[1] for ac in all_allele_counts)
        print(f"     ✓ 累计variants数: {total_variants}")
    
    if len(all_allele_counts) == 0:
        raise ValueError("No chromosomes processed successfully!")
    
    # 合并所有染色体
    print("\n" + "=" * 80)
    print("Merging all chromosome data...")
    print("=" * 80)
    
    merged_allele_counts = np.concatenate(all_allele_counts, axis=1)
    # 确保合并后的positions是int64类型
    merged_positions = np.concatenate(all_positions).astype(np.int64) if all_positions else None
    
    print(f"  After merging: {merged_allele_counts.shape[0]} samples, " +
          f"{merged_allele_counts.shape[1]} variants位点")
    
    # 保存合并后的缓存
    if use_cache:
        merged_cache_file = generate_merged_cache_filename(maf_threshold)
        print(f"\n💾 Saving merged data...")
        file_size = save_chromosome_data(
            merged_allele_counts, common_samples, merged_positions, merged_cache_file
        )
        print(f"  Cache file: {merged_cache_file.name}")
        print(f"  File size: {file_size:.1f} MB")
    
    # 释放内存
    del all_allele_counts
    gc.collect()
    
    return merged_allele_counts, common_samples, merged_positions


def ld_prune_parallel_chunk(args):
    """并行LD修剪的单个chunk处理函数"""
    gn_chunk, positions_chunk, window_bp, threshold = args
    try:
        chunk_mask = allel.locate_unlinked(
            gn_chunk,
            size=window_bp,
            step=window_bp,
            threshold=threshold,
            n_iter=1
        )
    except TypeError:
        chunk_mask = allel.locate_unlinked(
            gn_chunk,
            size=window_bp,
            step=window_bp,
            threshold=threshold
        )
    return chunk_mask


def ld_prune_with_thinning(allele_counts, positions, window_size=500, 
                           threshold=0.2, thin_bin_size=100000, 
                           max_per_bin=20, n_jobs=4):
    """
    带空间稀疏化的LD修剪（支持多进程并行）
    
    策略：
    1. 先空间稀疏化（每100kbmax 20 SNPs）
    2. 再进行LD修剪（支持多进程加速）
    
    参数:
        n_jobs: 并行进程数，默认4。设为1则单线程，-1则使用所有CPU
    """
    if positions is None:
        print("\nWarning: 无位置信息，跳过LD修剪")
        return allele_counts
    
    print(f"\nLD pruning (with spatial pre-filtering)...")
    print(f"  Input: {allele_counts.shape[1]}  variant sites")
    
    # 步骤1：空间稀疏化
    allele_counts_thin, positions_thin = spatial_thinning(
        allele_counts, positions, 
        bin_size=thin_bin_size, 
        max_per_bin=max_per_bin
    )
    
    # 步骤2：LD修剪
    print(f"\n  Executing LD pruning...")
    print(f"  Parameters: window={window_size}kb, r² threshold={threshold}")
    
    try:
        # Transpose
        gn_for_ld = allele_counts_thin.T
        
        window_bp = window_size * 1000
        
        # Adaptive step size
        if allele_counts_thin.shape[1] > 50000:
            step_bp = window_bp  # non-overlapping
            print(f"  Using fast step: {window_size}kb (non-overlapping)")
        else:
            step_bp = int(window_bp / 2)  # 50%overlapping
            print(f"  Using standard step: {window_size//2}kb (50%overlapping)")
        
        # Check if using parallel processing
        import os
        if n_jobs == -1:
            n_jobs = os.cpu_count()
        
        use_parallel = n_jobs > 1 and allele_counts_thin.shape[1] > 100000
        
        if use_parallel:
            print(f"  🚀 Multi-processing enabled (processes: {n_jobs})")
            print(f"  Calculating LD correlation...")
            
            from multiprocessing import Pool
            
            # Chunk processing
            n_variants = gn_for_ld.shape[0]
            chunk_size = max(10000, n_variants // (n_jobs * 2))
            
            chunks = []
            for i in range(0, n_variants, chunk_size):
                end_idx = min(i + chunk_size, n_variants)
                gn_chunk = gn_for_ld[i:end_idx]
                pos_chunk = positions_thin[i:end_idx]
                chunks.append((gn_chunk, pos_chunk, window_bp, threshold))
            
            print(f"  Split into {len(chunks)}  chunks for parallel processing...")
            
            # 并行处理
            with Pool(processes=n_jobs) as pool:
                chunk_masks = pool.map(ld_prune_parallel_chunk, chunks)
            
            # Merge results
            ld_mask = np.concatenate(chunk_masks)
        else:
            if n_jobs > 1:
                print(f"  variants数较少，使用单线程处理")
            print(f"  Calculating LD correlation...")
            
            # Single-thread LD pruning execution
            try:
                ld_mask = allel.locate_unlinked(
                    gn_for_ld,
                    size=window_bp,
                    step=step_bp,
                    threshold=threshold,
                    n_iter=1
                )
            except TypeError:
                ld_mask = allel.locate_unlinked(
                    gn_for_ld,
                    size=window_bp,
                    step=step_bp,
                    threshold=threshold
                )
        
        n_after = np.sum(ld_mask)
        removed = allele_counts_thin.shape[1] - n_after
        
        print(f"  ✓ LD pruning completed")
        print(f"  Output: {n_after} 个独立variants位点")
        print(f"  Removed: {removed} 个相关variants ({removed/allele_counts_thin.shape[1]*100:.1f}%)")
        
        allele_counts_pruned = allele_counts_thin[:, ld_mask]
        positions_pruned = positions_thin[ld_mask]
        
        # 总结
        total_removed = allele_counts.shape[1] - allele_counts_pruned.shape[1]
        print(f"\n  Total: {allele_counts.shape[1]} → {allele_counts_pruned.shape[1]} " +
              f"(reduced {total_removed/allele_counts.shape[1]*100:.1f}%)")
        
        return allele_counts_pruned, positions_pruned
    
    except Exception as e:
        print(f"  ✗ LD pruning failed: {e}")
        print(f"  → Returning spatially thinned data")
        return allele_counts_thin, positions_thin


def run_pca_analysis(allele_counts, n_components=3):
    """执行PCA分析"""
    print(f"\nUsing {n_components}  principal components for PCA...")
    print(f"  Input matrix: {allele_counts.shape[0]} samples x {allele_counts.shape[1]} variants")
    
    # Handle missing values
    allele_counts_clean = allele_counts.copy().astype(float)
    for i in range(allele_counts_clean.shape[1]):
        col = allele_counts_clean[:, i]
        mask = col != -1
        if np.sum(mask) > 0:
            mean_val = np.mean(col[mask])
            col[~mask] = mean_val
    
    # 标准化
    mean_vals = np.mean(allele_counts_clean, axis=0)
    std_vals = np.std(allele_counts_clean, axis=0)
    std_vals[std_vals == 0] = 1
    
    allele_counts_scaled = (allele_counts_clean - mean_vals) / std_vals
    
    # PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(allele_counts_scaled)
    
    print(f"\nPrincipal Component Variance Explained:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {var:.4f} ({var*100:.2f}%)")
    print(f"  Cumulative: {sum(pca.explained_variance_ratio_):.4f} " +
          f"({sum(pca.explained_variance_ratio_)*100:.2f}%)")
    
    # 返回PCA模型和标准化参数
    normalization_params = {
        'mean': mean_vals,
        'std': std_vals
    }
    
    return pca_result, pca, normalization_params


def plot_pca_results(pca_result, samples, pop_data=None, output_prefix='pca'):
    """绘制PCA结果"""
    print("\nGenerating visualizations...")
    
    n_components = pca_result.shape[1]
    pca_df = pd.DataFrame(
        pca_result,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    pca_df['Sample'] = samples
    
    # 合并人群信息
    if pop_data is not None:
        pca_df['Sample'] = pca_df['Sample'].astype(str)
        pop_data_copy = pop_data.copy()
        pop_data_copy['sample'] = pop_data_copy['sample'].astype(str)
        
        pca_df = pca_df.merge(
            pop_data_copy,
            left_on='Sample',
            right_on='sample',
            how='left'
        )
        
        matched = pca_df['super_pop'].notna().sum()
        if matched > 0:
            print(f"\nPopulation Distribution:")
            print(pca_df['super_pop'].value_counts().to_string())
    
    # Color definitions
    super_pop_colors = {
        'AFR': '#E74C3C',  # red
        'AMR': '#9B59B6',  # purple
        'EAS': '#3498DB',  # blue
        'EUR': '#F39C12',  # orange
        'SAS': '#2ECC71'   # green
    }
    
    # 2D plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    if pop_data is not None and 'super_pop' in pca_df.columns:
        for super_pop in sorted(pca_df['super_pop'].dropna().unique()):
            mask = pca_df['super_pop'] == super_pop
            ax.scatter(
                pca_df.loc[mask, 'PC1'],
                pca_df.loc[mask, 'PC2'],
                label=super_pop,
                alpha=0.7,
                s=50,
                color=super_pop_colors.get(super_pop, '#999999'),
                edgecolors='white',
                linewidths=0.5
            )
        ax.legend(title='Super Population', fontsize=12)
    else:
        ax.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.6, s=50)
    
    ax.set_xlabel('PC1', fontsize=14, fontweight='bold')
    ax.set_ylabel('PC2', fontsize=14, fontweight='bold')
    ax.set_title('1000 Genomes PCA - PC1 vs PC2', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_pc1_pc2.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_prefix}_pc1_pc2.png")
    plt.close()
    
    # 3D static plot
    if n_components >= 3:
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        if pop_data is not None and 'super_pop' in pca_df.columns:
            for super_pop in sorted(pca_df['super_pop'].dropna().unique()):
                mask = pca_df['super_pop'] == super_pop
                ax.scatter(
                    pca_df.loc[mask, 'PC1'],
                    pca_df.loc[mask, 'PC2'],
                    pca_df.loc[mask, 'PC3'],
                    c=super_pop_colors.get(super_pop, '#999999'),
                    label=super_pop,
                    alpha=0.7,
                    s=50
                )
            ax.legend(title='Super Population')
        else:
            ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], alpha=0.6, s=50)
        
        ax.set_xlabel('PC1', fontweight='bold')
        ax.set_ylabel('PC2', fontweight='bold')
        ax.set_zlabel('PC3', fontweight='bold')
        ax.set_title('1000 Genomes PCA - 3D', fontsize=15, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_3d.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_prefix}_3d.png")
        plt.close()
        
        # 交互式3D图
        if PLOTLY_AVAILABLE and pop_data is not None and 'super_pop' in pca_df.columns:
            print("  Generating interactive 3D plot...")
            
            fig_plotly = go.Figure()
            for super_pop in sorted(pca_df['super_pop'].dropna().unique()):
                mask = pca_df['super_pop'] == super_pop
                df_subset = pca_df[mask]
                
                fig_plotly.add_trace(go.Scatter3d(
                    x=df_subset['PC1'],
                    y=df_subset['PC2'],
                    z=df_subset['PC3'],
                    mode='markers',
                    name=super_pop,
                    marker=dict(
                        size=5,
                        color=super_pop_colors.get(super_pop, '#999999'),
                        opacity=0.7
                    ),
                    text=df_subset['Sample'].astype(str),
                    hovertemplate='<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<extra></extra>'
                ))
            
            fig_plotly.update_layout(
                title='1000 Genomes PCA - 交互式3D图',
                scene=dict(
                    xaxis=dict(title='PC1'),
                    yaxis=dict(title='PC2'),
                    zaxis=dict(title='PC3')
                ),
                width=1200,
                height=900
            )
            
            html_file = f'{output_prefix}_3d_interactive.html'
            fig_plotly.write_html(html_file)
            print(f"  Saved: {html_file}")
    
    # 保存CSV
    pca_df.to_csv(f'{output_prefix}_results.csv', index=False)
    print(f"  Saved: {output_prefix}_results.csv")
    
    return pca_df


def main(vcf_file, pop_file=None, output_prefix='pca_ultimate', n_components=3,
         region=None, maf_threshold=0.01, enable_ld_prune=True,
         ld_window_size=500, ld_threshold=0.2, use_cache=True,
         force_refresh=False, thin_bin_size=100, max_per_bin=20, n_jobs=4,
         save_snps=None, save_model=None):
    """
    主函数 - Ultimate Optimized Version
    
    参数:
        n_jobs: LD pruning parallel processes (1=single-thread, -1=all CPUs, default 4)
        save_snps: 保存选中的SNP位置列表的文件名（用于投影）
        save_model: 保存PCA模型和标准化参数的文件名（用于投影）
    """
    print("=" * 80)
    print("1000 Genomes PCA 分析 - Ultimate Optimized Version")
    print("Supports large-scale whole-genome analysis")
    print("=" * 80)
    
    print("\nOptimization features:")
    print("  ✓ Chromosome-wise processing - prevents memory overflow")
    print("  ✓ Smart caching system - resume capability")
    print("  ✓ Spatial pre-filtering - 10x variant reduction before LD pruning")
    print("  ✓ Automatic merging - seamlessly integrates all chromosomes")
    
    print("\nAnalysis Parameters:")
    print(f"  Number of PCs: {n_components}")
    print(f"  MAF threshold: {maf_threshold}")
    print(f"  LD pruning: {'enabled' if enable_ld_prune else 'disabled'}")
    if enable_ld_prune:
        print(f"    Window: {ld_window_size}kb, r²<{ld_threshold}")
        print(f"    Pre-filtering: every {thin_bin_size}kbmax {max_per_bin} SNPs")
    
    # Check if whole-genome analysis
    is_whole_genome = region is None
    
    if is_whole_genome:
        print(f"\n🌍 Whole-genome mode")
        
        # Check merged cache
        merged_cache = generate_merged_cache_filename(maf_threshold)
        if use_cache and not force_refresh and merged_cache.exists():
            print(f"\n✓ Found whole-genome merged cache: {merged_cache.name}")
            try:
                allele_counts, samples, positions = load_chromosome_data(merged_cache)
                print(f"  数据Shape: {allele_counts.shape}")
                print(f"  📂 Using merged cache, skipping all chromosome processing ⚡")
            except Exception as e:
                print(f"  Cache loading failed: {e}，re-processing")
                allele_counts, samples, positions = process_all_chromosomes(
                    vcf_file, maf_threshold, use_cache=use_cache
                )
        else:
            # Chromosome-wise processing
            allele_counts, samples, positions = process_all_chromosomes(
                vcf_file, maf_threshold, use_cache=use_cache
            )
    else:
        print(f"\n📍 Region mode: {region}")
        
        # 单个区域/染色体处理
        from pathlib import Path
        cache_file = CACHE_DIR / f"region_{region.replace(':', '_')}_maf{maf_threshold:.3f}.pkl.gz"
        
        if use_cache and not force_refresh and cache_file.exists():
            print(f"\n✓ 使用缓存")
            allele_counts, samples, positions = load_chromosome_data(cache_file)
        else:
            # 读取和过滤
            allele_counts, samples, positions = read_and_filter_chromosome(
                vcf_file, region, maf_threshold, use_cache=use_cache
            )
    
    # 读取人群信息
    pop_data = None
    if pop_file and os.path.exists(pop_file):
        print(f"\nReading population information: {pop_file}")
        pop_data = pd.read_csv(pop_file, sep='\t')
        pop_data['sample'] = pop_data['sample'].astype(str)
        print(f"  Loaded: {len(pop_data)}  samples")
    
    # LD修剪
    if enable_ld_prune:
        allele_counts, positions = ld_prune_with_thinning(
            allele_counts, positions,
            window_size=ld_window_size,
            threshold=ld_threshold,
            thin_bin_size=thin_bin_size * 1000,
            max_per_bin=max_per_bin,
            n_jobs=n_jobs
        )
    
    # PCA
    pca_result, pca_model, norm_params = run_pca_analysis(allele_counts, n_components)
    
    # 保存SNP位置（用于投影）
    if save_snps and positions is not None:
        print(f"\n💾 Saving SNP position list...")
        np.savez_compressed(save_snps, positions=positions)
        print(f"  Saved: {save_snps}")
        print(f"  Number of sites: {len(positions)}")
    
    # 保存PCA模型和标准化参数（用于投影）
    if save_model:
        print(f"\n💾 Saving PCA model and standardization parameters...")
        model_data = {
            'pca_model': pca_model,
            'mean': norm_params['mean'],
            'std': norm_params['std'],
            'n_components': n_components,
            'n_features': allele_counts.shape[1]
        }
        with open(save_model, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"  Saved: {save_model}")
        print(f"  Number of PCs: {n_components}")
        print(f"  Number of features: {allele_counts.shape[1]}")
    
    # Visualization
    pca_df = plot_pca_results(pca_result, samples, pop_data, output_prefix)
    
    print("\n" + "=" * 80)
    print("Analysis completed!")
    print("=" * 80)
    
    if save_snps or save_model:
        print("\n📦 Saved files for projection:")
        if save_snps:
            print(f"  ✓ SNP positions: {save_snps}")
        if save_model:
            print(f"  ✓ PCA model: {save_model}")
        print(f"\n💡 Use the following command to project new samples:")
        print(f"   python3 pca_projection.py \\")
        print(f"       --snps {save_snps or 'snps.npz'} \\")
        print(f"       --model {save_model or 'model.pkl'} \\")
        print(f"       --reference {output_prefix}_results.csv \\")
        print(f"       --query-vcf YOUR_VCF.gz \\")
        print(f"       --output query_projection")
    
    return pca_df, pca_model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='1000 Genomes PCA Ultimate Optimized Version - Supports large-scale whole-genome analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Whole-genome analysis (automatic chromosome-wise processing)
  python3 1000genomes_pca_ultimate.py \\
      all.1kg.phase3_shapeit2_mvncall_integrated_v1b.20130502.vcf.gz \\
      integrated_call_samples_v3.20130502.ALL.panel \\
      --maf 0.05 \\
      --output full_genome
  
  # Single chromosome
  python3 1000genomes_pca_ultimate.py \\
      all.1kg.phase3_shapeit2_mvncall_integrated_v1b.20130502.vcf.gz \\
      integrated_call_samples_v3.20130502.ALL.panel \\
      --region 20 \\
      --output chr20
  
  # Custom spatial filtering parameters
  python3 1000genomes_pca_ultimate.py \\
      all.1kg.phase3_shapeit2_mvncall_integrated_v1b.20130502.vcf.gz \\
      integrated_call_samples_v3.20130502.ALL.panel \\
      --region 20 \\
      --thin-bin 50 --max-per-bin 30
  
  # Use multi-processing for LD pruning (recommended)
  python3 1000genomes_pca_ultimate.py \\
      all.1kg.phase3_shapeit2_mvncall_integrated_v1b.20130502.vcf.gz \\
      integrated_call_samples_v3.20130502.ALL.panel \\
      --maf 0.15 --max-per-bin 10 \\
      --n-jobs 8 \\
      --output fast_result

Optimization features:
  ✓ Chromosome-wise processing - prevents whole-genome memory overflow
  ✓ Smart caching - independent cache per chromosome, supports resume
  ✓ Spatial pre-filtering - 90% variant reduction before LD pruning, prevents memory overflow
  ✓ Automatic merging - seamlessly integrates all chromosomes数据
  ✓ Memory optimized - suitable for whole-genome analysis on 8GB RAM machines
        """
    )
    
    parser.add_argument('vcf_file', help='VCF file path')
    parser.add_argument('pop_file', nargs='?', help='Population information file')
    parser.add_argument('-o', '--output', default='pca_ultimate',
                       help='Output prefix')
    parser.add_argument('-n', '--n-components', type=int, default=3,
                       help='Number of principal components')
    parser.add_argument('-r', '--region',
                       help='Chromosome region (leave empty for whole-genome)')
    parser.add_argument('--maf', type=float, default=0.01,
                       help='MAF threshold')
    parser.add_argument('--no-ld-prune', action='store_true',
                       help='disabledLD修剪')
    parser.add_argument('--ld-window', type=int, default=500,
                       help='LD window size (kb)')
    parser.add_argument('--ld-threshold', type=float, default=0.2,
                       help='LD r² threshold')
    parser.add_argument('--thin-bin', type=int, default=100,
                       help='Spatial bin size (kb), default 100')
    parser.add_argument('--max-per-bin', type=int, default=20,
                       help='每个区间max SNP数，默认20')
    parser.add_argument('--n-jobs', type=int, default=4,
                       help='LD pruning parallel processes (1=single-thread, -1=all CPUs, default 4)')
    parser.add_argument('--no-cache', action='store_true',
                       help='disabled缓存')
    parser.add_argument('--force-refresh', action='store_true',
                       help='Force refresh, ignore cache')
    parser.add_argument('--save-snps',
                       help='Save selected SNP positions (for projecting new samples)')
    parser.add_argument('--save-model',
                       help='Save PCA model and standardization params (for projecting new samples)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.vcf_file):
        print(f"Error: VCF file does not exist: {args.vcf_file}")
        exit(1)
    
    try:
        main(
            vcf_file=args.vcf_file,
            pop_file=args.pop_file,
            output_prefix=args.output,
            n_components=args.n_components,
            region=args.region,
            maf_threshold=args.maf,
            enable_ld_prune=not args.no_ld_prune,
            ld_window_size=args.ld_window,
            ld_threshold=args.ld_threshold,
            use_cache=not args.no_cache,
            force_refresh=args.force_refresh,
            thin_bin_size=args.thin_bin,
            max_per_bin=args.max_per_bin,
            n_jobs=args.n_jobs,
            save_snps=args.save_snps,
            save_model=args.save_model
        )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
