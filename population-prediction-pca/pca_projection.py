#!/usr/bin/env python3
"""
PCA投影脚本 - 将新样本投影到已训练的PCA空间

用于祖源推断：将未知种族的样本投影到1000 Genomes PCA空间，
并推断其种族归属。
"""

import numpy as np
import pandas as pd
import allel
import pickle
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
import traceback

# 尝试导入plotly
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def load_model_and_snps(model_file, snps_file):
    """
    加载PCA模型和SNP位置列表
    """
    print("=" * 80)
    print("加载PCA模型和SNP位置...")
    print("=" * 80)
    
    # 加载模型
    print(f"\n📂 加载PCA模型: {model_file}")
    with open(model_file, 'rb') as f:
        model_data = pickle.load(f)
    
    pca_model = model_data['pca_model']
    mean_vals = model_data['mean']
    std_vals = model_data['std']
    n_components = model_data['n_components']
    n_features = model_data['n_features']
    
    print(f"  ✓ 主成分数: {n_components}")
    print(f"  ✓ 特征数: {n_features}")
    print(f"  ✓ 方差解释率: {sum(pca_model.explained_variance_ratio_)*100:.2f}%")
    
    # 加载SNP位置
    print(f"\n📂 加载SNP位置列表: {snps_file}")
    snp_data = np.load(snps_file)
    positions = snp_data['positions']
    
    print(f"  ✓ SNP数量: {len(positions)}")
    
    # 检查positions和n_features是否一致
    if len(positions) != n_features:
        print(f"\n❌ 错误: SNP位置数({len(positions)})与模型特征数({n_features})不一致")
        print(f"\n可能原因:")
        print(f"  1. 使用的是旧版本训练脚本生成的模型文件")
        print(f"  2. snps.npz 和 model.pkl 来自不同的训练运行")
        print(f"\n解决方案:")
        print(f"  使用最新版本的训练脚本重新训练（已修复此bug）:")
        print(f"  python3 1000genomes_pca_ultimate.py ... \\")
        print(f"      --save-snps snps.npz --save-model model.pkl")
        
        raise ValueError(f"SNP位置列表与模型不匹配，请重新训练")
    
    return pca_model, mean_vals, std_vals, positions


def normalize_chrom_name(chrom):
    """标准化染色体名称，去除chr前缀"""
    chrom_str = str(chrom)
    if chrom_str.startswith('chr'):
        return chrom_str[3:]
    return chrom_str


def decode_position_with_chrom(position_offset):
    """
    从offset编码的位置解码出染色体和实际位置
    
    训练时编码规则：
    position_offset = chromosome * 1_000_000_000 + actual_position
    """
    chrom = int(position_offset // 1_000_000_000)
    actual_pos = int(position_offset % 1_000_000_000)
    return chrom, actual_pos


def extract_genotypes_from_vcf(vcf_file, positions, region=None):
    """
    从VCF文件提取指定位置的基因型
    支持自动处理染色体命名差异（chr1 vs 1）
    
    参数:
        vcf_file: VCF文件路径
        positions: 需要提取的位置列表（已编码offset）
        region: 染色体区域（可选）
    
    返回:
        genotypes: 基因型矩阵 (samples x variants)
        samples: 样本ID列表
        matched_positions: 实际匹配到的位置
    """
    print("\n" + "=" * 80)
    print("从VCF提取基因型数据...")
    print("=" * 80)
    
    print(f"\n📂 读取VCF: {vcf_file}")
    if region:
        print(f"  区域: {region}")
    
    # 读取VCF
    try:
        callset = allel.read_vcf(vcf_file, region=region)
    except Exception as e:
        raise ValueError(f"读取VCF失败: {e}")
    
    if callset is None or 'calldata/GT' not in callset:
        raise ValueError("VCF文件无数据或缺少基因型信息")
    
    # 获取VCF中的染色体和位置
    vcf_chroms = callset['variants/CHROM']
    vcf_positions = callset['variants/POS'].astype(np.int64)
    genotypes = allel.GenotypeArray(callset['calldata/GT'])
    samples = callset.get('samples', None)
    
    if samples is None:
        raise ValueError("VCF文件缺少样本信息")
    
    # 确保samples是字符串
    if isinstance(samples[0], bytes):
        samples = np.array([s.decode('utf-8') for s in samples])
    else:
        samples = np.array([str(s) for s in samples])
    
    print(f"  VCF中的样本数: {len(samples)}")
    print(f"  VCF中的变异数: {len(vcf_positions)}")
    
    # 检测VCF染色体命名格式
    first_chrom = vcf_chroms[0].decode('utf-8') if isinstance(vcf_chroms[0], bytes) else str(vcf_chroms[0])
    vcf_has_chr_prefix = first_chrom.startswith('chr')
    
    print(f"  染色体命名格式: {first_chrom} ({'有chr前缀' if vcf_has_chr_prefix else '无chr前缀'})")
    
    # 检测参考位置的格式（从第一个位点解码）
    ref_chrom_example, ref_pos_example = decode_position_with_chrom(positions[0])
    print(f"  参考数据格式: 染色体{ref_chrom_example} (无chr前缀)")
    
    # 决定是否需要转换
    need_normalize = vcf_has_chr_prefix
    
    if need_normalize:
        print(f"  🔄 自动转换: 将VCF的'chr{ref_chrom_example}'格式转为'{ref_chrom_example}'格式")
    else:
        print(f"  ✓ 命名格式一致，无需转换")
    
    # 标准化VCF的染色体名称（如果需要）
    if need_normalize:
        if isinstance(vcf_chroms[0], bytes):
            vcf_chroms_normalized = np.array([normalize_chrom_name(c.decode('utf-8')) for c in vcf_chroms])
        else:
            vcf_chroms_normalized = np.array([normalize_chrom_name(c) for c in vcf_chroms])
    else:
        if isinstance(vcf_chroms[0], bytes):
            vcf_chroms_normalized = np.array([c.decode('utf-8') for c in vcf_chroms])
        else:
            vcf_chroms_normalized = np.array([str(c) for c in vcf_chroms])
    
    # 创建VCF的染色体+位置索引
    vcf_chrom_pos_dict = {}
    for i, (chrom, pos) in enumerate(zip(vcf_chroms_normalized, vcf_positions)):
        key = (chrom, pos)
        vcf_chrom_pos_dict[key] = i
    
    # 匹配位置
    print(f"\n🔍 匹配SNP位置...")
    print(f"  参考位点数: {len(positions)}")
    
    matched_indices = []
    matched_positions = []
    
    print(f"  正在匹配位点（自动处理染色体命名差异）...")
    for i, pos_offset in tqdm(enumerate(positions), total=len(positions), desc="  匹配进度"):
        # 从offset解码出染色体和实际位置
        chrom, actual_pos = decode_position_with_chrom(pos_offset)
        chrom_str = str(chrom)
        
        # 在VCF中查找
        key = (chrom_str, actual_pos)
        if key in vcf_chrom_pos_dict:
            vcf_idx = vcf_chrom_pos_dict[key]
            matched_indices.append((i, vcf_idx))
            matched_positions.append(pos_offset)
    
    n_matched = len(matched_indices)
    match_rate = n_matched / len(positions)
    
    print(f"  匹配位点数: {n_matched}")
    print(f"  匹配率: {match_rate*100:.1f}%")
    
    if match_rate < 0.5:
        print(f"\n⚠️  警告: 匹配率过低 ({match_rate*100:.1f}%)")
        print(f"  建议: 检查VCF文件是否使用相同的参考基因组")
    elif match_rate < 0.8:
        print(f"\n⚠️  注意: 匹配率较低 ({match_rate*100:.1f}%)，结果可能不够准确")
    
    if n_matched == 0:
        raise ValueError("没有匹配的SNP位点！请检查VCF文件和参考位点列表")
    
    # 提取匹配位点的基因型
    print(f"\n📊 提取基因型数据...")
    ref_indices = [idx[0] for idx in matched_indices]
    vcf_indices = [idx[1] for idx in matched_indices]
    
    # 创建完整的基因型矩阵（包含缺失位点）
    n_samples = len(samples)
    n_ref_snps = len(positions)
    full_genotypes = np.full((n_samples, n_ref_snps), -1, dtype=np.int8)
    
    # 填充匹配的位点
    matched_genotypes = genotypes[vcf_indices].to_n_alt().T
    for i, ref_idx in enumerate(ref_indices):
        full_genotypes[:, ref_idx] = matched_genotypes[:, i]
    
    print(f"  提取完成: {n_samples} 样本 × {n_ref_snps} 位点")
    print(f"  缺失位点数: {n_ref_snps - n_matched}")
    
    return full_genotypes, samples, np.array(matched_positions)


def project_to_pca(genotypes, pca_model, mean_vals, std_vals):
    """
    将新样本投影到PCA空间
    
    参数:
        genotypes: 基因型矩阵 (samples x variants)
        pca_model: 训练好的PCA模型
        mean_vals: 标准化均值
        std_vals: 标准化标准差
    
    返回:
        pca_coords: PCA坐标 (samples x n_components)
    """
    print("\n" + "=" * 80)
    print("投影到PCA空间...")
    print("=" * 80)
    
    n_samples, n_variants = genotypes.shape
    print(f"\n输入: {n_samples} 样本 × {n_variants} 变异")
    
    # 处理缺失值（用均值填充）
    print(f"\n处理缺失值...")
    genotypes_clean = genotypes.copy().astype(float)
    
    missing_count = np.sum(genotypes_clean == -1)
    missing_rate = missing_count / (n_samples * n_variants)
    print(f"  缺失值数量: {missing_count}")
    print(f"  缺失率: {missing_rate*100:.2f}%")
    
    print(f"  填充缺失值...")
    for i in tqdm(range(n_variants), desc="  填充进度"):
        col = genotypes_clean[:, i]
        mask = col != -1
        if np.sum(mask) > 0:
            mean_val = np.mean(col[mask])
            col[~mask] = mean_val
        else:
            # 如果整列都缺失，用参考均值填充
            col[:] = mean_vals[i]
    
    # 应用与参考人群相同的标准化
    print(f"\n应用标准化...")
    genotypes_scaled = (genotypes_clean - mean_vals) / std_vals
    
    # 投影
    print(f"\n执行PCA投影...")
    pca_coords = pca_model.transform(genotypes_scaled)
    
    print(f"  ✓ 投影完成")
    print(f"  输出: {pca_coords.shape[0]} 样本 × {pca_coords.shape[1]} 主成分")
    
    return pca_coords


def infer_ancestry(query_pca, reference_pca, reference_populations, n_neighbors=20):
    """
    使用KNN推断种族归属
    
    参数:
        query_pca: 查询样本的PCA坐标
        reference_pca: 参考人群的PCA坐标
        reference_populations: 参考人群的种族标签
        n_neighbors: KNN的邻居数
    
    返回:
        predictions: 预测的种族
        probabilities: 各种族的概率
    """
    print("\n" + "=" * 80)
    print("推断种族归属...")
    print("=" * 80)
    
    print(f"\n使用KNN分类器 (K={n_neighbors})")
    print(f"  参考样本数: {len(reference_pca)}")
    print(f"  查询样本数: {len(query_pca)}")
    
    # 训练KNN
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(reference_pca, reference_populations)
    
    # 预测
    predictions = knn.predict(query_pca)
    probabilities = knn.predict_proba(query_pca)
    
    # 统计
    unique_pops, counts = np.unique(predictions, return_counts=True)
    print(f"\n预测结果统计:")
    for pop, count in zip(unique_pops, counts):
        print(f"  {pop}: {count} 样本 ({count/len(predictions)*100:.1f}%)")
    
    return predictions, probabilities, knn.classes_


def generate_ancestry_report(samples, predictions, probabilities, pop_labels, output_file):
    """
    生成详细的祖源推断报告
    """
    print(f"\n生成祖源推断报告: {output_file}")
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("祖源推断报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"样本数: {len(samples)}\n")
        f.write(f"参考人群: {', '.join(pop_labels)}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("详细结果\n")
        f.write("=" * 80 + "\n\n")
        
        for i, sample in enumerate(samples):
            pred_pop = predictions[i]
            probs = probabilities[i]
            confidence = np.max(probs)
            
            f.write(f"样本: {sample}\n")
            f.write(f"  预测种族: {pred_pop}\n")
            f.write(f"  置信度: {confidence*100:.1f}%\n")
            f.write(f"  概率分布:\n")
            
            for pop, prob in zip(pop_labels, probs):
                f.write(f"    {pop}: {prob*100:.1f}%\n")
            
            # 判断是否为混合人群
            sorted_probs = sorted(zip(pop_labels, probs), key=lambda x: x[1], reverse=True)
            if sorted_probs[1][1] > 0.2:  # 第二高的概率>20%
                f.write(f"  注释: 可能为混合人群 ({sorted_probs[0][0]}/{sorted_probs[1][0]})\n")
            
            f.write("\n")
    
    print(f"  ✓ 报告已保存")


def plot_combined_pca(reference_pca, reference_populations, query_pca, 
                      query_samples, predictions, probabilities,
                      output_prefix='projection'):
    """
    绘制参考人群和查询样本的组合PCA图
    """
    print("\n" + "=" * 80)
    print("生成可视化图表...")
    print("=" * 80)
    
    # 颜色定义
    pop_colors = {
        'AFR': '#E74C3C',  # 红色
        'AMR': '#9B59B6',  # 紫色
        'EAS': '#3498DB',  # 蓝色
        'EUR': '#F39C12',  # 橙色
        'SAS': '#2ECC71'   # 绿色
    }
    
    # 2D图
    print(f"\n生成2D散点图...")
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # 绘制参考人群（小圆点）
    for pop in np.unique(reference_populations):
        mask = reference_populations == pop
        ax.scatter(
            reference_pca[mask, 0],
            reference_pca[mask, 1],
            c=pop_colors.get(pop, '#999999'),
            label=f'{pop} (参考)',
            alpha=0.4,
            s=30,
            marker='o'
        )
    
    # 绘制查询样本（星号）
    for pop in np.unique(predictions):
        mask = predictions == pop
        ax.scatter(
            query_pca[mask, 0],
            query_pca[mask, 1],
            c=pop_colors.get(pop, '#999999'),
            label=f'{pop} (查询)',
            alpha=0.9,
            s=150,
            marker='*',
            edgecolors='black',
            linewidths=1.5
        )
    
    ax.set_xlabel('PC1', fontsize=14, fontweight='bold')
    ax.set_ylabel('PC2', fontsize=14, fontweight='bold')
    ax.set_title('PCA投影 - 参考人群与查询样本', fontsize=16, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = f'{output_prefix}_2d.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  保存: {output_file}")
    plt.close()
    
    # 3D图
    if reference_pca.shape[1] >= 3:
        print(f"\n生成3D散点图...")
        fig = plt.figure(figsize=(18, 14))
        ax = fig.add_subplot(111, projection='3d')
        
        # 参考人群
        for pop in np.unique(reference_populations):
            mask = reference_populations == pop
            ax.scatter(
                reference_pca[mask, 0],
                reference_pca[mask, 1],
                reference_pca[mask, 2],
                c=pop_colors.get(pop, '#999999'),
                label=f'{pop} (参考)',
                alpha=0.3,
                s=30,
                marker='o'
            )
        
        # 查询样本
        for pop in np.unique(predictions):
            mask = predictions == pop
            ax.scatter(
                query_pca[mask, 0],
                query_pca[mask, 1],
                query_pca[mask, 2],
                c=pop_colors.get(pop, '#999999'),
                label=f'{pop} (查询)',
                alpha=0.9,
                s=150,
                marker='*',
                edgecolors='black',
                linewidths=1.5
            )
        
        ax.set_xlabel('PC1', fontweight='bold')
        ax.set_ylabel('PC2', fontweight='bold')
        ax.set_zlabel('PC3', fontweight='bold')
        ax.set_title('PCA投影 - 3D视图', fontsize=16, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        
        plt.tight_layout()
        output_file = f'{output_prefix}_3d.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  保存: {output_file}")
        plt.close()
        
        # 交互式3D图
        if PLOTLY_AVAILABLE:
            print(f"\n生成交互式3D图...")
            fig_plotly = go.Figure()
            
            # 参考人群
            for pop in np.unique(reference_populations):
                mask = reference_populations == pop
                fig_plotly.add_trace(go.Scatter3d(
                    x=reference_pca[mask, 0],
                    y=reference_pca[mask, 1],
                    z=reference_pca[mask, 2],
                    mode='markers',
                    name=f'{pop} (参考)',
                    marker=dict(
                        size=4,
                        color=pop_colors.get(pop, '#999999'),
                        opacity=0.4
                    ),
                    hovertemplate=f'{pop}<br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<br>PC3: %{{z:.3f}}<extra></extra>'
                ))
            
            # 查询样本
            for pop in np.unique(predictions):
                mask = predictions == pop
                samples_subset = query_samples[mask]
                fig_plotly.add_trace(go.Scatter3d(
                    x=query_pca[mask, 0],
                    y=query_pca[mask, 1],
                    z=query_pca[mask, 2],
                    mode='markers',
                    name=f'{pop} (查询)',
                    marker=dict(
                        size=8,
                        color=pop_colors.get(pop, '#999999'),
                        opacity=0.9,
                        symbol='diamond',
                        line=dict(color='black', width=1)
                    ),
                    text=samples_subset,
                    hovertemplate='<b>%{text}</b><br>预测: ' + pop + '<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<extra></extra>'
                ))
            
            fig_plotly.update_layout(
                title='PCA投影 - 交互式3D图',
                scene=dict(
                    xaxis=dict(title='PC1'),
                    yaxis=dict(title='PC2'),
                    zaxis=dict(title='PC3')
                ),
                width=1400,
                height=1000
            )
            
            output_file = f'{output_prefix}_3d_interactive.html'
            fig_plotly.write_html(output_file)
            print(f"  保存: {output_file}")


def process_vcf_worker(args):
    """
    多进程Worker函数 - 处理单个VCF文件
    
    参数:
        args: 元组，包含所有process_single_vcf需要的参数
    
    返回:
        (vcf_name, simple_results, success, error_msg)
    """
    (vcf_file, output_path, snps_file, model_file, reference_csv,
     reference_pca, reference_populations, pca_model, mean_vals, 
     std_vals, positions, pop_labels, region, n_neighbors) = args
    
    vcf_basename = vcf_file.name.split('.')[0].split('-')[0].split('_')[0]
    output_subdir = output_path / vcf_basename
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    try:
        simple_results, query_df = process_single_vcf(
            snps_file, model_file, reference_csv, str(vcf_file),
            output_subdir, reference_pca, reference_populations,
            pca_model, mean_vals, std_vals, positions, pop_labels,
            region, n_neighbors
        )
        return (vcf_file.name, simple_results, True, None)
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        return (vcf_file.name, [], False, error_msg)


def process_single_vcf(snps_file, model_file, reference_csv, query_vcf,
                      output_subdir, reference_pca, reference_populations,
                      pca_model, mean_vals, std_vals, positions, pop_labels,
                      region=None, n_neighbors=20):
    """
    处理单个VCF文件
    """
    vcf_basename = Path(query_vcf).name.split('.')[0].split('-')[0].split('_')[0]
    
    print(f"\n{'='*80}")
    print(f"处理VCF: {Path(query_vcf).name}")
    print(f"{'='*80}")
    
    # 提取查询样本的基因型
    query_genotypes, query_samples, matched_positions = extract_genotypes_from_vcf(
        query_vcf, positions, region
    )
    
    # 投影到PCA空间
    query_pca = project_to_pca(query_genotypes, pca_model, mean_vals, std_vals)
    
    # 推断种族
    predictions, probabilities, _ = infer_ancestry(
        query_pca, reference_pca, reference_populations, n_neighbors
    )
    
    # 保存结果
    print("\n" + "=" * 80)
    print(f"保存结果到: {output_subdir}")
    print("=" * 80)
    
    # 完整CSV结果
    query_df = pd.DataFrame(query_pca, columns=[f'PC{i+1}' for i in range(query_pca.shape[1])])
    query_df['Sample'] = query_samples
    query_df['Predicted_Pop'] = predictions
    query_df['Confidence'] = np.max(probabilities, axis=1)
    
    for i, pop in enumerate(pop_labels):
        query_df[f'{pop}_prob'] = probabilities[:, i]
    
    csv_file = output_subdir / 'detailed_results.csv'
    query_df.to_csv(csv_file, index=False)
    print(f"  保存: {csv_file.name}")
    
    # 简洁的预测结果文件
    simple_results = []
    
    # 从VCF文件名提取样本名
    vcf_file_path = Path(query_vcf)
    base_name = vcf_file_path.name.split('.vcf')[0]  # 去除.vcf.gz或.vcf扩展名
    # 进一步简化：取第一个分隔符之前的部分
    simple_vcf_name = base_name.split('.')[0].split('-')[0].split('_')[0]
    
    # 如果VCF只有1个样本，直接用文件名；如果有多个样本，用"文件名_样本序号"
    if len(query_samples) == 1:
        simple_results.append((simple_vcf_name, predictions[0]))
    else:
        for idx, (sample, pred) in enumerate(zip(query_samples, predictions), 1):
            # 多样本VCF：使用"文件名_样本序号"格式
            simple_name = f"{simple_vcf_name}_sample{idx}"
            simple_results.append((simple_name, pred))
    
    simple_file = output_subdir / f'{vcf_basename}_predictions.txt'
    with open(simple_file, 'w') as f:
        f.write("Sample\tPredicted_Population\n")
        for name, pred in simple_results:
            f.write(f"{name}\t{pred}\n")
    print(f"  保存: {simple_file.name}")
    
    # 详细报告
    report_file = output_subdir / 'ancestry_report.txt'
    generate_ancestry_report(query_samples, predictions, probabilities, pop_labels, report_file)
    
    # 可视化
    plot_combined_pca(
        reference_pca, reference_populations,
        query_pca, query_samples, predictions, probabilities,
        str(output_subdir / 'pca')
    )
    
    return simple_results, query_df


def main(snps_file, model_file, reference_csv, query_vcf_or_dir, 
         output_dir='projection_output', region=None, n_neighbors=20, n_jobs=1):
    """
    主函数 - 支持单个VCF或VCF文件夹
    
    参数:
        query_vcf_or_dir: VCF文件路径 或 包含VCF文件的文件夹
        output_dir: 输出文件夹
        n_jobs: 并行进程数（默认1，单进程）
    """
    print("\n" + "=" * 80)
    print("PCA投影与祖源推断")
    print("=" * 80)
    
    # 创建输出文件夹
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 输出文件夹: {output_path.absolute()}")
    
    # 检测输入是文件还是文件夹
    input_path = Path(query_vcf_or_dir)
    
    if input_path.is_file():
        # 单个VCF文件
        vcf_files = [input_path]
        print(f"\n📄 输入: 单个VCF文件")
        print(f"   {input_path.name}")
    elif input_path.is_dir():
        # VCF文件夹
        vcf_files = sorted(list(input_path.glob('*.vcf.gz')) + list(input_path.glob('*.vcf')))
        if len(vcf_files) == 0:
            raise ValueError(f"文件夹中没有找到VCF文件: {input_path}")
        print(f"\n📁 输入: VCF文件夹")
        print(f"   找到 {len(vcf_files)} 个VCF文件")
        for vcf in vcf_files:
            print(f"   - {vcf.name}")
    else:
        raise ValueError(f"输入路径不存在: {input_path}")
    
    # 1. 加载模型和SNP位置（只需加载一次）
    pca_model, mean_vals, std_vals, positions = load_model_and_snps(model_file, snps_file)
    
    # 2. 加载参考人群PCA结果（只需加载一次）
    print("\n" + "=" * 80)
    print("加载参考人群数据...")
    print("=" * 80)
    print(f"\n📂 读取: {reference_csv}")
    reference_df = pd.read_csv(reference_csv)
    
    # 提取PCA坐标和种族信息
    pc_cols = [col for col in reference_df.columns if col.startswith('PC')]
    reference_pca = reference_df[pc_cols].values
    reference_populations = reference_df['super_pop'].values if 'super_pop' in reference_df.columns else None
    
    if reference_populations is None:
        raise ValueError("参考数据缺少'super_pop'列")
    
    print(f"  参考样本数: {len(reference_df)}")
    print(f"  主成分数: {len(pc_cols)}")
    print(f"  人群分布:")
    for pop, count in reference_df['super_pop'].value_counts().items():
        print(f"    {pop}: {count}")
    
    # 检查并过滤缺失值
    has_missing = reference_df['super_pop'].isna().sum()
    if has_missing > 0:
        print(f"  ⚠️  警告: {has_missing} 个样本缺少人群信息，已过滤")
        # 只使用有人群信息的样本
        valid_mask = reference_df['super_pop'].notna()
        reference_df_filtered = reference_df[valid_mask].copy()
        reference_pca = reference_df_filtered[pc_cols].values
        reference_populations = reference_df_filtered['super_pop'].values
    
    # 获取人群标签（过滤NaN后）
    pop_labels = sorted(reference_df['super_pop'].dropna().unique())
    
    # 3. 处理所有VCF文件
    all_predictions = []  # 存储所有样本的预测结果
    
    print(f"\n" + "=" * 80)
    print(f"开始处理 {len(vcf_files)} 个VCF文件...")
    if n_jobs > 1 and len(vcf_files) > 1:
        print(f"🚀 使用多进程加速 (进程数: {n_jobs})")
    print("=" * 80)
    
    if n_jobs > 1 and len(vcf_files) > 1:
        # 多进程并行处理
        print(f"\n准备 {len(vcf_files)} 个任务...")
        
        # 准备worker参数
        worker_args = [
            (vcf_file, output_path, snps_file, model_file, reference_csv,
             reference_pca, reference_populations, pca_model, mean_vals,
             std_vals, positions, pop_labels, region, n_neighbors)
            for vcf_file in vcf_files
        ]
        
        # 使用进程池处理
        print(f"启动 {n_jobs} 个进程...")
        with Pool(processes=n_jobs) as pool:
            # 使用imap_unordered以便及时获取结果
            results = []
            for result in tqdm(pool.imap_unordered(process_vcf_worker, worker_args),
                             total=len(vcf_files), desc="总体进度"):
                results.append(result)
        
        # 处理结果
        success_count = 0
        fail_count = 0
        
        print(f"\n" + "=" * 80)
        print("处理结果:")
        print("=" * 80)
        
        for vcf_name, simple_results, success, error_msg in results:
            if success:
                all_predictions.extend(simple_results)
                success_count += 1
                print(f"  ✅ {vcf_name}: {len(simple_results)} 个样本")
            else:
                fail_count += 1
                print(f"  ❌ {vcf_name}: {error_msg}")
        
        print(f"\n总计: 成功 {success_count}, 失败 {fail_count}")
        
    else:
        # 单进程顺序处理
        if n_jobs > 1:
            print(f"  ℹ️  只有1个VCF文件，使用单进程处理")
        
        for i, vcf_file in enumerate(vcf_files, 1):
            print(f"\n[{i}/{len(vcf_files)}] {vcf_file.name}")
            
            # 为每个VCF创建子文件夹
            vcf_basename = vcf_file.name.split('.')[0].split('-')[0].split('_')[0]
            output_subdir = output_path / vcf_basename
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            try:
                # 处理单个VCF
                simple_results, query_df = process_single_vcf(
                    snps_file, model_file, reference_csv, str(vcf_file),
                    output_subdir, reference_pca, reference_populations,
                    pca_model, mean_vals, std_vals, positions, pop_labels,
                    region, n_neighbors
                )
                
                # 收集预测结果
                all_predictions.extend(simple_results)
                
                print(f"  ✅ 完成: {len(simple_results)} 个样本")
                
            except Exception as e:
                print(f"  ❌ 处理失败: {e}")
                traceback.print_exc()
                continue
    
    # 4. 生成总结果文件
    if len(all_predictions) > 0:
        print("\n" + "=" * 80)
        print("生成总结果文件...")
        print("=" * 80)
        
        summary_file = output_path / 'all_samples_predictions.txt'
        with open(summary_file, 'w') as f:
            f.write("Sample\tPredicted_Population\n")
            for name, pred in all_predictions:
                f.write(f"{name}\t{pred}\n")
        
        print(f"\n📄 总结果文件: {summary_file.name}")
        print(f"   包含 {len(all_predictions)} 个样本")
        
        # 统计各人群数量
        from collections import Counter
        pop_counts = Counter([pred for _, pred in all_predictions])
        print(f"\n人群分布统计:")
        for pop in sorted(pop_counts.keys()):
            count = pop_counts[pop]
            print(f"  {pop}: {count} ({count/len(all_predictions)*100:.1f}%)")
    
    # 5. 最终总结
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)
    print(f"\n📁 输出文件夹: {output_path.absolute()}")
    print(f"\n核心文件:")
    print(f"  🎯 all_samples_predictions.txt - 总结果文件 ⭐⭐⭐")
    print(f"     (包含所有{len(all_predictions)}个样本的预测结果)")
    
    if len(vcf_files) > 1:
        print(f"\n各VCF子文件夹:")
        for vcf_file in vcf_files:
            vcf_basename = vcf_file.name.split('.')[0].split('-')[0].split('_')[0]
            print(f"  📂 {vcf_basename}/")
            print(f"     - {vcf_basename}_predictions.txt")
            print(f"     - detailed_results.csv")
            print(f"     - ancestry_report.txt")
            print(f"     - pca_2d.png, pca_3d.png")
    
    print(f"\n💡 快速查看总结果:")
    print(f"   cat {output_path}/all_samples_predictions.txt")
    print(f"\n💡 统计各人群数量:")
    print(f"   cut -f2 {output_path}/all_samples_predictions.txt | tail -n +2 | sort | uniq -c")
    
    return all_predictions


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='PCA投影与祖源推断 - 将新样本投影到已训练的PCA空间（支持单文件或批量处理）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

  # 1. 处理单个VCF文件
  python3 pca_projection.py \\
      --snps selected_snps.npz \\
      --model pca_model.pkl \\
      --reference reference_pca_results.csv \\
      --query-vcf new_cohort.vcf.gz \\
      --output-dir query_results
  
  # 2. 批量处理VCF文件夹（自动处理文件夹中所有VCF）🆕
  python3 pca_projection.py \\
      --snps selected_snps.npz \\
      --model pca_model.pkl \\
      --reference reference_pca_results.csv \\
      --query-vcf vcf_folder/ \\
      --output-dir batch_results
  
  # 3. 批量处理+多进程加速（推荐）🚀
  python3 pca_projection.py \\
      --snps selected_snps.npz \\
      --model pca_model.pkl \\
      --reference reference_pca_results.csv \\
      --query-vcf vcf_folder/ \\
      --output-dir batch_results \\
      --n-jobs 8
  
  # 4. 指定染色体区域
  python3 pca_projection.py \\
      --snps selected_snps.npz \\
      --model pca_model.pkl \\
      --reference reference_pca_results.csv \\
      --query-vcf new_cohort.vcf.gz \\
      --region 20 \\
      --output-dir query_results
  
  # 5. 调整KNN邻居数
  python3 pca_projection.py \\
      --snps selected_snps.npz \\
      --model pca_model.pkl \\
      --reference reference_pca_results.csv \\
      --query-vcf new_cohort.vcf.gz \\
      --n-neighbors 30 \\
      --output-dir query_results

输出文件结构:

  单个VCF文件:
    output_dir/
    ├── all_samples_predictions.txt   # 总结果文件 ⭐⭐⭐
    ├── <vcf>/                        # VCF子文件夹
    │   ├── <vcf>_predictions.txt     # 该VCF的预测结果
    │   ├── detailed_results.csv      # 完整结果（PCA坐标+概率）
    │   ├── ancestry_report.txt       # 详细报告
    │   ├── pca_2d.png               # 2D可视化
    │   ├── pca_3d.png               # 3D可视化
    │   └── pca_3d_interactive.html  # 交互式3D图
  
  批量处理VCF文件夹:
    output_dir/
    ├── all_samples_predictions.txt   # 总结果文件（所有样本）⭐⭐⭐
    ├── vcf1/                        # 第1个VCF的结果
    ├── vcf2/                        # 第2个VCF的结果
    └── vcf3/                        # 第3个VCF的结果

工作流程:
  1. 训练PCA模型（使用1000genomes_pca_ultimate.py）
     python3 1000genomes_pca_ultimate.py ... --save-snps snps.npz --save-model model.pkl
  
  2. 投影新样本（使用本脚本）
     python3 pca_projection.py --snps snps.npz --model model.pkl --reference ref.csv --query-vcf new.vcf.gz
        """
    )
    
    parser.add_argument('--snps', required=True,
                       help='SNP位置列表文件（.npz格式）')
    parser.add_argument('--model', required=True,
                       help='PCA模型文件（.pkl格式）')
    parser.add_argument('--reference', required=True,
                       help='参考人群PCA结果CSV文件')
    parser.add_argument('--query-vcf', required=True,
                       help='查询样本的VCF文件 或 包含VCF文件的文件夹（支持批量处理）')
    parser.add_argument('-o', '--output-dir', default='projection_output',
                       help='输出文件夹（默认: projection_output）')
    parser.add_argument('-r', '--region',
                       help='染色体区域（可选，如: 20 或 20:1000000-2000000）')
    parser.add_argument('--n-neighbors', type=int, default=20,
                       help='KNN分类器的邻居数（默认: 20）')
    parser.add_argument('--n-jobs', type=int, default=1,
                       help='并行处理的进程数（默认: 1）。多个VCF文件时建议设置为CPU核心数，如 --n-jobs 8')
    
    args = parser.parse_args()
    
    # 检查文件存在性
    for file, name in [(args.snps, 'SNP文件'), (args.model, '模型文件'), 
                       (args.reference, '参考CSV'), (args.query_vcf, '查询VCF')]:
        if not os.path.exists(file):
            print(f"错误: {name}不存在: {file}")
            exit(1)
    
    try:
        main(
            snps_file=args.snps,
            model_file=args.model,
            reference_csv=args.reference,
            query_vcf_or_dir=args.query_vcf,
            output_dir=args.output_dir,
            region=args.region,
            n_neighbors=args.n_neighbors,
            n_jobs=args.n_jobs
        )
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
