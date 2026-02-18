#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure SOM Classifier V3 with Configurable K-mer - 支持可配置的k-mer长度
基于pure_som_classifier_v3.py，参考som_classifier_from_excel.py的特征提取方式

改进：
- 支持4-mer、5-mer、6-mer三种k-mer长度
- 使用长度标准化的k-mer频率特征
- 去除复杂度和2-mer特征，只使用k-mer频率
"""

import numpy as np
import json
import os
import pickle
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import multiprocessing
import argparse

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class ConfigurableKmerEmbedder:
    """可配置k-mer长度的序列嵌入器
    
    特征：k-mer频率（除以序列长度标准化）+ Excel特征
    """
    
    def __init__(self, k=4, use_excel_features=True):
        """
        Args:
            k: k-mer长度 (4, 5, 或 6)
            use_excel_features: 是否使用Excel特征
        """
        self.k = k
        self.use_excel_features = use_excel_features
        self.nucleotides = ['A', 'T', 'G', 'C']
        
        # 生成所有可能的k-mer组合
        self.kmers = []
        num_kmers = 4 ** k
        for i in range(num_kmers):
            kmer = ""
            num = i
            for _ in range(k):
                kmer = self.nucleotides[num % 4] + kmer
                num //= 4
            self.kmers.append(kmer)
        
        self.kmer_dim = len(self.kmers)
        self.excel_feature_names = None
        self.excel_dim = 0
        
        if use_excel_features:
            print(f"初始化 {k}-mer 嵌入器 (k-mer特征维度: {self.kmer_dim}, 等待添加Excel特征)")
        else:
            self.feature_dim = self.kmer_dim
            print(f"初始化 {k}-mer 嵌入器 (纯k-mer，特征维度: {self.feature_dim})")
    
    def set_excel_features(self, excel_feature_names):
        """设置Excel特征名称"""
        self.excel_feature_names = excel_feature_names
        self.excel_dim = len(excel_feature_names)
        self.feature_dim = self.kmer_dim + self.excel_dim
        print(f"已添加Excel特征，总特征维度: {self.feature_dim} (k-mer: {self.kmer_dim} + Excel: {self.excel_dim})")
    
    def calculate_kmer_frequencies(self, sequence):
        """计算k-mer频率"""
        sequence = sequence.upper()
        if len(sequence) < self.k:
            return {}
            
        kmer_counts = Counter()
        for i in range(len(sequence) - self.k + 1):
            kmer = sequence[i:i+self.k]
            if all(base in 'ATGC' for base in kmer):
                kmer_counts[kmer] += 1
        
        total_kmers = sum(kmer_counts.values())
        if total_kmers == 0:
            return {}
            
        # 转换为频率
        kmer_frequencies = {}
        for kmer, count in kmer_counts.items():
            kmer_frequencies[kmer] = count / total_kmers
            
        return kmer_frequencies
    
    def calculate_gc_content(self, sequence):
        """计算GC含量"""
        sequence = sequence.upper()
        if len(sequence) == 0:
            return 0.0
        gc_count = sequence.count('G') + sequence.count('C')
        return gc_count / len(sequence)
    
    def embed_sequence(self, sequence, excel_features=None):
        """将单个序列转换为特征向量（k-mer频率 + Excel特征）
        
        Args:
            sequence: DNA序列
            excel_features: 对应的Excel特征向量（已标准化）
        """
        features = []
        
        # 1. 计算k-mer频率
        kmer_freqs = self.calculate_kmer_frequencies(sequence)
        
        # 按照预定义的kmer顺序提取特征
        for kmer in self.kmers:
            freq = kmer_freqs.get(kmer, 0.0)
            features.append(freq)
        
        # 使用序列长度标准化k-mer频率（参考excel版本）
        seq_length = len(sequence)
        if seq_length > 0:
            features = [f / seq_length for f in features]
        
        # 2. 如果启用Excel特征，添加它们
        if self.use_excel_features and excel_features is not None:
            features.extend(excel_features)
        
        return np.array(features)
    
    def embed_sequences(self, sequences, excel_features_matrix=None, verbose=False):
        """批量处理序列
        
        Args:
            sequences: 序列列表
            excel_features_matrix: Excel特征矩阵 (n_samples, n_excel_features)
            verbose: 是否详细输出
        """
        if verbose:
            if self.use_excel_features and excel_features_matrix is not None:
                print(f"正在嵌入 {len(sequences)} 个序列 (使用{self.k}-mer + Excel特征)...")
            else:
                print(f"正在嵌入 {len(sequences)} 个序列 (使用{self.k}-mer)...")
        
        embedded_sequences = []
        for i, seq in enumerate(sequences):
            if verbose and i % 100 == 0:
                print(f"  进度: {i}/{len(sequences)}")
            
            # 获取对应的Excel特征
            excel_feats = excel_features_matrix[i] if excel_features_matrix is not None else None
            
            features = self.embed_sequence(seq, excel_feats)
            embedded_sequences.append(features)
        
        embedded_matrix = np.array(embedded_sequences)
        if verbose:
            print(f"嵌入完成: {embedded_matrix.shape}")
            if self.use_excel_features and excel_features_matrix is not None:
                print(f"  - k-mer特征: {self.kmer_dim}维")
                print(f"  - Excel特征: {self.excel_dim}维")
        
        return embedded_matrix


class PureSOMClassifierKmer:
    """支持可配置k-mer长度的SOM分类器（支持Excel特征）"""
    
    def __init__(self, position_name, k=4, width=10, height=10, learning_rate=0.1, 
                 max_iter=5000, neighborhood_factor=1.0, use_excel_features=True):
        self.position_name = position_name
        self.k = k
        self.width = width
        self.height = height
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.neighborhood_factor = neighborhood_factor
        self.use_excel_features = use_excel_features
        
        # 初始化嵌入器
        self.embedder = ConfigurableKmerEmbedder(k=k, use_excel_features=use_excel_features)
        
        # 模型状态
        self.weights = None
        self.scaler = None
        self.trained = False
        self.training_sequences = None
        self.sequence_positions = None
        self.feature_matrix = None
        self.grid_analysis = None
    
    def find_bmu(self, sample):
        """找到最佳匹配单元(BMU)"""
        distances = np.sqrt(((self.weights - sample) ** 2).sum(axis=2))
        return np.unravel_index(np.argmin(distances), distances.shape)
    
    def update_weights(self, sample, bmu, iteration):
        """更新SOM权重"""
        # 计算学习率衰减
        learning_rate = self.learning_rate * np.exp(-iteration / self.max_iter)
        
        # 计算邻域半径衰减
        radius = self.neighborhood_factor * np.exp(-iteration / self.max_iter)
        
        # 更新权重
        for i in range(self.height):
            for j in range(self.width):
                # 计算到BMU的距离
                distance = np.sqrt((i - bmu[0])**2 + (j - bmu[1])**2)
                
                # 如果在邻域内，更新权重
                if distance <= radius:
                    influence = np.exp(-(distance**2) / (2 * radius**2))
                    self.weights[i, j] += learning_rate * influence * (sample - self.weights[i, j])
    
    def train(self, sequences, verbose=False, sample_counts=None, excel_features=None):
        """训练SOM模型
        
        Args:
            sequences: 训练序列列表
            verbose: 是否详细输出
            sample_counts: 序列样本计数字典 {序列: 样本数量}
            excel_features: Excel特征矩阵 (n_samples, n_excel_features)
        """
        if verbose:
            print(f"\n{'='*60}")
            if self.use_excel_features and excel_features is not None:
                print(f"为位点 {self.position_name} 训练SOM分类器 (k={self.k} + Excel特征)...")
            else:
                print(f"为位点 {self.position_name} 训练纯SOM分类器 (k={self.k})...")
            print(f"序列数量: {len(sequences)}")
            if sample_counts:
                total_alleles = sum(sample_counts.values())
                print(f"总等位基因数: {total_alleles}")
        
        # 存储训练序列
        self.training_sequences = sequences.copy()
        
        # 存储样本计数
        if sample_counts:
            self.sequence_sample_counts = sample_counts.copy()
        else:
            self.sequence_sample_counts = {seq: 1 for seq in sequences}
        
        # 1. 序列嵌入
        if verbose:
            feature_desc = f"{self.k}-mer"
            if self.use_excel_features and excel_features is not None:
                feature_desc += " + Excel特征"
            print(f"步骤1: 序列特征嵌入 ({feature_desc})")
        
        self.feature_matrix = self.embedder.embed_sequences(sequences, excel_features, verbose)
        if verbose:
            print(f"特征维度: {self.feature_matrix.shape}")
        
        # 2. 特征标准化
        if verbose:
            print(f"步骤2: 特征标准化")
        self.scaler = StandardScaler()
        feature_matrix_scaled = self.scaler.fit_transform(self.feature_matrix)
        
        # 3. 初始化SOM权重
        if verbose:
            print(f"步骤3: 初始化SOM网格 ({self.height}×{self.width})")
        feature_dim = feature_matrix_scaled.shape[1]
        self.weights = np.random.uniform(-1, 1, (self.height, self.width, feature_dim))
        
        # 4. SOM训练
        if verbose:
            print(f"步骤4: SOM训练 (迭代{self.max_iter}次)")
        for iteration in range(self.max_iter):
            if verbose and iteration % 500 == 0:
                print(f"  训练进度: {iteration}/{self.max_iter}")
                
            # 随机选择样本
            sample_idx = np.random.randint(0, len(feature_matrix_scaled))
            sample = feature_matrix_scaled[sample_idx]
            
            # 找到BMU并更新权重
            bmu = self.find_bmu(sample)
            self.update_weights(sample, bmu, iteration)
        
        # 5. 为训练序列分配SOM网格位置
        if verbose:
            print(f"步骤5: 分配SOM网格位置")
        self.sequence_positions = []
        for features in feature_matrix_scaled:
            bmu = self.find_bmu(features)
            self.sequence_positions.append(bmu)
            
        self.trained = True
        
        # 6. 分析SOM网格
        self.analyze_som_grid(verbose)
        self.calculate_position_quality(feature_matrix_scaled, verbose)
        
        return feature_matrix_scaled
    
    def analyze_som_grid(self, verbose=False):
        """分析SOM网格中每个位置的序列特征"""
        if verbose:
            print(f"步骤6: 分析SOM网格")
        
        # 统计每个网格位置的序列
        position_sequences = defaultdict(list)
        for seq_idx, position in enumerate(self.sequence_positions):
            position_sequences[position].append(seq_idx)
        
        occupied_positions = len(position_sequences)
        total_positions = self.height * self.width
        
        if verbose:
            print(f"占用的网格位置: {occupied_positions}/{total_positions}")
        
        # 分析每个占用的网格位置
        self.grid_analysis = {}
        for position, seq_indices in position_sequences.items():
            sequences_at_position = [self.training_sequences[i] for i in seq_indices]
            
            # 计算该位置序列的统计特征
            lengths = [len(seq) for seq in sequences_at_position]
            gc_contents = [self.embedder.calculate_gc_content(seq) for seq in sequences_at_position]
            
            self.grid_analysis[position] = {
                'count': len(sequences_at_position),
                'percentage': len(sequences_at_position) / len(self.training_sequences) * 100,
                'avg_length': np.mean(lengths),
                'length_range': (min(lengths), max(lengths)),
                'avg_gc_content': np.mean(gc_contents),
                'gc_range': (min(gc_contents), max(gc_contents)),
                'example_sequence': sequences_at_position[0][:50] + "..." if len(sequences_at_position[0]) > 50 else sequences_at_position[0]
            }
        
        # 打印网格分析结果
        if verbose:
            print(f"\nSOM网格分析结果:")
            for position, analysis in sorted(self.grid_analysis.items()):
                i, j = position
                print(f"位置({i},{j}): {analysis['count']}个序列 ({analysis['percentage']:.1f}%), "
                      f"平均长度{analysis['avg_length']:.1f}bp")
    
    def calculate_position_quality(self, feature_matrix_scaled, verbose=False):
        """计算每个SOM位置的内在质量"""
        if verbose:
            print(f"\n步骤7: 计算位置质量")
        
        self.position_quality = {}
        
        # 按位置分组特征
        position_features = defaultdict(list)
        for seq_idx, position in enumerate(self.sequence_positions):
            position_features[position].append(feature_matrix_scaled[seq_idx])
        
        # 计算每个位置的平均距离
        for position, features_list in position_features.items():
            if len(features_list) == 0:
                continue
                
            features_array = np.array(features_list)
            weight_vector = self.weights[position]
            
            # 计算该位置所有序列与权重向量的距离
            distances = []
            for features in features_array:
                distance = np.linalg.norm(features - weight_vector)
                distances.append(distance)
            
            avg_distance = np.mean(distances)
            std_distance = np.std(distances)
            
            # 质量等级
            if avg_distance < 3:
                quality_level = "优秀"
            elif avg_distance < 6:
                quality_level = "良好"
            elif avg_distance < 10:
                quality_level = "一般"
            else:
                quality_level = "较差"
            
            self.position_quality[position] = {
                'avg_distance': avg_distance,
                'std_distance': std_distance,
                'count': len(features_list),
                'quality_level': quality_level
            }
            
            if verbose:
                i, j = position
                print(f"  位置({i},{j}): 平均距离{avg_distance:.3f}, 质量{quality_level}")
    
    def merge_similar_positions(self, merge_threshold=0.8, verbose=False):
        """合并相似的网格位置"""
        if not self.trained:
            raise ValueError("模型尚未训练")
        
        if verbose:
            print(f"\n合并相似位置 (欧式距离阈值: {merge_threshold})")
        
        # 获取所有占用的位置
        occupied_positions = list(self.grid_analysis.keys())
        
        if len(occupied_positions) <= 1:
            if verbose:
                print("只有一个或没有占用位置，无需合并")
            self.merged_sequence_positions = self.sequence_positions.copy()
            return {}, {}
        
        # 提取权重向量用于欧式距离聚类
        position_weights = []
        for pos in occupied_positions:
            weight_vector = self.weights[pos].flatten()
            position_weights.append(weight_vector)
        
        position_weights = np.array(position_weights)
        
        # 使用欧式距离进行DBSCAN聚类
        clustering = DBSCAN(eps=merge_threshold, min_samples=1, metric='euclidean')
        cluster_labels = clustering.fit_predict(position_weights)
        
        # 创建位置映射
        position_mapping = {}
        cluster_centers = {}
        
        for cluster_id in set(cluster_labels):
            cluster_positions = [occupied_positions[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
            
            # 选择聚类中心（序列数最多的位置）
            center_position = max(cluster_positions, key=lambda pos: self.grid_analysis[pos]['count'])
            cluster_centers[cluster_id] = center_position
            
            # 映射所有位置到中心
            for pos in cluster_positions:
                position_mapping[pos] = center_position
        
        # 应用映射到序列位置
        self.merged_sequence_positions = []
        for pos in self.sequence_positions:
            merged_pos = position_mapping.get(pos, pos)
            self.merged_sequence_positions.append(merged_pos)
        
        if verbose:
            original_clusters = len(occupied_positions)
            merged_clusters = len(set(self.merged_sequence_positions))
            print(f"原始聚类数: {original_clusters}")
            print(f"合并后聚类数: {merged_clusters}")
            print(f"减少了 {original_clusters - merged_clusters} 个聚类")
        
        return position_mapping, cluster_centers
    
    def visualize_som_2d_with_patients(self, save_dir, sample_ids, patient_ids, population_map=None, verbose=False):
        """生成2D SOM散点图，标注patient样本和人群信息
        
        Args:
            save_dir: 保存目录
            sample_ids: 样本ID列表
            patient_ids: 患者ID列表
            population_map: 样本ID到人群代码的映射字典 {sample_id: population}
            verbose: 是否输出详细信息
        """
        if not self.trained:
            raise ValueError("模型尚未训练")
        
        # 使用合并后的位置
        if hasattr(self, 'merged_sequence_positions'):
            positions = self.merged_sequence_positions
        else:
            positions = self.sequence_positions
        
        # 创建图形（增加宽度以容纳更长的图例）
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        
        # 按位置分组样本
        position_samples = {}
        for idx, pos in enumerate(positions):
            if pos not in position_samples:
                position_samples[pos] = []
            position_samples[pos].append(idx)
        
        # 定义颜色
        colors = plt.cm.tab20(np.linspace(0, 1, max(20, len(position_samples))))
        patient_set = set(patient_ids) if patient_ids else set()
        
        # 为每个聚类绘制散点
        for cluster_idx, (pos, sample_indices) in enumerate(sorted(position_samples.items())):
            # 获取聚类坐标
            if isinstance(pos, tuple):
                cluster_x, cluster_y = pos[1], pos[0]
            else:
                cluster_x = pos % self.width
                cluster_y = pos // self.width
            
            # 添加抖动
            n_samples = len(sample_indices)
            jitter = 0.15
            x_coords = cluster_x + np.random.uniform(-jitter, jitter, n_samples)
            y_coords = cluster_y + np.random.uniform(-jitter, jitter, n_samples)
            
            # 分离patient和control
            patient_mask = [sample_ids[i] in patient_set if i < len(sample_ids) else False 
                           for i in sample_indices]
            
            # 统计数量
            n_patients = sum(patient_mask)
            n_controls = sum(not m for m in patient_mask)
            
            # 统计对照样本的人群分布
            population_counts = {}
            if population_map:
                for i in sample_indices:
                    if i < len(sample_ids):
                        sid = sample_ids[i]
                        # 检查是否是对照样本
                        if sid not in patient_set:
                            pop = population_map.get(sid, None)
                            if pop:
                                population_counts[pop] = population_counts.get(pop, 0) + 1
            
            # 构建聚类标签
            if n_controls > 0 and population_counts:
                # 按人群数量排序
                pop_strs = [f"{count} {pop}" for pop, count in sorted(population_counts.items(), 
                                                                       key=lambda x: (-x[1], x[0]))]
                pop_info = ", ".join(pop_strs)
                cluster_label = f"Cluster ({int(cluster_x)}, {int(cluster_y)}) ({n_patients} patients, {n_controls} controls: {pop_info})"
            else:
                cluster_label = f"Cluster ({int(cluster_x)}, {int(cluster_y)}) ({n_patients} patients, {n_controls} controls)"
            
            # 绘制control样本
            if n_controls > 0:
                non_patient_x = x_coords[[not m for m in patient_mask]]
                non_patient_y = y_coords[[not m for m in patient_mask]]
                ax.scatter(non_patient_x, non_patient_y, 
                          c=[colors[cluster_idx % len(colors)]], 
                          s=100, alpha=0.6, edgecolors='gray', linewidths=0.5,
                          label=cluster_label)
            
            # 绘制patient样本（黑色粗边框）
            if n_patients > 0:
                patient_x = x_coords[patient_mask]
                patient_y = y_coords[patient_mask]
                # 如果该聚类没有control样本，需要添加标签
                patient_label = cluster_label if n_controls == 0 else None
                ax.scatter(patient_x, patient_y, 
                          c=[colors[cluster_idx % len(colors)]], 
                          s=150, alpha=0.9, edgecolors='black', linewidths=3,
                          marker='o', zorder=10, label=patient_label)
        
        # 设置坐标轴
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(-0.5, self.height - 0.5)
        ax.invert_yaxis()
        
        # 添加网格
        ax.set_xticks(range(self.width))
        ax.set_yticks(range(self.height))
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_aspect('equal')
        
        # 标题
        k_mer = f"{self.k}-mer" if hasattr(self, 'k') else "SOM"
        ax.set_title(f'{self.position_name} - 2D SOM Visualization ({k_mer})\nPatient samples marked with thick black border', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('SOM Grid X', fontsize=12)
        ax.set_ylabel('SOM Grid Y', fontsize=12)
        
        # 添加统计信息
        patient_count = sum(1 for idx in range(len(sample_ids)) if sample_ids[idx] in patient_set)
        
        ax.text(0.02, 0.98, 
               f'Total: {len(sample_ids)} samples\nPatient: {patient_count} ({patient_count/len(sample_ids)*100:.1f}%)\nClusters: {len(position_samples)}',
               transform=ax.transAxes, fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # 添加图例（显示所有聚类）
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 0:
            # 根据聚类数量调整字体大小和列数（因为加了人群信息，图例会更长）
            if len(handles) > 30:
                fontsize = 5
                ncol = 2
            elif len(handles) > 20:
                fontsize = 6
                ncol = 2
            elif len(handles) > 15:
                fontsize = 7
                ncol = 1
            else:
                fontsize = 8
                ncol = 1
            
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                     fontsize=fontsize, ncol=ncol, framealpha=0.9)
        
        # 保存
        plot_file = os.path.join(save_dir, f"{self.position_name}_som_2d_patient.png")
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        if verbose:
            print(f"2D SOM patient图已保存: {plot_file}")
        
        return plot_file
    
    def visualize_som_grid(self, save_dir, verbose=False):
        """可视化SOM网格分析"""
        if not self.trained:
            raise ValueError("模型尚未训练")
            
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'{self.position_name} Pure SOM ({self.k}-mer) Grid Visualization', fontsize=16)
        
        # 1. 每个网格位置的序列数量
        count_map = np.zeros((self.height, self.width))
        for position in self.sequence_positions:
            i, j = position
            count_map[i, j] += 1
            
        im1 = axes[0,0].imshow(count_map, cmap='YlOrRd', interpolation='nearest')
        axes[0,0].set_title('Sequence Count per Grid Position')
        axes[0,0].set_xlabel('SOM Width')
        axes[0,0].set_ylabel('SOM Height')
        plt.colorbar(im1, ax=axes[0,0], label='Number of Sequences')
        
        # 添加非零位置的文本标注
        for i in range(self.height):
            for j in range(self.width):
                if count_map[i, j] > 0:
                    axes[0,0].text(j, i, f'{int(count_map[i, j])}', 
                                  ha='center', va='center', color='white', fontweight='bold')
        
        # 2. 网格占用情况
        occupancy_map = (count_map > 0).astype(int)
        im2 = axes[0,1].imshow(occupancy_map, cmap='RdYlGn', interpolation='nearest')
        axes[0,1].set_title('Grid Occupancy')
        axes[0,1].set_xlabel('SOM Width')
        axes[0,1].set_ylabel('SOM Height')
        plt.colorbar(im2, ax=axes[0,1], label='Occupied (1) / Empty (0)')
        
        # 3. 平均序列长度
        length_map = np.zeros((self.height, self.width))
        length_count = np.zeros((self.height, self.width))
        
        for seq_idx, position in enumerate(self.sequence_positions):
            i, j = position
            seq_length = len(self.training_sequences[seq_idx])
            length_map[i, j] += seq_length
            length_count[i, j] += 1
        
        avg_length_map = np.divide(length_map, length_count, 
                                  out=np.zeros_like(length_map), 
                                  where=length_count!=0)
        
        masked_length_map = np.ma.masked_where(length_count == 0, avg_length_map)
        im3 = axes[1,0].imshow(masked_length_map, cmap='viridis', interpolation='nearest')
        axes[1,0].set_title('Average Sequence Length')
        axes[1,0].set_xlabel('SOM Width')
        axes[1,0].set_ylabel('SOM Height')
        plt.colorbar(im3, ax=axes[1,0], label='Average Length (bp)')
        
        # 4. 网格位置标签
        label_map = np.full((self.height, self.width), -1, dtype=int)
        for i in range(self.height):
            for j in range(self.width):
                if count_map[i, j] > 0:
                    label_map[i, j] = i * self.width + j
        
        masked_label_map = np.ma.masked_where(count_map == 0, label_map)
        im4 = axes[1,1].imshow(masked_label_map, cmap='tab20', interpolation='nearest')
        axes[1,1].set_title('Grid Position Labels')
        axes[1,1].set_xlabel('SOM Width')
        axes[1,1].set_ylabel('SOM Height')
        plt.colorbar(im4, ax=axes[1,1], label='Position ID')
        
        # 为位置标签添加文本标注
        for i in range(self.height):
            for j in range(self.width):
                if count_map[i, j] > 0:
                    label = i * self.width + j
                    axes[1,1].text(j, i, f'{label}', 
                                  ha='center', va='center', color='white', fontweight='bold')
        
        # 保存图片
        plot_file = os.path.join(save_dir, f"{self.position_name}_{self.k}mer_som_visualization.png")
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        if verbose:
            print(f"SOM visualization saved: {plot_file}")
        return plot_file
    
    def save_detailed_classification_report(self, save_dir, verbose=False, use_merged=True):
        """生成详细的分类报告文件"""
        if not self.trained:
            raise ValueError("模型尚未训练")
        
        report_file = os.path.join(save_dir, f"{self.position_name}_{self.k}mer_som_detailed.txt")
        
        # 选择使用原始位置还是合并后的位置
        if use_merged and hasattr(self, 'merged_sequence_positions'):
            sequence_positions = self.merged_sequence_positions
            title_suffix = " (合并后)"
        else:
            sequence_positions = self.sequence_positions
            title_suffix = ""
        
        # 按SOM位置分组序列
        position_sequences = {}
        for seq_idx, position in enumerate(sequence_positions):
            if position not in position_sequences:
                position_sequences[position] = []
            position_sequences[position].append(seq_idx)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            # 标题信息
            f.write(f"位点: {self.position_name}\n")
            f.write(f"纯SOM分类详细结果 ({self.k}-mer){title_suffix}\n")
            f.write("=" * 80 + "\n")
            f.write(f"总序列数: {len(self.training_sequences)}\n")
            # 计算总等位基因数
            sample_counts_dict = getattr(self, 'sequence_sample_counts', {})
            total_alleles = sum(sample_counts_dict.get(seq, 1) for seq in self.training_sequences)
            f.write(f"总等位基因数: {total_alleles}\n")
            if use_merged and hasattr(self, 'merged_sequence_positions'):
                original_positions = len(set(self.sequence_positions))
                merged_positions = len(position_sequences)
                f.write(f"原始SOM位置数: {original_positions}\n")
                f.write(f"合并后位置数: {merged_positions}\n\n")
            else:
                f.write(f"SOM位置数: {len(position_sequences)}\n\n")
            
            # 按序列数量排序位置
            sorted_positions = sorted(position_sequences.items(), 
                                    key=lambda x: len(x[1]), reverse=True)
            
            for position_id, seq_indices in sorted_positions:
                sequences_at_position = [self.training_sequences[i] for i in seq_indices]
                
                # 计算统计信息
                lengths = [len(seq) for seq in sequences_at_position]
                gc_contents = [self.embedder.calculate_gc_content(seq) for seq in sequences_at_position]
                unique_sequences = list(set(sequences_at_position))
                
                # 写入位置信息
                if use_merged and hasattr(self, 'merged_sequence_positions'):
                    if isinstance(position_id, tuple):
                        pos_str = f"({int(position_id[0])}, {int(position_id[1])})"
                    else:
                        pos_str = str(int(position_id))
                    f.write(f"合并聚类 {pos_str}\n")
                else:
                    if isinstance(position_id, tuple):
                        pos_str = f"[{int(position_id[0])}, {int(position_id[1])}]"
                    else:
                        pos_str = f"[{int(position_id)}]"
                    f.write(f"SOM位置 {pos_str}\n")
                f.write("-" * 40 + "\n")
                f.write(f"序列数量: {len(sequences_at_position)}\n")
                # 计算该聚类的等位基因数
                cluster_allele_count = sum(sample_counts_dict.get(seq, 1) for seq in sequences_at_position)
                f.write(f"等位基因数: {cluster_allele_count}\n")
                f.write(f"长度范围: {min(lengths)}-{max(lengths)}bp\n")
                f.write(f"平均长度: {np.mean(lengths):.1f}bp\n")
                f.write(f"唯一序列: {len(unique_sequences)}\n")
                
                # 网格分析
                f.write("聚类分析:\n")
                f.write(f"  count: {len(sequences_at_position)}\n")
                f.write(f"  percentage: {len(sequences_at_position) / len(self.training_sequences) * 100:.2f}%\n")
                f.write(f"  avg_length: {np.mean(lengths):.1f}\n")
                f.write(f"  length_range: [{min(lengths)}, {max(lengths)}]\n")
                f.write(f"  avg_gc_content: {np.mean(gc_contents):.6f}\n")
                f.write(f"  gc_range: [{min(gc_contents):.6f}, {max(gc_contents):.6f}]\n")
                
                # 示例序列
                example_seq = sequences_at_position[0]
                if len(example_seq) > 50:
                    example_seq = example_seq[:50] + "..."
                f.write(f"  example_sequence: {example_seq}\n")
                
                # 序列列表
                f.write("序列列表:\n")
                for i, seq in enumerate(sequences_at_position, 1):
                    seq_allele_count = sample_counts_dict.get(seq, 1)
                    f.write(f"   {i}. [等位基因:{seq_allele_count}] {seq}\n")
                
                f.write("\n")
        
        if verbose:
            print(f"详细分类报告已保存: {report_file}")
        return report_file
    
    def save_model(self, save_dir, verbose=False):
        """保存模型参数"""
        if not self.trained:
            raise ValueError("模型尚未训练")
            
        model_file = os.path.join(save_dir, f"{self.position_name}_{self.k}mer_som_model.pkl")
        
        model_data = {
            'position_name': self.position_name,
            'k': self.k,
            'width': self.width,
            'height': self.height,
            'learning_rate': self.learning_rate,
            'max_iter': self.max_iter,
            'neighborhood_factor': self.neighborhood_factor,
            'position_quality': getattr(self, 'position_quality', {}),
            'weights': self.weights,
            'scaler': self.scaler,
            'embedder': self.embedder,
            'training_sequences': self.training_sequences,
            'sequence_sample_counts': getattr(self, 'sequence_sample_counts', {}),
            'sequence_positions': self.sequence_positions,
            'feature_matrix': self.feature_matrix,
            'grid_analysis': self.grid_analysis,
            'trained': self.trained
        }
        
        # 如果有合并后的分类结果，也保存
        if hasattr(self, 'merged_sequence_positions'):
            model_data['merged_sequence_positions'] = self.merged_sequence_positions
            model_data['has_merged_results'] = True
        else:
            model_data['has_merged_results'] = False
        
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)
            
        if verbose:
            print(f"SOM模型已保存: {model_file}")
        return model_file


def has_consecutive_repeats(sequence, max_repeat_length=10):
    """检查序列是否有连续的相同碱基重复超过指定长度"""
    if len(sequence) < max_repeat_length:
        return False
    
    sequence = sequence.upper()
    current_base = sequence[0]
    current_count = 1
    
    for i in range(1, len(sequence)):
        if sequence[i] == current_base:
            current_count += 1
            if current_count >= max_repeat_length:
                return True
        else:
            current_base = sequence[i]
            current_count = 1
    
    return False


def load_sequences_from_file(seq_file, sample_ids, verbose=False):
    """从序列文件中加载样本序列"""
    if verbose:
        print(f"\n加载序列文件: {seq_file}")
    
    # 读取序列文件
    sequences_dict = {}
    with open(seq_file, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                sample_id, sequence = line.split(':', 1)
                sequences_dict[sample_id] = sequence
    
    if verbose:
        print(f"序列文件中共有 {len(sequences_dict)} 个序列")
    
    # 按照sample_ids顺序提取序列
    sequences = []
    missing_samples = []
    for sid in sample_ids:
        if sid in sequences_dict:
            sequences.append(sequences_dict[sid])
        else:
            sid_stripped = sid.strip()
            if sid_stripped in sequences_dict:
                sequences.append(sequences_dict[sid_stripped])
            else:
                missing_samples.append(sid)
                sequences.append("")
    
    if missing_samples and verbose:
        print(f"警告: 以下样本在序列文件中未找到: {missing_samples[:5]}")
    
    if verbose:
        print(f"成功匹配 {len([s for s in sequences if s])} 个序列")
    
    return sequences


def load_excel_data_with_features(excel_file, seq_file, sheet_name='700bp', verbose=False):
    """从Excel文件加载样本ID、序列和Excel特征"""
    import pandas as pd
    
    if verbose:
        print(f"加载Excel文件: {excel_file}")
        print(f"Sheet: {sheet_name}")
    
    # 读取Excel文件，不使用header
    df_raw = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
    
    if verbose:
        print(f"原始数据形状: {df_raw.shape}")
    
    # 第一行是样本ID，第二行是分组，第三行及以后是特征
    sample_ids = df_raw.iloc[0, 1:].astype(str).tolist()  # 跳过第一个（'sample'）
    groups = df_raw.iloc[1, 1:].astype(str).tolist()  # 分组信息（patient/control）
    
    # 提取特征名称（第一列，从第3行开始，即index=2开始）
    feature_names = df_raw.iloc[2:, 0].astype(str).tolist()
    
    # 提取特征数据（从第3行开始，从第2列开始）
    feature_data = df_raw.iloc[2:, 1:].values.T  # 转置，使样本在行，特征在列
    
    if verbose:
        print(f"样本数量: {len(sample_ids)}")
        print(f"Excel特征数量: {len(feature_names)}")
        print(f"前10个特征名称: {feature_names[:10]}")
        print(f"分组信息: patient={groups.count('patient')}, control={groups.count('control')}")
    
    # 检查是否有length特征
    length_idx = None
    for i, name in enumerate(feature_names):
        if 'lenth' in name.lower() or 'length' in name.lower():
            length_idx = i
            break
    
    if length_idx is None:
        raise ValueError("未找到length或lenth特征")
    
    # 提取length列
    length_col = feature_data[:, length_idx].astype(float)
    
    if verbose:
        print(f"Length特征位于第{length_idx}列 ({feature_names[length_idx]})")
        print(f"Length范围: {np.min(length_col)}-{np.max(length_col)}")
    
    # 移除length列，获取其他特征
    feature_indices = [i for i in range(len(feature_names)) if i != length_idx]
    excel_features = feature_data[:, feature_indices].astype(float)
    excel_feature_names = [feature_names[i] for i in feature_indices]
    
    # 使用length标准化Excel特征（除以length）
    # 避免除以0
    length_col = np.where(length_col == 0, 1, length_col)
    normalized_excel_features = excel_features / length_col[:, np.newaxis]
    
    if verbose:
        print(f"已使用length标准化Excel特征")
        print(f"标准化后特征范围示例（第一个特征）: {np.min(normalized_excel_features[:, 0]):.6f}-{np.max(normalized_excel_features[:, 0]):.6f}")
    
    # 加载序列
    sequences = load_sequences_from_file(seq_file, sample_ids, verbose)
    
    # 过滤掉空序列，同时保留对应的Excel特征
    valid_data = []
    valid_excel_features = []
    
    for i, (sid, seq) in enumerate(zip(sample_ids, sequences)):
        if seq:
            valid_data.append((sid, seq))
            valid_excel_features.append(normalized_excel_features[i])
    
    valid_excel_features = np.array(valid_excel_features)
    
    if verbose:
        print(f"有效样本数: {len(valid_data)}")
        print(f"Excel特征矩阵形状: {valid_excel_features.shape}")
    
    return valid_data, valid_excel_features, excel_feature_names


def train_single_position_from_excel(excel_file, seq_file, sheet_name, position, save_dir, k, 
                                     width, height, learning_rate, max_iter, neighborhood_factor, 
                                     merge_threshold, verbose, force=False, use_excel_features=True):
    """从Excel文件训练单个位点的SOM模型"""
    try:
        # 检查结果文件是否已存在
        suffix = "excel" if use_excel_features else "pure"
        model_file = os.path.join(save_dir, f"{position}_{k}mer_{suffix}_som_model.pkl")
        report_file = os.path.join(save_dir, f"{position}_{k}mer_{suffix}_som_detailed.txt")
        
        if not force and os.path.exists(model_file) and os.path.exists(report_file):
            if verbose:
                print(f"⏭️  跳过位点 {position} ({k}-mer {suffix}模型已存在)")
            return {
                'position': position,
                'k': k,
                'status': 'skipped',
                'message': 'Files already exist'
            }
        
        if verbose:
            print(f"\n{'='*60}")
            feature_desc = f"{k}-mer"
            if use_excel_features:
                feature_desc += " + Excel特征"
            print(f"处理位点: {position} ({feature_desc})")
        
        # 从Excel加载数据（包含Excel特征）
        if use_excel_features:
            sample_data, excel_features, excel_feature_names = load_excel_data_with_features(
                excel_file, seq_file, sheet_name, verbose
            )
        else:
            # 简化版本，不使用Excel特征
            sample_data, excel_features, excel_feature_names = load_excel_data_with_features(
                excel_file, seq_file, sheet_name, verbose
            )
            excel_features = None  # 不使用Excel特征
        
        if not sample_data:
            if verbose:
                print(f"警告: 未找到有效数据")
            return None
        
        # 提取序列和sample_ids
        sample_ids_original = [sid for sid, _ in sample_data]
        sequences = [seq for _, seq in sample_data]
        sequence_sample_counts = {seq: 1 for seq in sequences}
        
        if verbose:
            print(f"原始序列数量: {len(sequences)}")
            total_alleles = len(sequences)
            print(f"总等位基因数: {total_alleles}")
            if use_excel_features and excel_features is not None:
                print(f"Excel特征维度: {excel_features.shape}")
        
        # 应用序列过滤（不依赖外部模块）
        min_length = k + 1  # 至少要比k-mer长度大1
        
        # 简单过滤：长度和同聚物
        def is_homopolymer_at_ends(seq, length=8):
            """检查序列两端是否有同聚物"""
            if len(seq) < length:
                return False
            # 检查前8个
            if len(set(seq[:length])) == 1:
                return True
            # 检查后8个
            if len(set(seq[-length:])) == 1:
                return True
            return False
        
        # 同步过滤sequences和sample_ids
        filtered_data = [(sid, seq) for sid, seq in zip(sample_ids_original, sequences) 
                        if len(seq) >= min_length and not is_homopolymer_at_ends(seq)]
        filtered_sequences = [seq for _, seq in filtered_data]
        filtered_sample_ids = [sid for sid, _ in filtered_data]
        
        if verbose:
            print(f"过滤后序列数量: {len(filtered_sequences)} (最小长度: {min_length})")
        
        # 不再过滤连续重复碱基，使用所有通过基本过滤的序列
        valid_sequences = filtered_sequences
        valid_sample_ids = filtered_sample_ids
        valid_sample_counts = {}
        for seq in valid_sequences:
            valid_sample_counts[seq] = sequence_sample_counts.get(seq, 1)
        
        if len(valid_sequences) < 10:
            if verbose:
                print(f"警告: 位点 {position} 有效序列太少 ({len(valid_sequences)}), 跳过")
            return None
            
        if verbose:
            print(f"最终有效序列数量: {len(valid_sequences)}")
            
            # 输出被过滤掉的样本（如果有）
            filtered_out = set(sample_ids_original) - set(valid_sample_ids)
            if filtered_out:
                print(f"\n⚠️  被过滤掉的样本 ({len(filtered_out)}个):")
                for sid in sorted(filtered_out):
                    print(f"   - {sid}")
        
        # 过滤Excel特征，只保留有效序列对应的特征
        if use_excel_features and excel_features is not None:
            # 创建原始序列到索引的映射
            seq_to_idx = {seq: i for i, seq in enumerate(sequences)}
            valid_excel_features = []
            for seq in valid_sequences:
                if seq in seq_to_idx:
                    valid_excel_features.append(excel_features[seq_to_idx[seq]])
            valid_excel_features = np.array(valid_excel_features)
            
            if verbose:
                print(f"过滤后Excel特征维度: {valid_excel_features.shape}")
        else:
            valid_excel_features = None
        
        # 创建并训练SOM模型
        som = PureSOMClassifierKmer(
            position_name=position,
            k=k,
            width=width,
            height=height,
            learning_rate=learning_rate,
            max_iter=max_iter,
            neighborhood_factor=neighborhood_factor,
            use_excel_features=use_excel_features
        )
        
        # 设置Excel特征名称
        if use_excel_features and excel_feature_names:
            som.embedder.set_excel_features(excel_feature_names)
        
        # 训练模型
        som.train(valid_sequences, verbose=verbose, sample_counts=valid_sample_counts, 
                 excel_features=valid_excel_features)
        
        # 合并相似聚类
        if merge_threshold > 0:
            merged_mapping, cluster_centers = som.merge_similar_positions(merge_threshold, verbose)
        
        # 生成可视化图片
        suffix = "excel" if use_excel_features else "pure"
        original_name = som.position_name
        som.position_name = f"{original_name}_{k}mer_{suffix}"
        
        plot_file = som.visualize_som_grid(save_dir, verbose=verbose)
        
        # 生成2D patient标注图（使用实际训练的样本）
        try:
            import pandas as pd
            df_raw = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
            excel_sample_ids_all = df_raw.iloc[0, 1:].astype(str).tolist()
            groups_all = df_raw.iloc[1, 1:].astype(str).tolist()
            
            # 创建sample_id到group的映射
            sid_to_group = {sid: grp for sid, grp in zip(excel_sample_ids_all, groups_all)}
            
            # 使用训练时过滤后的valid_sample_ids
            # 提取patient ID（从valid_sample_ids中找group为'patient'的）
            patient_id_list = [sid for sid in valid_sample_ids if sid_to_group.get(sid, '') == 'patient']
            
            # 加载人群信息
            population_map = {}
            ethics_file = 'ethics.txt'
            if os.path.exists(ethics_file):
                with open(ethics_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and '\t' in line:
                            parts = line.split('\t')
                            if len(parts) >= 2:
                                sample_name = parts[0].strip()
                                population = parts[1].strip()
                                population_map[sample_name] = population
                
                # 创建完整映射（处理_1和_2后缀）
                extended_map = {}
                for sid in valid_sample_ids:
                    # 先尝试直接匹配
                    if sid in population_map:
                        extended_map[sid] = population_map[sid]
                    else:
                        # 尝试去掉_1或_2后缀
                        base_name = sid.rsplit('_', 1)[0] if '_' in sid else sid
                        if base_name in population_map:
                            extended_map[sid] = population_map[base_name]
                
                if verbose:
                    print(f"加载人群信息: {len(population_map)} 个样本")
                    print(f"匹配到人群信息: {len(extended_map)} 个样本")
            else:
                extended_map = None
                if verbose:
                    print(f"未找到ethics.txt文件，跳过人群信息")
            
            if verbose:
                print(f"实际训练样本数: {len(valid_sample_ids)}")
                print(f"Patient样本数: {len(patient_id_list)}")
            
            # 生成2D patient图
            patient_plot = som.visualize_som_2d_with_patients(save_dir, valid_sample_ids, 
                                                             patient_id_list, 
                                                             population_map=extended_map,
                                                             verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to generate 2D patient plot: {e}")
            patient_plot = None
        
        # 生成详细分类报告
        report_file = som.save_detailed_classification_report(save_dir, verbose=verbose, 
                                                              use_merged=merge_threshold > 0)
        
        # 导出聚类组成信息（用于统计分析）
        try:
            import pandas as pd
            # 获取聚类位置
            if hasattr(som, 'merged_sequence_positions'):
                positions = som.merged_sequence_positions
            else:
                positions = som.sequence_positions
            
            # 创建聚类组成数据
            cluster_data = []
            for idx, (sid, grp) in enumerate(zip(valid_sample_ids, [sid_to_group.get(sid, 'unknown') for sid in valid_sample_ids])):
                if idx < len(positions):
                    cluster_data.append({
                        'sample_id': sid,
                        'group': grp,
                        'cluster': str(positions[idx])
                    })
            
            df_clusters = pd.DataFrame(cluster_data)
            cluster_csv = os.path.join(save_dir, f"{som.position_name}_cluster_composition.csv")
            df_clusters.to_csv(cluster_csv, index=False)
            
            if verbose:
                print(f"聚类组成信息已保存: {cluster_csv}")
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to export cluster composition: {e}")
        
        # 保存模型
        model_file = som.save_model(save_dir, verbose=verbose)
        
        # 恢复原始名称
        som.position_name = original_name
        
        if verbose:
            print(f"位点 {position} ({k}-mer) 训练完成:")
            print(f"  - 模型文件: {model_file}")
            print(f"  - 可视化图片: {plot_file}")
            print(f"  - 详细报告: {report_file}")
        
        return {
            'position': position,
            'k': k,
            'model_file': model_file,
            'plot_file': plot_file,
            'report_file': report_file,
            'merged_clusters': len(set(som.merged_sequence_positions)) if merge_threshold > 0 and hasattr(som, 'merged_sequence_positions') else len(set(som.sequence_positions))
        }
        
    except Exception as e:
        if verbose:
            print(f"训练位置 {position} ({k}-mer) 时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练可配置k-mer长度的纯SOM分类器')
    parser.add_argument('--excel-file', type=str, default='characte of seq.xlsx', help='Excel数据文件路径')
    parser.add_argument('--seq-file', type=str, default='seqs_700bp.txt', help='序列文件路径')
    parser.add_argument('--sheet-name', type=str, default='700bp', help='Excel sheet名称')
    parser.add_argument('--width', type=int, default=6, help='SOM网格宽度')
    parser.add_argument('--height', type=int, default=6, help='SOM网格高度')
    parser.add_argument('--learning-rate', type=float, default=0.02, help='学习率')
    parser.add_argument('--max-iter', type=int, default=5000, help='最大迭代次数')
    parser.add_argument('--neighborhood-factor', type=float, default=3.0, help='邻域因子')
    parser.add_argument('--merge-threshold', type=float, default=0.8, help='聚类合并阈值')
    parser.add_argument('--n-jobs', type=int, default=10, help='并行进程数')
    parser.add_argument('--all', action='store_true', help='处理所有位点')
    parser.add_argument('--force', action='store_true', help='强制重新处理已存在的文件')
    parser.add_argument('--no-excel-features', action='store_true', 
                       help='不使用Excel特征（仅使用k-mer）')
    parser.add_argument('-v', '--verbose', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    use_excel_features = not args.no_excel_features
    
    print("=== SOM Classifier with Configurable K-mer (from Excel) ===")
    print(f"Grid: {args.width}×{args.height}, LR: {args.learning_rate}, Iter: {args.max_iter}")
    print(f"Neighborhood Factor: {args.neighborhood_factor}, Merge Threshold: {args.merge_threshold}")
    print(f"K-mer lengths: 4, 5, 6")
    if use_excel_features:
        print(f"特征: k-mer频率 + Excel特征（长度标准化）")
    else:
        print(f"特征: 纯k-mer频率（不使用Excel特征）")
    
    # 检查文件是否存在
    if not os.path.exists(args.excel_file):
        print(f"❌ 错误: 找不到Excel文件 {args.excel_file}")
        return
    
    if not os.path.exists(args.seq_file):
        print(f"❌ 错误: 找不到序列文件 {args.seq_file}")
        return
    
    print(f"Excel文件: {args.excel_file}")
    print(f"序列文件: {args.seq_file}")
    print(f"Sheet: {args.sheet_name}")
    
    # 位点名称（从Excel文件推断）
    target_positions = ["HSF1"]  # 默认位点名称
    print(f"处理位点: {target_positions}")
    
    # 创建保存目录
    save_dir = "pure_som_kmer_models"
    os.makedirs(save_dir, exist_ok=True)
    
    # 训练三种k-mer模型（4-mer, 5-mer, 6-mer）
    all_results = []
    
    for k in [4, 5, 6]:
        print(f"\n{'='*80}")
        print(f"训练 {k}-mer 模型")
        print(f"{'='*80}")
        
        # 准备训练参数
        train_args = [(args.excel_file, args.seq_file, args.sheet_name, pos, save_dir, k, 
                      args.width, args.height, args.learning_rate, args.max_iter, 
                      args.neighborhood_factor, args.merge_threshold, args.verbose, args.force,
                      use_excel_features) 
                     for pos in target_positions]
        
        # 使用多进程训练（这里只有一个位点，实际不需要多进程）
        start_time = datetime.now()
        results = []
        for args_tuple in train_args:
            result = train_single_position_from_excel(*args_tuple)
            results.append(result)
        
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        valid_results = [r for r in results if r is not None]
        skipped_count = sum(1 for r in valid_results if isinstance(r, dict) and r.get('status') == 'skipped')
        processed_count = len(valid_results) - skipped_count
        
        print(f"\n✅ {k}-mer 模型训练完成！用时: {int(elapsed//60)}分{int(elapsed%60)}秒")
        print(f"   处理: {processed_count}个位点 | 跳过: {skipped_count}个位点")
        
        all_results.extend(valid_results)
    
    # 生成总结报告
    report_file = os.path.join(save_dir, "training_summary.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=== Pure SOM Classifier with Configurable K-mer - Training Summary ===\n\n")
        f.write(f"Parameters:\n")
        f.write(f"  Grid: {args.width}×{args.height}\n")
        f.write(f"  Learning Rate: {args.learning_rate}\n")
        f.write(f"  Max Iterations: {args.max_iter}\n")
        f.write(f"  Neighborhood Factor: {args.neighborhood_factor}\n")
        f.write(f"  Merge Threshold: {args.merge_threshold}\n\n")
        
        for k in [4, 5, 6]:
            k_results = [r for r in all_results if isinstance(r, dict) and r.get('k') == k and 'merged_clusters' in r]
            if k_results:
                f.write(f"\n{k}-mer Models:\n")
                f.write(f"  Positions: {len(k_results)}\n")
                avg_clusters = sum(r['merged_clusters'] for r in k_results) / len(k_results)
                f.write(f"  Average merged clusters: {avg_clusters:.1f}\n")
    
    print(f"\n训练总结已保存至: {report_file}")


if __name__ == "__main__":
    main()
