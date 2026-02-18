#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure SOM Classifier V3 with Configurable K-mer - Supports configurable k-mer length
Based on pure_som_classifier_v3.py, referencing the feature extraction approach from som_classifier_from_excel.py

Improvements:
- Supports three k-mer lengths: 4-mer, 5-mer, 6-mer
- Uses length-normalized k-mer frequency features
- Removes complexity and 2-mer features, uses only k-mer frequency
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

# Fix Chinese character display in plots
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class ConfigurableKmerEmbedder:
    """Configurable k-mer length sequence embedder
    
    Features: k-mer frequency (normalized by sequence length) + Excel features
    """
    
    def __init__(self, k=4, use_excel_features=True):
        """
        Args:
            k: k-mer length (4, 5, or 6)
            use_excel_features: Whether to use Excel features
        """
        self.k = k
        self.use_excel_features = use_excel_features
        self.nucleotides = ['A', 'T', 'G', 'C']
        
        # Generate all possible k-mer combinations
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
            print(f"Initializing {k}-mer embedder (k-mer feature dim: {self.kmer_dim}, waiting for Excel features)")
        else:
            self.feature_dim = self.kmer_dim
            print(f"Initializing {k}-mer embedder (pure k-mer, feature dim: {self.feature_dim})")
    
    def set_excel_features(self, excel_feature_names):
        """Set Excel feature names"""
        self.excel_feature_names = excel_feature_names
        self.excel_dim = len(excel_feature_names)
        self.feature_dim = self.kmer_dim + self.excel_dim
        print(f"Excel features added, total feature dim: {self.feature_dim} (k-mer: {self.kmer_dim} + Excel: {self.excel_dim})")
    
    def calculate_kmer_frequencies(self, sequence):
        """Calculate k-mer frequency"""
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
            
        # Convert to frequency
        kmer_frequencies = {}
        for kmer, count in kmer_counts.items():
            kmer_frequencies[kmer] = count / total_kmers
            
        return kmer_frequencies
    
    def calculate_gc_content(self, sequence):
        """Calculate GC content"""
        sequence = sequence.upper()
        if len(sequence) == 0:
            return 0.0
        gc_count = sequence.count('G') + sequence.count('C')
        return gc_count / len(sequence)
    
    def embed_sequence(self, sequence, excel_features=None):
        """Convert a single sequence to feature vector (k-mer frequency + Excel features)
        
        Args:
            sequence: DNA sequence
            excel_features: Corresponding Excel feature vector (already normalized)
        """
        features = []
        
        # 1. Calculate k-mer frequency
        kmer_freqs = self.calculate_kmer_frequencies(sequence)
        
        # Extract features in predefined kmer order
        for kmer in self.kmers:
            freq = kmer_freqs.get(kmer, 0.0)
            features.append(freq)
        
        # Normalize k-mer frequency by sequence length (reference: excel version)
        seq_length = len(sequence)
        if seq_length > 0:
            features = [f / seq_length for f in features]
        
        # 2. If Excel features enabled, append them
        if self.use_excel_features and excel_features is not None:
            features.extend(excel_features)
        
        return np.array(features)
    
    def embed_sequences(self, sequences, excel_features_matrix=None, verbose=False):
        """Batch process sequences
        
        Args:
            sequences: List of sequences
            excel_features_matrix: Excel feature matrix (n_samples, n_excel_features)
            verbose: Whether to output detailed info
        """
        if verbose:
            if self.use_excel_features and excel_features_matrix is not None:
                print(f"Embedding {len(sequences)} sequences (using {self.k}-mer + Excel features)...")
            else:
                print(f"Embedding {len(sequences)} sequences (using {self.k}-mer)...")
        
        embedded_sequences = []
        for i, seq in enumerate(sequences):
            if verbose and i % 100 == 0:
                print(f"  Progress: {i}/{len(sequences)}")
            
            # Get corresponding Excel features
            excel_feats = excel_features_matrix[i] if excel_features_matrix is not None else None
            
            features = self.embed_sequence(seq, excel_feats)
            embedded_sequences.append(features)
        
        embedded_matrix = np.array(embedded_sequences)
        if verbose:
            print(f"Embedding complete: {embedded_matrix.shape}")
            if self.use_excel_features and excel_features_matrix is not None:
                print(f"  - k-mer features: {self.kmer_dim} dim")
                print(f"  - Excel features: {self.excel_dim} dim")
        
        return embedded_matrix


class PureSOMClassifierKmer:
    """SOM classifier with configurable k-mer length (supports Excel features)"""
    
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
        
        # Initialize embedder
        self.embedder = ConfigurableKmerEmbedder(k=k, use_excel_features=use_excel_features)
        
        # Model state
        self.weights = None
        self.scaler = None
        self.trained = False
        self.training_sequences = None
        self.sequence_positions = None
        self.feature_matrix = None
        self.grid_analysis = None
    
    def find_bmu(self, sample):
        """Find Best Matching Unit (BMU)"""
        distances = np.sqrt(((self.weights - sample) ** 2).sum(axis=2))
        return np.unravel_index(np.argmin(distances), distances.shape)
    
    def update_weights(self, sample, bmu, iteration):
        """Update SOM weights"""
        # Calculate learning rate decay
        learning_rate = self.learning_rate * np.exp(-iteration / self.max_iter)
        
        # Calculate neighborhood radius decay
        radius = self.neighborhood_factor * np.exp(-iteration / self.max_iter)
        
        # Update weights
        for i in range(self.height):
            for j in range(self.width):
                # Calculate distance to BMU
                distance = np.sqrt((i - bmu[0])**2 + (j - bmu[1])**2)
                
                # If within neighborhood, update weights
                if distance <= radius:
                    influence = np.exp(-(distance**2) / (2 * radius**2))
                    self.weights[i, j] += learning_rate * influence * (sample - self.weights[i, j])
    
    def train(self, sequences, verbose=False, sample_counts=None, excel_features=None):
        """Train SOM model
        
        Args:
            sequences: List of training sequences
            verbose: Whether to output detailed info
            sample_counts: Sequence sample count dict {sequence: sample_count}
            excel_features: Excel feature matrix (n_samples, n_excel_features)
        """
        if verbose:
            print(f"\n{'='*60}")
            if self.use_excel_features and excel_features is not None:
                print(f"Training SOM classifier for position {self.position_name} (k={self.k} + Excel features)...")
            else:
                print(f"Training pure SOM classifier for position {self.position_name} (k={self.k})...")
            print(f"Number of sequences: {len(sequences)}")
            if sample_counts:
                total_alleles = sum(sample_counts.values())
                print(f"Total alleles: {total_alleles}")
        
        # Store training sequences
        self.training_sequences = sequences.copy()
        
        # Store sample counts
        if sample_counts:
            self.sequence_sample_counts = sample_counts.copy()
        else:
            self.sequence_sample_counts = {seq: 1 for seq in sequences}
        
        # 1. Sequence embedding
        if verbose:
            feature_desc = f"{self.k}-mer"
            if self.use_excel_features and excel_features is not None:
                feature_desc += " + Excel features"
            print(f"Step 1: Sequence feature embedding ({feature_desc})")
        
        self.feature_matrix = self.embedder.embed_sequences(sequences, excel_features, verbose)
        if verbose:
            print(f"Feature dimension: {self.feature_matrix.shape}")
        
        # 2. Feature standardization
        if verbose:
            print(f"Step 2: Feature standardization")
        self.scaler = StandardScaler()
        feature_matrix_scaled = self.scaler.fit_transform(self.feature_matrix)
        
        # 3. Initialize SOM weights
        if verbose:
            print(f"Step 3: Initialize SOM grid ({self.height}x{self.width})")
        feature_dim = feature_matrix_scaled.shape[1]
        self.weights = np.random.uniform(-1, 1, (self.height, self.width, feature_dim))
        
        # 4. SOM training
        if verbose:
            print(f"Step 4: SOM training ({self.max_iter} iterations)")
        for iteration in range(self.max_iter):
            if verbose and iteration % 500 == 0:
                print(f"  Training progress: {iteration}/{self.max_iter}")
                
            # Random sample selection
            sample_idx = np.random.randint(0, len(feature_matrix_scaled))
            sample = feature_matrix_scaled[sample_idx]
            
            # Find BMU and update weights
            bmu = self.find_bmu(sample)
            self.update_weights(sample, bmu, iteration)
        
        # 5. Assign SOM grid positions for training sequences
        if verbose:
            print(f"Step 5: Assign SOM grid positions")
        self.sequence_positions = []
        for features in feature_matrix_scaled:
            bmu = self.find_bmu(features)
            self.sequence_positions.append(bmu)
            
        self.trained = True
        
        # 6. Analyze SOM grid
        self.analyze_som_grid(verbose)
        self.calculate_position_quality(feature_matrix_scaled, verbose)
        
        return feature_matrix_scaled
    
    def analyze_som_grid(self, verbose=False):
        """Analyze sequence features at each SOM grid position"""
        if verbose:
            print(f"Step 6: Analyze SOM grid")
        
        # Count sequences per grid position
        position_sequences = defaultdict(list)
        for seq_idx, position in enumerate(self.sequence_positions):
            position_sequences[position].append(seq_idx)
        
        occupied_positions = len(position_sequences)
        total_positions = self.height * self.width
        
        if verbose:
            print(f"Occupied grid positions: {occupied_positions}/{total_positions}")
        
        # Analyze each occupied grid position
        self.grid_analysis = {}
        for position, seq_indices in position_sequences.items():
            sequences_at_position = [self.training_sequences[i] for i in seq_indices]
            
            # Calculate statistical features for sequences at this position
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
        
        # Print grid analysis results
        if verbose:
            print(f"\nSOM grid analysis results:")
            for position, analysis in sorted(self.grid_analysis.items()):
                i, j = position
                print(f"Position ({i},{j}): {analysis['count']} sequences ({analysis['percentage']:.1f}%), "
                      f"avg length {analysis['avg_length']:.1f}bp")
    
    def calculate_position_quality(self, feature_matrix_scaled, verbose=False):
        """Calculate intrinsic quality of each SOM position"""
        if verbose:
            print(f"\nStep 7: Calculate position quality")
        
        self.position_quality = {}
        
        # Group features by position
        position_features = defaultdict(list)
        for seq_idx, position in enumerate(self.sequence_positions):
            position_features[position].append(feature_matrix_scaled[seq_idx])
        
        # Calculate average distance for each position
        for position, features_list in position_features.items():
            if len(features_list) == 0:
                continue
                
            features_array = np.array(features_list)
            weight_vector = self.weights[position]
            
            # Calculate distance between all sequences at this position and weight vector
            distances = []
            for features in features_array:
                distance = np.linalg.norm(features - weight_vector)
                distances.append(distance)
            
            avg_distance = np.mean(distances)
            std_distance = np.std(distances)
            
            # Quality level
            if avg_distance < 3:
                quality_level = "Excellent"
            elif avg_distance < 6:
                quality_level = "Good"
            elif avg_distance < 10:
                quality_level = "Fair"
            else:
                quality_level = "Poor"
            
            self.position_quality[position] = {
                'avg_distance': avg_distance,
                'std_distance': std_distance,
                'count': len(features_list),
                'quality_level': quality_level
            }
            
            if verbose:
                i, j = position
                print(f"  Position ({i},{j}): avg distance {avg_distance:.3f}, quality {quality_level}")
    
    def merge_similar_positions(self, merge_threshold=0.8, verbose=False):
        """Merge similar grid positions"""
        if not self.trained:
            raise ValueError("Model not yet trained")
        
        if verbose:
            print(f"\nMerging similar positions (Euclidean distance threshold: {merge_threshold})")
        
        # Get all occupied positions
        occupied_positions = list(self.grid_analysis.keys())
        
        if len(occupied_positions) <= 1:
            if verbose:
                print("Only one or no occupied positions, no merge needed")
            self.merged_sequence_positions = self.sequence_positions.copy()
            return {}, {}
        
        # Extract weight vectors for Euclidean distance clustering
        position_weights = []
        for pos in occupied_positions:
            weight_vector = self.weights[pos].flatten()
            position_weights.append(weight_vector)
        
        position_weights = np.array(position_weights)
        
        # DBSCAN clustering using Euclidean distance
        clustering = DBSCAN(eps=merge_threshold, min_samples=1, metric='euclidean')
        cluster_labels = clustering.fit_predict(position_weights)
        
        # Create position mapping
        position_mapping = {}
        cluster_centers = {}
        
        for cluster_id in set(cluster_labels):
            cluster_positions = [occupied_positions[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
            
            # Select cluster center (position with most sequences)
            center_position = max(cluster_positions, key=lambda pos: self.grid_analysis[pos]['count'])
            cluster_centers[cluster_id] = center_position
            
            # Map all positions to center
            for pos in cluster_positions:
                position_mapping[pos] = center_position
        
        # Apply mapping to sequence positions
        self.merged_sequence_positions = []
        for pos in self.sequence_positions:
            merged_pos = position_mapping.get(pos, pos)
            self.merged_sequence_positions.append(merged_pos)
        
        if verbose:
            original_clusters = len(occupied_positions)
            merged_clusters = len(set(self.merged_sequence_positions))
            print(f"Original cluster count: {original_clusters}")
            print(f"Merged cluster count: {merged_clusters}")
            print(f"Reduced by {original_clusters - merged_clusters} clusters")
        
        return position_mapping, cluster_centers
    
    def visualize_som_2d_with_patients(self, save_dir, sample_ids, patient_ids, population_map=None, verbose=False):
        """Generate 2D SOM scatter plot with patient sample and population annotations
        
        Args:
            save_dir: Save directory
            sample_ids: List of sample IDs
            patient_ids: List of patient IDs
            population_map: Sample ID to population code mapping dict {sample_id: population}
            verbose: Whether to output detailed info
        """
        if not self.trained:
            raise ValueError("Model not yet trained")
        
        # Use merged positions if available
        if hasattr(self, 'merged_sequence_positions'):
            positions = self.merged_sequence_positions
        else:
            positions = self.sequence_positions
        
        # Create figure (wider to accommodate longer legend)
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        
        # Group samples by position
        position_samples = {}
        for idx, pos in enumerate(positions):
            if pos not in position_samples:
                position_samples[pos] = []
            position_samples[pos].append(idx)
        
        # Define colors
        colors = plt.cm.tab20(np.linspace(0, 1, max(20, len(position_samples))))
        patient_set = set(patient_ids) if patient_ids else set()
        
        # Plot scatter for each cluster
        for cluster_idx, (pos, sample_indices) in enumerate(sorted(position_samples.items())):
            # Get cluster coordinates
            if isinstance(pos, tuple):
                cluster_x, cluster_y = pos[1], pos[0]
            else:
                cluster_x = pos % self.width
                cluster_y = pos // self.width
            
            # Add jitter
            n_samples = len(sample_indices)
            jitter = 0.15
            x_coords = cluster_x + np.random.uniform(-jitter, jitter, n_samples)
            y_coords = cluster_y + np.random.uniform(-jitter, jitter, n_samples)
            
            # Separate patient and control
            patient_mask = [sample_ids[i] in patient_set if i < len(sample_ids) else False 
                           for i in sample_indices]
            
            # Count
            n_patients = sum(patient_mask)
            n_controls = sum(not m for m in patient_mask)
            
            # Count population distribution of control samples
            population_counts = {}
            if population_map:
                for i in sample_indices:
                    if i < len(sample_ids):
                        sid = sample_ids[i]
                        # Check if control sample
                        if sid not in patient_set:
                            pop = population_map.get(sid, None)
                            if pop:
                                population_counts[pop] = population_counts.get(pop, 0) + 1
            
            # Build cluster label
            if n_controls > 0 and population_counts:
                # Sort by population count
                pop_strs = [f"{count} {pop}" for pop, count in sorted(population_counts.items(), 
                                                                       key=lambda x: (-x[1], x[0]))]
                pop_info = ", ".join(pop_strs)
                cluster_label = f"Cluster ({int(cluster_x)}, {int(cluster_y)}) ({n_patients} patients, {n_controls} controls: {pop_info})"
            else:
                cluster_label = f"Cluster ({int(cluster_x)}, {int(cluster_y)}) ({n_patients} patients, {n_controls} controls)"
            
            # Plot control samples
            if n_controls > 0:
                non_patient_x = x_coords[[not m for m in patient_mask]]
                non_patient_y = y_coords[[not m for m in patient_mask]]
                ax.scatter(non_patient_x, non_patient_y, 
                          c=[colors[cluster_idx % len(colors)]], 
                          s=100, alpha=0.6, edgecolors='gray', linewidths=0.5,
                          label=cluster_label)
            
            # Plot patient samples (thick black border)
            if n_patients > 0:
                patient_x = x_coords[patient_mask]
                patient_y = y_coords[patient_mask]
                # If this cluster has no control samples, add label
                patient_label = cluster_label if n_controls == 0 else None
                ax.scatter(patient_x, patient_y, 
                          c=[colors[cluster_idx % len(colors)]], 
                          s=150, alpha=0.9, edgecolors='black', linewidths=3,
                          marker='o', zorder=10, label=patient_label)
        
        # Set axes
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(-0.5, self.height - 0.5)
        ax.invert_yaxis()
        
        # Add grid
        ax.set_xticks(range(self.width))
        ax.set_yticks(range(self.height))
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_aspect('equal')
        
        # Title
        k_mer = f"{self.k}-mer" if hasattr(self, 'k') else "SOM"
        ax.set_title(f'{self.position_name} - 2D SOM Visualization ({k_mer})\nPatient samples marked with thick black border', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('SOM Grid X', fontsize=12)
        ax.set_ylabel('SOM Grid Y', fontsize=12)
        
        # Add statistics
        patient_count = sum(1 for idx in range(len(sample_ids)) if sample_ids[idx] in patient_set)
        
        ax.text(0.02, 0.98, 
               f'Total: {len(sample_ids)} samples\nPatient: {patient_count} ({patient_count/len(sample_ids)*100:.1f}%)\nClusters: {len(position_samples)}',
               transform=ax.transAxes, fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # Add legend (show all clusters)
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 0:
            # Adjust font size and columns based on cluster count (legend is longer with population info)
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
        
        # Save
        plot_file = os.path.join(save_dir, f"{self.position_name}_som_2d_patient.png")
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        if verbose:
            print(f"2D SOM patient plot saved: {plot_file}")
        
        return plot_file
    
    def visualize_som_grid(self, save_dir, verbose=False):
        """Visualize SOM grid analysis"""
        if not self.trained:
            raise ValueError("Model not yet trained")
            
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'{self.position_name} Pure SOM ({self.k}-mer) Grid Visualization', fontsize=16)
        
        # 1. Sequence count per grid position
        count_map = np.zeros((self.height, self.width))
        for position in self.sequence_positions:
            i, j = position
            count_map[i, j] += 1
            
        im1 = axes[0,0].imshow(count_map, cmap='YlOrRd', interpolation='nearest')
        axes[0,0].set_title('Sequence Count per Grid Position')
        axes[0,0].set_xlabel('SOM Width')
        axes[0,0].set_ylabel('SOM Height')
        plt.colorbar(im1, ax=axes[0,0], label='Number of Sequences')
        
        # Add text annotations for non-zero positions
        for i in range(self.height):
            for j in range(self.width):
                if count_map[i, j] > 0:
                    axes[0,0].text(j, i, f'{int(count_map[i, j])}', 
                                  ha='center', va='center', color='white', fontweight='bold')
        
        # 2. Grid occupancy
        occupancy_map = (count_map > 0).astype(int)
        im2 = axes[0,1].imshow(occupancy_map, cmap='RdYlGn', interpolation='nearest')
        axes[0,1].set_title('Grid Occupancy')
        axes[0,1].set_xlabel('SOM Width')
        axes[0,1].set_ylabel('SOM Height')
        plt.colorbar(im2, ax=axes[0,1], label='Occupied (1) / Empty (0)')
        
        # 3. Average sequence length
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
        
        # 4. Grid position labels
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
        
        # Add text annotations for position labels
        for i in range(self.height):
            for j in range(self.width):
                if count_map[i, j] > 0:
                    label = i * self.width + j
                    axes[1,1].text(j, i, f'{label}', 
                                  ha='center', va='center', color='white', fontweight='bold')
        
        # Save figure
        plot_file = os.path.join(save_dir, f"{self.position_name}_{self.k}mer_som_visualization.png")
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        if verbose:
            print(f"SOM visualization saved: {plot_file}")
        return plot_file
    
    def save_detailed_classification_report(self, save_dir, verbose=False, use_merged=True):
        """Generate detailed classification report file"""
        if not self.trained:
            raise ValueError("Model not yet trained")
        
        report_file = os.path.join(save_dir, f"{self.position_name}_{self.k}mer_som_detailed.txt")
        
        # Choose original or merged positions
        if use_merged and hasattr(self, 'merged_sequence_positions'):
            sequence_positions = self.merged_sequence_positions
            title_suffix = " (merged)"
        else:
            sequence_positions = self.sequence_positions
            title_suffix = ""
        
        # Group sequences by SOM position
        position_sequences = {}
        for seq_idx, position in enumerate(sequence_positions):
            if position not in position_sequences:
                position_sequences[position] = []
            position_sequences[position].append(seq_idx)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            # Header info
            f.write(f"Position: {self.position_name}\n")
            f.write(f"Pure SOM classification detailed results ({self.k}-mer){title_suffix}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total sequences: {len(self.training_sequences)}\n")
            # Calculate total alleles
            sample_counts_dict = getattr(self, 'sequence_sample_counts', {})
            total_alleles = sum(sample_counts_dict.get(seq, 1) for seq in self.training_sequences)
            f.write(f"Total alleles: {total_alleles}\n")
            if use_merged and hasattr(self, 'merged_sequence_positions'):
                original_positions = len(set(self.sequence_positions))
                merged_positions = len(position_sequences)
                f.write(f"Original SOM positions: {original_positions}\n")
                f.write(f"Merged positions: {merged_positions}\n\n")
            else:
                f.write(f"SOM positions: {len(position_sequences)}\n\n")
            
            # Sort positions by sequence count
            sorted_positions = sorted(position_sequences.items(), 
                                    key=lambda x: len(x[1]), reverse=True)
            
            for position_id, seq_indices in sorted_positions:
                sequences_at_position = [self.training_sequences[i] for i in seq_indices]
                
                # Calculate statistics
                lengths = [len(seq) for seq in sequences_at_position]
                gc_contents = [self.embedder.calculate_gc_content(seq) for seq in sequences_at_position]
                unique_sequences = list(set(sequences_at_position))
                
                # Write position info
                if use_merged and hasattr(self, 'merged_sequence_positions'):
                    if isinstance(position_id, tuple):
                        pos_str = f"({int(position_id[0])}, {int(position_id[1])})"
                    else:
                        pos_str = str(int(position_id))
                    f.write(f"Merged cluster {pos_str}\n")
                else:
                    if isinstance(position_id, tuple):
                        pos_str = f"[{int(position_id[0])}, {int(position_id[1])}]"
                    else:
                        pos_str = f"[{int(position_id)}]"
                    f.write(f"SOM position {pos_str}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Sequence count: {len(sequences_at_position)}\n")
                # Calculate allele count for this cluster
                cluster_allele_count = sum(sample_counts_dict.get(seq, 1) for seq in sequences_at_position)
                f.write(f"Allele count: {cluster_allele_count}\n")
                f.write(f"Length range: {min(lengths)}-{max(lengths)}bp\n")
                f.write(f"Average length: {np.mean(lengths):.1f}bp\n")
                f.write(f"Unique sequences: {len(unique_sequences)}\n")
                
                # Grid analysis
                f.write("Cluster analysis:\n")
                f.write(f"  count: {len(sequences_at_position)}\n")
                f.write(f"  percentage: {len(sequences_at_position) / len(self.training_sequences) * 100:.2f}%\n")
                f.write(f"  avg_length: {np.mean(lengths):.1f}\n")
                f.write(f"  length_range: [{min(lengths)}, {max(lengths)}]\n")
                f.write(f"  avg_gc_content: {np.mean(gc_contents):.6f}\n")
                f.write(f"  gc_range: [{min(gc_contents):.6f}, {max(gc_contents):.6f}]\n")
                
                # Example sequence
                example_seq = sequences_at_position[0]
                if len(example_seq) > 50:
                    example_seq = example_seq[:50] + "..."
                f.write(f"  example_sequence: {example_seq}\n")
                
                # Sequence list
                f.write("Sequence list:\n")
                for i, seq in enumerate(sequences_at_position, 1):
                    seq_allele_count = sample_counts_dict.get(seq, 1)
                    f.write(f"   {i}. [alleles:{seq_allele_count}] {seq}\n")
                
                f.write("\n")
        
        if verbose:
            print(f"Detailed classification report saved: {report_file}")
        return report_file
    
    def save_model(self, save_dir, verbose=False):
        """Save model parameters"""
        if not self.trained:
            raise ValueError("Model not yet trained")
            
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
        
        # If merged classification results exist, save them too
        if hasattr(self, 'merged_sequence_positions'):
            model_data['merged_sequence_positions'] = self.merged_sequence_positions
            model_data['has_merged_results'] = True
        else:
            model_data['has_merged_results'] = False
        
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)
            
        if verbose:
            print(f"SOM model saved: {model_file}")
        return model_file


def has_consecutive_repeats(sequence, max_repeat_length=10):
    """Check if sequence has consecutive identical base repeats exceeding specified length"""
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
    """Load sample sequences from sequence file"""
    if verbose:
        print(f"\nLoading sequence file: {seq_file}")
    
    # Read sequence file
    sequences_dict = {}
    with open(seq_file, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                sample_id, sequence = line.split(':', 1)
                sequences_dict[sample_id] = sequence
    
    if verbose:
        print(f"Sequence file contains {len(sequences_dict)} sequences")
    
    # Extract sequences in sample_ids order
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
        print(f"Warning: The following samples were not found in sequence file: {missing_samples[:5]}")
    
    if verbose:
        print(f"Successfully matched {len([s for s in sequences if s])} sequences")
    
    return sequences


def load_excel_data_with_features(excel_file, seq_file, sheet_name='700bp', verbose=False):
    """Load sample IDs, sequences, and Excel features from Excel file"""
    import pandas as pd
    
    if verbose:
        print(f"Loading Excel file: {excel_file}")
        print(f"Sheet: {sheet_name}")
    
    # Read Excel file without header
    df_raw = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
    
    if verbose:
        print(f"Raw data shape: {df_raw.shape}")
    
    # Row 1: sample IDs, Row 2: groups, Row 3+: features
    sample_ids = df_raw.iloc[0, 1:].astype(str).tolist()  # Skip first ('sample')
    groups = df_raw.iloc[1, 1:].astype(str).tolist()  # Group info (patient/control)
    
    # Extract feature names (first column, from row 3, i.e. index=2)
    feature_names = df_raw.iloc[2:, 0].astype(str).tolist()
    
    # Extract feature data (from row 3, from column 2)
    feature_data = df_raw.iloc[2:, 1:].values.T  # Transpose: samples in rows, features in columns
    
    if verbose:
        print(f"Sample count: {len(sample_ids)}")
        print(f"Excel feature count: {len(feature_names)}")
        print(f"First 10 feature names: {feature_names[:10]}")
        print(f"Group info: patient={groups.count('patient')}, control={groups.count('control')}")
    
    # Check for length feature
    length_idx = None
    for i, name in enumerate(feature_names):
        if 'lenth' in name.lower() or 'length' in name.lower():
            length_idx = i
            break
    
    if length_idx is None:
        raise ValueError("Length or lenth feature not found")
    
    # Extract length column
    length_col = feature_data[:, length_idx].astype(float)
    
    if verbose:
        print(f"Length feature at column {length_idx} ({feature_names[length_idx]})")
        print(f"Length range: {np.min(length_col)}-{np.max(length_col)}")
    
    # Remove length column, get other features
    feature_indices = [i for i in range(len(feature_names)) if i != length_idx]
    excel_features = feature_data[:, feature_indices].astype(float)
    excel_feature_names = [feature_names[i] for i in feature_indices]
    
    # Normalize Excel features by length (divide by length)
    # Avoid division by zero
    length_col = np.where(length_col == 0, 1, length_col)
    normalized_excel_features = excel_features / length_col[:, np.newaxis]
    
    if verbose:
        print(f"Excel features normalized by length")
        print(f"Normalized feature range example (first feature): {np.min(normalized_excel_features[:, 0]):.6f}-{np.max(normalized_excel_features[:, 0]):.6f}")
    
    # Load sequences
    sequences = load_sequences_from_file(seq_file, sample_ids, verbose)
    
    # Filter out empty sequences while keeping corresponding Excel features
    valid_data = []
    valid_excel_features = []
    
    for i, (sid, seq) in enumerate(zip(sample_ids, sequences)):
        if seq:
            valid_data.append((sid, seq))
            valid_excel_features.append(normalized_excel_features[i])
    
    valid_excel_features = np.array(valid_excel_features)
    
    if verbose:
        print(f"Valid sample count: {len(valid_data)}")
        print(f"Excel feature matrix shape: {valid_excel_features.shape}")
    
    return valid_data, valid_excel_features, excel_feature_names


def train_single_position_from_excel(excel_file, seq_file, sheet_name, position, save_dir, k, 
                                     width, height, learning_rate, max_iter, neighborhood_factor, 
                                     merge_threshold, verbose, force=False, use_excel_features=True):
    """Train SOM model for a single position from Excel file"""
    try:
        # Check if result files already exist
        suffix = "excel" if use_excel_features else "pure"
        model_file = os.path.join(save_dir, f"{position}_{k}mer_{suffix}_som_model.pkl")
        report_file = os.path.join(save_dir, f"{position}_{k}mer_{suffix}_som_detailed.txt")
        
        if not force and os.path.exists(model_file) and os.path.exists(report_file):
            if verbose:
                print(f"⏭️  Skipping position {position} ({k}-mer {suffix} model already exists)")
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
                feature_desc += " + Excel features"
            print(f"Processing position: {position} ({feature_desc})")
        
        # Load data from Excel (including Excel features)
        if use_excel_features:
            sample_data, excel_features, excel_feature_names = load_excel_data_with_features(
                excel_file, seq_file, sheet_name, verbose
            )
        else:
            # Simplified version, no Excel features
            sample_data, excel_features, excel_feature_names = load_excel_data_with_features(
                excel_file, seq_file, sheet_name, verbose
            )
            excel_features = None  # Do not use Excel features
        
        if not sample_data:
            if verbose:
                print(f"Warning: No valid data found")
            return None
        
        # Extract sequences and sample_ids
        sample_ids_original = [sid for sid, _ in sample_data]
        sequences = [seq for _, seq in sample_data]
        sequence_sample_counts = {seq: 1 for seq in sequences}
        
        if verbose:
            print(f"Original sequence count: {len(sequences)}")
            total_alleles = len(sequences)
            print(f"Total alleles: {total_alleles}")
            if use_excel_features and excel_features is not None:
                print(f"Excel feature dimension: {excel_features.shape}")
        
        # Apply sequence filtering (no external module dependency)
        min_length = k + 1  # At least 1 longer than k-mer length
        
        # Simple filter: length and homopolymer
        def is_homopolymer_at_ends(seq, length=8):
            """Check if sequence has homopolymer at ends"""
            if len(seq) < length:
                return False
            # Check first 8
            if len(set(seq[:length])) == 1:
                return True
            # Check last 8
            if len(set(seq[-length:])) == 1:
                return True
            return False
        
        # Synchronize filtering of sequences and sample_ids
        filtered_data = [(sid, seq) for sid, seq in zip(sample_ids_original, sequences) 
                        if len(seq) >= min_length and not is_homopolymer_at_ends(seq)]
        filtered_sequences = [seq for _, seq in filtered_data]
        filtered_sample_ids = [sid for sid, _ in filtered_data]
        
        if verbose:
            print(f"Filtered sequence count: {len(filtered_sequences)} (min length: {min_length})")
        
        # No longer filter consecutive repeat bases, use all sequences passing basic filter
        valid_sequences = filtered_sequences
        valid_sample_ids = filtered_sample_ids
        valid_sample_counts = {}
        for seq in valid_sequences:
            valid_sample_counts[seq] = sequence_sample_counts.get(seq, 1)
        
        if len(valid_sequences) < 10:
            if verbose:
                print(f"Warning: Position {position} has too few valid sequences ({len(valid_sequences)}), skipping")
            return None
            
        if verbose:
            print(f"Final valid sequence count: {len(valid_sequences)}")
            
            # Output filtered samples (if any)
            filtered_out = set(sample_ids_original) - set(valid_sample_ids)
            if filtered_out:
                print(f"\n⚠️  Filtered out samples ({len(filtered_out)}):")
                for sid in sorted(filtered_out):
                    print(f"   - {sid}")
        
        # Filter Excel features, keep only valid sequence features
        if use_excel_features and excel_features is not None:
            # Create mapping from original sequence to index
            seq_to_idx = {seq: i for i, seq in enumerate(sequences)}
            valid_excel_features = []
            for seq in valid_sequences:
                if seq in seq_to_idx:
                    valid_excel_features.append(excel_features[seq_to_idx[seq]])
            valid_excel_features = np.array(valid_excel_features)
            
            if verbose:
                print(f"Filtered Excel feature dimension: {valid_excel_features.shape}")
        else:
            valid_excel_features = None
        
        # Create and train SOM model
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
        
        # Set Excel feature names
        if use_excel_features and excel_feature_names:
            som.embedder.set_excel_features(excel_feature_names)
        
        # Train model
        som.train(valid_sequences, verbose=verbose, sample_counts=valid_sample_counts, 
                 excel_features=valid_excel_features)
        
        # Merge similar clusters
        if merge_threshold > 0:
            merged_mapping, cluster_centers = som.merge_similar_positions(merge_threshold, verbose)
        
        # Generate visualization
        suffix = "excel" if use_excel_features else "pure"
        original_name = som.position_name
        som.position_name = f"{original_name}_{k}mer_{suffix}"
        
        plot_file = som.visualize_som_grid(save_dir, verbose=verbose)
        
        # Generate 2D patient annotation plot (using actual training samples)
        try:
            import pandas as pd
            df_raw = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
            excel_sample_ids_all = df_raw.iloc[0, 1:].astype(str).tolist()
            groups_all = df_raw.iloc[1, 1:].astype(str).tolist()
            
            # Create sample_id to group mapping
            sid_to_group = {sid: grp for sid, grp in zip(excel_sample_ids_all, groups_all)}
            
            # Use valid_sample_ids filtered during training
            # Extract patient IDs (from valid_sample_ids where group is 'patient')
            patient_id_list = [sid for sid in valid_sample_ids if sid_to_group.get(sid, '') == 'patient']
            
            # Load population info
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
                
                # Create extended mapping (handle _1 and _2 suffixes)
                extended_map = {}
                for sid in valid_sample_ids:
                    # Try direct match first
                    if sid in population_map:
                        extended_map[sid] = population_map[sid]
                    else:
                        # Try removing _1 or _2 suffix
                        base_name = sid.rsplit('_', 1)[0] if '_' in sid else sid
                        if base_name in population_map:
                            extended_map[sid] = population_map[base_name]
                
                if verbose:
                    print(f"Loaded population info: {len(population_map)} samples")
                    print(f"Matched population info: {len(extended_map)} samples")
            else:
                extended_map = None
                if verbose:
                    print(f"ethics.txt not found, skipping population info")
            
            if verbose:
                print(f"Actual training sample count: {len(valid_sample_ids)}")
                print(f"Patient sample count: {len(patient_id_list)}")
            
            # Generate 2D patient plot
            patient_plot = som.visualize_som_2d_with_patients(save_dir, valid_sample_ids, 
                                                             patient_id_list, 
                                                             population_map=extended_map,
                                                             verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to generate 2D patient plot: {e}")
            patient_plot = None
        
        # Generate detailed classification report
        report_file = som.save_detailed_classification_report(save_dir, verbose=verbose, 
                                                              use_merged=merge_threshold > 0)
        
        # Export cluster composition info (for statistical analysis)
        try:
            import pandas as pd
            # Get cluster positions
            if hasattr(som, 'merged_sequence_positions'):
                positions = som.merged_sequence_positions
            else:
                positions = som.sequence_positions
            
            # Create cluster composition data
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
                print(f"Cluster composition info saved: {cluster_csv}")
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to export cluster composition: {e}")
        
        # Save model
        model_file = som.save_model(save_dir, verbose=verbose)
        
        # Restore original name
        som.position_name = original_name
        
        if verbose:
            print(f"Position {position} ({k}-mer) training complete:")
            print(f"  - Model file: {model_file}")
            print(f"  - Visualization: {plot_file}")
            print(f"  - Detailed report: {report_file}")
        
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
            print(f"Error training position {position} ({k}-mer): {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train pure SOM classifier with configurable k-mer length')
    parser.add_argument('--excel-file', type=str, default='characte of seq.xlsx', help='Excel data file path')
    parser.add_argument('--seq-file', type=str, default='seqs_700bp.txt', help='Sequence file path')
    parser.add_argument('--sheet-name', type=str, default='700bp', help='Excel sheet name')
    parser.add_argument('--width', type=int, default=6, help='SOM grid width')
    parser.add_argument('--height', type=int, default=6, help='SOM grid height')
    parser.add_argument('--learning-rate', type=float, default=0.02, help='Learning rate')
    parser.add_argument('--max-iter', type=int, default=5000, help='Maximum iterations')
    parser.add_argument('--neighborhood-factor', type=float, default=3.0, help='Neighborhood factor')
    parser.add_argument('--merge-threshold', type=float, default=0.8, help='Cluster merge threshold')
    parser.add_argument('--n-jobs', type=int, default=10, help='Number of parallel processes')
    parser.add_argument('--all', action='store_true', help='Process all positions')
    parser.add_argument('--force', action='store_true', help='Force reprocess existing files')
    parser.add_argument('--no-excel-features', action='store_true', 
                       help='Do not use Excel features (k-mer only)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    use_excel_features = not args.no_excel_features
    
    print("=== SOM Classifier with Configurable K-mer (from Excel) ===")
    print(f"Grid: {args.width}x{args.height}, LR: {args.learning_rate}, Iter: {args.max_iter}")
    print(f"Neighborhood Factor: {args.neighborhood_factor}, Merge Threshold: {args.merge_threshold}")
    print(f"K-mer lengths: 4, 5, 6")
    if use_excel_features:
        print(f"Features: k-mer frequency + Excel features (length-normalized)")
    else:
        print(f"Features: pure k-mer frequency (no Excel features)")
    
    # Check if files exist
    if not os.path.exists(args.excel_file):
        print(f"❌ Error: Excel file not found: {args.excel_file}")
        return
    
    if not os.path.exists(args.seq_file):
        print(f"❌ Error: Sequence file not found: {args.seq_file}")
        return
    
    print(f"Excel file: {args.excel_file}")
    print(f"Sequence file: {args.seq_file}")
    print(f"Sheet: {args.sheet_name}")
    
    # Position names (inferred from Excel file)
    target_positions = ["HSF1"]  # Default position name
    print(f"Processing positions: {target_positions}")
    
    # Create save directory
    save_dir = "pure_som_kmer_models"
    os.makedirs(save_dir, exist_ok=True)
    
    # Train three k-mer models (4-mer, 5-mer, 6-mer)
    all_results = []
    
    for k in [4, 5, 6]:
        print(f"\n{'='*80}")
        print(f"Training {k}-mer model")
        print(f"{'='*80}")
        
        # Prepare training arguments
        train_args = [(args.excel_file, args.seq_file, args.sheet_name, pos, save_dir, k, 
                      args.width, args.height, args.learning_rate, args.max_iter, 
                      args.neighborhood_factor, args.merge_threshold, args.verbose, args.force,
                      use_excel_features) 
                     for pos in target_positions]
        
        # Multi-process training (only one position here, multi-process not needed)
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
        
        print(f"\n✅ {k}-mer model training complete! Time: {int(elapsed//60)}m {int(elapsed%60)}s")
        print(f"   Processed: {processed_count} positions | Skipped: {skipped_count} positions")
        
        all_results.extend(valid_results)
    
    # Generate summary report
    report_file = os.path.join(save_dir, "training_summary.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=== Pure SOM Classifier with Configurable K-mer - Training Summary ===\n\n")
        f.write(f"Parameters:\n")
        f.write(f"  Grid: {args.width}x{args.height}\n")
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
    
    print(f"\nTraining summary saved to: {report_file}")


if __name__ == "__main__":
    main()
