import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DysarthriaFusionModel(nn.Module):
    def __init__(self, audio_feature_size=256, visual_feature_size=512, hidden_size=256):
        super(DysarthriaFusionModel, self).__init__()
        
        # Feature alignment and projection
        self.audio_projection = nn.Sequential(
            nn.Linear(audio_feature_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2)
        )
        
        self.visual_projection = nn.Sequential(
            nn.Linear(visual_feature_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2)
        )
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            hidden_size,
            num_heads=4,
            batch_first=True,
            dropout=0.2
        )
        
        # Fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2)
        )
        
        # Multi-task prediction heads
        self.sync_confidence_head = nn.Sequential(
            nn.Linear(hidden_size // 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.overall_severity_head = nn.Sequential(
            nn.Linear(hidden_size // 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 5),  # 5 severity levels
            nn.Softmax(dim=1)
        )
        
        self.intelligibility_head = nn.Sequential(
            nn.Linear(hidden_size // 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.compensation_pattern_head = nn.Sequential(
            nn.Linear(hidden_size // 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3),  # 3 compensation patterns
            nn.Softmax(dim=1)
        )
        
        # Diagnostic features
        self.diagnostic_features = nn.Sequential(
            nn.Linear(hidden_size // 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
    def forward(self, audio_features, visual_features):
        # Project features to common space
        audio_proj = self.audio_projection(audio_features)
        visual_proj = self.visual_projection(visual_features)
        
        # Prepare for cross-attention
        audio_proj = audio_proj.unsqueeze(1)  # (batch, 1, hidden)
        visual_proj = visual_proj.unsqueeze(1)  # (batch, 1, hidden)
        
        # Cross-modal attention
        attended_audio, _ = self.cross_attention(
            audio_proj, visual_proj, visual_proj
        )
        attended_visual, _ = self.cross_attention(
            visual_proj, audio_proj, audio_proj
        )
        
        # Concatenate attended features
        fused = torch.cat([attended_audio.squeeze(1), 
                          attended_visual.squeeze(1)], dim=1)
        
        # Further fusion
        fused_features = self.fusion_layer(fused)
        
        # Make predictions
        sync_confidence = self.sync_confidence_head(fused_features)
        overall_severity = self.overall_severity_head(fused_features)
        intelligibility = self.intelligibility_head(fused_features)
        compensation_pattern = self.compensation_pattern_head(fused_features)
        diagnostic_features = self.diagnostic_features(fused_features)
        
        return {
            'sync_confidence': sync_confidence,
            'overall_severity': overall_severity,
            'intelligibility_score': intelligibility,
            'compensation_pattern': compensation_pattern,
            'diagnostic_features': diagnostic_features,
            'fused_features': fused_features
        }
    
    def analyze(self, audio_features_dict, visual_features_dict, 
                audio_analysis=None, visual_analysis=None):
        """Perform fusion analysis of audio and visual features"""
        try:
            # Extract feature vectors
            audio_feature_vector = self._extract_audio_feature_vector(audio_features_dict, audio_analysis)
            visual_feature_vector = self._extract_visual_feature_vector(visual_features_dict, visual_analysis)
            
            # Convert to tensors
            audio_tensor = torch.FloatTensor(audio_feature_vector).unsqueeze(0)
            visual_tensor = torch.FloatTensor(visual_feature_vector).unsqueeze(0)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.forward(audio_tensor, visual_tensor)
            
            # Extract and interpret results
            sync_confidence = outputs['sync_confidence'].item()
            intelligibility = outputs['intelligibility_score'].item()
            severity = torch.argmax(outputs['overall_severity']).item()
            compensation = torch.argmax(outputs['compensation_pattern']).item()
            
            # Calculate derived metrics
            derived_metrics = self._calculate_derived_metrics(
                audio_features_dict, 
                visual_features_dict,
                audio_analysis,
                visual_analysis
            )
            
            # Generate comprehensive analysis
            analysis = {
                'sync_confidence': sync_confidence,
                'sync_interpretation': self._interpret_sync_confidence(sync_confidence),
                'intelligibility_score': intelligibility,
                'intelligibility_interpretation': self._interpret_intelligibility(intelligibility),
                'overall_severity': severity,
                'severity_label': self._get_severity_label(severity),
                'compensation_pattern': compensation,
                'compensation_label': self._get_compensation_label(compensation),
                'derived_metrics': derived_metrics,
                'audio_visual_correlation': derived_metrics.get('audio_visual_correlation', 0),
                'recommendation_priority': self._calculate_priority(audio_analysis, visual_analysis),
                'confidence': (sync_confidence + intelligibility) / 2
            }
            
            return analysis
            
        except Exception as e:
            print(f"Fusion analysis error: {e}")
            return {
                'sync_confidence': 0.5,
                'intelligibility_score': 0.5,
                'overall_severity': 2,
                'severity_label': 'Moderate',
                'error': str(e),
                'confidence': 0.5
            }
    
    def _extract_audio_feature_vector(self, audio_features, audio_analysis):
        """Extract comprehensive audio feature vector"""
        features = []
        
        # Basic audio features
        if 'mfcc_mean' in audio_features:
            features.extend(audio_features['mfcc_mean'][:13])  # First 13 MFCCs
        
        # Analysis scores
        if audio_analysis:
            features.append(audio_analysis.get('clarity_score', 0.5))
            features.append(audio_analysis.get('articulation_score', 0.5))
            features.append(audio_analysis.get('fluency_score', 0.5))
            features.append(audio_analysis.get('pitch_stability', 0.5))
            features.append(audio_analysis.get('pitch_variability', 0.5))
        
        # Statistical features
        stats = ['pitch_mean', 'pitch_std', 'energy_mean', 'energy_variance',
                'articulation_rate', 'pause_ratio', 'speech_rate']
        for stat in stats:
            features.append(audio_features.get(stat, 0))
        
        # Pad or truncate to expected size
        expected_size = 256
        if len(features) < expected_size:
            features.extend([0] * (expected_size - len(features)))
        elif len(features) > expected_size:
            features = features[:expected_size]
        
        return np.array(features, dtype=np.float32)
    
    def _extract_visual_feature_vector(self, visual_features, visual_analysis):
        """Extract comprehensive visual feature vector"""
        features = []
        
        # Basic visual features
        if 'temporal_features' in visual_features:
            tf = visual_features['temporal_features']
            features.append(tf.get('mouth_opening_mean', 0))
            features.append(tf.get('mouth_opening_std', 0))
            features.append(tf.get('movement_smoothness', 0))
            features.append(tf.get('sync_consistency', 0))
        
        # Analysis scores
        if visual_analysis:
            features.append(visual_analysis.get('lip_sync_score', 0.5))
            features.append(visual_analysis.get('articulation_score', 0.5))
            features.append(visual_analysis.get('mouth_opening_score', 0.5))
            features.append(visual_analysis.get('movement_smoothness', 0.5))
            features.append(visual_analysis.get('expression_symmetry', 0.5))
        
        # Landmark-based features (simplified)
        if 'lip_landmarks' in visual_features and len(visual_features['lip_landmarks']) > 0:
            # Use first frame's landmarks
            landmarks = visual_features['lip_landmarks'][0].flatten()
            features.extend(landmarks[:20])  # Use first 20 values
        
        # Pad or truncate to expected size
        expected_size = 512
        if len(features) < expected_size:
            features.extend([0] * (expected_size - len(features)))
        elif len(features) > expected_size:
            features = features[:expected_size]
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_derived_metrics(self, audio_features, visual_features, 
                                 audio_analysis, visual_analysis):
        """Calculate metrics derived from both modalities"""
        metrics = {}
        
        # Calculate correlation between audio and visual metrics
        audio_scores = []
        visual_scores = []
        
        if audio_analysis:
            audio_scores.extend([
                audio_analysis.get('clarity_score', 0.5),
                audio_analysis.get('articulation_score', 0.5),
                audio_analysis.get('fluency_score', 0.5)
            ])
        
        if visual_analysis:
            visual_scores.extend([
                visual_analysis.get('lip_sync_score', 0.5),
                visual_analysis.get('articulation_score', 0.5),
                visual_analysis.get('mouth_opening_score', 0.5)
            ])
        
        if len(audio_scores) >= 2 and len(visual_scores) >= 2:
            # Simple correlation estimation
            min_len = min(len(audio_scores), len(visual_scores))
            correlation = np.corrcoef(audio_scores[:min_len], visual_scores[:min_len])[0, 1]
            metrics['audio_visual_correlation'] = max(0, correlation)  # Only positive correlations
        
        # Calculate timing alignment
        if 'duration' in audio_features and 'temporal_features' in visual_features:
            audio_duration = audio_features['duration']
            visual_frames = len(visual_features.get('lip_landmarks', []))
            if visual_frames > 0 and audio_duration > 0:
                expected_frames = audio_duration * 30  # Assuming 30 FPS
                timing_alignment = 1.0 - abs(visual_frames - expected_frames) / expected_frames
                metrics['timing_alignment'] = max(0, timing_alignment)
        
        # Calculate consistency scores
        consistency_scores = []
        if audio_analysis and visual_analysis:
            # Articulation consistency
            audio_artic = audio_analysis.get('articulation_score', 0.5)
            visual_artic = visual_analysis.get('articulation_score', 0.5)
            artic_consistency = 1.0 - abs(audio_artic - visual_artic)
            consistency_scores.append(artic_consistency)
            
            metrics['articulation_consistency'] = artic_consistency
        
        if consistency_scores:
            metrics['overall_consistency'] = np.mean(consistency_scores)
        
        return metrics
    
    def _interpret_sync_confidence(self, confidence):
        """Interpret audio-visual sync confidence"""
        if confidence >= 0.8:
            return "Excellent synchronization"
        elif confidence >= 0.6:
            return "Good synchronization"
        elif confidence >= 0.4:
            return "Moderate synchronization"
        elif confidence >= 0.2:
            return "Poor synchronization"
        else:
            return "Very poor synchronization"
    
    def _interpret_intelligibility(self, score):
        """Interpret intelligibility score"""
        if score >= 0.8:
            return "Highly intelligible"
        elif score >= 0.6:
            return "Mostly intelligible"
        elif score >= 0.4:
            return "Moderately intelligible"
        elif score >= 0.2:
            return "Poorly intelligible"
        else:
            return "Very poorly intelligible"
    
    def _get_severity_label(self, severity_level):
        """Convert severity level to label"""
        labels = ["Normal", "Mild", "Moderate", "Severe", "Very Severe"]
        return labels[min(severity_level, len(labels)-1)]
    
    def _get_compensation_label(self, pattern):
        """Convert compensation pattern to label"""
        patterns = [
            "Minimal compensation",
            "Articulatory compensation", 
            "Prosodic compensation"
        ]
        return patterns[min(pattern, len(patterns)-1)]
    
    def _calculate_priority(self, audio_analysis, visual_analysis):
        """Calculate recommendation priority"""
        priorities = []
        
        if audio_analysis:
            if audio_analysis.get('clarity_score', 1) < 0.6:
                priorities.append(('audio_clarity', 'high'))
            if audio_analysis.get('fluency_score', 1) < 0.6:
                priorities.append(('audio_fluency', 'medium'))
        
        if visual_analysis:
            if visual_analysis.get('lip_sync_score', 1) < 0.6:
                priorities.append(('visual_sync', 'high'))
            if visual_analysis.get('mouth_opening_score', 1) < 0.5:
                priorities.append(('visual_opening', 'medium'))
        
        return priorities[:3]  # Return top 3 priorities