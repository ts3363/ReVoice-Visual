import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class VisualAnalysisModel(nn.Module):
    def __init__(self, input_size=20*2, hidden_size=256):  # 20 lip points * (x,y)
        super(VisualAnalysisModel, self).__init__()
        
        # Lip landmark feature extractor
        self.landmark_encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )
        
        # Temporal processing with LSTM
        self.temporal_lstm = nn.LSTM(256, hidden_size, 
                                    batch_first=True, 
                                    bidirectional=True,
                                    num_layers=2,
                                    dropout=0.3)
        
        # Attention mechanism for important frames
        self.temporal_attention = nn.MultiheadAttention(
            hidden_size * 2,  # Bidirectional
            num_heads=4,
            batch_first=True,
            dropout=0.2
        )
        
        # Analysis heads
        self.lip_sync_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.articulation_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.mouth_opening_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.movement_smoothness_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.expression_symmetry_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Severity classifier
        self.severity_classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 5),  # 5 severity levels
            nn.Softmax(dim=1)
        )
        
        # Pooling layers
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, landmarks_sequence):
        # landmarks_sequence shape: (batch, seq_len, num_points * 2)
        batch_size, seq_len, _ = landmarks_sequence.shape
        
        # Encode each frame's landmarks
        encoded_frames = []
        for i in range(seq_len):
            frame_landmarks = landmarks_sequence[:, i, :]
            encoded = self.landmark_encoder(frame_landmarks)
            encoded_frames.append(encoded.unsqueeze(1))
        
        # Concatenate encoded frames
        encoded_sequence = torch.cat(encoded_frames, dim=1)  # (batch, seq_len, 256)
        
        # Temporal processing with LSTM
        lstm_out, _ = self.temporal_lstm(encoded_sequence)  # (batch, seq_len, hidden_size*2)
        
        # Apply temporal attention
        attn_out, attn_weights = self.temporal_attention(lstm_out, lstm_out, lstm_out)
        lstm_out = lstm_out + attn_out  # Residual connection
        
        # Global feature vector
        global_features, _ = torch.max(lstm_out, dim=1)
        
        # Calculate specific scores
        lip_sync_score = self.lip_sync_head(global_features)
        articulation_score = self.articulation_head(global_features)
        mouth_opening_score = self.mouth_opening_head(global_features)
        movement_smoothness = self.movement_smoothness_head(global_features)
        expression_symmetry = self.expression_symmetry_head(global_features)
        
        # Severity classification
        severity = self.severity_classifier(global_features)
        
        return {
            'lip_sync_score': lip_sync_score,
            'articulation_score': articulation_score,
            'mouth_opening_score': mouth_opening_score,
            'movement_smoothness': movement_smoothness,
            'expression_symmetry': expression_symmetry,
            'severity': severity,
            'attention_weights': attn_weights,
            'global_features': global_features
        }
    
    def analyze(self, visual_features):
        """Analyze visual features for dysarthria characteristics"""
        try:
            if 'lip_landmarks' in visual_features and len(visual_features['lip_landmarks']) > 0:
                # Prepare landmark sequence
                landmarks_seq = visual_features['lip_landmarks']
                
                # Flatten landmarks (20 points * 2 coordinates)
                flattened_landmarks = []
                for landmarks in landmarks_seq:
                    if len(landmarks) >= 20:  # Ensure we have enough points
                        flattened = landmarks[:20].flatten()  # Take first 20 points
                        if len(flattened) == 40:  # 20 * 2
                            flattened_landmarks.append(flattened)
                
                if len(flattened_landmarks) < 5:
                    return self._analyze_from_basic_features(visual_features)
                
                # Convert to tensor
                landmarks_tensor = torch.FloatTensor([flattened_landmarks]).unsqueeze(0)
                
                # Forward pass
                with torch.no_grad():
                    outputs = self.forward(landmarks_tensor)
                
                # Extract scores
                lip_sync = outputs['lip_sync_score'].item()
                articulation = outputs['articulation_score'].item()
                mouth_opening = outputs['mouth_opening_score'].item()
                movement_smoothness = outputs['movement_smoothness'].item()
                expression_symmetry = outputs['expression_symmetry'].item()
                severity = torch.argmax(outputs['severity']).item()
                
                # Calculate additional metrics from basic features
                additional_metrics = self._calculate_visual_metrics(visual_features)
                
                return {
                    'lip_sync_score': lip_sync,
                    'articulation_score': articulation,
                    'mouth_opening_score': mouth_opening,
                    'movement_smoothness': movement_smoothness,
                    'expression_symmetry': expression_symmetry,
                    'severity_level': severity,
                    'severity_label': self._get_severity_label(severity),
                    'additional_metrics': additional_metrics,
                    'mouth_opening_consistency': additional_metrics.get('opening_consistency', 0),
                    'movement_coordination': additional_metrics.get('movement_coordination', 0),
                    'facial_symmetry': additional_metrics.get('facial_symmetry', 0)
                }
            else:
                return self._analyze_from_basic_features(visual_features)
                
        except Exception as e:
            print(f"Visual analysis error: {e}")
            return {
                'lip_sync_score': 0.5,
                'articulation_score': 0.5,
                'mouth_opening_score': 0.5,
                'movement_smoothness': 0.5,
                'expression_symmetry': 0.5,
                'severity_level': 2,
                'severity_label': 'Moderate',
                'additional_metrics': {},
                'error': str(e)
            }
    
    def _calculate_visual_metrics(self, features):
        """Calculate additional visual metrics"""
        metrics = {}
        
        # Mouth opening consistency
        if 'mouth_openings' in features and len(features['mouth_openings']) > 0:
            openings = [o['height'] for o in features['mouth_openings']]
            metrics['opening_mean'] = np.mean(openings)
            metrics['opening_std'] = np.std(openings)
            metrics['opening_consistency'] = 1.0 - (np.std(openings) / max(np.mean(openings), 1))
            metrics['opening_range'] = np.max(openings) - np.min(openings)
        
        # Movement coordination
        if 'lip_movements' in features and len(features['lip_movements']) > 1:
            movements = features['lip_movements']
            metrics['movement_mean'] = np.mean(movements)
            metrics['movement_std'] = np.std(movements)
            metrics['movement_coordination'] = 1.0 - (np.std(movements) / max(np.mean(movements), 1))
            
            # Calculate periodicity
            if len(movements) > 10:
                autocorr = np.correlate(movements - np.mean(movements), 
                                       movements - np.mean(movements), 
                                       mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                if len(autocorr) > 5:
                    metrics['movement_periodicity'] = np.mean(autocorr[1:6])
        
        # Facial symmetry from orientations
        if 'face_orientations' in features and len(features['face_orientations']) > 0:
            orientations = features['face_orientations']
            tilt_values = [o['tilt'] for o in orientations if 'tilt' in o]
            if tilt_values:
                metrics['tilt_mean'] = np.mean(tilt_values)
                metrics['tilt_std'] = np.std(tilt_values)
                metrics['facial_symmetry'] = 1.0 - min(abs(metrics['tilt_mean']), 0.5) / 0.5
        
        # Expression features
        if 'expression_features' in features and len(features['expression_features']) > 0:
            expr_features = features['expression_features']
            mouth_distances = [e.get('mouth_corners_distance', 0) for e in expr_features]
            if mouth_distances:
                metrics['expression_variability'] = np.std(mouth_distances) / max(np.mean(mouth_distances), 1)
        
        # Temporal features
        if 'temporal_features' in features:
            metrics.update(features['temporal_features'])
        
        return metrics
    
    def _analyze_from_basic_features(self, features):
        """Fallback analysis using basic features"""
        scores = {}
        
        # Estimate scores from available features
        if 'mouth_openings' in features:
            openings = [o['height'] for o in features['mouth_openings']]
            if openings:
                opening_cv = np.std(openings) / max(np.mean(openings), 1)
                scores['mouth_opening_score'] = max(0, 1 - opening_cv)
        
        if 'lip_movements' in features:
            movements = features['lip_movements']
            if movements:
                movement_cv = np.std(movements) / max(np.mean(movements), 1)
                scores['movement_smoothness'] = max(0, 1 - movement_cv)
        
        # Default values
        scores['lip_sync_score'] = scores.get('lip_sync_score', 0.6)
        scores['articulation_score'] = scores.get('articulation_score', 0.55)
        scores['expression_symmetry'] = 0.65
        
        return scores
    
    def _get_severity_label(self, severity_level):
        """Convert severity level to label"""
        labels = ["Normal", "Mild", "Moderate", "Severe", "Very Severe"]
        return labels[min(severity_level, len(labels)-1)]