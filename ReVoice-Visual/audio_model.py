import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import signal
import librosa

class AudioAnalysisModel(nn.Module):
    def __init__(self, input_size=128, hidden_size=256, num_classes=5):
        super(AudioAnalysisModel, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(256, num_heads=4, batch_first=True)
        
        # LSTM for temporal analysis
        self.lstm = nn.LSTM(256, hidden_size, batch_first=True, bidirectional=True)
        
        # Dysarthria-specific analysis heads
        self.clarity_head = nn.Sequential(
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
        
        self.fluency_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.pitch_stability_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Classification head for severity levels
        self.severity_classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)
        )
        
        # Pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        # x shape: (batch, features, time)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)
        
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Apply attention
        x = x.transpose(1, 2)  # (batch, time, features)
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output  # Residual connection
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Global features
        global_features = torch.mean(lstm_out, dim=1)
        
        # Extract specific characteristics
        clarity_score = self.clarity_head(global_features)
        articulation_score = self.articulation_head(global_features)
        fluency_score = self.fluency_head(global_features)
        pitch_stability = self.pitch_stability_head(global_features)
        
        # Severity classification
        severity = self.severity_classifier(global_features)
        
        return {
            'clarity_score': clarity_score,
            'articulation_score': articulation_score,
            'fluency_score': fluency_score,
            'pitch_stability': pitch_stability,
            'severity': severity,
            'features': global_features
        }
    
    def analyze(self, audio_features):
        """Analyze audio features for dysarthria characteristics"""
        try:
            # Convert features to tensor
            if 'mfcc' in audio_features:
                mfcc = audio_features['mfcc']
                mfcc_tensor = torch.FloatTensor(mfcc).unsqueeze(0)
                
                # Forward pass
                with torch.no_grad():
                    outputs = self.forward(mfcc_tensor)
                
                # Extract scores
                clarity = outputs['clarity_score'].item()
                articulation = outputs['articulation_score'].item()
                fluency = outputs['fluency_score'].item()
                pitch_stability = outputs['pitch_stability'].item()
                severity = torch.argmax(outputs['severity']).item()
                
                # Additional feature-based calculations
                additional_metrics = self._calculate_additional_metrics(audio_features)
                
                return {
                    'clarity_score': clarity,
                    'articulation_score': articulation,
                    'fluency_score': fluency,
                    'pitch_stability': pitch_stability,
                    'severity_level': severity,
                    'severity_label': self._get_severity_label(severity),
                    'additional_metrics': additional_metrics,
                    'pitch_variability': audio_features.get('pitch_std', 0) / max(audio_features.get('pitch_mean', 1), 1),
                    'energy_consistency': 1.0 - (audio_features.get('energy_variance', 0) / max(audio_features.get('energy_mean', 1), 1)),
                    'articulation_rate': audio_features.get('articulation_rate', 0),
                    'pause_ratio': audio_features.get('pause_ratio', 0),
                    'speech_rate': audio_features.get('speech_rate', 0)
                }
            else:
                return self._analyze_from_raw_features(audio_features)
                
        except Exception as e:
            print(f"Audio analysis error: {e}")
            return {
                'clarity_score': 0.5,
                'articulation_score': 0.5,
                'fluency_score': 0.5,
                'pitch_stability': 0.5,
                'severity_level': 2,
                'severity_label': 'Moderate',
                'additional_metrics': {},
                'error': str(e)
            }
    
    def _calculate_additional_metrics(self, features):
        """Calculate additional dysarthria-relevant metrics"""
        metrics = {}
        
        # Jitter and shimmer (voice quality)
        if 'jitter' in features:
            metrics['jitter'] = features['jitter']
            metrics['jitter_interpretation'] = self._interpret_jitter(features['jitter'])
        
        if 'shimmer' in features:
            metrics['shimmer'] = features['shimmer']
            metrics['shimmer_interpretation'] = self._interpret_shimmer(features['shimmer'])
        
        # Formant analysis
        if 'formants' in features and len(features['formants']) >= 2:
            f1, f2 = features['formants'][:2]
            metrics['formant_frequencies'] = {'F1': f1, 'F2': f2}
            metrics['formant_ratio'] = f2 / f1 if f1 > 0 else 0
        
        # Voice onset/offset characteristics
        if 'onset_frames' in features:
            onsets = features['onset_frames']
            if len(onsets) > 1:
                intervals = np.diff(onsets)
                metrics['onset_regularity'] = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0
        
        # Harmonic to noise ratio approximation
        if 'harmonic' in features and 'percussive' in features:
            harmonic_energy = np.mean(features['harmonic']**2)
            percussive_energy = np.mean(features['percussive']**2)
            metrics['hnr_approximation'] = harmonic_energy / max(percussive_energy, 1e-10)
        
        # Vowel space area approximation (simplified)
        if 'mfcc' in features:
            mfcc_mean = np.mean(features['mfcc'], axis=1)
            metrics['mfcc_variance'] = np.var(mfcc_mean[:5])  # Variance in first 5 MFCCs
        
        return metrics
    
    def _interpret_jitter(self, jitter_value):
        """Interpret jitter value"""
        if jitter_value < 0.5:
            return "Normal pitch stability"
        elif jitter_value < 1.0:
            return "Mild pitch instability"
        elif jitter_value < 2.0:
            return "Moderate pitch instability"
        else:
            return "Severe pitch instability"
    
    def _interpret_shimmer(self, shimmer_value):
        """Interpret shimmer value"""
        if shimmer_value < 0.05:
            return "Normal amplitude stability"
        elif shimmer_value < 0.1:
            return "Mild amplitude instability"
        elif shimmer_value < 0.2:
            return "Moderate amplitude instability"
        else:
            return "Severe amplitude instability"
    
    def _get_severity_label(self, severity_level):
        """Convert severity level to label"""
        labels = ["Normal", "Mild", "Moderate", "Severe", "Very Severe"]
        return labels[min(severity_level, len(labels)-1)]
    
    def _analyze_from_raw_features(self, features):
        """Fallback analysis using raw features"""
        scores = {}
        
        # Calculate basic scores from features
        if 'pitch_mean' in features and 'pitch_std' in features:
            pitch_cv = features['pitch_std'] / max(features['pitch_mean'], 1)
            scores['pitch_stability'] = max(0, 1 - min(pitch_cv, 1))
        
        if 'energy_mean' in features and 'energy_variance' in features:
            energy_cv = np.sqrt(features['energy_variance']) / max(features['energy_mean'], 1)
            scores['energy_consistency'] = max(0, 1 - min(energy_cv, 1))
        
        # Estimate other scores
        scores['clarity_score'] = 0.7  # Default estimate
        scores['articulation_score'] = 0.6
        scores['fluency_score'] = 0.65
        
        return scores