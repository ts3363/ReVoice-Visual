import torch
import torch.nn as nn
import numpy as np


class VisualAnalysisModel(nn.Module):
    def __init__(self, input_size=20 * 2, hidden_size=256):
        super(VisualAnalysisModel, self).__init__()

        # Landmark encoder
        self.landmark_encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )

        # Temporal LSTM
        self.temporal_lstm = nn.LSTM(
            256,
            hidden_size,
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=0.3
        )

        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(
            hidden_size * 2,
            num_heads=4,
            batch_first=True,
            dropout=0.2
        )

        # Score heads
        def score_head():
            return nn.Sequential(
                nn.Linear(hidden_size * 2, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )

        self.lip_sync_head = score_head()
        self.articulation_head = score_head()
        self.mouth_opening_head = score_head()
        self.movement_smoothness_head = score_head()
        self.expression_symmetry_head = score_head()

        # Severity classifier
        self.severity_classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 5),
            nn.Softmax(dim=1)
        )

    # --------------------------------------------------
    # Utility
    # --------------------------------------------------
    def _ensure_float_landmarks(self, landmarks):
        return np.asarray(landmarks, dtype=np.float32)

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------
    def forward(self, landmarks_sequence):
        # landmarks_sequence: (batch, seq_len, 40)
        batch_size, seq_len, _ = landmarks_sequence.shape

        encoded_frames = []
        for i in range(seq_len):
            frame = landmarks_sequence[:, i, :]
            encoded = self.landmark_encoder(frame)
            encoded_frames.append(encoded.unsqueeze(1))

        encoded_sequence = torch.cat(encoded_frames, dim=1)

        lstm_out, _ = self.temporal_lstm(encoded_sequence)

        attn_out, attn_weights = self.temporal_attention(
            lstm_out, lstm_out, lstm_out
        )
        lstm_out = lstm_out + attn_out  # residual

        global_features, _ = torch.max(lstm_out, dim=1)

        return {
            'lip_sync_score': self.lip_sync_head(global_features),
            'articulation_score': self.articulation_head(global_features),
            'mouth_opening_score': self.mouth_opening_head(global_features),
            'movement_smoothness': self.movement_smoothness_head(global_features),
            'expression_symmetry': self.expression_symmetry_head(global_features),
            'severity': self.severity_classifier(global_features),
            'attention_weights': attn_weights,
            'global_features': global_features
        }

    # --------------------------------------------------
    # Analysis (FULLY FIXED)
    # --------------------------------------------------
    def analyze(self, visual_features):
        try:
            # STEP 4: Safe fallback
            if not isinstance(visual_features, dict):
                return self._analyze_from_basic_features({})

            if 'lip_landmarks' not in visual_features:
                return self._analyze_from_basic_features(visual_features)

            # STEP 2 + 3 + 5: Robust landmark handling
            landmarks_seq = [
                self._ensure_float_landmarks(lm)
                for lm in visual_features['lip_landmarks']
            ]

            flattened_landmarks = []

            for landmarks in landmarks_seq:
                landmarks = np.asarray(landmarks, dtype=np.float32)

                if landmarks.ndim != 2 or landmarks.shape[1] != 2:
                    continue

                if landmarks.shape[0] < 20:
                    continue

                flattened = landmarks[:20].reshape(-1)  # (40,)
                if flattened.shape[0] == 40:
                    flattened_landmarks.append(flattened)

            if len(flattened_landmarks) < 5:
                return self._analyze_from_basic_features(visual_features)

            # STEP 1: Correct tensor creation
            landmarks_np = np.array(flattened_landmarks, dtype=np.float32)
            landmarks_tensor = torch.from_numpy(landmarks_np).unsqueeze(0)

            with torch.no_grad():
                outputs = self.forward(landmarks_tensor)

            severity_idx = torch.argmax(outputs['severity']).item()

            additional_metrics = self._calculate_visual_metrics(visual_features)

            return {
                'lip_sync_score': outputs['lip_sync_score'].item(),
                'articulation_score': outputs['articulation_score'].item(),
                'mouth_opening_score': outputs['mouth_opening_score'].item(),
                'movement_smoothness': outputs['movement_smoothness'].item(),
                'expression_symmetry': outputs['expression_symmetry'].item(),
                'severity_level': severity_idx,
                'severity_label': self._get_severity_label(severity_idx),
                'additional_metrics': additional_metrics
            }

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

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------
    def _calculate_visual_metrics(self, features):
        metrics = {}

        if 'mouth_openings' in features and features['mouth_openings']:
            openings = [o['height'] for o in features['mouth_openings']]
            metrics['opening_consistency'] = 1.0 - (
                np.std(openings) / max(np.mean(openings), 1)
            )

        if 'lip_movements' in features and features['lip_movements']:
            movements = features['lip_movements']
            metrics['movement_coordination'] = 1.0 - (
                np.std(movements) / max(np.mean(movements), 1)
            )

        if 'face_orientations' in features:
            tilts = [
                o['tilt'] for o in features['face_orientations']
                if 'tilt' in o
            ]
            if tilts:
                metrics['facial_symmetry'] = 1.0 - min(abs(np.mean(tilts)), 0.5) / 0.5

        return metrics

    # --------------------------------------------------
    # Fallback
    # --------------------------------------------------
    def _analyze_from_basic_features(self, features):
        scores = {}

        if 'mouth_openings' in features:
            openings = [o['height'] for o in features['mouth_openings']]
            if openings:
                scores['mouth_opening_score'] = max(
                    0.0, 1.0 - (np.std(openings) / max(np.mean(openings), 1))
                )

        if 'lip_movements' in features:
            movements = features['lip_movements']
            if movements:
                scores['movement_smoothness'] = max(
                    0.0, 1.0 - (np.std(movements) / max(np.mean(movements), 1))
                )

        scores.setdefault('lip_sync_score', 0.6)
        scores.setdefault('articulation_score', 0.55)
        scores.setdefault('expression_symmetry', 0.65)

        return scores

    # --------------------------------------------------
    # Severity Label
    # --------------------------------------------------
    def _get_severity_label(self, severity_level):
        labels = ["Normal", "Mild", "Moderate", "Severe", "Very Severe"]
        return labels[min(severity_level, len(labels) - 1)]
