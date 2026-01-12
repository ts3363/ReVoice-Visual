import numpy as np
import librosa
import cv2
import torch
import torchvision.transforms as transforms
from scipy import signal
import mediapipe as mp
from moviepy.editor import VideoFileClip
import tempfile

class AudioVideoPreprocessor:
    def __init__(self, audio_sr=16000, visual_fps=30):
        self.audio_sr = audio_sr
        self.visual_fps = visual_fps
        
        # Initialize MediaPipe for face and lip detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Lip landmark indices (MediaPipe)
        self.lip_indices = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 408,  # Outer lips
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324,  # Inner lips
            191, 80, 81, 82, 13, 312, 311, 310, 415  # Additional lip points
        ]
        
        # Transform for visual features
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    
    def extract_audio_features(self, audio_path):
        """Extract comprehensive audio features"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.audio_sr)
            
            features = {}
            
            # 1. MFCC features
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            features['mfcc'] = mfcc
            features['mfcc_mean'] = np.mean(mfcc, axis=1)
            features['mfcc_std'] = np.std(mfcc, axis=1)
            
            # 2. Mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
            features['mel_spec'] = librosa.power_to_db(mel_spec, ref=np.max)
            
            # 3. Chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            features['chroma'] = chroma
            
            # 4. Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            
            features['spectral_centroid'] = spectral_centroid
            features['spectral_bandwidth'] = spectral_bandwidth
            features['spectral_rolloff'] = spectral_rolloff
            
            # 5. Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y=audio)
            features['zcr'] = zcr
            
            # 6. Pitch features using PYIN
            f0, voiced_flag, voiced_probs = librosa.pyin(y=audio, 
                                                         fmin=librosa.note_to_hz('C2'),
                                                         fmax=librosa.note_to_hz('C7'),
                                                         sr=sr)
            features['pitch'] = f0
            features['voiced_probability'] = voiced_probs
            
            # 7. RMS energy
            rms = librosa.feature.rms(y=audio)
            features['rms'] = rms
            
            # 8. Tempo and beat
            tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
            features['tempo'] = tempo
            features['beat_frames'] = beat_frames
            
            # 9. Harmonics and percussive components
            harmonic, percussive = librosa.effects.hpss(audio)
            features['harmonic'] = harmonic
            features['percussive'] = percussive
            
            # 10. Formant estimation (simplified)
            # Using LPC to estimate formants
            if len(audio) > 0:
                order = 4 + int(sr / 1000)
                a = librosa.lpc(audio, order=order)
                roots = np.roots(a)
                roots = roots[np.imag(roots) >= 0]
                ang = np.arctan2(np.imag(roots), np.real(roots))
                formants = ang * (sr / (2 * np.pi))
                features['formants'] = formants[:3]  # First 3 formants
            
            # 11. Jitter and shimmer (voice quality metrics)
            if voiced_probs is not None and len(voiced_probs) > 10:
                voiced_f0 = f0[voiced_probs > 0.5]
                if len(voiced_f0) > 2:
                    # Jitter: pitch period variability
                    jitter = np.abs(np.diff(voiced_f0)).mean() / np.mean(voiced_f0)
                    features['jitter'] = jitter
                    
                    # Simplified shimmer approximation using RMS
                    if len(rms[0]) > 2:
                        shimmer = np.abs(np.diff(rms[0])).mean() / np.mean(rms[0])
                        features['shimmer'] = shimmer
            
            # 12. Voice onset/offset detection
            onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
            features['onset_frames'] = onset_frames
            
            # Calculate statistics
            features['duration'] = len(audio) / sr
            features['energy_mean'] = np.mean(rms)
            features['energy_variance'] = np.var(rms)
            features['pitch_mean'] = np.nanmean(f0) if f0 is not None else 0
            features['pitch_std'] = np.nanstd(f0) if f0 is not None else 0
            features['pitch_range'] = np.nanmax(f0) - np.nanmin(f0) if f0 is not None else 0
            
            # Dysarthria-specific features
            features['articulation_rate'] = len(onset_frames) / features['duration'] if features['duration'] > 0 else 0
            features['pause_ratio'] = self._calculate_pause_ratio(audio, sr)
            features['speech_rate'] = self._estimate_speech_rate(audio, sr)
            
            return features
            
        except Exception as e:
            print(f"Audio feature extraction error: {e}")
            return {}
    
    def _calculate_pause_ratio(self, audio, sr):
        """Calculate ratio of silence/pauses in speech"""
        # Simple energy-based silence detection
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop
        
        energy = np.array([
            np.sum(audio[i:i+frame_length]**2)
            for i in range(0, len(audio)-frame_length, hop_length)
        ])
        
        if len(energy) == 0:
            return 0
        
        energy_threshold = np.percentile(energy, 25)  # Lower quartile as threshold
        silence_frames = np.sum(energy < energy_threshold)
        
        return silence_frames / len(energy) if len(energy) > 0 else 0
    
    def _estimate_speech_rate(self, audio, sr):
        """Estimate speech rate in syllables per second"""
        # Simplified estimation using onset detection
        onset_frames = librosa.onset.onset_detect(y=audio, sr=sr, units='time')
        
        if len(onset_frames) < 2:
            return 0
        
        duration = len(audio) / sr
        return len(onset_frames) / duration
    
    def extract_visual_features(self, video_path):
        """Extract visual features from video"""
        features = {
            'lip_landmarks': [],
            'mouth_openings': [],
            'lip_movements': [],
            'face_orientations': [],
            'expression_features': [],
            'temporal_features': []
        }
        
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every nth frame to reduce computation
                if frame_count % 2 == 0:
                    # Convert to RGB for MediaPipe
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.face_mesh.process(frame_rgb)
                    
                    if results.multi_face_landmarks:
                        face_landmarks = results.multi_face_landmarks[0]
                        
                        # Extract lip landmarks
                        lip_points = []
                        for idx in self.lip_indices:
                            landmark = face_landmarks.landmark[idx]
                            h, w = frame.shape[:2]
                            x, y = int(landmark.x * w), int(landmark.y * h)
                            lip_points.append([x, y])
                        
                        lip_points = np.array(lip_points)
                        features['lip_landmarks'].append(lip_points)
                        
                        # Calculate mouth opening
                        mouth_height = self._calculate_mouth_height(lip_points)
                        mouth_width = self._calculate_mouth_width(lip_points)
                        features['mouth_openings'].append({
                            'height': mouth_height,
                            'width': mouth_width,
                            'area': mouth_height * mouth_width
                        })
                        
                        # Calculate lip movement speed (if previous frame exists)
                        if len(features['lip_landmarks']) > 1:
                            prev_points = features['lip_landmarks'][-2]
                            movement = np.mean(np.linalg.norm(lip_points - prev_points, axis=1))
                            features['lip_movements'].append(movement)
                        
                        # Extract face orientation
                        orientation = self._estimate_face_orientation(face_landmarks, frame.shape)
                        features['face_orientations'].append(orientation)
                        
                        # Extract expression-related features
                        expression_feats = self._extract_expression_features(face_landmarks)
                        features['expression_features'].append(expression_feats)
                
                frame_count += 1
                if frame_count > 300:  # Limit to first 300 frames for performance
                    break
            
            cap.release()
            
            # Calculate temporal statistics
            if features['mouth_openings']:
                openings = [o['height'] for o in features['mouth_openings']]
                features['temporal_features'] = {
                    'mouth_opening_mean': np.mean(openings),
                    'mouth_opening_std': np.std(openings),
                    'mouth_opening_range': np.max(openings) - np.min(openings),
                    'movement_smoothness': self._calculate_movement_smoothness(features['lip_movements']),
                    'sync_consistency': self._calculate_sync_consistency(features['lip_movements'])
                }
            
            return features
            
        except Exception as e:
            print(f"Visual feature extraction error: {e}")
            return features
    
    def _calculate_mouth_height(self, lip_points):
        """Calculate mouth opening height"""
        # Upper lip points (indices 0-9 in our lip_indices)
        upper_lip = lip_points[:10]
        # Lower lip points (indices 10-19)
        lower_lip = lip_points[10:20]
        
        upper_center = np.mean(upper_lip, axis=0)
        lower_center = np.mean(lower_lip, axis=0)
        
        return np.linalg.norm(upper_center - lower_center)
    
    def _calculate_mouth_width(self, lip_points):
        """Calculate mouth width"""
        # Left corner (index around 61)
        left_corner = lip_points[0]
        # Right corner (index around 291)
        right_corner = lip_points[6]
        
        return np.linalg.norm(left_corner - right_corner)
    
    def _estimate_face_orientation(self, face_landmarks, frame_shape):
        """Estimate face orientation (simplified)"""
        # Use nose tip and face center for orientation estimation
        h, w = frame_shape[:2]
        
        nose_tip = face_landmarks.landmark[1]  # Nose tip landmark
        face_center = face_landmarks.landmark[0]  # Typically face center
        
        # Calculate offset from center
        offset_x = (nose_tip.x - face_center.x) * w
        offset_y = (nose_tip.y - face_center.y) * h
        
        return {
            'offset_x': offset_x,
            'offset_y': offset_y,
            'tilt': np.arctan2(offset_y, offset_x)
        }
    
    def _extract_expression_features(self, face_landmarks):
        """Extract features related to facial expressions"""
        features = {}
        
        # Mouth corner movements (smile/frown)
        left_corner = face_landmarks.landmark[61]
        right_corner = face_landmarks.landmark[291]
        
        features['mouth_corners_distance'] = np.sqrt(
            (left_corner.x - right_corner.x)**2 + 
            (left_corner.y - right_corner.y)**2
        )
        
        # Cheek movement approximation
        cheek_left = face_landmarks.landmark[123]
        cheek_right = face_landmarks.landmark[352]
        
        features['cheek_symmetry'] = abs(cheek_left.z - cheek_right.z)
        
        return features
    
    def _calculate_movement_smoothness(self, movements):
        """Calculate smoothness of lip movements"""
        if len(movements) < 3:
            return 0
        
        # Calculate jerk (derivative of acceleration)
        movements = np.array(movements)
        acceleration = np.diff(movements, n=2)
        
        if len(acceleration) == 0:
            return 0
        
        jerk = np.diff(acceleration)
        smoothness = 1.0 / (1.0 + np.mean(np.abs(jerk)))
        
        return smoothness
    
    def _calculate_sync_consistency(self, movements):
        """Calculate consistency of movement patterns"""
        if len(movements) < 5:
            return 0
        
        # Calculate autocorrelation for periodicity
        movements = np.array(movements)
        autocorr = np.correlate(movements - np.mean(movements), 
                               movements - np.mean(movements), 
                               mode='full')
        
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Look for periodicity in first few lags
        if len(autocorr) > 10:
            consistency = np.mean(autocorr[1:6])
        else:
            consistency = np.mean(autocorr[1:]) if len(autocorr) > 1 else 0
        
        return max(0, consistency)  # Ensure non-negative
    
    def synchronize_audio_video(self, audio_features, visual_features):
        """Synchronize audio and video features"""
        # Simple time-based synchronization
        audio_duration = audio_features.get('duration', 0)
        visual_frames = len(visual_features.get('lip_landmarks', []))
        
        if audio_duration > 0 and visual_frames > 0:
            visual_duration = visual_frames / self.visual_fps
            sync_ratio = audio_duration / visual_duration if visual_duration > 0 else 1
            
            return {
                'audio_duration': audio_duration,
                'visual_duration': visual_duration,
                'sync_ratio': sync_ratio,
                'is_synced': 0.8 < sync_ratio < 1.2  # Within 20% tolerance
            }
        
        return {'is_synced': False}