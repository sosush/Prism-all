"""
PRISM Core Engine - ML & Signal Processing Module
Author: Swarnim
Version: 2.0.0

Advanced physics-based liveness detection using:
- rPPG (Remote Photoplethysmography) with Welch's method
- HRV (Heart Rate Variability) for biological chaos detection
- Subsurface Scattering Spectroscopy
- Temporal Frequency Response Analysis
- Moiré Pattern Detection (anti-screen-replay)
- Multi-Modal Fusion Scoring
"""

from __future__ import annotations

import numpy as np
import cv2
import scipy.signal as signal
import scipy.fftpack as fft
from scipy.stats import entropy as scipy_entropy
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from enum import Enum
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PRISM_CORE")


# =============================================================================
# CONFIGURATION & DATA MODELS
# =============================================================================

class ScreenColor(str, Enum):
    """Screen colors used in chroma challenge."""
    RED = "RED"
    BLUE = "BLUE"
    WHITE = "WHITE"
    GREEN = "GREEN"


@dataclass
class PrismConfig:
    """Configuration for the PRISM engine. Tune these for different environments."""
    
    # Core settings
    fps: int = 30
    buffer_size: int = 150  # 5 seconds at 30fps
    
    # rPPG thresholds
    min_bpm: int = 45
    max_bpm: int = 190
    min_signal_quality: float = 0.25  # Lowered for indoor lighting
    
    # SSS (Subsurface Scattering) thresholds
    # Lowered from 1.05 to 0.88 for glasses/indoor conditions
    sss_ratio_threshold: float = 0.88
    
    # Chroma thresholds
    chroma_sensitivity: float = 1.1
    
    # Temporal response thresholds (milliseconds)
    temporal_delay_min_ms: float = 80   # Minimum biological delay
    temporal_delay_max_ms: float = 350  # Maximum biological delay
    
    # HRV thresholds
    hrv_min_rmssd: float = 8.0   # Lowered for short sample windows
    hrv_entropy_threshold: float = 0.25  # Lowered for noisy conditions
    
    # Moiré detection (lowered = more aggressive screen detection)
    moire_threshold: float = 0.04  # FFT peak threshold for screen detection (was 0.08)
    
    # BPM stability (anti-photo attack)
    bpm_stability_threshold: float = 10.0  # Max std dev of BPM for "stable" (was 15.0)
    
    # Static image detection (THE KEY ANTI-PHOTO DEFENSE)
    min_signal_variance: float = 0.8  # Minimum variance in green channel over time (was 0.5)
    # Real faces: variance > 2.0 due to blood flow
    # Photos/AI: variance < 0.8 (static image with only camera noise)
    
    # Screen replay detection (additional layer)
    screen_color_uniformity_threshold: float = 0.15  # Screens have very uniform color patches
    
    # Fusion model weights (adjusted for hackathon - prioritize rPPG+HRV)
    weight_physics_sss: int = 15      # Reduced (glasses cause SSS issues)
    weight_chroma: int = 20
    weight_rppg: int = 30             # Increased (most reliable signal)
    weight_hrv: int = 20              # Increased (strong liveness proof)
    weight_temporal: int = 10
    weight_moire: int = 5


@dataclass
class HRVMetrics:
    """Heart Rate Variability metrics for biological liveness."""
    rmssd: float = 0.0          # Root Mean Square of Successive Differences
    sdnn: float = 0.0           # Standard Deviation of NN intervals
    entropy: float = 0.0        # Shannon entropy of RR intervals
    is_biologically_valid: bool = False


@dataclass
class RPPGResult:
    """Results from rPPG heart rate analysis."""
    bpm: int = 0
    signal_quality: float = 0.0  # SNR-based quality score (0-1)
    raw_confidence: float = 0.0
    is_valid: bool = False
    hrv: HRVMetrics = field(default_factory=HRVMetrics)


@dataclass  
class PhysicsResult:
    """Results from physics-based checks."""
    sss_passed: bool = False
    sss_ratio: float = 0.0
    red_variance: float = 0.0
    blue_variance: float = 0.0


@dataclass
class TemporalResult:
    """Results from temporal frequency response analysis."""
    delay_ms: float = 0.0
    is_biological: bool = False
    response_detected: bool = False


@dataclass
class MoireResult:
    """Results from Moiré pattern detection."""
    is_screen: bool = False
    moire_score: float = 0.0


@dataclass
class StaticImageResult:
    """Results from static image detection."""
    is_static: bool = True  # Default to static until proven otherwise
    signal_variance: float = 0.0
    is_alive: bool = False


@dataclass
class LivenessResult:
    """Final liveness detection result. This is what Sohini's API consumes."""
    is_human: bool = False
    confidence: float = 0.0  # 0-100
    bpm: int = 0
    hrv_score: float = 0.0   # Biological chaos metric
    signal_quality: float = 0.0
    details: dict = field(default_factory=dict)


# =============================================================================
# PRISM ENGINE
# =============================================================================

class PrismEngine:
    """
    PRISM Physics-Based Liveness Detection Engine.
    
    Implements multi-modal fusion of:
    1. rPPG (Remote Photoplethysmography) - Heart rate from face color
    2. HRV (Heart Rate Variability) - Biological chaos signature
    3. SSS (Subsurface Scattering) - Light penetration through skin
    4. Temporal Response - Biological delay to stimuli
    5. Moiré Detection - Screen replay attack defense
    6. Chroma Sync - Color reflection verification
    
    Usage:
        engine = PrismEngine()
        result = engine.process_frame(forehead_roi, face_img, "RED")
    """
    
    def __init__(self, config: Optional[PrismConfig] = None):
        """
        Initialize the PRISM engine.
        
        Args:
            config: Configuration object. Uses defaults if None.
        """
        self.config = config or PrismConfig()
        
        # Signal buffers
        self.green_signal_buffer: deque = deque(maxlen=self.config.buffer_size)
        self.luminance_buffer: deque = deque(maxlen=60)  # 2 seconds for temporal
        self.color_change_timestamps: List[Tuple[str, float]] = []
        
        # State tracking  
        self.last_bpm = 0
        self.bpm_history: deque = deque(maxlen=10)
        self.last_screen_color: Optional[str] = None
        self.last_color_change_time: float = 0.0
        
        # RR interval buffer for HRV
        self.rr_intervals: deque = deque(maxlen=30)
        
        # BPM stability tracking (anti-photo attack)
        self.raw_bpm_history: deque = deque(maxlen=30)  # Track BPM over time
        
        logger.info(f"PRISM Engine initialized with buffer_size={self.config.buffer_size}")

    # =========================================================================
    # 1. ENHANCED rPPG (Heart Rate Detection)
    # =========================================================================
    
    def _get_heart_rate(self) -> RPPGResult:
        """
        Advanced rPPG using Welch's method for robust frequency estimation.
        
        Improvements over v1:
        - Welch's method instead of raw FFT (reduces noise)
        - Hamming window for spectral leakage reduction
        - SNR-based signal quality metric
        - Peak detection for RR interval extraction
        
        Returns:
            RPPGResult with BPM, signal quality, and validity flag.
        """
        result = RPPGResult()
        
        if len(self.green_signal_buffer) < self.config.buffer_size:
            return result  # Not enough data yet (warming up)
        
        # Convert buffer to numpy array
        raw_signal = np.array(self.green_signal_buffer)
        
        # 1. Detrending (remove slow DC drift from lighting changes)
        detrended = signal.detrend(raw_signal)
        
        # 2. Z-score normalization
        std = np.std(detrended)
        if std == 0:
            return result
        normalized = (detrended - np.mean(detrended)) / std
        
        # 3. Butterworth Bandpass Filter [0.75Hz - 3.0Hz] = [45 - 180 BPM]
        nyquist = 0.5 * self.config.fps
        low = 0.75 / nyquist
        high = 3.0 / nyquist
        
        # Clamp to valid range
        low = max(0.01, min(low, 0.99))
        high = max(low + 0.01, min(high, 0.99))
        
        b, a = signal.butter(3, [low, high], btype='bandpass')
        filtered_signal = signal.filtfilt(b, a, normalized)  # filtfilt for zero phase
        
        # 4. Welch's method for power spectral density (more robust than FFT)
        nperseg = min(len(filtered_signal), 128)
        freqs, psd = signal.welch(
            filtered_signal, 
            fs=self.config.fps,
            nperseg=nperseg,
            noverlap=nperseg // 2,
            window='hamming'
        )
        
        # 5. Find peak in valid HR range
        valid_mask = (freqs >= 0.75) & (freqs <= 3.0)
        valid_freqs = freqs[valid_mask]
        valid_psd = psd[valid_mask]
        
        if len(valid_psd) == 0:
            return result
        
        # Find dominant frequency
        peak_idx = np.argmax(valid_psd)
        peak_freq = valid_freqs[peak_idx]
        peak_power = valid_psd[peak_idx]
        
        # 6. Calculate Signal Quality (SNR: peak power / mean of rest)
        noise_mask = np.ones(len(valid_psd), dtype=bool)
        noise_mask[max(0, peak_idx-2):min(len(valid_psd), peak_idx+3)] = False
        noise_power = np.mean(valid_psd[noise_mask]) if np.sum(noise_mask) > 0 else 1e-10
        snr = peak_power / (noise_power + 1e-10)
        signal_quality = min(1.0, snr / 10.0)  # Normalize to 0-1
        
        # 7. Convert to BPM
        current_bpm = peak_freq * 60
        
        # 8. Temporal smoothing with quality weighting
        self.bpm_history.append((current_bpm, signal_quality))
        if len(self.bpm_history) > 0:
            # Weighted average by signal quality
            weighted_sum = sum(bpm * q for bpm, q in self.bpm_history)
            weight_sum = sum(q for _, q in self.bpm_history)
            smoothed_bpm = weighted_sum / (weight_sum + 1e-10)
        else:
            smoothed_bpm = current_bpm
        
        self.last_bpm = int(smoothed_bpm)
        
        # Track raw BPM for stability analysis (anti-photo attack)
        self.raw_bpm_history.append(current_bpm)
        
        # 9. Extract RR intervals for HRV (using peak detection on filtered signal)
        hrv_metrics = self._extract_hrv(filtered_signal)
        
        # Validate result
        is_valid = (
            self.config.min_bpm <= self.last_bpm <= self.config.max_bpm and
            signal_quality >= self.config.min_signal_quality
        )
        
        result.bpm = self.last_bpm
        result.signal_quality = round(signal_quality, 3)
        result.raw_confidence = round(peak_power, 3)
        result.is_valid = is_valid
        result.hrv = hrv_metrics
        
        return result

    # =========================================================================
    # 2. HRV (Heart Rate Variability) Extraction
    # =========================================================================
    
    def _extract_hrv(self, bvp_signal: np.ndarray) -> HRVMetrics:
        """
        Extract Heart Rate Variability metrics from BVP waveform.
        
        HRV is a key liveness indicator because:
        - Living hearts have natural variability (chaos)
        - Synthetic/static signals lack this biological randomness
        
        Metrics:
        - RMSSD: Root Mean Square of Successive Differences
        - SDNN: Standard Deviation of NN intervals
        - Entropy: Shannon entropy (higher = more biological chaos)
        
        Args:
            bvp_signal: Blood Volume Pulse signal (filtered green channel)
            
        Returns:
            HRVMetrics with computed values and validity flag.
        """
        result = HRVMetrics()
        
        if len(bvp_signal) < 30:
            return result
        
        # Find peaks (heartbeats) in the BVP signal
        # Use prominence to filter noise
        peaks, properties = signal.find_peaks(
            bvp_signal,
            distance=int(self.config.fps * 0.4),  # Min 0.4s between beats (150 BPM max)
            prominence=0.3 * np.std(bvp_signal)
        )
        
        if len(peaks) < 3:
            return result
        
        # Calculate RR intervals (time between peaks) in milliseconds
        rr_intervals = np.diff(peaks) * (1000 / self.config.fps)
        
        # Filter physiologically implausible intervals
        valid_mask = (rr_intervals > 333) & (rr_intervals < 1500)  # 40-180 BPM
        rr_intervals = rr_intervals[valid_mask]
        
        if len(rr_intervals) < 2:
            return result
        
        # RMSSD: Root Mean Square of Successive Differences
        successive_diffs = np.diff(rr_intervals)
        rmssd = np.sqrt(np.mean(successive_diffs ** 2))
        
        # SDNN: Standard Deviation of NN intervals
        sdnn = np.std(rr_intervals)
        
        # Shannon Entropy of RR intervals (binned)
        hist, _ = np.histogram(rr_intervals, bins=10, density=True)
        hist = hist[hist > 0]  # Remove zeros for log
        hrv_entropy = scipy_entropy(hist)
        
        # Determine biological validity
        is_valid = (
            rmssd >= self.config.hrv_min_rmssd and
            hrv_entropy >= self.config.hrv_entropy_threshold
        )
        
        result.rmssd = round(rmssd, 2)
        result.sdnn = round(sdnn, 2)
        result.entropy = round(hrv_entropy, 3)
        result.is_biologically_valid = is_valid
        
        return result

    # =========================================================================
    # 3. PHYSICS CHECK (Subsurface Scattering Spectroscopy)
    # =========================================================================
    
    def _check_physics_sss(self, face_img: np.ndarray) -> PhysicsResult:
        """
        Subsurface Scattering Spectroscopy analysis.
        
        Real skin physics:
        - Blue light reflects off surface (high sharpness, captures pores)
        - Red light penetrates ~1-3mm, scatters internally (blurry)
        
        Screens/photos:
        - Both channels have identical sharpness (flat surface)
        
        Args:
            face_img: BGR face image from Jai's CV pipeline.
            
        Returns:
            PhysicsResult with SSS ratio and pass/fail status.
        """
        result = PhysicsResult()
        
        if face_img is None or face_img.size == 0:
            return result
        
        # Split channels (OpenCV = BGR)
        b, g, r = cv2.split(face_img)
        
        # Apply Gaussian blur to reduce noise before Laplacian
        b_blur = cv2.GaussianBlur(b, (3, 3), 0)
        r_blur = cv2.GaussianBlur(r, (3, 3), 0)
        
        # Calculate Laplacian variance (sharpness) for each channel
        lap_b = cv2.Laplacian(b_blur, cv2.CV_64F)
        lap_r = cv2.Laplacian(r_blur, cv2.CV_64F)
        
        var_b = lap_b.var()
        var_r = lap_r.var()
        
        # Avoid division by zero
        if var_r < 0.001:
            var_r = 0.001
        
        # SSS Ratio: Blue sharpness / Red sharpness
        # Real skin: ratio > 1.05 (blue is sharper)
        # Screen: ratio ≈ 1.0 (both equal)
        ratio = var_b / var_r
        
        result.sss_ratio = round(ratio, 4)
        result.blue_variance = round(var_b, 2)
        result.red_variance = round(var_r, 2)
        result.sss_passed = ratio > self.config.sss_ratio_threshold
        
        return result

    # =========================================================================
    # 4. TEMPORAL FREQUENCY RESPONSE
    # =========================================================================
    
    def _check_temporal_response(
        self, 
        face_img: np.ndarray, 
        screen_color: str, 
        timestamp_ms: Optional[float] = None
    ) -> TemporalResult:
        """
        Temporal frequency response analysis.
        
        Biological response timing:
        - Human skin has 100-300ms delay to light stimulus
        - Pre-recorded video: response already baked in (0ms delay)
        - Real-time deepfake: processing lag in wrong direction
        
        Args:
            face_img: BGR face image
            screen_color: Current screen color from frontend
            timestamp_ms: Optional timestamp for precise timing
            
        Returns:
            TemporalResult with delay measurement and validity.
        """
        result = TemporalResult()
        
        if face_img is None or face_img.size == 0:
            return result
        
        current_time = timestamp_ms or (time.time() * 1000)
        
        # Calculate current face luminance
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        luminance = np.mean(gray)
        
        self.luminance_buffer.append((current_time, luminance, screen_color))
        
        # Detect color change
        if screen_color != self.last_screen_color and self.last_screen_color is not None:
            self.last_color_change_time = current_time
            self.color_change_timestamps.append((screen_color, current_time))
            
            # Keep only recent changes
            if len(self.color_change_timestamps) > 10:
                self.color_change_timestamps.pop(0)
        
        self.last_screen_color = screen_color
        
        # Analyze response if we have enough data
        if len(self.luminance_buffer) >= 30 and self.last_color_change_time > 0:
            # Find luminance change after color flash
            pre_flash = [lum for t, lum, _ in self.luminance_buffer 
                        if t < self.last_color_change_time]
            post_flash = [(t, lum) for t, lum, _ in self.luminance_buffer 
                         if t >= self.last_color_change_time]
            
            if len(pre_flash) >= 5 and len(post_flash) >= 5:
                baseline = np.mean(pre_flash[-5:])
                
                # Find when luminance changed significantly (>5% deviation)
                for t, lum in post_flash:
                    if abs(lum - baseline) > 0.05 * baseline:
                        delay = t - self.last_color_change_time
                        result.delay_ms = delay
                        result.response_detected = True
                        result.is_biological = (
                            self.config.temporal_delay_min_ms <= delay <= 
                            self.config.temporal_delay_max_ms
                        )
                        break
        
        return result

    # =========================================================================
    # 5. MOIRÉ PATTERN DETECTION
    # =========================================================================
    
    def _check_moire_pattern(self, face_img: np.ndarray) -> MoireResult:
        """
        Detect Moiré patterns indicating screen replay attack.
        
        When a camera films a screen, interference patterns appear
        in the frequency domain due to screen pixel grid interaction
        with camera sensor.
        
        Args:
            face_img: BGR face image
            
        Returns:
            MoireResult with screen detection status and score.
        """
        result = MoireResult()
        
        if face_img is None or face_img.size == 0:
            return result
        
        # Convert to grayscale
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # Apply 2D FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # Log transform for visualization
        log_magnitude = np.log1p(magnitude)
        
        # Normalize
        log_magnitude = log_magnitude / np.max(log_magnitude)
        
        # Look for periodic peaks that indicate screen pixels
        # Real faces have smooth frequency distribution
        # Screens have sharp peaks at regular intervals
        
        h, w = log_magnitude.shape
        center_h, center_w = h // 2, w // 2
        
        # Exclude DC component (center) and low frequencies
        mask = np.zeros_like(log_magnitude)
        mask[center_h-10:center_h+10, center_w-10:center_w+10] = 1
        masked_mag = log_magnitude * (1 - mask)
        
        # Calculate ratio of peak values to mean
        peak_value = np.max(masked_mag)
        mean_value = np.mean(masked_mag[masked_mag > 0])
        
        moire_score = peak_value / (mean_value + 1e-10)
        
        # High score indicates periodic pattern (screen)
        result.moire_score = round(moire_score, 3)
        result.is_screen = moire_score > (1 / self.config.moire_threshold)
        
        return result

    # =========================================================================
    # 6. CHROMA SYNC CHECK
    # =========================================================================
    
    def _check_chroma(self, face_img: np.ndarray, screen_color: str) -> bool:
        """
        Verify face reflection matches screen color.
        
        Physics: Real faces in real rooms reflect screen light.
        Pre-recorded video: Lighting doesn't match current screen.
        
        Args:
            face_img: BGR face image
            screen_color: Current screen color ("RED", "BLUE", "WHITE", "GREEN")
            
        Returns:
            True if reflection matches expected physics.
        """
        if face_img is None or face_img.size == 0:
            return False
        
        # Get average color of face (BGR)
        avg_color = np.mean(face_img, axis=(0, 1))
        blue_val, green_val, red_val = avg_color
        
        sensitivity = self.config.chroma_sensitivity
        
        if screen_color == "RED":
            return red_val > (blue_val * sensitivity)
        
        elif screen_color == "BLUE":
            # Lower threshold for blue due to skin absorption
            return blue_val > (red_val * 0.8)
        
        elif screen_color == "GREEN":
            return green_val > (red_val * 0.9) and green_val > (blue_val * 0.9)
        
        elif screen_color == "WHITE":
            # All channels should be relatively balanced and elevated
            return True
        
        return True

    # =========================================================================
    # 7. STATIC IMAGE DETECTION (THE KEY ANTI-PHOTO DEFENSE)
    # =========================================================================
    
    def _check_static_image(self) -> StaticImageResult:
        """
        Detect static images (photos) by analyzing temporal variance.
        
        THIS IS THE MOST IMPORTANT ANTI-PHOTO DEFENSE.
        
        Real faces: Green channel fluctuates due to blood volume pulse (~0.1-1%)
                   Variance over 5 seconds should be > 2.0
        
        Photos:    Green channel is STATIC (only camera sensor noise)
                   Variance over 5 seconds should be < 0.5
        
        AI images: Also static - no temporal variance at all
        
        Returns:
            StaticImageResult with variance and is_alive flag
        """
        result = StaticImageResult()
        
        if len(self.green_signal_buffer) < 60:  # Need at least 2 seconds
            return result
        
        # Get recent green channel values
        recent_signal = np.array(list(self.green_signal_buffer)[-90:])  # Last 3 seconds
        
        # Calculate variance (normalized by mean to handle different lighting)
        mean_val = np.mean(recent_signal)
        if mean_val < 1:
            mean_val = 1
        
        # Coefficient of variation (CV) - normalized variance
        variance = np.std(recent_signal) / mean_val * 100  # As percentage
        
        result.signal_variance = round(variance, 3)
        
        # Real faces: CV typically 1.0-3% due to blood flow
        # Photos/AI on screen: CV typically < 0.8% (just camera/screen noise)
        # Also check for suspiciously HIGH variance (screen flicker)
        too_stable = variance < self.config.min_signal_variance
        too_noisy = variance > 8.0  # Screen flicker can cause high variance but wrong pattern
        result.is_static = too_stable or too_noisy
        result.is_alive = not result.is_static
        
        return result

    def _check_screen_texture(self, face_img: np.ndarray) -> Tuple[bool, float]:
        """
        Detect screen replay by analyzing texture uniformity.
        
        Screens displaying images have unnaturally uniform color patches
        because of pixel rendering. Real skin has micro-texture variation.
        
        Returns:
            (is_screen_like, uniformity_score)
        """
        if face_img is None or face_img.size == 0:
            return False, 0.0
        
        # Convert to grayscale
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # Calculate local standard deviation using a small window
        # Real skin: high local variation due to pores, texture
        # Screen: low local variation (smooth pixel rendering)
        kernel_size = 5
        mean_local = cv2.blur(gray, (kernel_size, kernel_size))
        sqr_mean = cv2.blur(gray ** 2, (kernel_size, kernel_size))
        local_std = np.sqrt(np.maximum(sqr_mean - mean_local ** 2, 0))
        
        avg_local_std = np.mean(local_std)
        
        # Real skin typically has local_std > 8-12
        # Screens/AI images typically have local_std < 6
        is_screen_like = avg_local_std < 6.0
        
        return is_screen_like, round(avg_local_std, 2)

    # =========================================================================
    # 7. MULTI-MODAL FUSION
    # =========================================================================
    
    def _compute_fusion_score(
        self,
        rppg: RPPGResult,
        physics: PhysicsResult,
        chroma_passed: bool,
        temporal: TemporalResult,
        moire: MoireResult
    ) -> Tuple[bool, float, dict]:
        """
        Multi-modal fusion scoring with weighted confidence.
        
        Combines all detection layers into final liveness decision.
        Uses weighted scoring with anomaly detection for inconsistencies.
        
        Args:
            rppg: rPPG heart rate results
            physics: Subsurface scattering results
            chroma_passed: Chroma sync check result
            temporal: Temporal response results
            moire: Moiré detection results
            
        Returns:
            Tuple of (is_human, confidence_score, details_dict)
        """
        cfg = self.config
        score = 0.0
        max_score = 100.0
        
        details = {
            "bpm": rppg.bpm,
            "bpm_signal_quality": rppg.signal_quality,
            "hrv_rmssd": rppg.hrv.rmssd,
            "hrv_entropy": rppg.hrv.entropy,
            "physics_passed": physics.sss_passed,
            "sss_ratio": physics.sss_ratio,
            "chroma_passed": chroma_passed,
            "temporal_delay_ms": temporal.delay_ms,
            "temporal_biological": temporal.is_biological,
            "moire_detected": moire.is_screen,
            "moire_score": moire.moire_score,
        }
        
        # 1. rPPG Score (weighted by signal quality)
        if rppg.is_valid:
            rppg_score = cfg.weight_rppg * rppg.signal_quality
            score += rppg_score
            details["rppg_contribution"] = round(rppg_score, 1)
        
        # 2. HRV Score (biological chaos)
        if rppg.hrv.is_biologically_valid:
            hrv_score = cfg.weight_hrv
            score += hrv_score
            details["hrv_contribution"] = hrv_score
        
        # 3. Physics SSS Score  
        if physics.sss_passed:
            # Scale by how much ratio exceeds threshold
            sss_confidence = min(1.0, (physics.sss_ratio - cfg.sss_ratio_threshold) / 0.15)
            sss_confidence = max(0.3, sss_confidence)  # Minimum 30% if passed
            physics_score = cfg.weight_physics_sss * sss_confidence
            score += physics_score
            details["physics_contribution"] = round(physics_score, 1)
        
        # 4. Chroma Score
        if chroma_passed:
            score += cfg.weight_chroma
            details["chroma_contribution"] = cfg.weight_chroma
        
        # 5. Temporal Score
        if temporal.response_detected and temporal.is_biological:
            score += cfg.weight_temporal
            details["temporal_contribution"] = cfg.weight_temporal
        
        # 6. Moiré Penalty (negative if screen detected)
        if moire.is_screen:
            score -= cfg.weight_moire * 5  # 5x penalty for screen detection (was 3x)
            details["moire_penalty"] = -cfg.weight_moire * 5
        else:
            score += cfg.weight_moire
            details["moire_contribution"] = cfg.weight_moire
        
        # 7. BPM Stability Check (anti-photo attack)
        # Photos/AI produce random noise that causes BPM to jump around wildly
        # Real faces have stable, consistent heartbeats
        bpm_stability_penalty = 0
        if len(self.raw_bpm_history) >= 10:
            bpm_std = np.std(list(self.raw_bpm_history))
            details["bpm_stability_std"] = round(bpm_std, 1)
            
            if bpm_std > cfg.bpm_stability_threshold:
                # Unstable BPM = likely photo/screen/AI
                bpm_stability_penalty = min(50, (bpm_std - cfg.bpm_stability_threshold) * 2)  # 2x penalty
                score -= bpm_stability_penalty
                details["bpm_stability_penalty"] = round(-bpm_stability_penalty, 1)
        
        # 8. STATIC IMAGE CHECK (THE MOST IMPORTANT ANTI-PHOTO DEFENSE)
        static_result = self._check_static_image()
        details["signal_variance"] = static_result.signal_variance
        details["is_static_image"] = static_result.is_static
        
        if static_result.is_static:
            # CRITICAL: Static image = definitely not alive
            # Massive penalty - photos/AI images should NEVER pass
            score -= 70  # -70 points for static image (was -50)
            details["static_image_penalty"] = -70
        elif static_result.is_alive:
            # Bonus for showing signs of life (reduced to avoid false positives)
            score += 10
            details["alive_bonus"] = 10
        
        # 9. SCREEN TEXTURE CHECK (catches AI images on screens)
        # This uses a face_img that we need to pass - store it in instance for now
        if hasattr(self, "_last_face_img") and self._last_face_img is not None:
            is_screen_texture, texture_score = self._check_screen_texture(self._last_face_img)
            details["texture_uniformity"] = texture_score
            details["screen_texture_detected"] = is_screen_texture
            if is_screen_texture:
                score -= 30  # Penalty for screen-like texture
                details["screen_texture_penalty"] = -30
        
        # Normalize to 0-100
        confidence = max(0, min(100, score))
        
        # Final decision threshold
        # Generous for hackathon: 50% passes
        is_human = confidence >= 50
        
        # CRITICAL GATE: If static image detected, FORCE False regardless of score
        if static_result.is_static and len(self.green_signal_buffer) >= 60:
            is_human = False
            details["forced_false_reason"] = "static_image_detected"
        
        # Anomaly detection: if signals are inconsistent, reduce confidence
        anomalies = []
        if rppg.is_valid and not physics.sss_passed:
            anomalies.append("heartbeat_but_no_skin_physics")
        if physics.sss_passed and moire.is_screen:
            anomalies.append("skin_physics_but_screen_detected")
        
        if anomalies:
            confidence *= 0.8  # 20% penalty for anomalies
            details["anomalies"] = anomalies
        
        return is_human, round(confidence, 1), details

    # =========================================================================
    # MAIN PIPELINE
    # =========================================================================
    
    def process_frame(
        self,
        forehead_roi: Optional[np.ndarray],
        face_img: Optional[np.ndarray],
        screen_color: str,
        timestamp_ms: Optional[float] = None
    ) -> LivenessResult:
        """
        Main processing pipeline. Called by Sohini's FastAPI backend.
        
        Processes a single frame through all detection layers and
        returns a fused liveness result.
        
        Args:
            forehead_roi: Cropped forehead region from Jai's CV pipeline.
                         Used for rPPG heart rate detection.
            face_img: Full face BGR image from Jai's CV pipeline.
                     Used for physics and chroma checks.
            screen_color: Current screen color ("RED", "BLUE", "WHITE", "GREEN").
                         From Srijan's frontend via Sohini's API.
            timestamp_ms: Optional timestamp for temporal analysis.
            
        Returns:
            LivenessResult with is_human, confidence, BPM, HRV score, and details.
        """
        result = LivenessResult()
        
        # Store face_img for texture analysis in fusion scoring
        self._last_face_img = face_img
        
        # 1. Update rPPG buffer with forehead green channel
        if forehead_roi is not None and forehead_roi.size > 0:
            try:
                mean_green = np.mean(forehead_roi[:, :, 1])
                self.green_signal_buffer.append(mean_green)
            except (IndexError, ValueError) as e:
                logger.warning(f"Error extracting green channel: {e}")
        
        # 2. Run all detection algorithms
        rppg_result = self._get_heart_rate()
        physics_result = self._check_physics_sss(face_img)
        chroma_passed = self._check_chroma(face_img, screen_color)
        temporal_result = self._check_temporal_response(face_img, screen_color, timestamp_ms)
        moire_result = self._check_moire_pattern(face_img)
        
        # 3. Multi-modal fusion
        is_human, confidence, details = self._compute_fusion_score(
            rppg_result,
            physics_result,
            chroma_passed,
            temporal_result,
            moire_result
        )
        
        # 4. Build final result
        result.is_human = is_human
        result.confidence = confidence
        result.bpm = rppg_result.bpm
        result.hrv_score = rppg_result.hrv.entropy
        result.signal_quality = rppg_result.signal_quality
        result.details = details
        
        return result

    def reset(self) -> None:
        """Reset all buffers. Call when starting a new verification session."""
        self.green_signal_buffer.clear()
        self.luminance_buffer.clear()
        self.bpm_history.clear()
        self.rr_intervals.clear()
        self.color_change_timestamps.clear()
        self.last_bpm = 0
        self.last_screen_color = None
        self.last_color_change_time = 0.0
        logger.info("PRISM Engine buffers reset")


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

# Maintain backward compatibility with existing test_main.py
# The old API returned dict, new API returns dataclass
# This wrapper ensures both work

def _result_to_dict(result: LivenessResult) -> dict:
    """Convert LivenessResult to dict for legacy compatibility."""
    return {
        "is_human": result.is_human,
        "confidence": result.confidence,
        "details": {
            "bpm": result.bpm,
            "bpm_signal_strength": result.signal_quality,
            "physics_passed": result.details.get("physics_passed", False),
            "chroma_passed": result.details.get("chroma_passed", False),
            "sss_ratio": result.details.get("sss_ratio", 0),
            "hrv_entropy": result.hrv_score,
        }
    }