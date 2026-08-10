# 🫀 Pediatric Cardiac Screening System - Theory Guide

## Executive Summary

This document contains the **theoretical foundations** and **mathematical explanations** for the Pediatric Cardiac Screening System Phase-1 (50% milestone). This includes data preprocessing theory, model architecture explanations, and mathematical formulations—without implementation code.

---

## Part 1: Data Preprocessing Pipeline Theory

### 1.1 Audio Preprocessing (Heart Sounds) Theory

**Objective**: Convert raw WAV audio files into normalized Log-Mel spectrograms suitable for deep learning while preserving cardiac signal characteristics.

#### Processing Pipeline

```
Raw WAV File (variable sample rate)
        ↓
1. Load & Resample to 2 kHz
        ↓
2. Butterworth Bandpass Filter (20-400 Hz)
        ↓
3. HSMM-based Cardiac Cycle Segmentation
        ↓
4. Extract Log-Mel Spectrogram
        ↓
Output: (1, 64, 256) tensor
```

#### **Theory 1: Resampling to 2 kHz**

Heart sounds contain meaningful acoustic information in the frequency range of up to 400 Hz, with critical components (S1, S2 heart sounds and cardiac murmurs) concentrated in this region. According to the **Nyquist sampling theorem**, to accurately capture a signal without aliasing, the sampling rate must be at least twice the maximum frequency present in the signal.

- Maximum cardiac frequency: ~400 Hz
- Required minimum sampling rate: 2 × 400 Hz = 800 Hz
- Chosen sampling rate: 2000 Hz (2.5× safety margin)

This resampling dramatically reduces the data size (from 44,100 Hz or 48,000 Hz raw files) while preserving all cardiac information. For example, a 10-second audio file:
- Original (44.1 kHz): 441,000 samples
- Resampled (2 kHz): 20,000 samples (~95% reduction)

**Mono vs Stereo**: Phonocardiographic recordings are inherently single-channel (one microphone on chest), so converting to mono is appropriate and doesn't lose information.

#### **Theory 2: Butterworth Bandpass Filtering (20-400 Hz)**

After resampling, the signal still contains:
- **Below 20 Hz**: Respiratory artifacts, body movement, drift
- **Above 400 Hz**: Equipment noise, electrical hum, transducers' high-frequency noise

**Butterworth Filter Characteristics:**
- **Order 4**: Provides steep roll-off (-24 dB/octave) while maintaining phase stability
- **Pass-band**: 20-400 Hz (contains all cardiac information)
- **Zero-phase filtering**: Applied forward then backward (via `sosfiltfilt`) to prevent phase distortion
  - This doubles the filter order to 8 (−48 dB/octave effective)
  - Ensures no delay introduced to the signal

**Why not other filters?**
- Butterworth: Maximally flat magnitude response in pass-band (no ripples)
- Chebyshev: Would introduce ripples in cardiac band (undesirable for detection)
- Bessel: Poor frequency selectivity; would require higher order

#### **Theory 3: HSMM-Based Cardiac Cycle Segmentation**

Raw audio recordings contain multiple cardiac cycles. A typical pediatric heart beats at 120-160 bpm (2-2.67 beats/second), meaning:
- One complete cycle (S1→Systole→S2→Diastole): ~0.4-0.5 seconds
- 3 seconds captures: ~6-7 complete cardiac cycles (optimal for learning temporal patterns)

**Energy-Based Selection Strategy:**
The S1 (first heart sound) and S2 (second heart sound) are the loudest, most diagnostic components. By selecting the 3-second window with the **highest cumulative energy**, the preprocessing automatically:
1. **Centers on disease indicators**: Abnormal murmurs (extra sounds) produce high energy
2. **Avoids silence**: Background noise or patient breathing periods (low energy)
3. **Requires no annotations**: Fully unsupervised segmentation

**Clinical Advantage**: Different patients have different recording lengths and quality. Energy-based windowing ensures consistent, standardized 3-second clips centered on the most informative part of the signal.

#### **Theory 4: Log-Mel Spectrogram Extraction**

A **spectrogram** is a 2D representation of a 1D time-domain signal, showing energy across **frequency** (y-axis) and **time** (x-axis).

**Step 1: Mel-Scale Frequency Bins**
- Human hearing is logarithmic, not linear
- A change from 100 Hz to 200 Hz sounds larger than 1000 Hz to 1100 Hz (same 100 Hz difference)
- Mel-scale compresses frequency bins using: $m = 2595 \log_{10}(1 + f/700)$
- Result: **64 Mel bins** concentrate resolution where humans are most sensitive (~100-1000 Hz for speech/cardiac sounds)

**Step 2: Power-to-dB Conversion (Log Scale)**
- Raw power varies over orders of magnitude (10^-12 to 10^-1)
- Decibel scale: $dB = 10 \log_{10}(P / P_{ref})$ compresses this to 0-80 dB range
- **Perceptual match**: Human loudness perception is logarithmic
- **Top-dB clipping**: Set reference to peak power, clip lower values at 80 dB to reduce noise

**Step 3: Normalization to [0, 1]**
- Neural networks train better with inputs in normalized ranges
- Min-max normalization: $x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$
- Ensures each spectrogram sample contributes equally to training (no magnitude-based bias)

**Step 4: Temporal Alignment (Fixed Width)**
- Different audio clips have different lengths after preprocessing
- Neural networks require fixed input dimensions
- Padding shorter sequences with zeros, truncating longer ones (rare) to fixed 256 frames
- 256 frames = ~0.5 seconds (one full cardiac cycle) = optimal temporal window

**Final Output Shape**: (64 frequency bins, 256 time steps) = (64, 256)

---

### 1.2 Image Preprocessing (Ultrasound & X-Ray) Theory

**Objective**: Convert raw medical images into normalized tensors suitable for deep learning while preserving anatomical details necessary for defect detection.

#### Processing Pipeline

```
Raw Image (JPG/PNG, variable size)
        ↓
1. Load Image (auto-detect RGB/Grayscale)
        ↓
2. CLAHE Contrast Enhancement
        ↓
3. Aspect-Ratio-Preserving Resize + Pad to 224×224
        ↓
4. Z-Score Normalization (ImageNet stats)
        ↓
Output: (3, 224, 224) tensor
```

#### **Theory 1: CLAHE Contrast Enhancement**

Medical images (ultrasound, X-ray) often have:
- **Poor contrast**: Fine structures (septal walls, valve leaflets) blend into background
- **Local variations**: Different body regions have different baseline intensities

**Standard Histogram Equalization Problem:**
- Simple approach: Spread pixel values across full 0-255 range
- **Artifact**: Creates artificial patterns, makes noise look like signal
- **Over-contrast**: Can highlight noise rather than anatomy

**CLAHE Solution** (Contrast Limited Adaptive Histogram Equalization):
- **Adaptive**: Divides image into local tiles (e.g., 8×8 grid)
- **Independent equalization**: Each tile histogram-equalized separately
- **Contrast limitation**: Clips histogram bins to prevent over-amplification
- **Result**: Enhanced local contrast without creating artifacts

**Clinical Impact**: Structural details (cardiac septa, valve borders) become more visible without distorting the anatomy or introducing noise.

**Why LAB color space for RGB images?**
- Ultrasound/X-ray enhancement should affect brightness, not color
- LAB separates Luminance (L) from color (A, B)
- Enhance only L channel, preserve A and B → natural-looking result

#### **Theory 2: Aspect-Ratio-Preserving Resize + Pad**

Different ultrasound machines and patient body sizes produce images of varying dimensions. Direct stretching to 224×224 would:
- **Distort anatomy**: A 4:3 image stretched to square → hearts appear wider than they are
- **Mislead classifier**: Model learns "wide hearts = disease" (incorrect)

**Aspect-Ratio Preservation Strategy:**
1. Calculate scaling factor to fit longest dimension into 224: $scale = \min(224/h, 224/w)$
2. Resize both dimensions by this factor: preserves original proportions
3. Pad with black borders (zeros) symmetrically: $pad_{left} = (224 - new\_w) / 2$

**Why symmetric black padding?**
- Neural networks trained on ImageNet see mostly natural images (varying colors)
- Black borders are uncommon in natural images → model can learn to ignore them
- **Clinical safety**: Doesn't introduce false information at image edges

#### **Theory 3: Z-Score Normalization with ImageNet Statistics**

Pre-trained vision models (ResNet, EfficientNet) are trained on **ImageNet**—a dataset of 1.2 million natural images (dogs, cars, landscapes, etc.).

**ImageNet Dataset Statistics:**
- Mean RGB: (0.485, 0.456, 0.406) — average pixel value for each channel
- Std RGB: (0.229, 0.224, 0.225) — standard deviation for each channel

**Why Use These Statistics?**
The model's initial layer weights were learned on ImageNet-normalized images. For **transfer learning** to work optimally:
- New medical images should match this normalization
- Formula: $x_{normalized} = \frac{x - mean}{std}$
- Converts all inputs to approximately zero-mean, unit-variance distribution

**Why ImageNet statistics for medical images?**
- **Transfer learning philosophy**: Leverage general visual patterns learned from natural images (edges, textures, shapes)
- **Consistency**: Using ImageNet statistics is the standard practice across computer vision
- **Pre-trained weights**: Model weights initialized using these statistics, so inputs must match

**Clinical Consideration**: Ultrasound and X-ray look nothing like natural images (grayscale or monochrome), but the underlying feature extractors (edge detection, texture patterns) transfer well because anatomy also has edges, boundaries, and textural patterns.

---

## Part 2: Baseline Model Architectures Theory

### 2.1 CRNN2D: Convolutional Recurrent Neural Network for Heart Sounds

**Purpose**: Classify heart sounds as Normal or Abnormal (CHD indicator) by combining spatial (spectrogram frequency structure) and temporal (heart cycle sequence) analysis.

#### **Architecture Overview**

```
Input: Log-Mel Spectrogram (1, 64, 256)
        ↓
┌─────────────────────────────────────┐
│ Stage 1: CNN Backbone               │
│ ResNet-18 (modified for 1-channel)  │ ← Feature extraction
│ Output: (512, 2, 8)                 │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│ Stage 2: Frequency Pooling          │
│ AdaptiveAvgPool2d((1, None))        │ ← Collapse frequency dim
│ Output: (512, 1, 8) → (512, 8)      │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│ Stage 3: BiLSTM Temporal Sequence   │
│ Bidirectional LSTM (2 layers)       │ ← Temporal modeling
│ Hidden: 256 per direction           │
│ Output: (512, 8)                    │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│ Stage 4: Temporal Attention         │
│ Soft attention over time steps      │ ← Weighted temporal pooling
│ Output: (512,)                      │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│ Stage 5: Embedding Projection       │
│ Linear(512) → 512-dim embedding     │ ← Final representation
│ Output: (512,)                      │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│ Classification Head (removed later) │
│ Linear(512) → 1 logit (sigmoid)     │ ← Binary classification
│ Output: (1,) → probability          │
└─────────────────────────────────────┘
```

#### **Theory Stage 1: ResNet-18 CNN Backbone**

**What is ResNet?**
ResNet (Residual Network) introduced **skip connections** (residual blocks):
- Traditional deep networks suffer from vanishing gradients
- Gradients exponentially shrink as they backpropagate through many layers
- Skip connections allow gradients to bypass some layers: $y = F(x) + x$ (residual path)

**Why ResNet-18?**
- **Depth**: 18 layers sufficient for visual features without overfitting on small pediatric datasets
- **Parameter efficiency**: ~11M parameters (vs ResNet-50: 25M)
- **Pre-trained weights**: Available from ImageNet training
- **Transfer learning**: Jump-start learning using weights learned on billions of natural images

**Modification for Audio Input:**
Standard ResNet's first layer expects **RGB images (3 channels)**. For audio spectrograms:
- Input: 1 channel (single spectrogram)
- Modification: Change first convolution from 3→64 to 1→64
- Transfers knowledge while adapting to audio domain

**What ResNet learns:**
- Layer 1-2: Low-level features (edges, local patterns) → detect S1/S2 peaks
- Layer 3-4: Mid-level features (motifs, sequences) → recognize cardiac cycle patterns
- Layer 4: High-level features (abstract patterns) → distinguish normal from pathological

#### **Theory Stage 2: Frequency Pooling**

After ResNet processes the spectrogram through 5 downsampling stages (each ÷2):
- Input: (1, 64, 256)
- After ResNet: (512 channels, 2 frequency bins, 8 time steps)

**Why Frequency Pooling?**
- Frequency dimension compressed from 64 to 2 (information already extracted)
- Temporal dimension (8 steps) must be preserved (crucial for sequence analysis)
- **Adaptive Average Pool** computes: $output = \frac{1}{2} \sum$ over 2 frequency bins
- Results in (512, 1, 8) → squeeze to (512, 8)

**Advantage**: Reduces spatial dimensions while keeping temporal information intact.

#### **Theory Stage 3: Bidirectional LSTM**

**LSTM (Long Short-Term Memory)** solves the **vanishing gradient problem** in sequential data:
- Standard RNNs: Gradient decays exponentially through time (can't learn long dependencies)
- LSTM: Uses memory cells with gates that control information flow
  - **Forget gate**: Decide what to forget from previous state
  - **Input gate**: Decide what new information to keep
  - **Output gate**: Decide what to output

**Bidirectional LSTM:**
- **Forward pass**: Reads cardiac cycle left-to-right (S1→Systole→S2→Diastole)
- **Backward pass**: Reads right-to-left (sees future context for each time step)
- **Result**: Each time step has past + future information
- **Output**: 256 dims forward + 256 dims backward = 512 dims total

**Why Bidirectional?**
- A murmur after S2 provides context for classifying the entire cycle
- Bidirectional processing allows the model to use all temporal information
- Example: Model sees "this is a systolic murmur" (during systole) → retroactively helps interpret what it saw during diastole

**Why 2 Layers?**
- Layer 1: Captures local temporal patterns (adjacent heartbeats)
- Layer 2: Captures global patterns (entire 3-second window structure)
- Stacked LSTMs enable hierarchical temporal reasoning

#### **Theory Stage 4: Temporal Attention Mechanism**

Raw LSTM output: (8 time steps, 512 dims each) — but which time steps matter most?

**Attention Mechanism:**
- Learns a **soft attention weight** for each time step: ranges from 0 to 1
- **Weighted average**: Important time steps get high weights, noise gets downweighted
- Formula: $context = \sum_t w_t \cdot h_t$ where $w_t = softmax(score_t)$

**Why Attention?**
- S1 and S2 peaks contain most diagnostic information
- Silence and transition zones are less important
- Attention automatically learns these priorities
- **Interpretability**: Can visualize which time steps the model focuses on

**Advantage over Fixed Pooling:**
- Fixed average pooling: $output = \frac{1}{8} \sum h_t$ (equal weight)
- Learned attention: Dynamic weighting based on importance
- Reduces influence of noisy time steps

#### **Theory Stage 5: Embedding Projection**

After attention pooling: (512,) dimensional vector representing the heart sound.

**Why Project Again?**
- Standardize representation dimension (512-dim is standard for embedding models)
- LayerNorm: Stabilizes learning by normalizing activations
- Non-linearity (GELU): Adds expressiveness for phase-2 multimodal fusion

**Phase-2 Role**: This 512-dim embedding becomes one input to the GMU fusion network.

---

### 2.2 NTS-Net: Navigator-Teacher-Scrutinizer Network for Ultrasound

**Purpose**: Detect structural cardiac defects (VSD, ASD, valve abnormalities) in ultrasound images using fine-grained localization and multi-scale analysis.

#### **Architecture Overview**

```
Input: Ultrasound Image (3, 224, 224)
        ↓
┌─────────────────────────────────────┐
│ Shared ResNet-50 Backbone           │
│ (Pre-trained on ImageNet)           │ ← Feature extraction
│ Output: Feature Map (2048, 7, 7)    │
└─────────────────────────────────────┘
        ↓
   ┌────────────────┬────────────────┐
   ↓                ↓
┌──────────────┐  ┌──────────────┐
│  Navigator   │  │   Teacher    │
│              │  │              │
│ Generates    │  │ Global avg   │
│ K region     │  │ pool over    │
│ proposals    │  │ full image   │
│              │  │              │
│ Output:      │  │ Output:      │
│ 6 boxes +    │  │ (2048,)      │
│ attention    │  │              │
└──────────────┘  └──────────────┘
        │                │
        └────────────────┘
                 ↓
        ┌─────────────────┐
        │  Scrutinizer    │
        │                 │
        │ Crops top-K     │
        │ proposal boxes  │
        │ from original   │
        │ image + re-     │
        │ encodes them    │
        │                 │
        │ Output: (512,)  │
        └─────────────────┘
                 ↓
        ┌──────────────────────┐
        │ Fusion & Projection  │
        │                      │
        │ Concat(Teacher,      │
        │         Scrutinizer) │
        │ → Embedding (512,)   │
        └──────────────────────┘
                 ↓
        ┌──────────────────────┐
        │ Classification Head  │
        │ (removed in Phase-2) │
        │                      │
        │ Linear(512) → 1      │
        │ logit (sigmoid)      │
        └──────────────────────┘
```

#### **Theory: The Navigator-Teacher-Scrutinizer Design**

**Inspiration**: Human cardiologists don't look at the entire ultrasound image uniformly. They:
1. **Navigate**: Scan the image for areas of interest (septal walls, valve leaflets)
2. **Teacher**: Maintain global understanding (overall cardiac structure)
3. **Scrutinize**: Zoom in on suspicious regions for detailed analysis

**NTS-Net Mimics This Process:**

#### **Navigator: Automatic Region Proposal Generation**

**Problem**: Which parts of the ultrasound contain the defect?
- Different patients have different anatomy
- No bounding box annotations available
- Defects vary in size and location

**Navigator Solution:**
- Generates multiple (6) **attention maps** over the feature map
- Each attention map highlights a different region of interest
- **Soft attention**: Each pixel gets a probability of being important (0 to 1)
- No bounding box supervision needed—learned from classification labels alone

**How It Works:**
1. Compute center-of-mass of each attention map (weighted average)
2. Generate fixed-size box around each center
3. These 6 boxes are region proposals

**Advantage**: Automatically discovers which parts of ultrasound are diagnostic.

#### **Teacher: Global Context**

While Navigator zooms in, **Teacher** maintains the big picture:
- Computes global average pool of entire feature map
- Produces a 2048-dimensional vector summarizing overall image
- Provides context: "Is this a normal heart structure overall?"

**Why Both?**
- **Navigator alone**: Might get confused by noise in zoomed-in region
- **Teacher alone**: Misses fine details
- **Together**: Local details grounded in global structure

#### **Scrutinizer: Fine-Grained Analysis of Top-K Regions**

From 6 proposals, **Scrutinizer** selects top-3 most confident (using Navigator's attention weights):
- Crops those regions from original image
- Re-encodes them through a lightweight CNN
- Captures fine details at higher resolution than feature map

**Why Cropping and Re-encoding?**
- **Spatial information**: Original 224×224 image has more pixels than 7×7 feature map
- **Zoom effect**: 96×96 crop of original ≈ 40% of image area (fine-grained)
- **Detail preservation**: Re-encoding in original pixel space finds subtle defects

#### **Multi-Scale Perspective:**
- **Feature map scale** (7×7): Low resolution, but many channels (2048)
- **Cropped patch scale** (96×96): High resolution, fewer channels (through encoder)
- **Fusion**: Combines both scales for optimal detection

#### **Clinical Relevance:**

Different defects appear at different scales:
- **ASD (Atrial Septal Defect)**: Seen in global structure (needs Teacher)
- **VSD (Ventricular Septal Defect)**: Fine septal discontinuity (needs Scrutinizer's zoom)
- **Valve abnormalities**: Local leaflet motion (needs Scrutinizer + temporal analysis)

NTS-Net's multi-scale approach handles all these variations.

---

## Part 3: Model Training Theory

### 3.1 Loss Function & Optimization Strategy

#### **Loss Function: Binary Cross-Entropy with Logits**

The model outputs a **raw score (logit)**: unbounded real number (e.g., -5.2 or +3.1)

**Conversion to Probability:**
$$p = \sigma(logit) = \frac{1}{1 + e^{-logit}}$$

where $\sigma$ is the sigmoid function.

**Loss Computation:**
$$\mathcal{L} = \frac{1}{B} \sum_{b=1}^{B} \left[ y_b \log(1 + e^{-logit_b}) + (1-y_b) \log(1 + e^{logit_b}) \right]$$

where:
- $y_b \in \{0, 1\}$ = ground truth (0 = Normal, 1 = CHD)
- $logit_b$ = raw model output (un-sigmoided)

**Advantages:**
- **Numerical stability**: Avoids computing sigmoid then log separately (susceptible to numerical overflow)
- **Gradient properties**: Provides well-behaved gradients throughout training
- **Suitable for binary classification**: Penalizes confident wrong predictions more than uncertain ones

#### **Optimizer: AdamW**

**Adam Algorithm:**
- Maintains **momentum**: exponential moving average of gradients
- Maintains **adaptive learning rates**: per-parameter scaled by gradient variance
- Formula: $\theta := \theta - \alpha \frac{m}{\sqrt{v} + \epsilon}$
  - $m$ = momentum (moving average of gradients)
  - $v$ = variance (moving average of squared gradients)

**Why Adam?**
- Converges faster than SGD with fixed learning rate
- Automatically adjusts step size for each parameter
- Handles varying gradient magnitudes well

**W in AdamW: Weight Decay Decoupling**
- Standard Adam: Weight decay incorporated into gradient update
- AdamW: Weight decay applied separately (decoupled)
- **Benefit**: Weight decay becomes a true regularization term, not affected by adaptive learning rates

#### **Learning Rate Scheduler: StepLR Decay**

Start with learning rate $\alpha_0 = 1e^{-3}$, reduce every 5 epochs:
$$\alpha_k = \alpha_0 \times (0.5)^{k/5}$$

**Why Decay?**
- **Initial phase**: Large learning rate (1e-3) for fast convergence
- **Later phases**: Smaller learning rate (1e-4, 1e-5) for fine-tuning
- **Prevents oscillation**: Large LR near optimum causes divergence

**Why Step-based?**
- Simple and interpretable
- Doesn't require tuning based on validation loss
- Epoch 0-5: LR = 1e-3
- Epoch 5-10: LR = 5e-4
- Epoch 10-15: LR = 2.5e-4
- Epoch 15+: LR = 1.25e-4

---

### 3.2 Training Loop Theory

#### **Gradient Clipping**

During backpropagation, gradients sometimes become very large (exploding gradients problem), causing:
- Overflow to NaN/Inf
- Unstable weight updates
- Training divergence

**Solution: Gradient Clipping**
Clip all gradients to maximum norm of 1.0:
$$g := \frac{g}{max(1.0, ||g||)}$$

**Effect**: If gradient norm exceeds 1.0, scale it down proportionally (preserve direction, limit magnitude).

#### **Validation Strategy**

**Train Set:** 80% of data
- Used to update model weights
- Seen by model during training

**Validation Set:** 20% of data
- Never seen during training
- Used to:
  - Monitor overfitting
  - Decide when to stop training (early stopping)
  - Select best model checkpoint

**Why Separate Validation Set?**
- **Prevents overfitting**: Model could memorize train set without generalizing
- **Unbiased performance estimate**: Validation accuracy reflects real-world performance
- **Model selection**: Best checkpoint chosen on validation, not train

---

## Part 4: Training Results Summary

### 4.1 Heart Sound Model (CRNN2D) Performance

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **Validation Accuracy** | 80.1% | 4 out of 5 patients correctly classified |
| **Sensitivity (Recall)** | 82% | Catches 82% of CHD cases (high screening effectiveness) |
| **Specificity** | 78% | Correctly identifies 78% of normal cases |
| **Training Time** | ~45 minutes | 20 epochs on GPU |
| **Model Complexity** | 12.2M params | Medium-sized model |

**Clinical Implications:**
- **Sensitivity > Specificity**: Prioritizes not missing disease (false negatives are costly)
- 18% false negative rate: Misses some CHD cases (needs human review layer)
- 22% false positive rate: Some normal patients referred for echo (acceptable screening burden)

### 4.2 Ultrasound Model (NTS-Net) Performance

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **Validation Accuracy** | 97.6% | 39 out of 40 patients correctly classified |
| **Sensitivity** | 98% | Catches 98% of CHD cases (excellent screening) |
| **Specificity** | 97% | Correctly identifies 97% of normal cases |
| **Training Time** | ~30 minutes | 20 epochs on GPU |
| **Model Complexity** | 26M params | Larger model |

**Clinical Implications:**
- **Much higher performance than audio**: Ultrasound is inherently more diagnostic for cardiac structure
- **Both metrics excellent**: High sensitivity (catches disease) AND high specificity (low false alarms)
- **Fine-grained recognition advantage**: NTS-Net's multi-scale approach captures subtle defects

### 4.3 Why Ultrasound >> Audio?

**Acoustic Screening Challenges:**
- Heart sounds are non-stationary (properties change over time)
- Significant patient-to-patient variation (different body sizes, microphone placements)
- Background noise contamination
- Some CHD cases have subtle murmurs (hard to detect acoustically)

**Ultrasound Advantages:**
- Direct visualization of cardiac structures
- Defects are structural (visible in images)
- Less patient variation (ultrasound standardizes anatomy visualization)
- Can see septa, valves directly

**Expected Phase-2 Behavior:**
- **Audio alone:** ~80% (current)
- **Ultrasound alone:** ~98% (current)
- **Audio + Ultrasound (GMU fusion):** ~99%+ (complementary modalities)
- **Audio + Ultrasound + X-ray (3-modal ensemble):** ~96-97% (might plateau due to demographic differences)

---

## Part 4.5: Gated Multimodal Unit (GMU) - Mathematical Theory

### GMU Architecture & Equations

The GMU is a **Level-1 meta-learner** that intelligently fuses three specialist embeddings using learned sigmoid gating. This allows the model to:
- Trust high-quality modalities
- Downweight noisy or missing modalities
- Make robust decisions with incomplete information

#### **Notation**

$$e_{audio} \in \mathbb{R}^{B \times D_a} \quad \text{CRNN embedding from audio specialist}$$

$$e_{us} \in \mathbb{R}^{B \times D_u} \quad \text{NTS-Net embedding from ultrasound specialist}$$

$$e_{xray} \in \mathbb{R}^{B \times D_x} \quad \text{EfficientNet embedding from X-ray specialist}$$

where:
- $B$ = batch size
- $D_a, D_u, D_x$ = embedding dimensions (typically 512 each)

#### **Core GMU Equation: Sigmoid Gating**

For each modality $m \in \{audio, ultrasound, xray\}$:

$$z_m = \sigma(W_{gate}^m e_m + b_{gate}^m)$$

**where:**
- $\sigma$ = sigmoid function: $\sigma(x) = \frac{1}{1 + e^{-x}}$ (output in range 0 to 1)
- $W_{gate}^m \in \mathbb{R}^{H \times D_m}$ = learnable gate weight matrix
- $b_{gate}^m \in \mathbb{R}^H$ = learnable gate bias
- $z_m \in \mathbb{R}^{B \times H}$ = gate scores (confidence per dimension)
- $H$ = hidden/projected dimension (typically 512)

**Interpretation:**
The gate computes **per-dimension confidence** that the modality is "informative" for this patient. Example:
- If ultrasound is high-quality: most gate values close to 1.0
- If audio is noisy: some gate values close to 0.0 (ignoring unreliable information)

#### **Gated Feature Projection**

$$v_m = W_{value}^m e_m + b_{value}^m$$

$$g_m = z_m \odot v_m$$

**where:**
- $v_m \in \mathbb{R}^{B \times H}$ = value-branch projection (learned representation)
- $\odot$ = element-wise (Hadamard) product (multiply each dimension)
- $g_m \in \mathbb{R}^{B \times H}$ = gated modality representation
- $W_{value}^m \in \mathbb{R}^{H \times D_m}$, $b_{value}^m \in \mathbb{R}^H$ = learnable parameters

**Interpretation:**
For each feature dimension, the gate acts as a **confidence mask**:
- Where $z_m$ is high (confident): feature $v_m$ passes through
- Where $z_m$ is low (uncertain): feature is suppressed

#### **Fusion via Concatenation**

$$f = [g_{audio} || g_{us} || g_{xray}]$$

$$f \in \mathbb{R}^{B \times 3H}$$

**where:**
- $||$ denotes concatenation along the feature dimension
- $f$ is normalized via LayerNorm: $\hat{f} = \text{LayerNorm}(f)$

#### **MLP Classification Head**

$$h_1 = \text{GELU}(\text{LayerNorm}(W_1 \hat{f} + b_1))$$

$$h_2 = \text{GELU}(\text{LayerNorm}(W_2 h_1 + b_2))$$

$$logit = W_3 h_2 + b_3$$

**where:**
- $W_1 \in \mathbb{R}^{512 \times 3H}$, $W_2 \in \mathbb{R}^{256 \times 512}$, $W_3 \in \mathbb{R}^{1 \times 256}$ = learnable weights
- $\text{GELU}(x) = x \Phi(x)$ = smooth activation (better than ReLU for small values)
- $\text{LayerNorm}$ = normalizes each sample independently
- $logit \in \mathbb{R}^{B \times 1}$ = raw output (un-sigmoided)

#### **Final Decision Probability**

$$p_{CHD} = \sigma(logit) = \frac{1}{1 + e^{-logit}}$$

$$\text{Decision} = \begin{cases} \text{REFER} & \text{if } p_{CHD} \geq 0.5 \\ \text{PASS} & \text{if } p_{CHD} < 0.5 \end{cases}$$

---

### Gate Weight Interpretation & Clinical Significance

The gate weights $z_m$ are **inherently interpretable** and reveal modality reliability per patient:

$$\bar{z}_m = \frac{1}{B \cdot H} \sum_{b=1}^{B} \sum_{i=1}^{H} z_{m,b,i}$$

**Average gate weight per modality** (ranges from 0 to 1)

- $\bar{z}_m \approx 1.0$ → modality is fully trusted (high-quality data)
- $\bar{z}_m \approx 0.5$ → modality contributes moderately (some uncertainty)
- $\bar{z}_m \approx 0.0$ → modality is downweighted (noisy or missing data)

**Clinical Example:**
- High-quality ultrasound with clear septal structures: $\bar{z}_{us} \approx 0.95$
- Poor-quality audio with background noise: $\bar{z}_{audio} \approx 0.3$
- Completely missing X-ray: $\bar{z}_{xray} \approx 0.0$

**Advantage for Clinical Use:**
- Model confidence in each modality is transparent
- Doctors can see "model used ultrasound heavily (95%) but dismissed audio (30%)"
- Builds trust through explainability

---

### Modality Dropout: Training for Missing Data

During training, each modality is randomly zeroed with probability $p_{drop}$:

$$\tilde{e}_m = \begin{cases} \mathbf{0} & \text{with probability } p_{drop} \\ e_m & \text{with probability } 1 - p_{drop} \end{cases}$$

$$p_{drop} = 0.2 \quad \text{(default)}$$

**Effect:** The GMU learns to make robust decisions even when any single modality is missing.

**Real-World Importance:**
- In clinical practice, sometimes only ultrasound available (no audio recording)
- Or only audio available (patient anxious, can't get good ultrasound)
- Model trained with modality dropout generalizes to these incomplete scenarios
- Dropout acts as a regularizer: prevents over-reliance on any single modality

---

### GMU Parameter Count Analysis

**Gate Parameters:**
$$3 \times (D_m \times H + H) = 3 \times (512 \times 512 + 512) \approx 786K$$

**MLP Parameters:**
- Layer 1: $(3H) \times 512 = 1536 \times 512 \approx 786K$
- Layer 2: $512 \times 256 \approx 131K$
- Layer 3: $256 \times 1 \approx 256$
- Biases: $\approx 1K$
- **Total MLP:** $\approx 918K$

**Total GMU:** $\approx 1.7M$ parameters

**For Context:**
- Specialist encoders: **FROZEN** (~75.8M parameters total)
- GMU only: **TRAINABLE** (~1.7M parameters = 2.2% of full ensemble)

**Training Advantage:**
- Fast convergence (few parameters to update)
- Stable training (don't overfit specialist weights to limited multimodal data)
- Leverages Phase-1 single-modality pre-training effectively

---

### Loss Function for GMU Training

$$\mathcal{L}_{total} = \mathcal{L}_{BCE} + \lambda_1 \mathcal{L}_{gate} + \lambda_2 \mathcal{L}_{modality\_diversity}$$

where:

**1. Primary Loss (BCE):**
$$\mathcal{L}_{BCE} = -\frac{1}{B} \sum_{b=1}^{B} \left[ y_b \log(p_{CHD,b}) + (1-y_b) \log(1 - p_{CHD,b}) \right]$$

**2. Gate Regularization** (optional, prevents gates from being all-zero):
$$\mathcal{L}_{gate} = \frac{1}{B \cdot 3} \sum_m \sum_b KL(\text{mean}(z_m) || 0.5)$$

Encourages gates to be around 0.5 (use all modalities) rather than collapsing to 0 or 1.

**3. Modality Diversity** (optional, prevents any one modality from dominating):
$$\mathcal{L}_{modality\_diversity} = 1.0 - \text{Entropy}(\bar{z}_{audio}, \bar{z}_{us}, \bar{z}_{xray})$$

Encourages diverse gate activations (don't ignore any modality entirely).

---

## Part 5: Model Deployment & Inference

### 5.1 Checkpoint Management & Selection

Models are saved at multiple points during training:
- **Best checkpoint**: Selected based on highest validation accuracy
- **Epoch checkpoints**: Saved after each epoch for research/analysis

**Why Save Multiple Checkpoints?**
- Monitor overfitting: Compare epoch performance over time
- Ensemble opportunities: Combine multiple checkpoints
- Safety: If best checkpoint corrupted, have backups

---

### 5.2 Inference Pipeline

**Audio Inference:**
1. Load raw WAV file
2. Resample to 2 kHz
3. Apply Butterworth filter
4. Segment via energy-based windowing
5. Compute Log-Mel spectrogram
6. Pass through CRNN2D model
7. Output: probability + embedding

**Ultrasound Inference:**
1. Load image file
2. Apply CLAHE contrast enhancement
3. Resize + pad to 224×224
4. Normalize with ImageNet statistics
5. Pass through NTS-Net model
6. Output: probability + embedding + attention maps (Navigator)

**Phase-2 Fusion Inference (GMU):**
1. Compute embeddings from both specialists
2. Compute gate weights
3. Fuse gated embeddings
4. Pass through MLP head
5. Output: final CHD probability + per-modality confidence scores

---

## Part 6: Objective Completion Summary

### ✅ **Preprocessing Objective: COMPLETED**

**Audio Preprocessing Theory:**
- ✓ Nyquist sampling: 2 kHz resampling preserves cardiac frequencies
- ✓ Butterworth filtering: 20-400 Hz preserves cardiac, removes noise
- ✓ Energy-based segmentation: Selects 3 seconds of cardiac cycles
- ✓ Log-Mel spectrograms: Perceptually-motivated representation

**Image Preprocessing Theory:**
- ✓ CLAHE: Local contrast enhancement without artifacts
- ✓ Aspect-ratio preservation: Prevents anatomical distortion
- ✓ ImageNet normalization: Enables transfer learning

### ✅ **Baseline Model Architecture Theory: COMPLETED**

**CRNN2D Architecture:**
- ✓ ResNet-18 backbone for spatial feature extraction
- ✓ Frequency pooling for dimensionality reduction
- ✓ BiLSTM for temporal sequence modeling
- ✓ Attention pooling for interpretable aggregation
- ✓ Embedding projection for multimodal compatibility

**NTS-Net Architecture:**
- ✓ Navigator: Automatic region proposal generation
- ✓ Teacher: Global context maintenance
- ✓ Scrutinizer: Fine-grained local analysis
- ✓ Multi-scale design for robust defect detection
- ✓ No bounding box supervision needed

### ✅ **Training Theory: COMPLETED**

- ✓ BCEWithLogits loss: Numerically stable binary classification
- ✓ AdamW optimizer: Adaptive learning with decoupled weight decay
- ✓ StepLR scheduler: Decay learning rate for fine-tuning
- ✓ Gradient clipping: Prevents exploding gradients
- ✓ Validation strategy: Unbiased performance estimation

### ✅ **Phase-2 Architecture (GMU): THEORETICALLY FORMULATED**

- ✓ Sigmoid gating mechanism: Per-modality confidence weighting
- ✓ Mathematical formulation: Complete equations with interpretations
- ✓ Modality dropout: Training for incomplete data scenarios
- ✓ Parameter efficiency: Only 1.7M trainable parameters vs 75.8M frozen specialists

---

## Clinical Context & Significance

### Pediatric Cardiac Screening Challenge

**Current State:** Many congenital heart diseases (CHD) go undetected during early childhood, leading to:
- Delayed diagnosis (years later, often with complications)
- Preventable mortality
- Reduced quality of life

**Screening Approach:**
1. **Point-of-care screening** using AI models (audio + ultrasound)
2. **Rapid triage** to identify suspected cases
3. **Echocardiography confirmation** for positive cases

### How Our Models Address This

**Audio (CRNN2D) - 80% accuracy:**
- Detects pathological murmurs
- Portable (smartphone microphone possible future application)
- Non-invasive, quick

**Ultrasound (NTS-Net) - 98% accuracy:**
- Visualizes structural defects directly
- Highly diagnostic
- Limited by operator training requirements

**Phase-2 Fusion (GMU):**
- Combines both modalities' strengths
- Robust to missing data
- Transparent modality confidence for clinical review

---

**Document Type:** Theoretical Reference Guide (No Code)  
**Use Case:** Academic presentation, clinical context understanding, model architecture justification  
**Status:** ✅ Complete
