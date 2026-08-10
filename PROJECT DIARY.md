# 📓 PROJECT DIARY - PEDIATRIC CARDIAC SCREENING SYSTEM

## Academic Year: 2024-2025
## Department: Computer Science and Engineering
## From: [Start Date] To: [End Date]

---

| SI. No. | Work Done | Reference / Remarks | Sign with Date |
|---------|-----------|-------------------|----------------|
| **1** | **Research on Various Medical Related Problems** - Researched critical healthcare challenges affecting pediatric populations including congenital disorders and cardiovascular anomalies. Analyzed prevalence rates, mortality statistics, and diagnostic limitations in resource-limited settings. | WHO reports; Medical journals | |
| **2** | **Finalized the Idea as CHD (Congenital Heart Disease) Based Screening & Classification** - Selected CHD as project focus (8-9 per 1,000 live births). Decided to build AI system for non-invasive, rapid detection using multiple diagnostic modalities. Established scope: detect VSD, ASD, PDA, cardiomegaly. | CHD prevalence: 8-9/1000; Project scope defined | |
| **3** | **Research on Standardized Datasets for CHD Detection with Audio, Image & Clinical Data** - Searched for public datasets containing heart sounds, echocardiography images, and chest X-rays. Identified: ZCHSound (1,259 recordings), CARDIUM (19,674 frames), CHD-CXR (828 images). | Dataset characteristics documented | |
| **4** | **Finalized New Datasets from the Research Gap Identified** - Confirmed ZCHSound (1,154 valid clips) and CARDIUM (19,674 frames) for Phase-1. Defined train-validation-test splits: 70-15-15 for audio/X-ray; 88.9-5.5-5.5 for ultrasound. | Class distribution analyzed; Splits confirmed | |
| **5** | **Identified 12+ Relevant Papers on Heart Sound (PCG) Analysis Using Deep Learning** - Reviewed papers on phonocardiogram analysis. Found: 2D spectrograms superior to 1D (AUC 0.91 vs 0.84), CRNN architectures effective, transfer learning works for pediatric data. | Partovi, Shuvo, Wazed, Li et al. cited | |
| **6** | **Identified 9+ Relevant Papers on Echocardiography (Ultrasound) Analysis Using Fine-Grained Recognition** - Researched ultrasound analysis. Key finding: NTS-Net achieves 92.6% accuracy without bounding-box annotations. Multi-view analysis improves accuracy to AUC 0.93. | Yang, Zhang, Wang, Jiang et al. cited | |
| **7** | **Identified 12+ Papers on Chest X-Ray Based Cardiac Assessment & Automated CTR Measurement** - Reviewed chest X-ray literature. EfficientNetV2 offers best accuracy-efficiency trade-off. Automated CTR achieves ICC=0.982. Domain adaptation needed for pediatric X-rays. | Tan & Le, Li, Saiviroonporn et al. cited | |
| **8** | **Reviewed Identified Papers on PCG Analysis & Derived Pertinent Information** - Extracted insights: 2D CNNs outperform 1D, attention mechanisms improve performance (97% accuracy). Best practices: 2 kHz resampling, 20-400 Hz bandpass, HSMM segmentation. | CRNN2D architecture selected | |
| **9** | **Reviewed Identified Papers on Echocardiography & Derived Pertinent Information** - Found: NTS-Net's Navigator-Teacher-Scrutinizer mechanism automatically locates defects. Multi-view analysis effective. CLAHE preprocessing critical for varying image quality. | NTS-Net confirmed as best choice | |
| **10** | **Reviewed Identified Papers on X-Ray Analysis & Derived Pertinent Information** - Documented: EfficientNetV2 most efficient, FCN-based CTR segmentation reaches ICC=0.982, semi-supervised learning effective with limited data. Pediatric domain adaptation necessary. | EfficientNetV2-S selected for Phase-2 | |
| **11** | **Identified the Research Gap and Concluded to Use Two Different Datasets** - Identified gaps: No single dataset spans 3 modalities, limited pediatric datasets, missing modality handling unexplored. Decided Phase-1: Audio + US only; Phase-2: Add X-ray + GMU fusion. | Research gaps documented; Dual-dataset justified | |
| **12** | **Finalized Two Datasets for Phase-1 Implementation: ZCHSound & CARDIUM** - Locked datasets: ZCHSound (Normal 60.1%, CHD 39.9%) and CARDIUM (Non-CHD 83.7%, CHD 16.3%). Created validation scripts and preprocessing pipelines for both modalities. | Datasets validated; Preprocessing ready | |
| **13** | **Ideated a Hierarchical Model that Detects and Classifies CHD on Multi-Level** - Conceptualized two-level architecture: Level-0 (specialist encoders: CRNN2D audio, NTS-Net US, EfficientNetV2 X-ray) → Level-1 (GMU meta-learner for intelligent fusion). Expected accuracy: 94-96%. | Architecture designed; Block diagrams created | |
| **14** | **Drafted the Synopsis Based on the Proposed Idea & Architecture** - Documented problem (8-9/1000 births), solution (multimodal AI), technical approach (Level-0 + GMU), expected outcomes (94-96% accuracy, <500ms inference), and societal impact. | Synopsis reviewed and finalized | |
| **15** | **Converted Audio Data to Standardized Format and Preprocessing Pipeline** - Converted raw WAV files to 2 kHz mono. Applied Butterworth bandpass (20-400 Hz). Implemented HSMM segmentation for 3-second windows. Computed Log-Mel spectrograms (64×256). | Preprocessing validated on samples | |
| **16** | **Isolated and Selected Common Channels and Applied Bandpass and Notch Filtering** - Applied bandpass filtering (20-400 Hz). Added notch filters at 50, 100, 150 Hz for electrical interference. SNR improved: 8dB → 18dB. HSMM segmentation accuracy: 87% → 94%. | Filtering effectiveness validated | |
| **17** | **Resampled to a Common Frequency and Applied Artifact Rejection Technique** - Resampled variable rates (8-44 kHz) to 2 kHz using polyphase filtering. Applied 3-step artifact rejection (statistical, spectral, manual). Removed 5-7% noisy recordings (1,259 → 1,154 clips). | Dataset: 1,259 → 1,154 valid clips | |
| **18** | **Applied Normalization and Segmented Data into Smaller Windows & Assigned Labels** - Z-score normalized at 3 levels (per-recording, per-window, per-feature). Segmented into 3-second sliding windows (50% overlap): 1,154 clips → 4,231 samples. Assigned labels (Normal, ASD, VSD, PDA). | Windowing: 3.67x augmentation achieved | |
| **19** | **Built and Trained 1D CNN and 2D CNN Model with One Epoch for Testing** - Tested 1D CNN (72% acc, AUC 0.78) vs 2D CNN (81% acc, AUC 0.84). Confirmed 2D spectrogram superior to raw waveforms. Both <100ms inference time. Selected 2D CNN approach. | 2D CNN confirmed as baseline | |
| **20** | **Modified Parameters and Epochs for Better Results** - Trained CRNN2D: 80.1% accuracy, AUC 0.88. Trained NTS-Net on ultrasound: 97.6% accuracy, AUC 0.96. Both checkpoints saved. Phase-1 complete; ready for Phase-2 GMU fusion. | Phase-1 complete; Phase-2 ready | |

---

## Summary

**Phase-1 Status: ✅ COMPLETE**
- ✅ 20 entries (research → training)
- ✅ CRNN2D: 80.1% | NTS-Net: 97.6%
- ✅ Ready for GMU fusion + X-ray