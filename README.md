# Utility-Based Resource Allocation for Multi-Stream Video Analytics

This repository contains scripts and summarized experimental results for the comparative evaluation of utility-based resource allocation strategies in multi-stream video analytics systems.

The study compares three representative allocation approaches:
- Alpha Fairness–based allocation
- Markov Chain–based allocation
- Lyapunov drift-plus-penalty optimization

The experiments are conducted using YOLO-based object detection models under multiple bitrate configurations, focusing on the trade-offs among bitrate, detection accuracy, and normalized utility.

---

## Datasets

The video datasets used in this study consist of real-world surveillance and traffic monitoring videos commonly adopted in video analytics research.  
To ensure transparency and reproducibility, the original video sources are publicly available on YouTube.

**Example video sources include:**
- Road Traffic   
  https://youtu.be/289rvo6RQDc?si=XBhbLbxgk_qNZg3B  
- Pedestrian Road
  https://youtu.be/TNVlw4r8sIo?si=GPwkZMMfHEbznAzY
- Indoor CCTV
  https://youtu.be/xtffU4dk4YI?si=fOBTofx-o0Anu-G0   

All videos are downloaded and re-encoded under multiple bitrate, resolution, and frame-rate configurations prior to inference.  
Details on preprocessing and encoding settings are described in the paper.

> **Note:** The raw video files are not hosted in this repository due to size constraints. Researchers are encouraged to retrieve the videos directly from the provided sources and follow the preprocessing steps described in the paper.

---

## Experimental Scripts

This repository provides Python scripts used to profile video analytics performance and evaluate different utility-based resource allocation strategies.

### 1. Video Profiling and Accuracy–Bitrate Characterization

The following script profiles video streams using YOLO-based object detection models under multiple encoding configurations:

```bash
python profile_videos.py \
  --map streams_semicolon.csv \
  --out profiling.csv \
  --models yolov8s yolov8m yolov8l \
  --qps 22 20 18 16 14 12 10 8 6 \
  --fps 20 25 \
  --res 1280x720 \
  --sample_stride 2 \
  --teacher yolov8x \
  --teacher_on test \
  --conf 0.25 \
  --iou 0.50
```

### 2. Alpha Fairness–Based Allocation
Alpha Fairness–based resource allocation is evaluated using the following script, which enforces bandwidth constraints and fairness objectives:

```bash
python allocator_alpha_constraints.py \
  --csv profiling_yolov8.csv \
  --B_mode abs \
  --B 24000 \
  --u_mode gradfair \
  --alpha 1 \
  --ai_map "1:5,2:5,3:5" \
  --out results_gradfair_alpha1.csv \
  --out_delim ';'
```

Experiments with different fairness parameters and stream importance mappings can be conducted as follows:

```bash
python allocator_alpha_constraints.py \
  --csv profiling_yolov8.csv \
  --B_mode abs \
  --B 24000 \
  --u_mode gradfair \
  --alpha 2 \
  --ai_map "1:5,2:6,3:12" \
  --out results_gradfair_alpha2.csv \
  --out_delim ';'
```

### 3. Markov Chain–Based Allocation
Markov Chain–based allocation models bandwidth dynamics using discrete states and transition probabilities:

```bash
python allocator_markov_chain.py \
  --csv profiling_yolov8.csv \
  --states Low,Med,High \
  --B_states 8000,16000,24000 \
  --P "0.7 0.3 0.0; 0.2 0.6 0.2; 0.0 0.3 0.7" \
  --step 10 \
  --G 1e9 \
  --omega "1:1,2:1,3:1" \
  --out markov_expected_yolov8.csv
```

### 4. Lyapunov Optimization–Based Allocation
Lyapunov drift-plus-penalty optimization is evaluated using the following script:

```bash
python allocator_lyapunov.py \
  --csv profiling_yolov8.csv \
  --states Low,Med,High \
  --B_states 8000,16000,24000 \
  --P "0.7 0.3 0.0; 0.2 0.6 0.2; 0.0 0.3 0.7" \
  --T 500 \
  --V 200 \
  --step 10 \
  --G 1e9 \
  --omega "1:1,2:1,3:1" \
  --out lyap_expected_yolov8.csv \
  --out_trace lyap_trace_yolov8.csv
```
