# 🧠 MoE Research Hub - Streamlit Dashboard

An interactive web dashboard for the **Geometric Constrained Learning & Advanced MoE Architectures** research framework.

## 🚀 Features

### **Live Training Monitoring**
- **Real-time loss curves** with trend analysis
- **Geometric training metrics**: rotation angles, orthogonality preservation, expert specialization
- **Ghost expert monitoring**: activation patterns, saturation levels
- **Expert activation heatmaps** and connection matrices
- **3D rotation visualization** for geometric training

### **Interactive Configuration**
- **Visual parameter tuning** with sliders and dropdowns
- **Architecture presets**: Geometric Lambda, Ghost Expert Test, Basic Comparison
- **Real-time validation** with parameter estimation
- **Advanced settings** for fine-grained control

### **Background Training**
- **Non-blocking training** with progress monitoring
- **Start/stop controls** with graceful interruption
- **Automatic checkpointing** and model management
- **Error handling** with detailed reporting

### **Comprehensive Analysis**
- **Interactive inference** with parameter controls
- **Model comparison** and checkpoint management
- **Dataset upload** and preprocessing status
- **Performance visualization** with Plotly charts

## 🎯 Dashboard Layout

```
┌─────────────────────────────────────────────────────────────┐
│                     🧠 MoE Research Hub                     │
├─────────────────┬───────────────────────────────────────────┤
│   Configuration │              Live Graph Center            │
│   Panel         │                                           │
│                 │   ┌─────────────────────────────────────┐ │
│ Architecture    │   │     [Selectable Live Plot]         │ │
│ ○ gnn           │   │   • Training Loss                   │ │
│ ○ hgnn          │   │   • Rotation Angles (Geometric)    │ │
│ ● geometric     │   │   • Expert Activations             │ │
│                 │   │   • Ghost Saturation               │ │
│ Model Settings  │   │   • Orthogonality Metrics         │ │
│ [sliders...]    │   │                                     │ │
│                 │   └─────────────────────────────────────┘ │
│ Training        │                                           │
│ [start/stop]    │        Training Controls & Status         │
│                 │   [▶️ Start] [⏸️ Pause] [⏹️ Stop] [💾 Save]  │
│ Dataset         │                                           │
│ [upload/select] │              Tabs Below                   │
│                 │   [Analysis] [Inference] [Models] [Data]  │
│ Geometric       │                                           │
│ [special opts]  │                                           │
└─────────────────┴───────────────────────────────────────────┘
```

## 📦 Installation

### **1. Install Dashboard Dependencies**
```bash
pip install -r streamlit_requirements.txt
```

### **2. Verify Core Dependencies**
Make sure your main project dependencies are installed:
```bash
pip install -r requirements.txt
```

### **3. Launch Dashboard**
```bash
streamlit run streamlit_dashboard.py
```

The dashboard will open in your web browser at `http://localhost:8501`

## 🎮 Quick Start Guide

### **Step 1: Configure Your Experiment**
1. **Select Architecture**: Choose from GNN, HGNN, Orthogonal, Ghost, or Geometric
2. **Set Parameters**: Use sliders for model size, batch size, epochs
3. **Choose Dataset**: Upload local files or select HuggingFace datasets
4. **Validate Config**: Click "✅ Validate Configuration" to check settings

### **Step 2: Start Training**
1. **Click "🚀 Start Training"** in the sidebar
2. **Monitor Progress**: Watch live plots update in real-time
3. **View Metrics**: Check the right panel for current loss and progress
4. **Geometric Features**: If using geometric mode, see rotation visualizations

### **Step 3: Analyze Results**
1. **Analysis Tab**: View comprehensive training plots
2. **Inference Tab**: Test your model with interactive text generation
3. **Models Tab**: Load and compare different checkpoints
4. **Datasets Tab**: Manage your data and preprocessing

## 🔄 Geometric Constrained Learning Dashboard

The dashboard provides specialized visualizations for the revolutionary **Geometric Constrained Learning** system:

### **Rotation Monitoring**
- **Real-time rotation angles** for each expert
- **3D trajectory visualization** showing rotation evolution
- **Rotation efficiency metrics** and magnitude tracking

### **Orthogonality Tracking**
- **Live orthogonality preservation** scores
- **Expert specialization** divergence patterns
- **Geometric loss components** breakdown

### **Lambda Calculus Support**
- **Cognitive rotation scheduling** visualization
- **Lambda-specific rotation patterns**
- **Reasoning task optimization** metrics

## 👻 Ghost Expert Features

When ghost experts are enabled, the dashboard provides:

### **Activation Monitoring**
- **Ghost activation patterns** over time
- **Saturation level tracking** with threshold indicators
- **Primary vs Ghost learning** comparison

### **Adaptive Capacity Visualization**
- **Dynamic expert scaling** representation
- **Activation threshold** tuning interface
- **Ghost learning rate** coupling display

## 📊 Live Plot Types

The center graph can display various real-time visualizations:

### **Core Metrics**
- **Training Loss**: Real-time loss with trend analysis
- **Expert Activations**: Heatmap of expert usage patterns
- **Learning Rate Schedule**: Dynamic learning rate visualization

### **Geometric Training** (when enabled)
- **Rotation Angles Evolution**: Per-expert rotation tracking
- **Orthogonality Preservation**: Geometric constraint monitoring
- **Expert Specialization**: Divergence and specialization metrics
- **Geometric Loss Components**: Multi-component loss breakdown
- **3D Rotation Visualization**: Spatial rotation trajectories

### **Ghost Experts** (when enabled)
- **Ghost Expert Activation**: Dynamic capacity scaling
- **Saturation Monitoring**: Expert saturation levels
- **Ghost vs Primary Learning**: Comparative learning analysis

### **Architecture Features**
- **Expert Connection Heatmap**: Inter-expert communication strength
- **Hypergraph Edge Weights**: HGNN coupling visualization

## 🔧 Configuration Presets

### **Geometric Lambda** 🔥
Optimized for lambda calculus reasoning with geometric constrained learning:
```python
Architecture: geometric
Training Mode: geometric
Experts: 4 primary
Geometric LR: 1e-3
Expert LR: 1e-4
Rotation Dims: 4
Lambda Cognitive: enabled
```

### **Ghost Expert Test** 👻
Adaptive capacity with ghost experts:
```python
Architecture: ghost
Training Mode: standard
Experts: 4 primary + 2 ghost
Ghost Threshold: 0.01
Ghost LR: 1e-4
```

### **Basic Comparison** 📊
Simple baseline for comparisons:
```python
Architecture: hgnn
Training Mode: standard
Experts: 4 primary
Model Size: 64d
```

## 💾 Data Management

### **Dataset Upload**
- **Drag & drop** JSON, JSONL, or TXT files
- **Automatic format detection** and preprocessing
- **Preview functionality** for data validation

### **HuggingFace Integration**
- **Direct dataset loading** from HF Hub
- **Compatibility validation** with automatic checks
- **Config-specific** dataset handling

### **Preprocessing Pipeline**
- **Automatic tokenization** with caching
- **24x speedup** with pretokenized datasets
- **Fingerprint validation** for cache integrity

## 🎯 Best Practices

### **For Research**
1. **Use presets** as starting points for experiments
2. **Save configurations** before major parameter changes
3. **Monitor geometric metrics** for GCL experiments
4. **Compare models** using the Models tab

### **For Development**
1. **Start with small models** for rapid iteration
2. **Use local datasets** for development
3. **Check preprocessing status** before training
4. **Validate configurations** before long runs

### **For Production**
1. **Use larger models** for serious experiments
2. **Enable checkpointing** for long training runs
3. **Monitor system resources** during training
4. **Export configurations** for reproducibility

## 🚨 Troubleshooting

### **Common Issues**

**Dashboard won't start:**
```bash
# Check Streamlit installation
streamlit --version

# Reinstall if needed
pip install streamlit>=1.28.0
```

**Training fails to start:**
- Check configuration validation errors
- Verify dataset file exists and is readable
- Ensure sufficient system memory

**Live plots not updating:**
- Training may have stopped due to error
- Check the error message in training controls
- Try stopping and restarting training

**Model loading fails:**
- Verify checkpoint file exists
- Check config.json is present in checkpoint directory
- Ensure model architecture matches checkpoint

### **Performance Tips**

**For faster training:**
- Use smaller batch sizes on limited hardware
- Enable mixed precision training
- Use pretokenized datasets for faster loading

**For better visualization:**
- Reduce log_every frequency for smoother plots
- Limit max_batches_per_epoch for quick experiments
- Use geometric visualizations sparingly on large models

## 🔗 Integration with CLI

The dashboard complements the existing CLI tools:

- **`app.py`**: Interactive CLI menu system
- **`run.py`**: Command-line training interface
- **`streamlit_dashboard.py`**: Web-based visual interface

All three interfaces share the same `core/` modules and configuration system.

## 📈 Advanced Features

### **Real-time Training**
- **Background threading** for non-blocking training
- **Progress polling** with automatic updates
- **Graceful interruption** with state preservation

### **Interactive Analysis**
- **Plotly integration** for interactive charts
- **Zoom, pan, hover** functionality
- **Export capabilities** for publication

### **Model Management**
- **Checkpoint comparison** across runs
- **Configuration versioning** and export
- **Model parameter** estimation and validation

## 🏆 Success Stories

The dashboard has been successfully used for:

- **Geometric Constrained Learning** research and validation
- **Lambda calculus reasoning** task optimization
- **Ghost expert** adaptive capacity experiments  
- **Multi-architecture** comparison studies
- **Real-time experiment** monitoring and adjustment

---

**Ready to revolutionize your MoE research with visual, interactive training?**

```bash
streamlit run streamlit_dashboard.py
```

🚀 **Welcome to the future of machine learning research dashboards!**
