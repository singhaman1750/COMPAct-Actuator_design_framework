# COMPAct Framework

**Paper:** [*COMPAct: Computational Optimization and Automated Modular design of Planetary Actuators*](https://www.arxiv.org/abs/2510.07197)

The **COMPAct Framework** streamlines the design of robotic actuators by automating the calculation of optimal gear parameters and updating 3D CAD models.

## ✅ Supported Hardware

### Supported Motors
| Motor Code | Description |
| :--- | :--- |
| **U8** | T-motor U8 |
| **U10** | T-motor U10+ |
| **U12** | T-motor U12 |
| **MN8014** | T-motor MN8014 |
| **VT8020** | Vector Techniques 8020 |
| **MAD_M6C12**| MAD Components M6C12 |

### Supported Gearbox Topologies
| Type | Description |
| :--- | :--- |
| **sspg** | Single-Stage Planetary Gearbox |
| **cpg** | Compound Planetary Gearbox |
| **wpg** | Wolfrom Planetary Gearbox (3K) |
| **dspg** | Double-Stage Planetary Gearbox |

---

## 💻 Setup & Requirements

### System Requirements
* **Python 3.x**
* **SolidWorks 2024** (or higher) – *Required for CAD automation features.*

### Recommended Terminal
* **Windows:** [Git Bash](https://git-scm.com/downloads) is highly recommended for running the automation scripts.
* **Linux/macOS:** Default terminal.

### 1. Installation
Clone the repository and install the dependencies:

```bash
# Clone the repository
git clone https://github.com/singhaman1750/COMPAct-Actuator_design_framework.git

# Enter the directory
cd COMPAct-Actuator_design_framework 

# Install required packages
pip install numpy matplotlib pandas
```

### 2. Extract CAD Files

Due to file size limits, CAD files are zipped. You must extract them before running the framework.
  1. Navigate to the `CADs` directory.
  2. Inside each gearbox folder (e.g., `CADs/SSPG/`), unzip the archive (e.g., `sspg_actuator.zip`) into the same directory.

Your directory structure should look like this after extraction:

```
COMPAct-Actuator_design_framework/
└── CADs/
    ├── SSPG/sspg_actuator/sspg_actuator/...
    ├── CPG/cpg_actuator/cpg_actuator/...
    ├── DSPG/dspg_actuator/dspg_actuator/...
    └── WPG/wpg_actuator/wpg_actuator/...

```
