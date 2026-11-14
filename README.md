<h1 align="center">🤖 Indeed Automation - AI Job Applier</h1>

<p align="center">
  <b>Autonomous, stealth, and AI-powered job application bot built specifically for Indeed (with LinkedIn parity).</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" alt="Python Version">
  <img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Engine-Patchright-orange" alt="Patchright Engine">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen" alt="Status">
</p>

---

## 📖 Overview

**Indeed Automation AI Job Applier** automates the entire job search and application pipeline on Indeed. It searches for relevant vacancies, evaluates job descriptions against eligibility criteria and resume match thresholds, streams tailored resumes on the fly via RenderCV, answers application questions using LLMs, and submits applications with human-like randomized timing to prevent bot detection.

---

## ✨ Key Features

* **⚡ SmartApply & Easily Apply Automation**:
  - Automatically identifies and handles Indeed Easy Apply / SmartApply multi-step modal forms.
  - Interacts with radio buttons, checkboxes, dropdowns, text inputs, and experience forms without getting stuck.
* **📄 Live RenderCV Tailored Resumes & CV Options Upload**:
  - Live console streaming during tailored resume generation with RenderCV.
  - Automatically handles the SmartApply `CV options` &rarr; `Upload a different file` menu and file chooser to ensure the generated PDF is uploaded.
* **🛡️ Stealth Automation with Patchright**:
  - Built on [Patchright](https://github.com/Kaliiiiiiiiii-Vinyzu/patchright) (hardened Playwright build) with persistent browser profiles to bypass Cloudflare and bot-detection challenges.
* **🎲 Human-Like Randomized Click Timing**:
  - Randomized pre-click and post-click hesitation strictly under 5 seconds to simulate real human behavior and prevent account flagging.
* **🪟 Floating Minimizable Bot Controller**:
  - Floating desktop banner with standard Windows minimize (`WS_MINIMIZEBOX`) and drag support.
  - `Ctrl+X` global shortcut to instantly pause and resume execution.
* **⏱️ Self-Activating Venv & Task Scheduler Ready**:
  - [run.bat](run.bat) launcher automatically activates `.venv` and forwards CLI arguments.
  - Built-in Python self-re-execution inside `main.py` detects external Python interpreters and re-executes cleanly inside `.venv` regardless of working directory.
* **🚀 High-Speed Scraping & Zero-Freeze Caching**:
  - Scrapes job descriptions directly from Indeed's 2-pane list without opening unnecessary tabs.
  - Session-level in-memory skip caching avoids slow synchronous YAML dumps and keeps bot execution fast and responsive.

---

## 🛠️ Getting Started

### Prerequisites

- **Python 3.12+**
- Google Chrome or Chromium
- Git

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/UdayRaj2003/Indeed_Automatiom.git
   cd Indeed_Automatiom
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux / macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   patchright install chromium
   ```

---

## 🔧 Configuration

1. **Environment Variables**:
   Copy the example environment file and configure your credentials:
   ```bash
   cp .env.example .env
   ```
   Add your Indeed email and LLM API keys:
   ```env
   indeed_email="your_email@example.com"
   indeed_password="your_password"
   ```

2. **Search Preferences**:
   Copy and customize `config/search_config.yaml`:
   ```bash
   cp config/search_config.example.yaml config/search_config.yaml
   ```
   Configure target positions, remote/hybrid preferences, experience levels, and locations.

---

## ▶️ Usage

### Run Manually

- **Normal run**:
  ```bash
  .\run.bat
  # or:
  python main.py
  ```

- **Test mode (fills forms without final submission)**:
  ```bash
  .\run.bat --test
  # or:
  python main.py --test
  ```

- **Single cycle execution**:
  ```bash
  .\run.bat --once
  ```

- **Headless mode without UI controller**:
  ```bash
  .\run.bat --no-ui
  ```

### Windows Task Scheduler Setup

You can schedule the bot to run automatically on system logon or at specified daily intervals:

```powershell
$action = New-ScheduledTaskAction -Execute "D:\Path\To\Indeed_Bot_Repo\run.bat" -WorkingDirectory "D:\Path\To\Indeed_Bot_Repo"
$trigger = New-ScheduledTaskTrigger -Daily -At "22:30"
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -StartWhenAvailable -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName "Indeed_Easy_Apply_Bot" -Action $action -Trigger $trigger -Settings $settings
```

---

## 🧪 Running Tests

Run the comprehensive unit test suite:
```bash
pytest
```

---

## 📜 License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for complete details.

---

## 🙏 Credits & Acknowledgements

This project builds upon and integrates work from several exceptional open-source projects:

* **[LinkedIn-AI-Job-Applier-Ultimate](https://github.com/beatwad/LinkedIn-AI-Job-Applier-Ultimate)** by **beatwad** — The architectural foundation for autonomous AI-driven job applications.
* **[Auto_job_applier_linkedIn](https://github.com/GodsScion/Auto_job_applier_linkedIn)** by **Sai Vignesh Golla (GodsScion)** — Question handling, configuration structures, and application parity.
* **[Jobs_Applier_AI_Agent_AIHawk](https://github.com/feder-cr/Jobs_Applier_AI_Agent_AIHawk)** by **feder-cr** — Pioneer of open-source automated job application agents.
* **[Patchright](https://github.com/Kaliiiiiiiiii-Vinyzu/patchright)** by **Kaliiiiiiiiii-Vinyzu** — Undetected browser automation framework.
* **[RenderCV](https://github.com/sinaatalay/rendercv)** by **Sina Atalay** — LaTeX and PDF programmatic resume generation.
