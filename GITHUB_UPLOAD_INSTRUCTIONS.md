# GitHub Upload Instructions for Safe Haven Stress Test

Since Git is not available in the command line, here are manual steps to upload your project to GitHub:

## Option 1: GitHub Desktop (Recommended)

1. **Download GitHub Desktop** (if not installed): https://desktop.github.com/
2. **Install and sign in** to your GitHub account
3. **Add the repository**:
   - Click "File" → "Add local repository"
   - Browse to: `c:\Users\silic\.gemini\antigravity\scratch\portfolio-analyzer-py`
   - Click "Add Repository"
4. **Create repository on GitHub**:
   - Click "Publish repository"
   - Name: `safe_haven_stress_test`
   - Description: "Monte Carlo portfolio stress testing with IBKR Portfolio Margin"
   - Uncheck "Keep this code private" (or keep checked for private repo)
   - Click "Publish Repository"

## Option 2: GitHub Web Interface

1. **Create new repository** on GitHub.com:
   - Go to https://github.com/new
   - Repository name: `safe_haven_stress_test`
   - Description: "Monte Carlo portfolio stress testing with IBKR Portfolio Margin"
   - Choose Public or Private
   - **Do NOT** initialize with README, .gitignore, or license
   - Click "Create repository"

2. **Install Git** (if needed):
   - Download from: https://git-scm.com/download/win
   - Install with default settings
   - Restart terminal/PowerShell

3. **Open PowerShell** in project directory:
   ```powershell
   cd c:\Users\silic\.gemini\antigravity\scratch\portfolio-analyzer-py
   ```

4. **Run these commands**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Safe Haven Stress Test"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/safe_haven_stress_test.git
   git push -u origin main
   ```

## Option 3: VS Code

1. **Open folder** in VS Code:
   - `c:\Users\silic\.gemini\antigravity\scratch\portfolio-analyzer-py`
2. **Source Control panel** (Ctrl+Shift+G)
3. **Initialize Repository**
4. **Stage all changes** (+ icon)
5. **Commit** with message: "Initial commit: Safe Haven Stress Test"
6. **Publish to GitHub** (click button in Source Control panel)

## Files Ready for Upload

✓ README.md - Comprehensive documentation
✓ .gitignore - Excludes temporary files
✓ app.py - Streamlit interface
✓ full_simulation.py - Batch processor
✓ src/ - Core simulation engine
✓ requirements.txt - Dependencies

## Next Steps After Upload

1. Add GitHub topics: `monte-carlo`, `portfolio-optimization`, `risk-management`, `streamlit`
2. Consider adding a license file (MIT recommended)
3. Share the repository URL!
