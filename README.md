# PDF Report Grading Script

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your configurations in the script:
   - Open `grading_script.py`
   - Replace `API_KEY = "YOUR_API_KEY_HERE"` with your actual API key
   - Replace `FOLDER_PATH = "."` with your PDF folder path

3. Run the script:
```bash
python grading_script.py
```

## Output Format
- Console displays detailed scores for each student
- Generates Excel file with all grading results
- Scores based on Milestone 2 rubric's 8 detailed criteria
- Shows brief deduction reasons (if any)
- Excel columns: Report Name, Total Score, and for each rubric item: Score and Reason