#!/usr/bin/env python3
"""
Automated grading script for Milestone 4 Vehicle Routing Problem (VRP) reports
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any
import PyPDF2
import openai
import pandas as pd
from datetime import datetime

class ReportGrader:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.rubric = {
            "Distance Matrix": 5,
            "Scenario 1 Route": 5,
            "Scenario 2 Route": 5,
            "Sequence of Locations": 5,
            "Total Travel Distance": 5,
            "Road Closure Modeling": 5,
            "Reflection on Results": 5,
            "Code for Both Scenarios": 5,
            "Overall Presentation": 5
        }
        self.total_points = 45
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
            return ""
    
    def create_grading_prompt(self, report_text: str) -> str:
        """Create the GPT prompt for grading"""
        prompt = f"""
You are an experienced professor grading Milestone 4 Vehicle Routing Problem (VRP) reports. Please evaluate based on the following rubric:

Grading Rubric (Total 45 points):
1. Distance Matrix (5 points): Completed distance matrix with correct calculations
2a. Scenario 1: Optimized Route (5 points): Route minimizes distance, visits each customer exactly once, returns to warehouse
2b. Scenario 2: Route with Road Closure (5 points): Correctly accounts for road closure, optimized under constraints
3a. Sequence of Locations (5 points): Clear sequences stated for both scenarios
3b. Total Travel Distance (5 points): Correctly calculated distances for both scenarios
4. Modeling of Road Closure (5 points): Clear explanation of how road closure was handled
5. Reflection on Results (5 points): Thoughtful discussion of differences between scenarios and insights
6. Code for Both Scenarios (5 points): Complete, clear code provided for both scenarios
Overall Presentation and Professionalism (5 points): Professional, well-structured presentation

Scoring Guidelines:
- Excellent (5): Meets all requirements excellently
- Good (4): Meets requirements with minor issues
- Satisfactory (3): Meets basic requirements but with noticeable issues
- Needs Improvement (1-2): Significant problems or missing key elements
- Missing (0): Component not provided

Grading Guidelines:
- Apply VERY GENEROUS standards - be lenient and encouraging
- Only give low scores (1-2) for severely deficient work
- Give 4-5 points for any reasonable attempt that meets basic requirements
- Give substantial partial credit for efforts
- Focus on completion and effort rather than perfection
- BE CONSISTENT: Similar quality work should receive similar scores
- Err on the side of giving higher scores when in doubt
- IMPORTANT: Never exceed the maximum points for each criterion

Return the grading results in JSON format:
{{
    "student_name": "report filename without extension",
    "total_score": total_score,
    "detailed_scores": {{
        "Distance Matrix": {{"score": score, "reason": "brief deduction reason (if any)"}},
        "Scenario 1 Route": {{"score": score, "reason": "brief deduction reason (if any)"}},
        "Scenario 2 Route": {{"score": score, "reason": "brief deduction reason (if any)"}},
        "Sequence of Locations": {{"score": score, "reason": "brief deduction reason (if any)"}},\n        "Total Travel Distance": {{"score": score, "reason": "brief deduction reason (if any)"}},
        "Road Closure Modeling": {{"score": score, "reason": "brief deduction reason (if any)"}},
        "Reflection on Results": {{"score": score, "reason": "brief deduction reason (if any)"}},
        "Code for Both Scenarios": {{"score": score, "reason": "brief deduction reason (if any)"}},
        "Overall Presentation": {{"score": score, "reason": "brief deduction reason (if any)"}}
    }}
}}

Report Content:
{report_text}
"""
        return prompt
    
    def grade_report(self, report_text: str, filename: str) -> Dict[str, Any]:
        """Grade a single report using GPT-4o-mini"""
        prompt = self.create_grading_prompt(report_text)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an experienced professor grading VRP reports. Always respond in valid JSON format. Be VERY generous in scoring - give high scores unless work is severely deficient."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,  # Make scoring deterministic
                max_tokens=1500,
                seed=42  # Use a fixed seed for consistency
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Clean up the response to ensure valid JSON
            if result_text.startswith("```json"):
                result_text = result_text[7:-3]
            elif result_text.startswith("```"):
                result_text = result_text[3:-3]
            
            result = json.loads(result_text)
            
            # Force use filename instead of content title
            result["student_name"] = Path(filename).stem
            
            # Enforce score limits and calculate total
            detailed_scores = result.get("detailed_scores", {})
            score_limits = {
                "Distance Matrix": 5,
                "Scenario 1 Route": 5,
                "Scenario 2 Route": 5,
                "Sequence of Locations": 5,
                "Total Travel Distance": 5,
                "Road Closure Modeling": 5,
                "Reflection on Results": 5,
                "Code for Both Scenarios": 5,
                "Overall Presentation": 5
            }
            
            # Cap scores at their maximum values
            for category, max_score in score_limits.items():
                if category in detailed_scores:
                    current_score = detailed_scores[category].get("score", 0)
                    if current_score > max_score:
                        detailed_scores[category]["score"] = max_score
                        if not detailed_scores[category].get("reason"):
                            detailed_scores[category]["reason"] = f"Score capped at maximum {max_score}"
            
            result["detailed_scores"] = detailed_scores
            calculated_total = sum(item.get("score", 0) for item in detailed_scores.values())
            result["total_score"] = min(calculated_total, 45)  # Cap total at 45
                
            return result
            
        except Exception as e:
            print(f"Error grading {filename}: {e}")
            return {
                "student_name": Path(filename).stem,
                "total_score": 0,
                "detailed_scores": {},
                "error": str(e)
            }
    
    def grade_all_reports(self, pdf_directory: str = "."):
        """Grade all PDF reports in the directory"""
        pdf_files = list(Path(pdf_directory).glob("*.pdf"))
        
        if not pdf_files:
            print("No PDF files found in the directory.")
            return
        
        print(f"Found {len(pdf_files)} PDF files to grade.")
        results = []
        
        for pdf_file in pdf_files:
            print(f"Grading: {pdf_file.name}")
            
            # Extract text from PDF
            report_text = self.extract_text_from_pdf(str(pdf_file))
            
            if not report_text.strip():
                print(f"Warning: No text extracted from {pdf_file.name}")
                continue
            
            # Grade the report
            result = self.grade_report(report_text, pdf_file.name)
            results.append(result)
        
        # Output results
        self.output_results(results)
    
    def output_results(self, results: List[Dict[str, Any]]):
        """Output grading results in a formatted manner"""
        print("\n" + "="*80)
        print("MILESTONE 4 GRADING RESULTS")
        print("="*80)
        
        # Prepare data for Excel
        excel_data = []
        
        for result in results:
            if "error" in result:
                print(f"\n{result['student_name']}: ERROR - {result['error']}")
                continue
                
            print(f"\n{result['student_name']}: {result['total_score']}/45")
            print("-" * 50)
            
            detailed = result.get('detailed_scores', {})
            categories = [
                ("Distance Matrix", 5),
                ("Scenario 1 Route", 5),
                ("Scenario 2 Route", 5),
                ("Sequence of Locations", 5),
                ("Total Travel Distance", 5),
                ("Road Closure Modeling", 5),
                ("Reflection on Results", 5),
                ("Code for Both Scenarios", 5),
                ("Overall Presentation", 5)
            ]
            
            # Prepare row for Excel
            percentage_score = round((result['total_score'] / 45) * 100)
            row_data = {
                "Report Name": result['student_name'],
                "Total Score": f"{result['total_score']}/45",
                "Percentage (100)": f"{percentage_score}/100"
            }
            
            for category, max_score in categories:
                if category in detailed:
                    score_info = detailed[category]
                    score = score_info.get('score', 0)
                    reason = score_info.get('reason', '')
                    
                    print(f"  {category}: {score}/{max_score}", end="")
                    if reason and score < max_score:
                        print(f" - {reason}")
                    else:
                        print()
                    
                    # Add to Excel data
                    row_data[f"{category} Score"] = f"{score}/{max_score}"
                    row_data[f"{category} Reason"] = reason if reason else ""
                else:
                    row_data[f"{category} Score"] = f"0/{max_score}"
                    row_data[f"{category} Reason"] = "Not evaluated"
            
            excel_data.append(row_data)
        
        # Save to Excel file
        if excel_data:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            excel_file = f"grading_results_{timestamp}.xlsx"
            
            df = pd.DataFrame(excel_data)
            
            # Reorder columns for better readability
            column_order = ["Report Name", "Total Score", "Percentage (100)"]
            categories = [
                "Distance Matrix", "Scenario 1 Route", "Scenario 2 Route",
                "Sequence of Locations", "Total Travel Distance", "Road Closure Modeling",
                "Reflection on Results", "Code for Both Scenarios", "Overall Presentation"
            ]
            
            for category in categories:
                column_order.extend([f"{category} Score", f"{category} Reason"])
            
            df = df[column_order]
            
            # Save to Excel with formatting
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Milestone 4 Results')
                
                # Get the workbook and worksheet
                workbook = writer.book
                worksheet = writer.sheets['Milestone 4 Results']
                
                # Auto-adjust column widths
                for column in df:
                    column_length = max(df[column].astype(str).map(len).max(), len(column))
                    col_idx = df.columns.get_loc(column)
                    worksheet.column_dimensions[chr(65 + col_idx)].width = min(column_length + 2, 50)
            
            print(f"\nMilestone 4 grading results saved to Excel: {excel_file}")
        else:
            print("\nNo valid results to save.")

def main():
    # Set your OpenAI API key here
    API_KEY = "YOUR_API_KEY_HERE"  # Replace with your actual API key
    
    # Set your PDF folder path here
    FOLDER_PATH = "Tuesday"  # Replace with your folder path, e.g., "/path/to/reports" or "C:/Reports"
    
    if API_KEY == "YOUR_API_KEY_HERE":
        print("Please set your OpenAI API key in the script.")
        return
    
    # Check if the folder exists
    if not os.path.exists(FOLDER_PATH):
        print(f"Error: Folder '{FOLDER_PATH}' does not exist.")
        return
    
    if not os.path.isdir(FOLDER_PATH):
        print(f"Error: '{FOLDER_PATH}' is not a directory.")
        return
    
    print(f"Grading Milestone 4 reports in: {os.path.abspath(FOLDER_PATH)}")
    
    grader = ReportGrader(API_KEY)
    grader.grade_all_reports(FOLDER_PATH)

if __name__ == "__main__":
    main()