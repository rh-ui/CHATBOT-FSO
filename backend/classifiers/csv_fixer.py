import pandas as pd
import csv
import re

def analyze_csv_file(file_path):

    
    issues = []
    line_count = 0
    problematic_lines = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line_count += 1
                line = line.strip()
                
                if not line:
                    continue
                
                comma_count = line.count(',')
                
                quote_count = line.count('"')
                
                if comma_count > 1:
                    if quote_count == 0 or quote_count % 2 != 0:
                        problematic_lines.append((line_num, line[:100] + '...' if len(line) > 100 else line))
                
                if line_num <= 5:
                    print(f"Line {line_num}: {line[:100]}{'...' if len(line) > 100 else ''}")
    
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return
    
    
    if problematic_lines:
        for line_num, line_content in problematic_lines[:10]:
            print(f"Line {line_num}: {line_content}")
    
    return problematic_lines

def fix_csv_file(input_file, output_file):
    
    fixed_lines = []
    errors = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                
                if not line:
                    continue
                
                try:
                    # Find the last comma (assuming intent is after last comma)
                    if ',' in line:
                        last_comma_idx = line.rfind(',')
                        question = line[:last_comma_idx].strip()
                        intent = line[last_comma_idx + 1:].strip()
                        
                        # Clean up quotes
                        question = question.strip('"\'')
                        intent = intent.strip('"\'')
                        
                        # Only add if both parts exist
                        if question and intent:
                            # Escape any remaining quotes in the question
                            question = question.replace('"', '""')
                            
                            # Write in proper CSV format with quotes around question
                            fixed_line = f'"{question}",{intent}'
                            fixed_lines.append(fixed_line)
                        else:
                            errors += 1
                    else:
                        errors += 1
                        
                except Exception as e:
                    print(f"Error processing line {line_num}: {str(e)}")
                    errors += 1
        
        # Write fixed file
        with open(output_file, 'w', encoding='utf-8', newline='') as file:
            for line in fixed_lines:
                file.write(line + '\n')
        
        try:
            test_df = pd.read_csv(output_file, encoding='utf-8')
            print(f"Fixed file loads successfully {len(test_df)} rows and {len(test_df.columns)} columns")
            
            if len(test_df.columns) == 2:
                test_df.columns = ['question', 'intent']
            
        except Exception as e:
            print(f"Fixed file still has issues: {str(e)}")
        
        return output_file
        
    except Exception as e:
        print(f"Error fixing file: {str(e)}")
        return None

def validate_csv_format(file_path):
    
    try:
        # Try to read with pandas
        df = pd.read_csv(file_path, encoding='utf-8')
        
        if len(df.columns) != 2:
            return False
        
        print(f"CSV format is valid")
        
        # Check for missing values
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            print(f"Found {missing_values} missing values")
        
        return True
        
    except Exception as e:
        print(f"CSV validation failed: {str(e)}")
        return False

def main():

    input_file = input("Enter your CSV file path: ").strip().strip('"\'')
    
    if not input_file:
        input_file = "questions_intents.csv"  # default
    
    print(f"Working with file: {input_file}")
    
    # Step 1: Analyze the file
    problematic_lines = analyze_csv_file(input_file)
    
    # Step 2: Check if we need to fix it
    if not validate_csv_format(input_file):
        print(f"File needs fixing...")
        
        output_file = input_file.replace('.csv', '_fixed.csv')
        fixed_file = fix_csv_file(input_file, output_file)
        
        if fixed_file:
            replace = input(f"\nReplace original file? (y/n): ").strip().lower()
            if replace == 'y':
                import shutil
                shutil.copy2(fixed_file, input_file)
                print(f"Original file replaced with fixed version")
    else:
        print(f"File is already in correct format!")

if __name__ == "__main__":
    main()