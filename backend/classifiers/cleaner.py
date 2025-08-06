import csv
import re

def clean_question(question):
    """
    Clean the question by:
    1. Removing spaces
    2. Converting to lowercase
    3. Removing punctuation and special characters
    """
    # Remove quotes if they exist at the beginning and end
    question = question.strip().strip('"')
    
    # Remove spaces and convert to lowercase
    cleaned = question.replace(" ", "").lower()
    
    # Remove punctuation and special characters, keep only letters and numbers
    cleaned = re.sub(r'[^\w]', '', cleaned)
    
    return cleaned

def detect_duplicates(file_path):
    """
    Detect duplicates in the dataset by comparing each question with all subsequent questions
    """
    questions = []
    
    # Read the dataset
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Handle CSV format
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if row:  # Skip empty rows
                    # Extract just the question part (first column)
                    question = row[0]
                    questions.append(question)
    except FileNotFoundError:
        print(f"File {file_path} not found. Using sample data for demonstration.")
        # Use your sample data
        sample_data = [
            "Qui est actuellement le doyen de la faculté des Sciences?",
            "Pouvez-vous me donner le nom complet du doyen de la FSO?",
            "Qui dirige actuellement la faculté des Sciences de l'UMP?",
            "Quel professeur occupe le poste de doyen à la FSO cette année?",
            "Je cherche les coordonnées du doyen de la faculté des Sciences.",
            "Qui est le responsable administratif en chef de la FSO?",
            "Quel est le nom du doyen actuel de la faculté des Sciences?",
            "Comment s'appelle le doyen en fonction cette année académique?",
            "Qui représente officiellement la faculté des Sciences comme doyen?",
            "Quelle est la personne à contacter pour le mot du doyen?",
            "Who is currently the dean of the Faculty of Science?",
            "Can you provide the full name of the FSO dean?",
            "Who is currently leading the Faculty of Science at UMP?",
            "Which professor holds the dean position at FSO this year?",
            "I'm looking for the contact details of the Faculty of Science dean.",
            "Who is the chief administrative officer of FSO?",
            "What is the name of the current Faculty of Science dean?",
            "What is the name of the dean in office this academic year?",
            "Who officially represents the Faculty of Science as dean?",
            "Who is the person to contact for the dean's message?",
            "من هو عميد كلية العلوم الحالي؟",
            "هل يمكنك تقديم الاسم الكامل لعميد كلية العلوم؟",
            "من هو المسؤول عن كلية العلوم في جامعة محمد الأول حالياً؟",
            "أي أستاذ يشغل منصب العميد هذا العام؟",
            "أبحث عن تفاصيل الاتصال بعميد الكلية.",
            "من هو المسؤول الإداري الأول في كلية العلوم؟",
            "ما اسم عميد كلية العلوم الحالي؟",
            "كيف يسمى العميد في هذه السنة الدراسية؟",
            "من يمثل كلية العلوم رسمياً كعميد؟",
            "من هو عميد كلية العلوم الحالي؟",
            "من هو الشخص الذي يجب الاتصال به لكلمة العميد؟",
            "Anwa i d amenzu n temdint n Tussna akka tura?",
            "Tzemreḍ ad d-tesεiḍ isem ummid n FSO?",
            "Anwa i d anebdaḍ n temdint n Tussna deg UMP akka tura?",
            "Anwa aselmad i yeṭṭef uḥuddu n umenzu deg FSO aseggas-a?"
        ]
        questions = sample_data
    
    # Clean all questions
    cleaned_questions = [clean_question(q) for q in questions]
    
    # Find duplicates
    duplicates_found = []
    duplicate_counter = 0
    
    print("Comparing questions for duplicates...")
    print("=" * 50)
    
    for i in range(len(cleaned_questions)):
        for j in range(i + 1, len(cleaned_questions)):
            if cleaned_questions[i] == cleaned_questions[j]:
                duplicate_counter += 1
                duplicates_found.append((i, j, questions[i], questions[j]))
                print(f"DUPLICATE FOUND #{duplicate_counter}:")
                print(f"  Question {i+1}: {questions[i]}")
                print(f"  Question {j+1}: {questions[j]}")
                print(f"  Cleaned form: {cleaned_questions[i]}")
                print("-" * 30)
    
    print("=" * 50)
    print(f"TOTAL DUPLICATES FOUND: {duplicate_counter}")
    print(f"TOTAL QUESTIONS ANALYZED: {len(questions)}")
    print(f"UNIQUE QUESTIONS: {len(questions) - duplicate_counter}")
    
    return duplicate_counter, duplicates_found

# Example usage
if __name__ == "__main__":
    # Replace 'your_dataset.csv' with your actual file path
    #file_path = 'your_dataset.csv'
    
    print("Dataset Duplicate Detection Tool")
    print("=" * 50)
    
    duplicate_count, duplicates = detect_duplicates("cleaned_dataset.csv")
    
    if duplicate_count == 0:
        #print("No duplicates found in the dataset!")
        pass
    else:
        print(f"\nSummary: Found {duplicate_count} duplicate pairs in your dataset.")
        
        #ptional: Show detailed breakdown
        #print("\nDetailed breakdown:")
        # for i, (idx1, idx2, q1, q2) in enumerate(duplicates, 1):
            # print(f"{i}. Questions {idx1+1} and {idx2+1} are duplicates")