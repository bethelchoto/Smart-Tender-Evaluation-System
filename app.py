import streamlit as st
import os
import PyPDF2 as pdf
import importlib.util
import json

# Import Model
spec = importlib.util.spec_from_file_location("train", "model/gpt.py")
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)

# Extract PDF text function
def input_pdf_text(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
    return text

# Extract match percentage from response
def extract_match_percentage(response):
    try:
        data = json.loads(response)
        if "Criteria Match" in data:
            match_percentage = data["Criteria Match"]
            if isinstance(match_percentage, (int, float)):
                return match_percentage
            elif isinstance(match_percentage, str) and match_percentage.endswith('%'):
                return float(match_percentage.strip('%'))
    except ValueError:
        pass
    return None

# Extract missing keywords from response
def extract_missing_keywords(response):
    try:
        data = json.loads(response)
        if "MissingKeywords" in data and isinstance(data["MissingKeywords"], list):
            return data["MissingKeywords"]
    except ValueError:
        pass
    return []

# Generate evaluation summary from response
def generate_evaluation_summary(response):
    try:
        data = json.loads(response)
        if "Evaluation Summary" in data:
            return data["Evaluation Summary"]
    except ValueError:
        pass
    return "Evaluation summary not available"

# Generate company name from response
def generate_company_name(response):
    try:
        data = json.loads(response)
        if "Company Name" in data:
            return data["Company Name"]
    except ValueError:
        pass
    return "Unknown Company"

# Streamlit app
with st.sidebar:
    st.title("Smart Tender Evaluation System")
    st.subheader("About")
    st.write("""
        This sophisticated tender evaluation system, developed using an LLM and Streamlit,
        seamlessly incorporates advanced features including match percentage, keyword analysis
        to identify missing criteria, and the generation of comprehensive evaluation summaries,
        enhancing the efficiency and precision of the tender evaluation process for procurement professionals.
    """)
    
    st.write("Made with ❤ by Mercy Manyani")

st.title("Smart Tender Evaluation System")
st.text("Improve Your Tender Document Evaluation")

# Input fields
general_tender_description = st.text_area("General Tender Description")
project_timeline = st.number_input("Desired Project Timeline (years)")
bid_amount = st.number_input("Desired Bid Amount or Total Project Price (USD)")
required_experience = st.number_input("Total Required Experience (years)")
additional_criteria = st.text_area("Additional Criteria or Specific Aspects")

# File uploader for multiple PDFs
uploaded_files = st.file_uploader("Upload Your Tender Documents", type="pdf", accept_multiple_files=True)

submit = st.button("Submit")

if submit and uploaded_files:
    # Initialize a list to store evaluation results
    evaluation_results = []

    # Process each uploaded tender document
    for idx, file in enumerate(uploaded_files):
        text = input_pdf_text(file)
        input_prompt = f"""
                        Hey, act like a skilled tender evaluation system with a deep understanding of procurement processes, contract management, and evaluation criteria. Your task is to evaluate the tender document based on the given evaluation criteria.
                        Assign the percentage Matching based on the criteria and provide feedback. The feedback shouldn't mention areas of improvement i.e. To improve the evaluation, these criteria should be included and evaluated thoroughly. It should just be feedback on why it got the score (match percentage) and why it was approved and not approved based on Criteria for evaluation. Be comprehensive and exhaustive. Your score (match percentage) should be thought critically.  
                        The Project Timeline, Required Experience, and Bid Amount have higher importance in evaluation. The additional criteria are an icing on the cake.
                        Tender Document: {text}
                        General Tender Description: {general_tender_description}
                        Project Timeline: should be less than {project_timeline} years
                        Bid Amount: should be less than ${bid_amount}
                        Required Experience: should be greater than or equal to {required_experience} years
                        Additional Criteria Description: {additional_criteria}

                        I want the response as per below structure:
                        {{"Company Name" : "","Criteria Match": "%", "MissingKeywords": [], "Evaluation Summary": ""}}
                        """

        
        response = model.get_gemini_response(input_prompt)
        
        # Extract match percentage, missing keywords, and evaluation summary from the response
        company_name = generate_company_name(response)
        match_percentage = extract_match_percentage(response)
        missing_keywords = extract_missing_keywords(response)
        evaluation_summary = generate_evaluation_summary(response)

        # Check if match_percentage is None to avoid TypeError
        if match_percentage is not None:
            approved_status = "Approved" if match_percentage >= 70 else "Not Approved"
        else:
            approved_status = "Not Approved"  # Default to "Not Approved" if match_percentage is None

        # Collect evaluation results
        evaluation_results.append({
            "Company": company_name,
            "Score" : match_percentage,
            "Missing Keywords" : missing_keywords,
            "Approved/Not": approved_status,
            "Evaluation Summary": evaluation_summary
        })

    # Sort evaluation results based on match_percentage (descending order)
    evaluation_results.sort(key=lambda x: x["Score"] if x["Score"] is not None else float('-inf'), reverse=True)

    # Display evaluation results in a tabular format
    st.subheader("Tender Evaluation Results")
    st.table(evaluation_results)
