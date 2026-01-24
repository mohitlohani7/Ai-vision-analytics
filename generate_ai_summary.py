from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Table, Spacer, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from openai import OpenAI

def get_ai_summary(api_key, analytics, duration):
    """
    Sends video data to OpenAI to generate a professional summary.
    """
    if not api_key:
        return "OpenAI API Key not provided. AI summary could not be generated."

    client = OpenAI(api_key=api_key)
    
    # Data ko string format mein convert kar rahe taaki GPT samajh sake
    data_str = f"Duration: {duration:.2f} seconds. Detected Objects (Unique IDs): {dict(analytics['class_count'])}."

    prompt = f"""
    You are a Senior Video Analytics Expert. Analyze the following video detection data:
    {data_str}

    Write a professional, 3-paragraph executive summary for a PDF report. 
    1. First paragraph: Summarize the overall activity and volume of objects.
    2. Second paragraph: Analyze the specific class distribution (what is most/least common).
    3. Third paragraph: Provide a brief insight or security/traffic recommendation based on these numbers.
    
    Keep the tone formal and concise. Do not use markdown symbols like ** or # in the output, just plain text.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Ya gpt-4 use kar agar budget hai
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI Summary Error: {str(e)}"

def generate_pdf(report_path, analytics, duration, graph_path, ai_summary):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(report_path, pagesize=A4)

    story = []

    # Title
    story.append(Paragraph("🎥 AI Video Analytics Report", styles["Title"]))
    story.append(Spacer(1, 12))
    
    # AI Summary Section (The New Magic Part)
    story.append(Paragraph("Executive Summary (AI Generated):", styles["Heading2"]))
    story.append(Spacer(1, 6))
    
    # AI ka text yahan paragraph ban ke aayega
    # Split by newline in case GPT gives multiple paras
    for para in ai_summary.split('\n'):
        if para.strip():
            story.append(Paragraph(para, styles["Normal"]))
            story.append(Spacer(1, 8))
    
    story.append(Spacer(1, 12))

    # Statistics Section
    story.append(Paragraph("Statistical Data:", styles["Heading2"]))
    story.append(Spacer(1, 6))

    table_data = [["Object Class", "Unique Count", "Avg Confidence"]]
    
    for cls, count in analytics["class_count"].items():
        if analytics["confidence_count"][cls] > 0:
            avg_conf = analytics["confidence_sum"][cls] / analytics["confidence_count"][cls]
        else:
            avg_conf = 0.0
            
        table_data.append([cls, str(count), f"{avg_conf:.2f}"])

    table = Table(table_data, colWidths=[2.5*inch, 2*inch, 2*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    
    story.append(table)
    story.append(Spacer(1, 20))

    # Graph Section
    story.append(Paragraph("Visual Distribution:", styles["Heading2"]))
    story.append(Image(graph_path, 6*inch, 4*inch))

    doc.build(story)