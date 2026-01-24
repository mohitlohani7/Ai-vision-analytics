import os
from datetime import datetime
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
)
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from groq import Groq
from openai import OpenAI

# --------------------------------------------------
# CONFIGURATION & ASSETS
# --------------------------------------------------
COMPANY_HEADER_BG = r"L:\Object detection\blue.png"
COMPANY_LOGO = r"L:\Object detection\Sriya_new_logo.png"

# Brand Colors
PRIMARY_BLUE = colors.HexColor("#1e3c72")
SECONDARY_BLUE = colors.HexColor("#2a5298")
LIGHT_BLUE = colors.HexColor("#e8f4f8")
ULTRA_LIGHT_BLUE = colors.HexColor("#f5f9fc")
TEXT_COLOR = colors.HexColor("#2c3e50")

# --------------------------------------------------
# AI SUMMARY GENERATION
# --------------------------------------------------
def get_ai_summary(api_key, analytics, duration, provider, model_name):
    """Generates summary using either Groq or OpenAI."""
    if not api_key or not api_key.strip():
        return f"Error: {provider} API Key is missing. Report generated without AI insights."

    data_str = f"Video Duration: {duration:.2f} sec. Objects Detected (Unique IDs): {dict(analytics['class_count'])}."

    prompt = f"""
    You are a Senior Video Analytics Expert. Analyze this data:
    {data_str}

    Write a 3-paragraph executive summary for a professional PDF report:
    1. Overall activity/volume analysis (High/Low traffic, etc.).
    2. Specific class distribution insights (What objects dominate?).
    3. Security or operational recommendations based on this data.
    
    Format: Plain text only. No markdown, no asterisks (**), no hashes (#). Keep it formal.
    """

    try:
        if provider == "Groq":
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model_name, 
            )
            return response.choices[0].message.content

        elif provider == "OpenAI":
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model_name,
            )
            return response.choices[0].message.content
        
        else:
            return "Error: Invalid Provider Selected."
            
    except Exception as e:
        return f"AI Error ({provider}): {str(e)}"

# --------------------------------------------------
# STYLING UTILITIES
# --------------------------------------------------
def get_custom_styles():
    """Defines professional paragraph styles."""
    styles = getSampleStyleSheet()
    
    return {
        'title': ParagraphStyle(
            "CustomTitle", parent=styles['Title'],
            fontSize=26, textColor=PRIMARY_BLUE, alignment=TA_CENTER, spaceAfter=20
        ),
        'subtitle': ParagraphStyle(
            "CustomSubtitle", parent=styles['Heading2'],
            fontSize=12, textColor=SECONDARY_BLUE, alignment=TA_CENTER, spaceAfter=30
        ),
        'heading': ParagraphStyle(
            "CustomHeading", parent=styles['Heading2'],
            fontSize=16, textColor=PRIMARY_BLUE, spaceBefore=20, spaceAfter=10
        ),
        'body': ParagraphStyle(
            "CustomBody", parent=styles['Normal'],
            fontSize=10, leading=14, alignment=TA_JUSTIFY, spaceAfter=10, textColor=TEXT_COLOR
        ),
        'caption': ParagraphStyle(
            "Caption", parent=styles['Normal'],
            fontSize=9, textColor=colors.grey, alignment=TA_CENTER, spaceAfter=10
        )
    }

def create_divider(width_inches):
    """Creates a horizontal divider line."""
    return Table([[""]], colWidths=[width_inches * inch], rowHeights=[2],
                 style=[("BACKGROUND", (0, 0), (-1, -1), PRIMARY_BLUE)])

# --------------------------------------------------
# HEADER & FOOTER
# --------------------------------------------------
def draw_header_footer(canvas, doc):
    """Draws the Company Header and Footer on every page."""
    canvas.saveState()
    page_width, page_height = A4
    
    # --- HEADER ---
    if os.path.exists(COMPANY_HEADER_BG):
        try:
            canvas.drawImage(COMPANY_HEADER_BG, 0, page_height - 1.5 * inch, 
                           width=page_width, height=1.5 * inch, mask='auto')
        except:
            canvas.setFillColor(PRIMARY_BLUE)
            canvas.rect(0, page_height - 1.5 * inch, page_width, 1.5 * inch, fill=True, stroke=False)
    else:
        canvas.setFillColor(PRIMARY_BLUE)
        canvas.rect(0, page_height - 1.5 * inch, page_width, 1.5 * inch, fill=True, stroke=False)

    # Logo
    if os.path.exists(COMPANY_LOGO):
        try:
            canvas.drawImage(COMPANY_LOGO, page_width - 2.5 * inch,
                           page_height - 1.1 * inch, width=150, height=50,
                           mask='auto', preserveAspectRatio=True)
        except:
            pass

    # Header Text
    canvas.setFont("Helvetica-Bold", 18)
    canvas.setFillColor(colors.white)
    canvas.drawString(0.5 * inch, page_height - 0.9 * inch, "SRIYA.AI ANALYTICS")
    
    canvas.setFont("Helvetica", 10)
    canvas.drawString(0.5 * inch, page_height - 1.1 * inch, "Intelligent Video Object Detection")

    # --- FOOTER ---
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.grey)
    canvas.drawRightString(page_width - 0.5 * inch, 0.5 * inch, f"Page {doc.page}")
    canvas.drawString(0.5 * inch, 0.5 * inch, f"Report Generated: {datetime.now().strftime('%Y-%m-%d')}")
    
    canvas.restoreState()

# --------------------------------------------------
# REPORT SECTIONS
# --------------------------------------------------
def create_cover_page(story, styles):
    story.append(Spacer(1, 60))
    story.append(Paragraph("AI Video Analytics Report", styles['title']))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Automated Object Detection & Tracking System</b>", styles['subtitle']))
    story.append(Spacer(1, 20))
    story.append(create_divider(7))
    story.append(Spacer(1, 40))
    
    current_time = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    report_info = [
        ["Report Generated:", current_time],
        ["Detection Engine:", "YOLOv8 + ByteTrack"],
        ["Analysis Type:", "Object Tracking & Counting"],
        ["Generated By:", "Sriya.AI Automated System"],
    ]
    info_table = Table(report_info, colWidths=[2.5*inch, 3.5*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), LIGHT_BLUE),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('PADDING', (0, 0), (-1, -1), 12),
        ('ROWBACKGROUNDS', (1, 0), (1, -1), [colors.white, ULTRA_LIGHT_BLUE])
    ]))
    story.append(info_table)
    story.append(Spacer(1, 50))
    story.append(Paragraph("This document contains a detailed analysis of video/image footage processed using state-of-the-art AI.", styles['body']))
    story.append(PageBreak())

def create_methodology_section(story, styles):
    story.append(Paragraph("1. Methodology & Technical Approach", styles['heading']))
    story.append(create_divider(5))
    story.append(Spacer(1, 10))
    story.append(Paragraph("The analysis is derived using a multi-stage Computer Vision pipeline.", styles['body']))
    method_data = [
        ["Component", "Technology", "Function"],
        ["Object Detection", "YOLOv8 (Nano)", "Identifies objects in real-time."],
        ["Object Tracking", "BoT-SORT / ByteTrack", "Assigns unique IDs (Video only)."],
        ["Insight Generation", "LLM (Llama3 / GPT-4)", "Analyzes data for insights."]
    ]
    table = Table(method_data, colWidths=[1.5*inch, 1.5*inch, 3.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY_BLUE),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [LIGHT_BLUE, ULTRA_LIGHT_BLUE])
    ]))
    story.append(table)
    story.append(Spacer(1, 20))

def create_ai_summary_section(story, styles, ai_summary):
    story.append(Paragraph("2. Executive Summary (AI Analysis)", styles['heading']))
    story.append(create_divider(5))
    story.append(Spacer(1, 10))
    if ai_summary:
        for para in ai_summary.split('\n'):
            if para.strip():
                story.append(Paragraph(para, styles['body']))
                story.append(Spacer(1, 6))
    else:
        story.append(Paragraph("No AI Summary available.", styles['body']))
    story.append(Spacer(1, 20))

def create_statistics_section(story, styles, analytics):
    story.append(Paragraph("3. Statistical Data", styles['heading']))
    story.append(create_divider(5))
    story.append(Spacer(1, 10))
    table_data = [["Object Class", "Count", "Avg Confidence"]]
    for cls, count in analytics["class_count"].items():
        avg_conf = 0.0
        if analytics["confidence_count"][cls] > 0:
            avg_conf = analytics["confidence_sum"][cls] / analytics["confidence_count"][cls]
        table_data.append([cls, str(count), f"{avg_conf:.2f}"])
    table = Table(table_data, colWidths=[3*inch, 2*inch, 2*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY_BLUE),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), ULTRA_LIGHT_BLUE),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    story.append(table)
    story.append(PageBreak())

def create_visuals_section(story, styles, graph_path):
    story.append(Paragraph("4. Visual Distribution", styles['heading']))
    story.append(create_divider(5))
    story.append(Spacer(1, 10))
    if os.path.exists(graph_path):
        img = Image(graph_path, width=6*inch, height=4*inch)
        story.append(img)
    else:
        story.append(Paragraph("Visual graph not available.", styles['body']))

# --------------------------------------------------
# MAIN PDF BUILDER
# --------------------------------------------------
def generate_pdf(report_path, analytics, duration, graph_path, ai_summary):
    doc = SimpleDocTemplate(
        report_path, pagesize=A4,
        topMargin=2 * inch, bottomMargin=1 * inch,
        leftMargin=0.5 * inch, rightMargin=0.5 * inch
    )
    styles = get_custom_styles()
    story = []
    
    create_cover_page(story, styles)
    create_methodology_section(story, styles)
    create_ai_summary_section(story, styles, ai_summary)
    create_statistics_section(story, styles, analytics)
    create_visuals_section(story, styles, graph_path)
    
    try:
        doc.build(story, onFirstPage=draw_header_footer, onLaterPages=draw_header_footer)
        print(f"✅ Report Generated: {report_path}")
    except Exception as e:
        print(f"❌ PDF Generation Error: {e}")