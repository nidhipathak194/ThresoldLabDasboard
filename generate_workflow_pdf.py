import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#6b7280"))
        
        # Header
        self.drawString(54, letter[1] - 36, "Threshold Lab — System Workflow & Architecture Document")
        self.setStrokeColor(colors.HexColor("#e5e7eb"))
        self.setLineWidth(0.5)
        self.line(54, letter[1] - 42, letter[0] - 54, letter[1] - 42)
        
        # Footer
        self.line(54, 45, letter[0] - 54, 45)
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 54, 30, page_str)
        self.drawString(54, 30, "COMP702 M.Sc. Project — University of Liverpool (Nidhi Kumari)")
        self.restoreState()

def create_workflow_pdf(filename="Threshold_Lab_Workflow.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Title'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=colors.HexColor('#1e1b4b'),
        alignment=0,
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#4f46e5'),
        spaceAfter=15
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=colors.HexColor('#1e1b4b'),
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=colors.HexColor('#312e81'),
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#1f2937'),
        spaceAfter=8
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=15,
        bulletIndent=5,
        spaceAfter=4
    )

    code_style = ParagraphStyle(
        'Code_Custom',
        fontName='Courier',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#1e293b'),
        backColor=colors.HexColor('#f1f5f9'),
        borderColor=colors.HexColor('#cbd5e1'),
        borderWidth=0.5,
        borderPadding=6,
        spaceBefore=6,
        spaceAfter=8
    )

    story = []

    # Title & Header Block
    story.append(Paragraph("Threshold Lab: Workflow & System Architecture", title_style))
    story.append(Paragraph("Calibrating the Sorites Paradox in Credit Card Fraud Detection | COMP702 M.Sc. Project Proposal", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#4f46e5'), spaceAfter=12))

    # Executive Overview
    story.append(Paragraph("1. Executive Summary & Core Objective", h1_style))
    story.append(Paragraph(
        "<b>Threshold Lab</b> is an interactive web-based machine learning framework designed to demonstrate that the binary decision boundary "
        "between 'Genuine' and 'Fraudulent' credit card transactions is a <b>constructed boundary</b> derived from cost assumptions, rather than a fixed fact. "
        "By applying Eubulides' classical <i>Sorites Paradox</i> and Wright's (1975) <i>tolerance principle</i> alongside Elkan's (2001) cost-sensitive expected-loss formula, "
        "the application allows stakeholders to adjust financial cost ratios live and observe the movement of the optimal decision boundary <b>τ*</b>, "
        "confusion matrices, and the resulting <i>Contested Zone</i>.", body_style
    ))

    story.append(Spacer(1, 8))

    # Part 1: User Interactive Workflow
    story.append(Paragraph("2. User Interactive Workflow (Step-by-Step)", h1_style))
    story.append(Paragraph("The end-user interacts with the dashboard through a structured 5-stage workflow:", body_style))

    user_steps = [
        ("Step 1: Configuration & Dataset Selection", "User selects between the <b>Synthetic ULB Benchmark Dataset</b> (50,000 samples, 0.172% fraud prevalence) or uploads a custom <b>Kaggle creditcard.csv</b> dataset via the sidebar uploader."),
        ("Step 2: Model & Calibration Choice", "User selects the classifier architecture (<i>Logistic Regression</i> or <i>Random Forest Ensemble</i>) and active probability calibration method (<i>Raw Scores</i>, <i>Platt Scaling</i>, or <i>Isotonic Regression</i>)."),
        ("Step 3: Philosophical Sorites Exploration", "User inspects a live worked example of twin transactions straddling τ*. Although their risk scores differ by less than 0.001, one transaction is silently approved while the other triggers a card block."),
        ("Step 4: Live Cost-Sensitive Tuning", "User adjusts the sliders for False Positive Cost <b>C(FP)</b> (£1 to £200) and False Negative Cost <b>C(FN)</b> (£100 to £5000) to observe real-time updating of τ*, expected monetary loss, PR curves, and confusion heatmaps."),
        ("Step 5: Contested Zone & CSV Export", "User sets a plausible range of cost assumptions [C(FP)_min, C(FP)_max] to quantify all transactions whose label flips, downloading the resulting contested transaction table as a CSV file.")
    ]

    for title, desc in user_steps:
        story.append(Paragraph(f"• <b>{title}</b>", bullet_style))
        story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;{desc}", body_style))

    story.append(Spacer(1, 10))

    # Part 2: Technical System Architecture
    story.append(Paragraph("3. Technical System Architecture & Execution Flow", h1_style))
    story.append(Paragraph("Under the hood, data and computation flow across four modular software layers:", body_style))

    arch_table_data = [
        [Paragraph("<b>Layer</b>", h2_style), Paragraph("<b>Module / File</b>", h2_style), Paragraph("<b>Primary Responsibilities</b>", h2_style)],
        [
            Paragraph("<b>1. Data Layer</b>", body_style),
            Paragraph("<code>src/data_loader.py</code>", code_style),
            Paragraph("Generates 31 benchmark features (Time, Amount, V1-V28 PCA), performs stratified 80/20 train/test split, and applies standard scaling.", body_style)
        ],
        [
            Paragraph("<b>2. ML & Calibration</b>", body_style),
            Paragraph("<code>src/ml_engine.py</code>", code_style),
            Paragraph("Trains baseline models, applies 3-fold Platt scaling (Sigmoidal) & Isotonic Regression, and calculates PR-AUC & Brier scores.", body_style)
        ],
        [
            Paragraph("<b>3. Sorites Engine</b>", body_style),
            Paragraph("<code>src/sorites_engine.py</code>", code_style),
            Paragraph("Computes Elkan's τ*, evaluates confusion matrices & monetary loss, filters the Contested Zone, and finds twin sorites pairs.", body_style)
        ],
        [
            Paragraph("<b>4. UI & Dashboard</b>", body_style),
            Paragraph("<code>app.py</code>", code_style),
            Paragraph("Renders dark-themed Streamlit interface, LaTeX equations, Plotly PR curves, reliability diagrams, and CSV export handlers.", body_style)
        ]
    ]

    arch_table = Table(arch_table_data, colWidths=[1.3*inch, 1.8*inch, 3.4*inch])
    arch_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e0e7ff')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#1e1b4b')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
    ]))

    story.append(arch_table)

    story.append(Spacer(1, 12))

    # Part 3: Mathematical Formulation
    story.append(Paragraph("4. Key Mathematical Formulations", h1_style))
    
    math_box_content = (
        "<b>1. Elkan's Optimal Cost-Sensitive Threshold Formula (Elkan, 2001):</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>τ* = C(FP) / [ C(FP) + C(FN) ]</b><br/><br/>"
        "<b>2. Expected Total Financial Loss Function:</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>Loss(τ) = FP(τ) · C(FP) + FN(τ) · C(FN)</b><br/><br/>"
        "<b>3. Platt Sigmoidal Probability Scaling (Platt, 1999):</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>P(Y = 1 | s) = 1 / [ 1 + exp( A · s + B ) ]</b><br/><br/>"
        "<b>4. Contested Zone Transaction Set:</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>C_zone = { x | τ*(R_min) ≤ p(x) < τ*(R_max) }</b>"
    )
    story.append(Paragraph(math_box_content, code_style))

    story.append(Spacer(1, 10))

    # Document Footer / Author Note
    story.append(Paragraph("5. Author & Academic Context", h1_style))
    story.append(Paragraph(
        "<b>Student:</b> Nidhi Kumari (201960034) &nbsp;|&nbsp; <b>Supervisor:</b> Dr. Vitaliy Kurlin<br/>"
        "<b>Module:</b> COMP702 — M.Sc. Project (2025/26) &nbsp;|&nbsp; <b>Department:</b> School of Computer Science and Informatics, University of Liverpool",
        body_style
    ))

    doc.build(story, canvasmaker=NumberedCanvas)
    return filename

if __name__ == "__main__":
    out_pdf = create_workflow_pdf("Threshold_Lab_Workflow.pdf")
    print(f"Workflow PDF successfully generated: {out_pdf}")
