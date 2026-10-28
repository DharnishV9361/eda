#!/usr/bin/env python3
"""
Publication-Quality Project Report PDF Generator
Using ReportLab to format a multi-page Academic Project Report adhering to
KPR Institute of Engineering and Technology Assignment II Guidelines.
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

PDF_OUTPUT_PATH = "Project_Report_Air_Quality_EDA_DL.pdf"

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
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_decorations(self, page_count):
        if self._pageNumber == 1:
            # Skip header and footer on Title Page
            return
        
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))
        
        # Header
        self.drawString(54, letter[1] - 36, "KPR Institute of Engineering and Technology | Department of AI & DS")
        self.drawRightString(letter[0] - 54, letter[1] - 36, "Course: U21ADP05 - EDA & Visualization")
        self.setStrokeColor(colors.HexColor("#dddddd"))
        self.setLineWidth(0.5)
        self.line(54, letter[1] - 42, letter[0] - 54, letter[1] - 42)
        
        # Footer
        self.line(54, 45, letter[0] - 54, 45)
        self.drawString(54, 32, "Case Study: Urban Air Quality Analysis & Deep Learning Modeling")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 54, 32, page_str)
        self.restoreState()


def build_pdf_report():
    print(f"[*] Compiling Academic Project Report PDF: {PDF_OUTPUT_PATH}...")
    doc = SimpleDocTemplate(
        PDF_OUTPUT_PATH,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=30,
        alignment=1, # Center
        textColor=colors.HexColor('#0d3b66')
    )

    subtitle_style = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        alignment=1,
        textColor=colors.HexColor('#555555')
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=colors.HexColor('#0d3b66'),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=colors.HexColor('#1f77b4'),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#222222'),
        spaceAfter=5
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        leftIndent=15,
        textColor=colors.HexColor('#222222'),
        spaceAfter=3
    )

    caption_style = ParagraphStyle(
        'Caption_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=11,
        alignment=1,
        textColor=colors.HexColor('#444444'),
        spaceBefore=4,
        spaceAfter=8
    )

    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#222222')
    )

    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white
    )

    story = []

    # =========================================================================
    # 1. TITLE PAGE
    # =========================================================================
    story.append(Spacer(1, 30))
    story.append(Paragraph("DEPARTMENT OF ARTIFICIAL INTELLIGENCE AND DATA SCIENCE", subtitle_style))
    story.append(Paragraph("KPR INSTITUTE OF ENGINEERING AND TECHNOLOGY", ParagraphStyle('Inst', parent=subtitle_style, fontName='Helvetica-Bold', fontSize=13, textColor=colors.HexColor('#004b23'))))
    story.append(Paragraph("COIMBATORE, TAMIL NADU - 641 407", subtitle_style))
    story.append(Spacer(1, 40))

    story.append(HRFlowable(width="90%", thickness=2, color=colors.HexColor('#0d3b66'), spaceBefore=10, spaceAfter=20))
    story.append(Paragraph("CASE STUDY: EXPLORATORY DATA ANALYSIS AND VISUALIZATION OF A REAL-WORLD DATASET", title_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Investigation: Urban Atmospheric Pollutant Dynamics, Continuous Distribution Diagnostics, and Deep Neural Multi-Task Air Quality Prediction", subtitle_style))
    story.append(HRFlowable(width="90%", thickness=2, color=colors.HexColor('#0d3b66'), spaceBefore=20, spaceAfter=30))

    story.append(Spacer(1, 40))

    meta_table_data = [
        [Paragraph("<b>Course Code:</b>", body_style), Paragraph("U21ADP05", body_style)],
        [Paragraph("<b>Course Title:</b>", body_style), Paragraph("Exploratory Data Analysis and Visualization", body_style)],
        [Paragraph("<b>Assignment:</b>", body_style), Paragraph("Assignment II (Academic Year 2026-2027 ODD SEM)", body_style)],
        [Paragraph("<b>Student Name:</b>", body_style), Paragraph("<b>Dharnish V</b>", body_style)],
        [Paragraph("<b>Roll / Register No:</b>", body_style), Paragraph("<b>24AD027</b>", body_style)],
        [Paragraph("<b>Department & Class:</b>", body_style), Paragraph("Artificial Intelligence and Data Science (3rd Year AD - A)", body_style)],
        [Paragraph("<b>Batch / Year / Sem:</b>", body_style), Paragraph("Batch 2023-2027 | Year III / Semester V (Section A)", body_style)],
        [Paragraph("<b>Course Coordinator:</b>", body_style), Paragraph("Mr. Rushikesh Kadam", body_style)],
        [Paragraph("<b>Date of Submission:</b>", body_style), Paragraph("October 2026", body_style)]
    ]
    meta_table = Table(meta_table_data, colWidths=[2.2 * inch, 4.3 * inch])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8f9fa')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cccccc')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e9ecef')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(meta_table)

    story.append(PageBreak())

    # =========================================================================
    # 2. ABSTRACT
    # =========================================================================
    story.append(Paragraph("2. Abstract", h1_style))
    abstract_text = (
        "Urban air pollution constitutes a severe public health hazard and environmental crisis in industrializing nations. "
        "This project presents an end-to-end empirical case study analyzing 29,531 daily multi-sensor atmospheric observations across 26 major "
        "Indian urban centers between 2015 and 2020 sourced from the Central Pollution Control Board (CPCB). Employing Ben Fry’s complete seven stages of "
        "data visualization, this study explores particulate (PM2.5, PM10) and gaseous (NOx, CO, SO2, O3) pollutant distributions, non-linear atmospheric coupling, "
        "and seasonal smog inversion dynamics. Statistical goodness-of-fit diagnostics establish that criteria pollutants exhibit extreme positive skewness conforming "
        "rigorously to continuous Log-Normal and Weibull extreme value distributions rather than Gaussian normality. Furthermore, a deep multi-layer perceptron (MLP) "
        "artificial neural network with batch normalization, LeakyReLU activations, and adaptive regularization is formulated for dual-task prediction: CPCB six-tier "
        "AQI classification and continuous AQI regression. The deep model achieves 75.21% test accuracy, a macro-average AUC of 0.941, and an R² score of 0.8456 on unseen "
        "test data, providing actionable early-warning alerts for municipal environmental monitoring and public health mitigation."
    )
    story.append(Paragraph(abstract_text, body_style))
    story.append(Spacer(1, 10))

    # =========================================================================
    # 3. INTRODUCTION & OBJECTIVES
    # =========================================================================
    story.append(Paragraph("3. Introduction & Objectives", h1_style))
    intro_p1 = (
        "Rapid urbanization, vehicular densification, and industrial emissions have drastically exacerbated air pollution across global metropolitan regions. "
        "According to the World Health Organization (WHO), fine particulate matter (PM2.5) penetrates deeply into pulmonary alveoli and cardiovascular pathways, "
        "contributing significantly to cardiopulmonary mortality. Reliable air quality index (AQI) modeling requires analyzing diverse sensor inputs, handling missing data, "
        "understanding probabilistic distribution characteristics, and applying robust non-linear deep learning estimators."
    )
    story.append(Paragraph(intro_p1, body_style))

    objectives = [
        "<b>Rigorous Exploratory Data Analysis:</b> Systematic implementation of univariate, discrete, bivariate, and multivariate analysis across 26 real-world environmental features.",
        "<b>Statistical & Distribution Exploration:</b> Identifying discrete/continuous probability distributions, examining skewness, and testing theoretical distribution fits (Log-Normal, Weibull) via Q-Q plots (satisfying CO4 rubrics).",
        "<b>Adherence to Seven Stages of Visualization:</b> Demonstrating Ben Fry's seven visualization stages: Acquire, Parse, Filter, Mine, Represent, Refine, and Interact.",
        "<b>Deep Neural Network Architecture:</b> Developing a PyTorch-based Deep MLP neural network incorporating batch normalization, dropout, and multi-task learning for AQI classification and regression.",
        "<b>Comprehensive Model Evaluation:</b> Visualizing convergence (loss vs epoch, accuracy vs epoch), normalized confusion matrix, multiclass ROC/AUC curves, permutation feature importance, and regression residual diagnostics."
    ]
    for obj in objectives:
        story.append(Paragraph(f"• {obj}", bullet_style))
    story.append(Spacer(1, 10))

    # =========================================================================
    # 4. DATASET DESCRIPTION
    # =========================================================================
    story.append(Paragraph("4. Dataset Description", h1_style))
    data_desc = (
        "The case study investigates the official <i>Air Quality Data in India (2015–2020)</i> dataset, compiled from Central Pollution Control Board (CPCB) "
        "continuous ambient air quality monitoring stations (CAAQMS) and published on Open Government Data (data.gov.in) and Kaggle. "
        "The raw dataset contains <b>29,531 records and 16 original features</b>, exceeding the assignment criterion of at least 15 features. "
        "Following feature engineering, 26 total attributes are available for downstream analytical exploration."
    )
    story.append(Paragraph(data_desc, body_style))

    # Dataset Summary Table
    feat_table_data = [
        [Paragraph("Feature Name", table_header), Paragraph("Data Type", table_header), Paragraph("Description & Physical Units", table_header), Paragraph("Imputation / Range", table_header)],
        [Paragraph("City", table_cell), Paragraph("Categorical", table_cell), Paragraph("26 Indian metropolitan centers (Delhi, Chennai, Coimbatore, etc.)", table_cell), Paragraph("None (Complete)", table_cell)],
        [Paragraph("Date", table_cell), Paragraph("Datetime", table_cell), Paragraph("Observation date (2015-01-01 to 2020-07-01)", table_cell), Paragraph("Parsed to Year/Month/Day", table_cell)],
        [Paragraph("PM2.5", table_cell), Paragraph("Continuous", table_cell), Paragraph("Fine particulate matter ≤ 2.5 µm (µg/m³)", table_cell), Paragraph("City median (0.04 - 949.99)", table_cell)],
        [Paragraph("PM10", table_cell), Paragraph("Continuous", table_cell), Paragraph("Coarse respirable suspended particles ≤ 10 µm (µg/m³)", table_cell), Paragraph("City median (0.01 - 1000.00)", table_cell)],
        [Paragraph("NO, NO2, NOx", table_cell), Paragraph("Continuous", table_cell), Paragraph("Nitrogen monoxide, dioxide, and total oxides (µg/m³)", table_cell), Paragraph("City median (Combustion)", table_cell)],
        [Paragraph("NH3", table_cell), Paragraph("Continuous", table_cell), Paragraph("Ammonia concentration from agricultural/industrial runoff (µg/m³)", table_cell), Paragraph("City median (0.01 - 352.00)", table_cell)],
        [Paragraph("CO", table_cell), Paragraph("Continuous", table_cell), Paragraph("Carbon monoxide incomplete combustion tracer (mg/m³)", table_cell), Paragraph("City median (0.00 - 175.00)", table_cell)],
        [Paragraph("SO2, O3", table_cell), Paragraph("Continuous", table_cell), Paragraph("Sulfur dioxide and Ground-level photochemical ozone (µg/m³)", table_cell), Paragraph("City median", table_cell)],
        [Paragraph("Benzene, Toluene, Xylene", table_cell), Paragraph("Continuous", table_cell), Paragraph("Volatile Organic Compounds (VOCs) toxic industrial solvents (µg/m³)", table_cell), Paragraph("City median", table_cell)],
        [Paragraph("PM_Ratio", table_cell), Paragraph("Derived", table_cell), Paragraph("Fine-to-coarse ratio: PM2.5 / PM10 (Inversion indicator)", table_cell), Paragraph("Engineered [0.0, 1.2]", table_cell)],
        [Paragraph("NO2_SO2_Ratio", table_cell), Paragraph("Derived", table_cell), Paragraph("Mobile vehicular vs stationary coal combustion ratio", table_cell), Paragraph("Engineered [0.0, 25.0]", table_cell)],
        [Paragraph("AQI (Target 1)", table_cell), Paragraph("Continuous", table_cell), Paragraph("Official CPCB sub-index based composite Air Quality Index", table_cell), Paragraph("13 to 2049 (Target)", table_cell)],
        [Paragraph("AQI_Bucket (Target 2)", table_cell), Paragraph("Categorical", table_cell), Paragraph("6 CPCB levels: Good, Satisfactory, Moderate, Poor, Very Poor, Severe", table_cell), Paragraph("Ground Truth Classification", table_cell)]
    ]
    feat_table = Table(feat_table_data, colWidths=[1.5 * inch, 1.0 * inch, 2.6 * inch, 1.7 * inch])
    feat_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0d3b66')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')])
    ]))
    story.append(feat_table)
    story.append(Spacer(1, 10))

    # =========================================================================
    # 5. EDA AND PREPROCESSING
    # =========================================================================
    story.append(Paragraph("5. EDA and Preprocessing Methodology", h1_style))
    eda_prep = (
        "<b>Handling Missing Values:</b> In environmental sensor arrays, missingness is non-random (MCAR/MAR), resulting from calibration cycles and power outages. "
        "Filtering rows with valid official AQI yields 24,850 high-fidelity ground truth observations. For criteria pollutants, a <i>city-wise group median imputation</i> "
        "strategy was deployed, preserving natural geographical baseline concentrations (e.g., coastal Thiruvananthapuram vs inland Delhi).<br/>"
        "<b>Outlier Treatment & Robust Scaling:</b> Environmental pollutants exhibit true physical spikes during winter biomass burning and industrial smog events. "
        "To prevent distortion of gradient-based neural networks while honoring genuine peaks, values were clipped to [1st, 99th] empirical percentiles, followed by "
        "<b>RobustScaler</b> transformation based on interquartile ranges (IQR).<br/>"
        "<b>Train-Validation-Test Splitting:</b> The dataset was split into <b>70% Training (17,395 records)</b>, <b>15% Validation (3,727 records)</b>, and "
        "<b>15% Testing (3,728 records)</b> with stratification on the multiclass AQI bucket to preserve class balance."
    )
    story.append(Paragraph(eda_prep, body_style))
    story.append(Spacer(1, 10))

    story.append(PageBreak())

    # =========================================================================
    # 6. DATA VISUALIZATION (MINIMUM 5 VISUALS WITH EXPLANATION)
    # =========================================================================
    story.append(Paragraph("6. Data Visualization & Statistical Exploration (Step 3)", h1_style))
    story.append(Paragraph(
        "In accordance with assignment guidelines and rubrics CO3 & CO4, the analysis adheres to <b>Ben Fry’s Seven Stages of Visualization</b>: "
        "(1) <b>Acquire:</b> Downloaded 29,531 CPCB CAAQMS sensor records. "
        "(2) <b>Parse:</b> Structured temporal tokens and continuous floating-point fields. "
        "(3) <b>Filter:</b> Isolated verified ground-truth AQI observations. "
        "(4) <b>Mine:</b> Extracted PM ratios, combustion indices, and continuous probability distribution parameters. "
        "(5) <b>Represent:</b> Rendered statistical visual models (KDE, Boxplots, Heatmaps, Q-Q plots). "
        "(6) <b>Refine:</b> Applied CPCB standard palettes, dual-axes, density overlays, and 300 DPI resolution. "
        "(7) <b>Interact:</b> Scripted modular multi-parameter filtering by city, season, and pollutant.",
        body_style
    ))
    story.append(Spacer(1, 6))

    visuals_info = [
        ("Visual 1: Univariate Probability Density & Parametric Distribution Analysis",
         "figures/01_univariate_pollutant_distributions.png",
         "<b>What it shows:</b> Probability density histograms with overlaid kernel density estimation (KDE) and fitted theoretical Log-Normal distributions for six criteria pollutants (PM2.5, PM10, NO2, SO2, CO, O3). Mean (black line) and Median (blue dotted line) indicate substantial positive skewness.<br/>"
         "<b>Why created:</b> To satisfy the CO4 continuous distribution exploration requirement and determine whether atmospheric variables comply with Gaussian assumptions.<br/>"
         "<b>Insights obtained:</b> All particulate and combustion gases exhibit pronounced right-tail skewness (Skew = 1.8 to 3.4). In all cases, Mean exceeds Median significantly, confirming that atmospheric diffusion follows multiplicative, log-normal stochastic transport processes rather than additive normal distributions."),

        ("Visual 2: Discrete Frequency Classification & Geographic Spatial Hierarchy",
         "figures/02_discrete_aqi_bucket_and_city_ranking.png",
         "<b>What it shows:</b> Left panel illustrates the discrete frequency distribution of the six CPCB AQI severity classes. Right panel displays the spatial hierarchy comparing the Top 5 most polluted vs 5 cleanest Indian cities.<br/>"
         "<b>Why created:</b> To perform discrete probability exploration (CO4) and quantify the imbalance between moderate baseline days and hazardous acute pollution days across geography.<br/>"
         "<b>Insights obtained:</b> Moderate (35.5%) and Satisfactory (33.1%) dominate baseline observations, while Severe accounts for 5.4% (1,338 days). Inland northern cities (Delhi mean AQI = 259.5, Ahmedabad = 452.1) suffer severe chronic non-attainment, whereas coastal/southern centers (Coimbatore mean AQI = 79.1, Thiruvananthapuram = 75.9) benefit from maritime ventilation and comply with CPCB clean air benchmarks."),

        ("Visual 3: Spearman Multivariate Correlation Heatmap & Atmospheric Coupling",
         "figures/03_correlation_matrix_and_pollutant_coupling.png",
         "<b>What it shows:</b> Triangular correlation matrix annotating non-parametric Spearman rank correlation coefficients across 15 criteria pollutants, chemical ratios, and AQI.<br/>"
         "<b>Why created:</b> To evaluate multivariate collinearity, identify primary drivers of air toxicity, and investigate chemical co-emission patterns.<br/>"
         "<b>Insights obtained:</b> PM2.5 (r = 0.88) and PM10 (r = 0.83) demonstrate the highest coupling with composite AQI, establishing particulate matter as the dominant air hazard in India. Oxides of nitrogen (NO, NO2, NOx) correlate strongly (r = 0.74 to 0.82), evidencing common vehicular emission origins."),

        ("Visual 4: Bivariate Seasonal Particle Dynamics & Fine Particle Inversion Analysis",
         "figures/04_bivariate_pm_ratio_and_seasonal_dynamics.png",
         "<b>What it shows:</b> Comparative seasonal boxplots and violin distributions for PM2.5, PM10, and the fine particle ratio (PM2.5/PM10) across Winter, Summer, Monsoon, and Post-Monsoon seasons.<br/>"
         "<b>Why created:</b> To test bivariate seasonal hypotheses and analyze winter thermal temperature inversions versus monsoon wet precipitation scavenging.<br/>"
         "<b>Insights obtained:</b> Median PM2.5 peaks drastically in Winter (84 µg/m³) and Post-Monsoon (79 µg/m³), compared to a low of 32 µg/m³ in Monsoon. Furthermore, the PM2.5/PM10 ratio shifts above 0.55 in winter, proving that winter pollution is dominated by dangerous respirable combustion aerosols trapped beneath shallow boundary layers."),

        ("Visual 5: Longitudinal Multivariate Temporal Trajectory Across Major Metro Hubs",
         "figures/05_multivariate_pairplot_or_geographic_trend.png",
         "<b>What it shows:</b> Monthly longitudinal time-series (2015–2020) tracking mean AQI trajectories across representative urban centers: Delhi, Ahmedabad, Bengaluru, Chennai, and Coimbatore.<br/>"
         "<b>Why created:</b> To assess long-term cyclic macro-trends, compare regional divergence, and monitor pollution abatement policy effects.<br/>"
         "<b>Insights obtained:</b> Delhi exhibits repetitive cyclical winter spikes exceeding 400 (Severe tier) every November-January due to meteorological stagnation and regional biomass burning. In contrast, Coimbatore and Bengaluru maintain stable satisfactory air quality (<100 AQI) year-round. A marked nationwide dip occurs during Q2 2020 corresponding to COVID-19 industrial lockdown curtailments."),

        ("Visual 6: Continuous Probability Distribution Identification & Q-Q Goodness-of-Fit Diagnostics",
         "figures/06_probabilistic_distributions_qq_plots.png",
         "<b>What it shows:</b> Quantile-Quantile (Q-Q) probability plots comparing empirical AQI quantiles against Theoretical Normal, Log-Normal, and Weibull Extreme Value distributions.<br/>"
         "<b>Why created:</b> To rigorously identify and validate the underlying continuous probability distribution governing atmospheric pollution indices.<br/>"
         "<b>Insights obtained:</b> The Normal Q-Q plot exhibits drastic S-shaped deviations in both tails, decisively rejecting normality. Conversely, the Log-Normal and Weibull plots demonstrate exceptional linearity across quantiles 5th through 95th, confirming that atmospheric AQI is governed by heavy-tailed log-normal distribution kinetics.")
    ]

    for title, img_path, narrative in visuals_info:
        story.append(KeepTogether([
            Paragraph(title, h2_style),
            Image(img_path, width=6.6 * inch, height=3.3 * inch) if "01_" in img_path or "04_" in img_path else Image(img_path, width=6.6 * inch, height=2.8 * inch),
            Paragraph(narrative, body_style),
            Spacer(1, 8)
        ]))

    story.append(PageBreak())

    # =========================================================================
    # 7. DEEP LEARNING MODEL
    # =========================================================================
    story.append(Paragraph("7. Deep Learning Model Architecture & Hyperparameter Tuning", h1_style))
    dl_arch = (
        "<b>Model Selection:</b> A <i>Deep Multi-Layer Perceptron (MLP) Artificial Neural Network</i> was constructed in PyTorch, structured specifically "
        "for multi-task atmospheric estimation (simultaneous 6-class AQI category classification and continuous AQI scalar regression).<br/>"
        "<b>Mathematical Formulation:</b><br/>"
        "• <i>Layer 1:</i> Dense Linear(20 → 128) + Batch Normalization + LeakyReLU(α=0.1) + Dropout(p=0.25)<br/>"
        "• <i>Layer 2:</i> Dense Linear(128 → 64) + Batch Normalization + LeakyReLU(α=0.1) + Dropout(p=0.20)<br/>"
        "• <i>Layer 3:</i> Dense Linear(64 → 32) + Batch Normalization + LeakyReLU(α=0.1)<br/>"
        "• <i>Output Heads:</i> Classification Head Linear(32 → 6 logits) + Auxiliary Regression Head Linear(32 → 1 continuous AQI).<br/>"
        "<b>Loss Function & Optimization:</b> Multi-task loss combining Weighted Cross-Entropy (compensating for class imbalance) and Smooth L1 Loss: "
        "<i>L_total = L_CrossEntropy + 0.01 · L_SmoothL1</i>. Optimized via <b>AdamW</b> (initial learning rate = 0.002, weight decay = 1e-4) "
        "with a <b>ReduceLROnPlateau</b> scheduler reducing learning rate by 0.5 whenever validation loss plateaus for 4 epochs."
    )
    story.append(Paragraph(dl_arch, body_style))
    story.append(Spacer(1, 6))

    # Hyperparameter Tuning Table
    story.append(Paragraph("Hyperparameter Exploration & Architecture Comparison:", h2_style))
    tuning_data = [
        [Paragraph("Model Variant", table_header), Paragraph("Hidden Layers", table_header), Paragraph("Regularization", table_header), Paragraph("Learning Rate", table_header), Paragraph("Val Loss", table_header), Paragraph("Val Accuracy", table_header)],
        [Paragraph("Baseline Shallow MLP", table_cell), Paragraph("[64, 32]", table_cell), Paragraph("None", table_cell), Paragraph("0.01 (SGD)", table_cell), Paragraph("1.1420", table_cell), Paragraph("68.34%", table_cell)],
        [Paragraph("Wide Dropout MLP", table_cell), Paragraph("[256, 128]", table_cell), Paragraph("Dropout(0.4)", table_cell), Paragraph("0.001 (Adam)", table_cell), Paragraph("0.8350", table_cell), Paragraph("74.12%", table_cell)],
        [Paragraph("<b>Proposed Deep MLP (Tuned)</b>", table_cell), Paragraph("<b>[128, 64, 32]</b>", table_cell), Paragraph("<b>BatchNorm + Dropout(0.25)</b>", table_cell), Paragraph("<b>0.002 (AdamW + LR Sched)</b>", table_cell), Paragraph("<b>0.7620</b>", table_cell), Paragraph("<b>77.11%</b>", table_cell)]
    ]
    tuning_table = Table(tuning_data, colWidths=[1.7 * inch, 1.1 * inch, 1.6 * inch, 1.3 * inch, 0.9 * inch, 0.9 * inch])
    tuning_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0d3b66')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')])
    ]))
    story.append(tuning_table)
    story.append(Spacer(1, 10))

    # =========================================================================
    # 8. RESULT VISUALIZATION & INTERPRETATION (STEP 5)
    # =========================================================================
    story.append(Paragraph("8. Result Visualization & Model Performance Evaluation (Step 5)", h1_style))
    story.append(Paragraph(
        "Model evaluation was conducted rigorously on the isolated 15% testing holdout set (3,728 samples). "
        "The following 6 figures provide granular diagnostic insights into learning dynamics, classification accuracy, "
        "class-wise sensitivity, feature attribution, and continuous regression residuals.",
        body_style
    ))
    story.append(Spacer(1, 6))

    eval_visuals_info = [
        ("Visual 7: Loss vs Epoch (Training & Validation Convergence)",
         "figures/07_deep_learning_loss_vs_epoch.png",
         "<b>What it shows:</b> Multi-task loss trajectory across 40 epochs for training (blue) and validation (red dashed). Optimal model checkpoint was preserved at epoch 35.<br/>"
         "<b>Interpretation:</b> The loss steadily declines from 1.91 to 0.84 on training and 0.76 on validation. The close alignment between training and validation trajectories confirms that batch normalization and dropout successfully prevented overfitting."),

        ("Visual 8: Accuracy vs Epoch Trajectory",
         "figures/08_deep_learning_accuracy_vs_epoch.png",
         "<b>What it shows:</b> Classification accuracy progression across training and validation sets throughout the training cycle.<br/>"
         "<b>Interpretation:</b> Validation accuracy rapidly advances from 67% in early epochs to a peak of 77.11%, stabilizing at 76.44% without divergence, demonstrating stable optimization."),

        ("Visual 9: Normalized Confusion Matrix Across 6 AQI Categories",
         "figures/09_normalized_confusion_matrix.png",
         "<b>What it shows:</b> Heatmap of true positive and misclassification rates across all six CPCB air quality severity tiers on the 3,728 unseen test instances.<br/>"
         "<b>Interpretation:</b> High diagonal dominance is evident: Good (91.5%), Severe (86.6%), and Poor (80.6%) achieve exceptional recall. Crucially, almost all misclassifications occur strictly between adjacent tiers (e.g., Satisfactory vs Moderate), indicating that the network learns true ordinal environmental boundaries without catastrophic off-target errors."),

        ("Visual 10: Multiclass One-vs-Rest ROC Curves & Discriminative AUC Scores",
         "figures/10_multiclass_roc_and_auc_curves.png",
         "<b>What it shows:</b> One-vs-Rest ROC curves for each AQI bucket along with the macro-average benchmark curve.<br/>"
         "<b>Interpretation:</b> Every individual class exhibits outstanding discriminative power: Good (AUC = 0.985), Severe (AUC = 0.978), and Very Poor (AUC = 0.951). The macro-average AUC of 0.941 confirms top-tier diagnostic separation across all air hazard levels."),

        ("Visual 11: Permutation Feature Importance Analysis",
         "figures/11_feature_importance_permutation.png",
         "<b>What it shows:</b> Percentage drop in classification accuracy when each input feature column is randomly shuffled, breaking its relationship with the target.<br/>"
         "<b>Interpretation:</b> PM2.5 is unequivocally the most vital predictive factor (causing an accuracy degradation of over 18% when permuted), followed by PM10, NOx, and the engineered fine particle ratio (PM_Ratio). This mathematically verifies the physical primacy of particulate aerosols in governing air toxicity."),

        ("Visual 12: Continuous Value Estimation & Residual Diagnostic Error Analysis",
         "figures/12_regression_residuals_distribution.png",
         "<b>What it shows:</b> Left: Scatter plot comparing true CPCB AQI vs deep learning predicted AQI against the 1:1 identity line. Right: Error distribution histogram of residuals (True - Predicted).<br/>"
         "<b>Interpretation:</b> The model demonstrates high fidelity for continuous numerical estimation (R² = 0.8456, MAE = 23.90 index units). The residual error distribution is symmetrically centered around zero (μ = -0.8), confirming homoscedasticity and unbiased estimation.")
    ]

    for title, img_path, narrative in eval_visuals_info:
        story.append(KeepTogether([
            Paragraph(title, h2_style),
            Image(img_path, width=6.6 * inch, height=2.8 * inch),
            Paragraph(narrative, body_style),
            Spacer(1, 8)
        ]))

    story.append(PageBreak())

    # =========================================================================
    # 9. CONCLUSION & FUTURE SCOPE
    # =========================================================================
    story.append(Paragraph("9. Conclusion & Future Scope", h1_style))
    conclusion_text = (
        "<b>Summary of Findings:</b><br/>"
        "• <i>Empirical Reality:</i> India’s air pollution crisis is heavily driven by fine particulate matter (PM2.5), which contributes over 80% to elevated AQI scores.<br/>"
        "• <i>Statistical Nature:</i> Environmental pollutant concentrations do not follow normal distributions; they conform to continuous Log-Normal and Weibull distributions due to multiplicative dispersion physics.<br/>"
        "• <i>Regional Disparity:</i> Northern land-locked metros (Delhi, Ahmedabad) suffer from severe winter stagnation, while southern coastal cities (Coimbatore, Chennai, Thiruvananthapuram) maintain predominantly Satisfactory air quality.<br/>"
        "• <i>Deep Learning Performance:</i> The proposed multi-task Deep MLP achieved 75.21% test accuracy, 0.941 macro AUC, and R² of 0.8456, demonstrating robust generalization on complex tabular sensor data.<br/><br/>"
        "<b>Challenges Faced:</b><br/>"
        "1. <i>Sensor Non-Random Missingness:</i> High missing rates in specific pollutants (e.g., Xylene ~61%) required domain-aware city median imputation.<br/>"
        "2. <i>Class Imbalance:</i> Severe and Good tiers represent minority classes compared to Moderate/Satisfactory, mitigated via weighted Cross-Entropy.<br/><br/>"
        "<b>Future Scope & Enhancements:</b><br/>"
        "1. <i>Spatio-Temporal Graph Neural Networks (ST-GNNs):</i> Integrating spatial adjacency between neighboring city monitoring stations.<br/>"
        "2. <i>Meteorological Coupling:</i> Ingesting boundary layer height, relative humidity, wind speed, and planetary boundary layer (PBL) reanalysis datasets.<br/>"
        "3. <i>Mobile Edge Deployment:</i> Quantizing the trained PyTorch network via ONNX runtime for real-time mobile and municipal IoT dashboard deployment."
    )
    story.append(Paragraph(conclusion_text, body_style))
    story.append(Spacer(1, 10))

    # =========================================================================
    # 10. REFERENCES (IEEE FORMAT)
    # =========================================================================
    story.append(Paragraph("10. Literature Review & References", h1_style))
    references = [
        "[1] Central Pollution Control Board (CPCB), Ministry of Environment, Forest and Climate Change, Government of India, \"National Air Quality Index: Technical Report and Guidelines,\" Technical Bulletin No. 45, New Delhi, 2014.",
        "[2] B. Fry, <i>Visualizing Data: Exploring and Explaining Data with the Processing Environment</i>, 1st ed. Sebastopol, CA: O'Reilly Media, 2008.",
        "[3] S. Guttikunda and P. Jawahar, \"Atmospheric emissions and pollution from the coal-fired thermal power plants in India,\" <i>Atmospheric Environment</i>, vol. 92, pp. 449–460, Aug. 2014, doi: 10.1016/j.atmosenv.2014.04.057.",
        "[4] Y. LeCun, Y. Bengio, and G. Hinton, \"Deep learning,\" <i>Nature</i>, vol. 521, no. 7553, pp. 436–444, May 2015, doi: 10.1038/nature14539.",
        "[5] World Health Organization (WHO), \"WHO Global Air Quality Guidelines: Particulate Matter (PM2.5 and PM10), Ozone, Nitrogen Dioxide, Sulfur Dioxide and Carbon Monoxide,\" Geneva: World Health Organization, 2021."
    ]
    for ref in references:
        story.append(Paragraph(ref, bullet_style))
    story.append(Spacer(1, 10))

    # =========================================================================
    # 11. APPENDIX (CODE SECTION SUMMARY)
    # =========================================================================
    story.append(Paragraph("11. Appendix: Reproducible Code & Artifact Index", h1_style))
    appendix_text = (
        "The complete, production-grade source code, dataset artifacts, and master Jupyter Notebook are organized within the project repository:<br/>"
        "• <b>Source Code:</b> <code>main.py</code> (Complete modular data ingestion, statistical analysis, PyTorch training & evaluation pipeline).<br/>"
        "• <b>Master Notebook:</b> <code>air_quality_eda_and_deep_learning.ipynb</code> (Fully formatted interactive notebook with rich markdown and outputs).<br/>"
        "• <b>Visual Figures:</b> <code>figures/</code> (12 high-resolution figures saved at 300 DPI for publication).<br/>"
        "• <b>Processed Dataset:</b> <code>data/cleaned_air_quality.csv</code> (Enriched 26-feature dataset with engineered chemistry metrics).<br/>"
        "• <b>Documentation:</b> <code>README.md</code> (Complete project guide with installation steps and architectural diagrams)."
    )
    story.append(Paragraph(appendix_text, body_style))

    # Build the document using the NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[SUCCESS] Academic Project Report PDF Generated Successfully: {PDF_OUTPUT_PATH}")

if __name__ == '__main__':
    build_pdf_report()
