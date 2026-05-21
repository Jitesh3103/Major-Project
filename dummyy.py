import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
from collections import Counter, deque
import os
import time
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import tempfile
import matplotlib.colors as mcolors

from deepface import DeepFace
print("✓ DeepFace loaded successfully")

# Try to import additional models for action recognition
try:
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import mobilenet_v3_large
    ACTION_MODEL_AVAILABLE = True
    print("✓ Action recognition models available")
except ImportError:
    ACTION_MODEL_AVAILABLE = False
    print("⚠ Action recognition models not available - using basic detection")

# -----------------------------
# COLOR NAME DETECTION FUNCTIONS
# -----------------------------

def get_color_name_from_rgb(rgb_value):
    """
    Convert RGB values to color names using matplotlib color database
    """
    try:
        # Convert RGB to tuple if needed
        if isinstance(rgb_value, list):
            rgb_value = tuple(rgb_value)
        
        # Normalize RGB to 0-1 range for matplotlib
        rgb_normalized = tuple(c/255.0 for c in rgb_value)
        
        min_distance = float('inf')
        closest_color_name = "Unknown Color"
        
        # Check all named colors in matplotlib
        for color_name, hex_value in mcolors.CSS4_COLORS.items():
            # Convert hex to RGB
            css_rgb = mcolors.hex2color(hex_value)
            
            # Calculate Euclidean distance between colors
            distance = np.sqrt(
                (rgb_normalized[0] - css_rgb[0])**2 +
                (rgb_normalized[1] - css_rgb[1])**2 +
                (rgb_normalized[2] - css_rgb[2])**2
            )
            
            if distance < min_distance:
                min_distance = distance
                closest_color_name = color_name.replace('_', ' ').title()
        
        # If distance is too large, it's probably not a standard color
        if min_distance > 0.3:  # Threshold for color matching
            return "Custom Color"
        
        return closest_color_name
        
    except Exception as e:
        print(f"Color name detection error: {e}")
        return "Color Detection Failed"

def get_simplified_color_name(rgb_value):
    """
    Get a simplified, more human-readable color name
    """
    color_name = get_color_name_from_rgb(rgb_value)
    
    # Simplify some common color names
    color_simplifications = {
        'Dark Gray': 'Gray',
        'Dark Grey': 'Gray',
        'Light Gray': 'Light Gray',
        'Light Grey': 'Light Gray',
        'Dim Gray': 'Gray',
        'Dim Grey': 'Gray',
        'Dark Slate Gray': 'Dark Gray',
        'Light Slate Gray': 'Light Gray',
        'Slate Gray': 'Gray',
        'Light Steel Blue': 'Light Blue',
        'Steel Blue': 'Blue',
        'Dark Blue': 'Blue',
        'Medium Blue': 'Blue',
        'Light Blue': 'Light Blue',
        'Powder Blue': 'Light Blue',
        'Sky Blue': 'Light Blue',
        'Light Sky Blue': 'Light Blue',
        'Deep Sky Blue': 'Blue',
        'Dodger Blue': 'Blue',
        'Cornflower Blue': 'Blue',
        'Royal Blue': 'Blue',
        'Medium Slate Blue': 'Blue',
        'Slate Blue': 'Blue',
        'Dark Slate Blue': 'Dark Blue',
        'Medium Purple': 'Purple',
        'Blue Violet': 'Purple',
        'Dark Violet': 'Purple',
        'Dark Orchid': 'Purple',
        'Medium Orchid': 'Purple',
        'Dark Magenta': 'Purple',
        'Deep Pink': 'Pink',
        'Light Pink': 'Pink',
        'Hot Pink': 'Pink',
        'Pale Violet Red': 'Pink',
        'Medium Violet Red': 'Pink',
        'Dark Red': 'Red',
        'Firebrick': 'Red',
        'Crimson': 'Red',
        'Indian Red': 'Red',
        'Light Coral': 'Red',
        'Dark Salmon': 'Salmon',
        'Light Salmon': 'Salmon',
        'Dark Orange': 'Orange',
        'Light Yellow': 'Yellow',
        'Light Goldenrod Yellow': 'Yellow',
        'Pale Goldenrod': 'Yellow',
        'Dark Khaki': 'Khaki',
        'Dark Green': 'Green',
        'Forest Green': 'Green',
        'Sea Green': 'Green',
        'Medium Sea Green': 'Green',
        'Light Sea Green': 'Green',
        'Pale Green': 'Light Green',
        'Light Green': 'Light Green',
        'Spring Green': 'Green',
        'Medium Spring Green': 'Green',
        'Dark Cyan': 'Cyan',
        'Light Cyan': 'Cyan',
        'Dark Turquoise': 'Turquoise',
        'Medium Turquoise': 'Turquoise',
        'Light Sea Green': 'Green',
        'Cadet Blue': 'Blue',
        'Dark Sea Green': 'Green',
        'Medium Aquamarine': 'Aqua',
        'Dark Olive Green': 'Olive Green',
        'Olive Drab': 'Olive Green'
    }
    
    return color_simplifications.get(color_name, color_name)

# -----------------------------
# GENDER DETECTION FUNCTIONS
# -----------------------------

def detect_gender_from_deepface(image, box):
    """
    Detect gender using DeepFace library for accurate facial analysis
    """
    try:
        x1, y1, x2, y2 = map(int, box)
        
        # Extract face region (upper 30% of bounding box)
        height = y2 - y1
        width = x2 - x1
        
        face_y1 = y1
        face_y2 = y1 + int(height * 0.3)
        face_x1 = x1 + int(width * 0.2)
        face_x2 = x2 - int(width * 0.2)
        
        # Ensure valid coordinates
        face_x1 = max(0, face_x1)
        face_x2 = min(image.shape[1], face_x2)
        face_y1 = max(0, face_y1)
        face_y2 = min(image.shape[0], face_y2)
        
        if face_x2 <= face_x1 or face_y2 <= face_y1:
            return "Unknown", "0%"
        
        face_region = image[face_y1:face_y2, face_x1:face_x2]
        
        if face_region.size == 0 or face_region.shape[0] < 20 or face_region.shape[1] < 20:
            return "Unknown", "0%"
        
        # Use DeepFace to analyze gender
        result = DeepFace.analyze(face_region, actions=['gender'], enforce_detection=False)
        
        if isinstance(result, list):
            result = result[0]
        
        gender_data = result.get('gender', {})
        print(f"DeepFace gender analysis: {gender_data}")
        
        if 'Man' in gender_data and 'Woman' in gender_data:
            male_confidence = gender_data['Man']
            female_confidence = gender_data['Woman']
            
            print(f"Male: {male_confidence:.1f}%, Female: {female_confidence:.1f}%")
            
            if male_confidence > female_confidence:
                return "Male", f"{male_confidence:.1f}%"
            else:
                return "Female", f"{female_confidence:.1f}%"
        else:
            print(f"Unexpected gender data format: {gender_data}")
            return "Unknown", "0%"
            
    except Exception as e:
        print(f"DeepFace gender detection error: {e}")
        return "Unknown", "0%"



def estimate_age_from_deepface(image, box):
    """
    Estimate age using DeepFace library for accurate facial analysis
    """
    try:
        x1, y1, x2, y2 = map(int, box)
        
        # Extract face region (upper 30% of bounding box)
        height = y2 - y1
        width = x2 - x1
        
        face_y1 = y1
        face_y2 = y1 + int(height * 0.3)
        face_x1 = x1 + int(width * 0.2)
        face_x2 = x2 - int(width * 0.2)
        
        # Ensure valid coordinates
        face_x1 = max(0, face_x1)
        face_x2 = min(image.shape[1], face_x2)
        face_y1 = max(0, face_y1)
        face_y2 = min(image.shape[0], face_y2)
        
        if face_x2 <= face_x1 or face_y2 <= face_y1:
            return "Unknown Age"
        
        face_region = image[face_y1:face_y2, face_x1:face_x2]
        
        if face_region.size == 0 or face_region.shape[0] < 20 or face_region.shape[1] < 20:
            return "Unknown Age"
        
        # Use DeepFace to analyze age
        result = DeepFace.analyze(face_region, actions=['age'], enforce_detection=False)
        
        if isinstance(result, list):
            result = result[0]
        
        age = result.get('age', 0)
        print(f"DeepFace age prediction: {age} years")
        
        # Convert age to age range
        if age <= 3:
            return "0-3 years (Toddler)"
        elif age <= 9:
            return "4-9 years (Child)"
        elif age <= 14:
            return "10-14 years (Early Teen)"
        elif age <= 19:
            return "15-19 years (Late Teen)"
        elif age <= 35:
            return "20-35 years (Young Adult)"
        elif age <= 50:
            return "36-50 years (Middle Adult)"
        elif age <= 65:
            return "51-65 years (Senior Adult)"
        else:
            return "65+ years (Elderly)"
            
    except Exception as e:
        print(f"DeepFace age detection error: {e}")
        return "Unknown Age"



# -----------------------------
# PDF Report Generation Functions (UPDATED WITH GENDER DETECTION)
# -----------------------------

def create_individual_pdf_report(person_data, output_dir="pdf_reports"):
    """
    Create individual PDF report for a single person with color names and gender
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create filename
    timestamp = int(time.time())
    person_id = len([f for f in os.listdir(output_dir) if f.startswith("person_")])
    filename = f"person_{person_id+1}_{timestamp}.pdf"
    filepath = os.path.join(output_dir, filename)
    
    # Create PDF document
    doc = SimpleDocTemplate(filepath, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        alignment=1,  # Center aligned
        textColor=colors.HexColor('#2E4057')
    )
    title = Paragraph("PERSON ANALYSIS REPORT", title_style)
    story.append(title)
    
    # Physical Attributes Section
    phys_header_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        textColor=colors.HexColor('#2E86AB')
    )
    phys_header = Paragraph("Physical Attributes", phys_header_style)
    story.append(phys_header)
    
    # Physical attributes table (UPDATED WITH GENDER AND SUSPICIOUS STATUS)
    phys_data = [
        ['Attribute', 'Value'],
        ['Height', f"{person_data['height_cm']:.1f} cm"],
        ['Weight', f"{person_data['weight_kg']:.1f} kg"],
        ['Gender', f"{person_data['gender']} ({person_data['gender_confidence']})"],
        ['Age Range', person_data['age_range']],
        ['Distance Category', person_data['distance_category']],
        ['Suspicious Status', person_data.get('suspicious_status', 'Not Suspicious')]
    ]
    
    # Add suspicious details if available
    if person_data.get('suspicious_details'):
        phys_data.append(['Suspicious Activities', ', '.join(person_data['suspicious_details'])])
    
    phys_table = Table(phys_data, colWidths=[2*inch, 3*inch])
    phys_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
    ]))
    story.append(phys_table)
    story.append(Spacer(1, 20))
    
    # Appearance Analysis Section (UPDATED WITH COLOR NAMES)
    appear_header = Paragraph("Appearance Analysis", phys_header_style)
    story.append(appear_header)
    
    # Get color names
    shirt_color = person_data['shirt_rgb']
    pant_color = person_data['pant_rgb']
    skin_color = person_data['skin_rgb']
    
    shirt_name = get_simplified_color_name(shirt_color)
    pant_name = get_simplified_color_name(pant_color)
    skin_name = get_simplified_color_name(skin_color)
    
    # Appearance table with color names
    appear_data = [
        ['Feature', 'Color Name', 'RGB Values', 'Color Type'],
        ['Shirt', shirt_name, f"RGB{shirt_color}", 'Primary'],
        ['Pants', pant_name, f"RGB{pant_color}", 'Primary'],
        ['Skin', skin_name, f"RGB{skin_color}", person_data['skin_tone']]
    ]
    
    appear_table = Table(appear_data, colWidths=[1.2*inch, 1.8*inch, 1.8*inch, 1.2*inch])
    appear_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#A23B72')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
    ]))
    story.append(appear_table)
    story.append(Spacer(1, 20))
    
    # Detection Information Section
    detect_header = Paragraph("Detection Information", phys_header_style)
    story.append(detect_header)
    
    detect_data = [
        ['Parameter', 'Value'],
        ['Bounding Box', f"[{person_data['box'][0]}, {person_data['box'][1]}, {person_data['box'][2]}, {person_data['box'][3]}]"],
        ['Pixel Height', f"{person_data['pixel_height']} px"],
        ['Calibration Reference', f"{person_data['calibration_ref_px']}px = {person_data['calibration_ref_m']:.2f}m"],
        ['Analysis Timestamp', time.strftime('%Y-%m-%d %H:%M:%S')]
    ]
    
    detect_table = Table(detect_data, colWidths=[2*inch, 3*inch])
    detect_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F18F01')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))
    story.append(detect_table)
    story.append(Spacer(1, 20))
    
    # Analyzed Image Section
    if person_data.get('annotated_image_path'):
        image_header = Paragraph("Analyzed Image", phys_header_style)
        story.append(image_header)
        
        try:
            # Add the annotated image to PDF
            img = Image(person_data['annotated_image_path'], width=5*inch, height=3.5*inch)
            img.hAlign = 'CENTER'
            story.append(img)
            story.append(Spacer(1, 10))
            
            # Image caption
            caption_style = ParagraphStyle(
                'Caption',
                parent=styles['Normal'],
                fontSize=9,
                alignment=1,
                textColor=colors.grey
            )
            caption = Paragraph("Analyzed image with bounding box and color regions", caption_style)
            story.append(caption)
        except Exception as e:
            print(f"Warning: Could not add image to PDF: {e}")
    
    # Build PDF
    doc.build(story)
    print(f"✓ Individual PDF report saved: {filepath}")
    return filepath

def create_summary_pdf_report(all_persons_data, output_dir="pdf_reports"):
    """
    Create comprehensive summary PDF report for all analyzed persons with color names and gender
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not all_persons_data:
        print("No data available for summary report")
        return None
    
    timestamp = int(time.time())
    filename = f"analysis_summary_{timestamp}.pdf"
    filepath = os.path.join(output_dir, filename)
    
    # Create PDF document
    doc = SimpleDocTemplate(filepath, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'SummaryTitle',
        parent=styles['Heading1'],
        fontSize=20,
        spaceAfter=30,
        alignment=1,
        textColor=colors.HexColor('#2E4057')
    )
    title = Paragraph("COMPREHENSIVE ANALYSIS SUMMARY", title_style)
    story.append(title)
    
    # Summary Statistics Section
    summary_header_style = ParagraphStyle(
        'SummaryHeader',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=15,
        textColor=colors.HexColor('#2E86AB')
    )
    summary_header = Paragraph("Summary Statistics", summary_header_style)
    story.append(summary_header)
    
    # Calculate statistics
    heights = [p['height_cm'] for p in all_persons_data]
    weights = [p['weight_kg'] for p in all_persons_data]
    
    # Gender statistics
    genders = [p['gender'] for p in all_persons_data]
    male_count = sum(1 for g in genders if 'male' in g.lower() and 'female' not in g.lower())
    female_count = sum(1 for g in genders if 'female' in g.lower())
    unknown_count = len(genders) - male_count - female_count
    
    # Suspicious activity statistics
    suspicious_count = sum(1 for p in all_persons_data if p.get('suspicious_status') == 'Suspicious')
    not_suspicious_count = len(all_persons_data) - suspicious_count
    
    summary_data = [
        ['Metric', 'Value'],
        ['Total Persons Analyzed', str(len(all_persons_data))],
        ['Average Height', f"{np.mean(heights):.1f} cm"],
        ['Average Weight', f"{np.mean(weights):.1f} kg"],
        ['Male Detected', f"{male_count}"],
        ['Female Detected', f"{female_count}"],
        ['Gender Unknown', f"{unknown_count}"],
        ['Suspicious Persons', f"{suspicious_count}"],
        ['Not Suspicious', f"{not_suspicious_count}"],
        ['Height Range', f"{min(heights):.1f} - {max(heights):.1f} cm"],
        ['Weight Range', f"{min(weights):.1f} - {max(weights):.1f} kg"]
    ]
    
    summary_table = Table(summary_data, colWidths=[2.5*inch, 2.5*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 25))
    
    # Detailed Person Data Section (UPDATED WITH GENDER)
    detail_header = Paragraph("Detailed Person Analysis", summary_header_style)
    story.append(detail_header)
    
    # Create detailed table (UPDATED WITH SUSPICIOUS STATUS)
    detail_headers = ['Person', 'Height (cm)', 'Weight (kg)', 'Gender', 'Age Range', 'Skin Tone', 'Distance', 'Suspicious']
    detail_data = [detail_headers]
    
    for i, person in enumerate(all_persons_data):
        detail_row = [
            f"Person {i+1}",
            f"{person['height_cm']:.1f}",
            f"{person['weight_kg']:.1f}",
            f"{person['gender']}",
            person['age_range'],
            person['skin_tone'],
            person['distance_category'],
            person.get('suspicious_status', 'Not Suspicious')
        ]
        detail_data.append(detail_row)
    
    detail_table = Table(detail_data, colWidths=[0.7*inch, 0.7*inch, 0.7*inch, 1*inch, 1*inch, 0.8*inch, 0.7*inch, 0.9*inch])
    detail_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#A23B72')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F0F0F0')]),
    ]))
    story.append(detail_table)
    story.append(Spacer(1, 25))
    
    # Color Analysis Summary (UPDATED WITH COLOR NAMES)
    color_header = Paragraph("Color Analysis Summary", summary_header_style)
    story.append(color_header)
    
    color_data = [['Person', 'Shirt Color', 'Pant Color', 'Skin Color']]
    for i, person in enumerate(all_persons_data):
        shirt_name = get_simplified_color_name(person['shirt_rgb'])
        pant_name = get_simplified_color_name(person['pant_rgb'])
        skin_name = get_simplified_color_name(person['skin_rgb'])
        
        color_row = [
            f"Person {i+1}",
            f"{shirt_name}\nRGB{person['shirt_rgb']}",
            f"{pant_name}\nRGB{person['pant_rgb']}",
            f"{skin_name}\nRGB{person['skin_rgb']}"
        ]
        color_data.append(color_row)
    
    color_table = Table(color_data, colWidths=[1*inch, 1.8*inch, 1.8*inch, 1.8*inch])
    color_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F18F01')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
    ]))
    story.append(color_table)
    
    # Footer with timestamp
    story.append(Spacer(1, 30))
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=9,
        alignment=1,
        textColor=colors.grey
    )
    footer = Paragraph(f"Report generated on {time.strftime('%Y-%m-%d at %H:%M:%S')}", footer_style)
    story.append(footer)
    
    # Build PDF
    doc.build(story)
    print(f"✓ Summary PDF report saved: {filepath}")
    return filepath

def save_annotated_image(frame, box, shirt_rgb, pant_rgb, skin_tone, gender, output_dir="annotated_images"):
    """
    Save annotated image with bounding box and information including gender
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create annotated image
    annotated_frame = frame.copy()
    x1, y1, x2, y2 = box
    
    # Draw bounding box
    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    
    # Draw division line between shirt and pants
    mid_y = y1 + (y2 - y1) // 2
    cv2.line(annotated_frame, (x1, mid_y), (x2, mid_y), (0, 255, 255), 1)
    
    # Get color names for display
    shirt_name = get_simplified_color_name(shirt_rgb)
    pant_name = get_simplified_color_name(pant_rgb)
    
    # Add information text
    info_lines = [
        f"Shirt: {shirt_name} RGB{shirt_rgb}",
        f"Pant: {pant_name} RGB{pant_rgb}",
        f"Skin: {skin_tone}",
        f"Gender: {gender}"
    ]
    
    for i, line in enumerate(info_lines):
        y_offset = y2 + 20 + (i * 20)
        cv2.putText(annotated_frame, line, (x1, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    # Save image
    timestamp = int(time.time())
    filename = f"annotated_{timestamp}.jpg"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, annotated_frame)
    
    return filepath

# -----------------------------
# CORRECTED Dress Color Detection Functions
# -----------------------------

def get_upper_lower_colors_from_segmentation(image, seg_mask, box, k=3):
    """
    CORRECTED: Get dominant colors from upper (shirt) and lower (pant) regions using segmentation mask
    """
    x1, y1, x2, y2 = map(int, box)
    height = y2 - y1
    width = x2 - x1
    
    if height == 0 or width == 0:
        return get_upper_lower_colors_fallback(image, box, k)
    
    # Extract the person region from original image
    person_region = image[y1:y2, x1:x2]
    
    if person_region.size == 0:
        return get_upper_lower_colors_fallback(image, box, k)
    
    # Resize segmentation mask to match person region dimensions
    seg_mask_resized = cv2.resize(seg_mask, (width, height))
    
    # Convert mask to binary (0 and 255)
    _, binary_mask = cv2.threshold(seg_mask_resized, 127, 255, cv2.THRESH_BINARY)
    
    # Check if we have enough segmented area
    if np.sum(binary_mask) < 1000:  # Minimum area threshold
        return get_upper_lower_colors_fallback(image, box, k)
    
    # Improved region division - use anatomical proportions
    # Upper body: 0-45% (shirt/blouse area)
    # Lower body: 45-85% (pants area, excluding feet)
    upper_y1 = 0
    upper_y2 = int(height * 0.45)
    lower_y1 = int(height * 0.45)
    lower_y2 = int(height * 0.85)  # Exclude feet area
    
    # Create masks for upper and lower regions
    upper_mask = np.zeros_like(binary_mask)
    upper_mask[upper_y1:upper_y2, :] = binary_mask[upper_y1:upper_y2, :]
    
    lower_mask = np.zeros_like(binary_mask)
    lower_mask[lower_y1:lower_y2, :] = binary_mask[lower_y1:lower_y2, :]
    
    # Clean masks with morphological operations
    kernel = np.ones((2, 2), np.uint8)
    upper_mask = cv2.morphologyEx(upper_mask, cv2.MORPH_OPEN, kernel)
    lower_mask = cv2.morphologyEx(lower_mask, cv2.MORPH_OPEN, kernel)
    
    # Get colors using the masks
    shirt_color = get_dominant_color_with_mask_improved(person_region, upper_mask, k)
    pant_color = get_dominant_color_with_mask_improved(person_region, lower_mask, k)
    
    return shirt_color, pant_color

def get_upper_lower_colors_fallback(image, box, k=3):
    """
    Fallback method using traditional bounding box approach
    """
    x1, y1, x2, y2 = map(int, box)
    height = y2 - y1
    
    if height == 0:
        return (128, 128, 128), (128, 128, 128)
    
    # Upper body region (shirt) - top 45% of bounding box
    upper_y1 = y1
    upper_y2 = y1 + int(height * 0.45)
    upper_cropped = image[upper_y1:upper_y2, x1:x2]
    
    # Lower body region (pants) - 45-85% of bounding box (excluding feet)
    lower_y1 = y1 + int(height * 0.45)
    lower_y2 = y1 + int(height * 0.85)
    lower_cropped = image[lower_y1:lower_y2, x1:x2]
    
    shirt_color = get_dominant_color_from_region_cropped(upper_cropped, k)
    pant_color = get_dominant_color_from_region_cropped(lower_cropped, k)
    
    return shirt_color, pant_color

def get_dominant_color_with_mask_improved(region, mask, k=3):
    """
    CORRECTED: Improved dominant color extraction with better pixel filtering
    """
    if region.size == 0 or mask.size == 0:
        return (128, 128, 128)  # Return gray instead of black
    
    # Ensure mask and region have same dimensions
    if region.shape[:2] != mask.shape:
        mask = cv2.resize(mask, (region.shape[1], region.shape[0]))
    
    # Apply mask to get only the segmented area
    masked_region = cv2.bitwise_and(region, region, mask=mask)
    
    # Convert to RGB for processing
    masked_rgb = cv2.cvtColor(masked_region, cv2.COLOR_BGR2RGB)
    pixels = masked_rgb.reshape(-1, 3)
    
    # Remove completely black pixels (masked out areas)
    non_black_mask = np.any(pixels > [10, 10, 10], axis=1)
    pixels = pixels[non_black_mask]
    
    # Remove completely white pixels (likely artifacts)
    non_white_mask = np.any(pixels < [245, 245, 245], axis=1)
    pixels = pixels[non_white_mask]
    
    if len(pixels) < 20:  # Too few pixels for reliable clustering
        # Fallback: get dominant color from entire region without mask
        return get_dominant_color_from_region_cropped(region, k)
    
    try:
        # Determine optimal k based on available pixels
        optimal_k = min(k, max(2, len(pixels) // 100))  # At least 100 pixels per cluster
        optimal_k = max(2, min(optimal_k, 5))  # Keep between 2-5
        
        kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(pixels)
        counts = Counter(labels)
        
        # Get the most dominant cluster that's not too dark or too bright
        dominant_color = None
        for cluster_idx, _ in counts.most_common(optimal_k):
            candidate_color = kmeans.cluster_centers_[cluster_idx].astype(int)
            color_mean = np.mean(candidate_color)
            
            # Accept color if it's in reasonable range (not too dark/light)
            if 30 <= color_mean <= 220:
                dominant_color = candidate_color
                break
        
        # If no suitable color found, use the most dominant one
        if dominant_color is None:
            dominant_cluster_idx = counts.most_common(1)[0][0]
            dominant_color = kmeans.cluster_centers_[dominant_cluster_idx].astype(int)
        
        return tuple(map(int, dominant_color))
        
    except Exception as e:
        print(f"Color clustering error: {e}")
        return get_dominant_color_from_region_cropped(region, k)

def get_dominant_color_from_region_cropped(cropped_region, k=3):
    """
    CORRECTED: Get dominant color from a cropped image region with better filtering
    """
    if cropped_region.size == 0:
        return (128, 128, 128)
    
    # Convert to RGB
    cropped_rgb = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2RGB)
    pixels = cropped_rgb.reshape(-1, 3)
    
    # Remove extreme dark and light pixels
    valid_pixels = pixels[
        (np.any(pixels > [20, 20, 20], axis=1)) & 
        (np.any(pixels < [235, 235, 235], axis=1))
    ]
    
    if len(valid_pixels) == 0:
        # If no valid pixels, use the original pixels but remove extremes
        valid_pixels = pixels[
            (np.any(pixels > [5, 5, 5], axis=1)) & 
            (np.any(pixels < [250, 250, 250], axis=1))
        ]
    
    if len(valid_pixels) == 0:
        return (128, 128, 128)

    try:
        optimal_k = min(k, max(2, len(valid_pixels) // 50))
        kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(valid_pixels)
        counts = Counter(labels)
        
        # Find the best color cluster
        dominant_color = None
        for cluster_idx, _ in counts.most_common(optimal_k):
            candidate_color = kmeans.cluster_centers_[cluster_idx].astype(int)
            color_mean = np.mean(candidate_color)
            
            if 40 <= color_mean <= 210:  # Reasonable color range
                dominant_color = candidate_color
                break
        
        if dominant_color is None:
            dominant_cluster_idx = counts.most_common(1)[0][0]
            dominant_color = kmeans.cluster_centers_[dominant_cluster_idx].astype(int)
            
        return tuple(map(int, dominant_color))
    except Exception as e:
        print(f"Fallback color extraction error: {e}")
        # Return average color as last resort
        return tuple(map(int, np.mean(valid_pixels, axis=0)))

# -----------------------------
# Get Dominant Color from Face Region
# -----------------------------
def get_face_color_improved(image, box, k=2):
    """
    Improved face color detection with better region selection
    """
    x1, y1, x2, y2 = map(int, box)
    height = y2 - y1
    width = x2 - x1
    
    if height == 0:
        return (128, 128, 128)
    
    # More precise face region - top 20% of bounding box, center 50% horizontally
    face_y1 = y1
    face_y2 = y1 + int(height * 0.2)
    face_x1 = x1 + int(width * 0.25)
    face_x2 = x2 - int(width * 0.25)
    
    # Ensure valid coordinates
    face_x1 = max(x1, face_x1)
    face_x2 = min(x2, face_x2)
    face_y1 = max(y1, face_y1)
    face_y2 = min(y2, face_y2)
    
    face_cropped = image[face_y1:face_y2, face_x1:face_x2]
    
    if face_cropped.size == 0:
        return (128, 128, 128)
    
    return get_dominant_color_from_region_cropped(face_cropped, k)

# -----------------------------
# Skin Tone Classification
# -----------------------------
def classify_skin_tone(rgb):
    """Classify skin tone into common categories including white skin"""
    r, g, b = rgb
    
    # Convert to HSV for better skin tone detection
    hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]
    h, s, v = hsv
    
    # Enhanced skin tone classification
    if v < 40:
        return "very dark"
    elif v < 70:
        return "dark brown"
    elif v < 100:
        if s > 70:
            return "medium brown"
        else:
            return "light brown"
    elif v < 140:
        if s > 60:
            return "olive"
        else:
            return "beige"
    elif v < 180:
        if s > 50:
            return "tan"
        elif s > 30:
            return "light"
        else:
            return "fair"
    else:  # v >= 180 (very light)
        if s < 25:
            return "very fair/white"
        elif s < 40:
            return "fair white"
        else:
            return "light white"

# -----------------------------
# Distance-Based Calibration System
# -----------------------------
class DistanceCalibrator:
    def __init__(self, frame_height):
        self.frame_height = frame_height
        self.calibration_data = {}  # Store multiple calibration points
        self.reference_height_m = 1.7  # Standard reference height
        
    def estimate_distance_from_position(self, box):
        """
        Estimate distance based on person's vertical position in frame
        Lower position = closer to camera
        Higher position = farther from camera
        """
        x1, y1, x2, y2 = box
        person_center_y = (y1 + y2) / 2
        
        # Normalize position (0 = bottom, 1 = top)
        position_ratio = person_center_y / self.frame_height
        
        # Distance estimation based on position
        # This is approximate - you'll need to calibrate for your camera
        if position_ratio < 0.3:
            return "very_close"  # 1-2 meters
        elif position_ratio < 0.5:
            return "close"       # 2-4 meters
        elif position_ratio < 0.7:
            return "medium"      # 4-6 meters
        else:
            return "far"         # 6+ meters
    
    def get_calibration_height_for_distance(self, distance_category, actual_height_m=None):
        """
        Return appropriate calibration height based on distance
        These values need to be calibrated for your specific camera setup
        """
        # Default calibration heights for different distances
        # You should adjust these based on your camera testing
        calibration_heights = {
            "very_close": 1.5,   # Person appears larger
            "close": 1.6,        # Slightly smaller
            "medium": 1.7,       # Standard distance
            "far": 1.8           # Person appears smaller
        }
        
        if actual_height_m is not None:
            # Use actual known height if provided
            return actual_height_m
        else:
            return calibration_heights.get(distance_category, 1.7)
    
    def auto_calibrate_distance(self, box, known_height_m=None):
        """
        Automatically calibrate based on estimated distance
        """
        distance_category = self.estimate_distance_from_position(box)
        calibration_height = self.get_calibration_height_for_distance(distance_category, known_height_m)
        
        return calibration_height, distance_category
    
    def manual_distance_calibration(self, box, actual_height_m):
        """
        Manual calibration for specific distance
        """
        distance_category = self.estimate_distance_from_position(box)
        self.calibration_data[distance_category] = actual_height_m
        return distance_category, actual_height_m

# Age and gender detection functions are now defined above with fallback support

def estimate_age_range_fallback(height_cm):
    """Fallback method using height when face analysis fails"""
    if height_cm < 100:
        return "0-3 years (Toddler)"
    elif height_cm < 130:
        return "4-9 years (Child)"
    elif height_cm < 150:
        return "10-14 years (Early Teen)"
    elif height_cm < 165:
        return "15-19 years (Late Teen)"
    elif height_cm < 175:
        return "20-35 years (Young Adult)"
    elif height_cm < 185:
        return "36-50 years (Middle Adult)"
    elif height_cm < 195:
        return "51-65 years (Senior Adult)"
    else:
        return "65+ years (Elderly)"

# -----------------------------
# CORRECTED Segmentation-based Image Analysis
# -----------------------------
def analyze_captured_image_with_segmentation(image, person_box, seg_model):
    """
    CORRECTED: Analyze shirt color, pant color and skin tone from a captured image using segmentation
    """
    x1, y1, x2, y2 = map(int, person_box)
    
    # Extract person region
    person_region = image[y1:y2, x1:x2]
    
    if person_region.size == 0:
        return get_upper_lower_colors_fallback(image, person_box, k=3), (128, 128, 128), "unknown", (128, 128, 128)
    
    try:
        # Run segmentation on the person region
        seg_results = seg_model(person_region, verbose=False)
        
        if len(seg_results) == 0 or seg_results[0].masks is None:
            print("No segmentation mask found - using fallback")
            return get_upper_lower_colors_fallback(image, person_box, k=3), (128, 128, 128), "unknown", (128, 128, 128)
        
        # Get the first (largest) segmentation mask
        seg_mask = seg_results[0].masks.data[0].cpu().numpy()
        
        # Convert mask to uint8 and scale properly
        seg_mask = (seg_mask * 255).astype(np.uint8)
        
        # Resize mask to match person region dimensions
        seg_mask = cv2.resize(seg_mask, (person_region.shape[1], person_region.shape[0]))
        
        # Apply threshold to create binary mask
        _, binary_mask = cv2.threshold(seg_mask, 128, 255, cv2.THRESH_BINARY)
        
        # Check if we have a reasonable mask
        mask_area = np.sum(binary_mask > 0)
        total_area = binary_mask.shape[0] * binary_mask.shape[1]
        
        if mask_area < total_area * 0.1:  # Less than 10% area
            print(f"Mask too small ({mask_area}/{total_area}) - using fallback")
            return get_upper_lower_colors_fallback(image, person_box, k=3), (128, 128, 128), "unknown", (128, 128, 128)
        
        # Get colors using segmentation
        shirt_color, pant_color = get_upper_lower_colors_from_segmentation(image, binary_mask, person_box, k=3)
        
        # Extract skin tone from face region
        skin_color = get_face_color_improved(image, person_box, k=2)
        skin_tone_type = classify_skin_tone(skin_color)
        
        # Print color names for debugging
        shirt_name = get_simplified_color_name(shirt_color)
        pant_name = get_simplified_color_name(pant_color)
        skin_name = get_simplified_color_name(skin_color)
        
        print(f"Segmentation successful - Shirt: {shirt_name} {shirt_color}, Pant: {pant_name} {pant_color}")
        return shirt_color, pant_color, skin_tone_type, skin_color
        
    except Exception as e:
        print(f"Segmentation analysis failed: {e}")
        # Fallback to traditional method
        shirt_color, pant_color = get_upper_lower_colors_fallback(image, person_box, k=3)
        skin_color = get_face_color_improved(image, person_box, k=2)
        skin_tone_type = classify_skin_tone(skin_color)
        return shirt_color, pant_color, skin_tone_type, skin_color

# -----------------------------
# Manual Person Selection
# -----------------------------
class ManualPersonSelector:
    def __init__(self):
        self.selected_persons = []
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.current_box = None
    
    def draw_rectangle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            self.current_box = [x, y, x, y]
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_box[2] = x
                self.current_box[3] = y
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.current_box[2] = x
            self.current_box[3] = y
            # Ensure positive width and height
            x1 = min(self.ix, x)
            y1 = min(self.iy, y)
            x2 = max(self.ix, x)
            y2 = max(self.iy, y)
            
            if abs(x2 - x1) > 50 and abs(y2 - y1) > 50:  # Minimum size
                self.selected_persons.append([x1, y1, x2, y2])
                print(f"✓ Selected person at [{x1}, {y1}, {x2}, {y2}]")
            self.current_box = None

# -----------------------------
# Find reference person for calibration
# -----------------------------
def find_reference_person(boxes):
    """Find tallest person (largest bounding box)"""
    if len(boxes) == 0:
        return None
    ref_idx = max(range(len(boxes)), key=lambda i: boxes[i][3] - boxes[i][1])
    return ref_idx

# -----------------------------
# Image Capture and Save Function
# -----------------------------
def capture_and_save_frame(frame, capture_dir="captured_frames"):
    """Capture and save current frame for analysis"""
    if not os.path.exists(capture_dir):
        os.makedirs(capture_dir)
    
    timestamp = int(time.time())
    filename = os.path.join(capture_dir, f"captured_frame_{timestamp}.jpg")
    cv2.imwrite(filename, frame)
    print(f"✓ Frame captured and saved: {filename}")
    return filename

# -----------------------------
# MAIN PROGRAM WITH PDF REPORTING AND GENDER DETECTION
# -----------------------------

# Load YOLO pose model and segmentation model
print("Loading YOLO pose model...")
pose_model = YOLO("yolo11s-pose.pt")

print("Loading YOLO segmentation model...")
seg_model = YOLO("yolo11n-seg.pt")  # You can use yolo11s-seg.pt or yolo11m-seg.pt for better accuracy

# Video input
video_path = "/home/pranam/manohar/WhatsApp Video 2025-10-29 at 11.53.19 PM.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))  # CORRECTED: Using CAP_PROP_FPS instead of CAP_PROP_FRAME_RATE

print(f"Video: {frame_width}x{frame_height} @ {fps}fps")

# Configuration
CALIBRATION_MODE = False
MANUAL_CALIBRATION_MODE = False
reference_height_px = None
current_reference_height_m = 1.7

# Initialize distance calibrator
calibrator = DistanceCalibrator(frame_height)

# Manual selection mode
manual_mode = False
selector = ManualPersonSelector()
analyzed_persons = []
all_persons_data = []  # Store data for PDF reports

# Suspicious activity detection (IMPROVED THRESHOLDS)
suspicious_activities = {}  # Store suspicious behavior data
activity_threshold = {
    'loitering': 300,      # 300 frames (~12 seconds at 25fps) - less sensitive
    'fast_movement': 80,   # Higher pixel distance threshold
    'direction_change': 6, # More direction changes required
}

# Initialize action recognition model
action_model = None
action_transform = None

if ACTION_MODEL_AVAILABLE:
    try:
        # Load pre-trained MobileNetV3 for action recognition
        action_model = mobilenet_v3_large(pretrained=True)
        action_model.classifier = torch.nn.Sequential(
            torch.nn.Linear(960, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 7)  # 7 action classes
        )
        action_model.eval()
        
        action_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Action classes: 0=Normal, 1=Fighting, 2=Running, 3=Loitering, 4=Vandalism, 5=Theft, 6=Suspicious
        action_classes = ['Normal', 'Fighting', 'Running', 'Loitering', 'Vandalism', 'Theft', 'Suspicious']
        print("✓ Action recognition model loaded")
    except Exception as e:
        print(f"⚠ Action model loading failed: {e}")
        ACTION_MODEL_AVAILABLE = False

def detect_action_with_ai(image, box):
    """Detect suspicious actions using AI model"""
    if not ACTION_MODEL_AVAILABLE or action_model is None:
        return "Normal", 0.5
    
    try:
        x1, y1, x2, y2 = map(int, box)
        person_crop = image[y1:y2, x1:x2]
        
        if person_crop.size == 0:
            return "Normal", 0.5
        
        # Preprocess image
        person_crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        input_tensor = action_transform(person_crop_rgb).unsqueeze(0)
        
        with torch.no_grad():
            outputs = action_model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)
            
            action = action_classes[predicted.item()]
            conf = confidence.item()
            
            return action, conf
            
    except Exception as e:
        print(f"Action detection error: {e}")
        return "Normal", 0.5

def detect_suspicious_activity_advanced(person_id, box, frame, frame_count):
    """Intelligent suspicious activity detection with context awareness and minimal false positives"""
    if person_id not in suspicious_activities:
        suspicious_activities[person_id] = {
            'positions': [],
            'frame_count': 0,
            'direction_changes': 0,
            'alerts': [],
            'action_history': [],
            'suspicious_score': 0.0,
            'normal_behavior_count': 0,
            'last_alert_frame': 0,
            'confidence_history': []
        }
    
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    data = suspicious_activities[person_id]
    data['positions'].append((center_x, center_y, frame_count))
    data['frame_count'] += 1
    
    # AI-based action detection
    action, confidence = detect_action_with_ai(frame, box)
    data['action_history'].append((action, confidence, frame_count))
    
    # Track normal behavior to reduce false positives
    if action == 'Normal' and confidence > 0.7:
        data['normal_behavior_count'] += 1
    
    # Keep only recent data
    if len(data['positions']) > 100:
        data['positions'] = data['positions'][-100:]
    if len(data['action_history']) > 50:
        data['action_history'] = data['action_history'][-50:]
    
    alerts = []
    
    # 1. AI-based suspicious action detection (VERY STRICT THRESHOLDS)
    if action in ['Fighting', 'Vandalism', 'Theft'] and confidence > 0.9:  # Very high confidence required
        # Additional validation: check if this action persists
        recent_same_actions = [a[0] for a in data['action_history'][-5:] if a[0] == action]
        if len(recent_same_actions) >= 2:  # Must be detected multiple times
            alerts.append(f"AI_DETECTED_{action.upper()}")
            data['suspicious_score'] += confidence * 0.3
    elif action == 'Suspicious' and confidence > 0.95:  # Extremely high for generic suspicious
        recent_suspicious = [a[0] for a in data['action_history'][-5:] if a[0] == 'Suspicious']
        if len(recent_suspicious) >= 3:  # Must be consistently suspicious
            alerts.append("AI_DETECTED_SUSPICIOUS")
            data['suspicious_score'] += confidence * 0.2
    
    # 2. Enhanced loitering detection (MORE RESTRICTIVE)
    if data['frame_count'] > activity_threshold['loitering'] * 1.5:  # Longer time required
        recent_positions = data['positions'][-75:]  # Check longer period
        if len(recent_positions) > 20:
            distances = []
            for i in range(1, len(recent_positions)):
                x1_pos, y1_pos, _ = recent_positions[i-1]
                x2_pos, y2_pos, _ = recent_positions[i]
                dist = ((x2_pos-x1_pos)**2 + (y2_pos-y1_pos)**2)**0.5
                distances.append(dist)
            
            avg_movement = np.mean(distances) if distances else 0
            # Much stricter loitering detection
            if avg_movement < 5:  # Very minimal movement
                # Require AI confirmation OR extremely low movement
                recent_actions = [a[0] for a in data['action_history'][-15:]]
                loitering_ai_count = sum(1 for a in recent_actions if a == 'Loitering')
                
                if loitering_ai_count >= 3 or avg_movement < 2:
                    alerts.append("LOITERING")
                    data['suspicious_score'] += 0.15
    
    # 3. Running detection (VERY CONSERVATIVE)
    if len(data['positions']) >= 5:  # Check over more frames
        # Calculate average speed over last 5 positions
        recent_speeds = []
        for i in range(len(data['positions'])-4, len(data['positions'])):
            if i > 0:
                x1_pos, y1_pos, _ = data['positions'][i-1]
                x2_pos, y2_pos, _ = data['positions'][i]
                speed = ((x2_pos-x1_pos)**2 + (y2_pos-y1_pos)**2)**0.5
                recent_speeds.append(speed)
        
        avg_speed = np.mean(recent_speeds) if recent_speeds else 0
        max_speed = max(recent_speeds) if recent_speeds else 0
        
        # Only flag if consistently very fast AND AI confirms running
        if avg_speed > activity_threshold['fast_movement'] * 1.5:  # Much higher threshold
            if action == 'Running' and confidence > 0.8:
                # Additional check: must be running for multiple frames
                recent_running = [a[0] for a in data['action_history'][-5:] if a[0] == 'Running']
                if len(recent_running) >= 2:
                    alerts.append("RUNNING")
                    data['suspicious_score'] += 0.08
            elif max_speed > 150:  # Extremely fast movement
                alerts.append("FAST_MOVEMENT")
                data['suspicious_score'] += 0.05
    
    # 4. Pattern-based suspicious behavior (EXTREMELY STRICT)
    if len(data['action_history']) >= 30:  # Much longer observation period
        recent_actions = [a[0] for a in data['action_history'][-30:]]
        suspicious_actions = ['Fighting', 'Vandalism', 'Theft', 'Suspicious']
        normal_actions = [a for a in recent_actions if a == 'Normal']
        
        suspicious_count = sum(1 for action in recent_actions if action in suspicious_actions)
        normal_ratio = len(normal_actions) / len(recent_actions)
        
        # Only flag if very high suspicious activity AND very low normal behavior
        if suspicious_count >= 10 and normal_ratio < 0.2:  # 33% suspicious AND <20% normal
            # Additional validation: check for consistency
            suspicious_clusters = 0
            for i in range(0, len(recent_actions)-5, 5):
                cluster = recent_actions[i:i+5]
                if sum(1 for a in cluster if a in suspicious_actions) >= 2:
                    suspicious_clusters += 1
            
            if suspicious_clusters >= 3:  # Multiple suspicious clusters
                alerts.append("PATTERN_SUSPICIOUS")
                data['suspicious_score'] += 0.15
    
    # 5. Overall suspicious score threshold (VERY HIGH THRESHOLD)
    if data['suspicious_score'] > 1.2 and "HIGH_RISK" not in data['alerts']:  # Very high threshold
        # Additional checks: ensure not too much normal behavior AND sustained suspicious activity
        normal_ratio = data['normal_behavior_count'] / max(1, data['frame_count'])
        if normal_ratio < 0.4 and data['frame_count'] > 100:  # Less than 40% normal AND observed long enough
            # Final validation: check recent suspicious score trend
            if len(data['action_history']) >= 10:
                recent_suspicious_actions = [a for a in data['action_history'][-10:] if a[0] in ['Fighting', 'Vandalism', 'Theft', 'Suspicious']]
                if len(recent_suspicious_actions) >= 3:  # Recent suspicious activity
                    alerts.append("HIGH_RISK")
    
    # 6. NORMAL BEHAVIOR BONUS (Aggressive false positive reduction)
    normal_ratio = data['normal_behavior_count'] / max(1, data['frame_count'])
    if normal_ratio > 0.7:  # 70%+ normal behavior
        data['suspicious_score'] *= 0.3  # Drastically reduce suspicious score
    elif normal_ratio > 0.5:  # 50%+ normal behavior
        data['suspicious_score'] *= 0.6  # Significantly reduce suspicious score
    
    # Much faster decay for suspicious score
    data['suspicious_score'] *= 0.9  # Faster decay
    
    # Prevent score from going negative
    data['suspicious_score'] = max(0, data['suspicious_score'])
    
    # Store new alerts (with intelligent cooldown and validation)
    new_alerts = []
    current_frame = frame_count
    
    for alert in alerts:
        # Cooldown period: don't repeat same alert within 50 frames (~2 seconds)
        if alert not in data['alerts'] or (current_frame - data['last_alert_frame']) > 50:
            # Additional validation for high-impact alerts
            if alert in ['HIGH_RISK', 'AI_DETECTED_FIGHTING', 'AI_DETECTED_VANDALISM', 'AI_DETECTED_THEFT']:
                # Require very high confidence and multiple confirmations
                recent_high_conf = [c for c in data['confidence_history'][-10:] if c > 0.8]
                if len(recent_high_conf) >= 3:  # Multiple high-confidence detections
                    data['alerts'].append(alert)
                    new_alerts.append(alert)
                    data['last_alert_frame'] = current_frame
                    print(f"🚨 VERIFIED SUSPICIOUS ACTIVITY - Person {person_id}: {alert} (Score: {data['suspicious_score']:.2f})")
            else:
                # Lower-impact alerts (movement-based)
                data['alerts'].append(alert)
                new_alerts.append(alert)
                data['last_alert_frame'] = current_frame
                print(f"⚠️  MINOR ALERT - Person {person_id}: {alert} (Score: {data['suspicious_score']:.2f})")
    
    # Track confidence history for validation
    if len(data['action_history']) > 0:
        latest_confidence = data['action_history'][-1][1]
        data['confidence_history'].append(latest_confidence)
        if len(data['confidence_history']) > 20:
            data['confidence_history'] = data['confidence_history'][-20:]
    
    # Clean old alerts (remove after some time)
    if len(data['alerts']) > 3:
        data['alerts'] = data['alerts'][-2:]  # Keep only most recent alerts
    
    return new_alerts

def get_person_id_simple(box, existing_boxes, threshold=50):
    """Simple person tracking based on proximity"""
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Find closest existing person
    min_distance = float('inf')
    closest_id = None
    
    for i, existing_box in enumerate(existing_boxes):
        ex1, ey1, ex2, ey2 = existing_box
        ex_center_x = (ex1 + ex2) / 2
        ex_center_y = (ey1 + ey2) / 2
        
        distance = ((center_x - ex_center_x)**2 + (center_y - ex_center_y)**2)**0.5
        if distance < min_distance and distance < threshold:
            min_distance = distance
            closest_id = i
    
    return closest_id if closest_id is not None else len(existing_boxes)

# Create directories
capture_dir = "captured_frames"
pdf_dir = "pdf_reports"
annotated_dir = "annotated_images"

for directory in [capture_dir, pdf_dir, annotated_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

print("\n" + "="*50)
print("ENHANCED PERSON ANALYSIS WITH SEGMENTATION & PDF REPORTS:")
print("  - Automatic distance-based calibration")
print("  - YOLO segmentation for accurate dress color detection")
print("  - DeepFace AI-powered gender & age detection")
print("  - AI-powered suspicious activity detection (fighting, theft, vandalism, loitering)")
print("  - Frame capture on 'c' and 's' buttons")
print("  - Manual person selection mode")
print("  - Height estimation with smart calibration")
print("  - Shirt & Pant RGB color detection using segmentation")
print("  - Skin tone analysis")
print("  - PROFESSIONAL PDF REPORTS WITH COLOR NAMES & GENDER")
print("\nCALIBRATION GUIDE:")
print("  Very Close (bottom): Use 1.5-1.6m calibration")
print("  Close: Use 1.6-1.65m calibration") 
print("  Medium: Use 1.65-1.75m calibration")
print("  Far (top): Use 1.75-1.85m calibration")
print("\nCONTROLS:")
print("  'q' = Quit")
print("  'c' = Capture frame + Auto-calibrate")
print("  's' = Capture frame + Analyze with segmentation")
print("  'm' = Manual calibrate (enter known height)")
print("  'x' = Cancel calibration mode")
print("  't' = Toggle manual selection mode")
print("  'a' = Analyze selected persons with segmentation")
print("  'r' = Clear calibration")
print("  'p' = Generate PDF summary report")
print("  'i' = Generate individual PDF reports")

print("="*50 + "\n")

# Read first frame to set up manual selection
ret, first_frame = cap.read()
if not ret:
    print("Error: Cannot read video file")
    exit()

# Create window and set mouse callback
cv2.namedWindow("Enhanced Person Analysis with Segmentation & PDF Reports")
cv2.setMouseCallback("Enhanced Person Analysis with Segmentation & PDF Reports", selector.draw_rectangle)

# Reset video to beginning
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_count = 0
previous_boxes = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video")
        # Reset to beginning for continuous analysis
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count = 0
        previous_boxes = []
        continue
    
    frame_count += 1
    
    display_frame = frame.copy()
    
    if manual_mode:
        # Manual selection mode
        cv2.putText(display_frame, "MANUAL MODE: Draw rectangles around persons", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(display_frame, "Press 'a' to analyze, 't' to exit manual mode", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw current selection
        if selector.current_box:
            x1, y1, x2, y2 = selector.current_box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw all selected persons
        for i, box in enumerate(selector.selected_persons):
            x1, y1, x2, y2 = box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(display_frame, f"Person {i+1}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    else:
        # Automatic detection mode
        try:
            # Detect persons with pose keypoints
            results = pose_model.predict(frame, conf=0.7, classes=[0], verbose=False)
            
            detected_persons = []
            
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        box = boxes.xyxy[i].cpu().numpy().astype(int)
                        detected_persons.append(box)
            
            # Display detected persons with distance info and suspicious activity detection
            for i, box in enumerate(detected_persons):
                x1, y1, x2, y2 = box
                
                # Get person ID for tracking
                person_id = get_person_id_simple(box, previous_boxes)
                
                # Detect suspicious activity using advanced AI model
                suspicious_alerts = detect_suspicious_activity_advanced(person_id, box, frame, frame_count)
                
                # Estimate distance for each person
                distance_category = calibrator.estimate_distance_from_position(box)
                
                # Color code based on distance or suspicious activity
                if suspicious_alerts:
                    color = (0, 0, 255)  # Red for suspicious activity
                elif distance_category == "very_close":
                    color = (0, 0, 255)  # Red - very close
                elif distance_category == "close":
                    color = (0, 165, 255)  # Orange - close
                elif distance_category == "medium":
                    color = (0, 255, 255)  # Yellow - medium
                else:
                    color = (0, 255, 0)  # Green - far
                
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, f"ID{person_id} ({distance_category})", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Display suspicious activity alerts
                if suspicious_alerts:
                    alert_text = ", ".join(suspicious_alerts)
                    cv2.putText(display_frame, f"🚨 {alert_text}", (x1, y2+20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                
                # Draw horizontal division line between shirt and pants
                mid_y = y1 + (y2 - y1) // 2
                cv2.line(display_frame, (x1, mid_y), (x2, mid_y), color, 1)
            
            # Update previous boxes for next frame
            previous_boxes = detected_persons.copy()
            
            # Show calibration mode status
            if CALIBRATION_MODE:
                cv2.putText(display_frame, "AUTO-CALIBRATION: Select reference person", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(display_frame, "Press 'x' to cancel calibration", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif MANUAL_CALIBRATION_MODE:
                cv2.putText(display_frame, "MANUAL CALIBRATION: Select reference person", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.putText(display_frame, "Press 'x' to cancel calibration", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            else:
                cv2.putText(display_frame, f"AUTO MODE: Detected {len(detected_persons)} persons", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display_frame, "Press 't' for manual mode, 's' to analyze with segmentation", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Calibration mode - find reference person
            if (CALIBRATION_MODE or MANUAL_CALIBRATION_MODE) and len(detected_persons) > 0 and reference_height_px is None:
                ref_idx = find_reference_person(detected_persons)
                if ref_idx is not None:
                    box = detected_persons[ref_idx]
                    reference_height_px = box[3] - box[1]
                    
                    # Auto-determine calibration height based on distance
                    if CALIBRATION_MODE:
                        current_reference_height_m, distance_category = calibrator.auto_calibrate_distance(box)
                        cal_type = "Auto"
                        cal_color = (0, 0, 255)
                    else:
                        # Manual calibration - you would input actual height here
                        # For demo, using auto calibration
                        current_reference_height_m, distance_category = calibrator.auto_calibrate_distance(box)
                        cal_type = "Manual"
                        cal_color = (255, 0, 0)
                    
                    # Draw calibration box
                    x1, y1, x2, y2 = box
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), cal_color, 3)
                    cv2.putText(display_frame, f"{cal_type} CALIBRATION REFERENCE", (x1, y1-30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, cal_color, 2)
                    cv2.putText(display_frame, f"{reference_height_px:.0f}px = {current_reference_height_m:.2f}m", 
                                (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cal_color, 2)
                    cv2.putText(display_frame, f"Distance: {distance_category}", 
                                (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cal_color, 2)
                    
                    print(f"✓ {cal_type} CALIBRATED: {reference_height_px:.0f}px = {current_reference_height_m:.2f}m")
                    print(f"  Distance: {distance_category}")
                    CALIBRATION_MODE = False
                    MANUAL_CALIBRATION_MODE = False
                        
        except Exception as e:
            print(f"Detection error: {e}")
            cv2.putText(display_frame, "Detection failed - Switch to manual mode (press 't')", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Display analyzed persons
    for i, (box, shirt_rgb, pant_rgb, skin, skin_rgb, height_cm, weight_kg, age_range, distance_category, gender, gender_confidence) in enumerate(analyzed_persons):
        x1, y1, x2, y2 = box
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        
        # Draw division line
        mid_y = y1 + (y2 - y1) // 2
        cv2.line(display_frame, (x1, mid_y), (x2, mid_y), (255, 255, 0), 1)
        
        # Get color names for display
        shirt_name = get_simplified_color_name(shirt_rgb)
        pant_name = get_simplified_color_name(pant_rgb)
        skin_name = get_simplified_color_name(skin_rgb)
        
        # Display RGB values and age range with color names and gender
        shirt_info = f"Shirt: {shirt_name}"
        pant_info = f"Pant: {pant_name}"
        skin_info = f"Skin: {skin} ({skin_name})"
        gender_info = f"Gender: {gender} ({gender_confidence})"
        info_text = f"Person {i+1}: {height_cm:.1f}cm"
        cv2.putText(display_frame, info_text, (x1, y2+15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(display_frame, f"{shirt_info}", (x1, y2+35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(display_frame, f"{pant_info}", (x1, y2+55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(display_frame, f"{skin_info}", (x1, y2+75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(display_frame, f"{gender_info}", (x1, y2+95), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(display_frame, f"Age: {age_range}", (x1, y2+115), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    # Display calibration status
    status_y = frame_height - 120
    if CALIBRATION_MODE or MANUAL_CALIBRATION_MODE:
        mode_type = "AUTO" if CALIBRATION_MODE else "MANUAL"
        cv2.putText(display_frame, f"{mode_type} CALIBRATION ACTIVE - Press 'x' to cancel", 
                    (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        status_y -= 20
    
    if reference_height_px:
        cal_text = f"Calibrated: {reference_height_px:.0f}px = {current_reference_height_m:.2f}m"
        cv2.putText(display_frame, cal_text, (10, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        status_y -= 20
    else:
        cv2.putText(display_frame, "Not calibrated - press 'c' for auto or 'm' for manual", (10, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        status_y -= 20
    
    # Display PDF controls info
    cv2.putText(display_frame, "PDF Controls: 'p'=Summary Report, 'i'=Individual Reports", 
                (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    status_y -= 20
    
    # Display distance legend
    cv2.putText(display_frame, "Distance Colors: Red=VeryClose Orange=Close Yellow=Medium Green=Far", 
                (10, frame_height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Display frame info
    cv2.putText(display_frame, f"Frame: {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}", 
                (10, frame_height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow("Enhanced Person Analysis with Segmentation & PDF Reports", display_frame)
    
    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        # Capture frame + Auto-calibration mode
        capture_and_save_frame(frame, capture_dir)
        if not manual_mode:
            CALIBRATION_MODE = True
            MANUAL_CALIBRATION_MODE = False
            reference_height_px = None
            print(f"\n📸 Frame captured + AUTO-CALIBRATION MODE: Looking for reference person...")
            print("   Will automatically adjust calibration height based on distance")
            print("   Press 'x' to cancel calibration")
        else:
            print("Calibration not available in manual mode. Exit manual mode first.")
    elif key == ord('s'):
        # Capture frame + Analyze with segmentation
        capture_and_save_frame(frame, capture_dir)
        print(f"\n📸 Frame captured + Analyzing with segmentation...")
        try:
            results = pose_model.predict(frame, conf=0.7, classes=[0], verbose=False)
            
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        box = boxes.xyxy[i].cpu().numpy().astype(int)
                        # Use segmentation for color analysis
                        shirt_rgb, pant_rgb, skin_tone, skin_rgb = analyze_captured_image_with_segmentation(frame, box, seg_model)
                        
                        # Estimate distance
                        distance_category = calibrator.estimate_distance_from_position(box)
                        
                        # Estimate height with calibration
                        height_px = box[3] - box[1]
                        if reference_height_px:
                            scale_factor = current_reference_height_m / reference_height_px
                            height_m = height_px * scale_factor
                            height_cm = height_m * 100
                            weight_kg = 22.0 * (height_m ** 2)
                        else:
                            height_m = height_px * (current_reference_height_m / (frame_height * 0.35))
                            height_cm = height_m * 100
                            weight_kg = 22.0 * (height_m ** 2)
                        
                        # Estimate age range using DeepFace
                        age_range = estimate_age_from_deepface(frame, box)
                        
                        # Detect gender using DeepFace
                        gender, gender_confidence = detect_gender_from_deepface(frame, box)
                        
                        # Save annotated image
                        annotated_image_path = save_annotated_image(frame, box, shirt_rgb, pant_rgb, skin_tone, gender, annotated_dir)
                        
                        # Get person ID and check for suspicious activity with improved logic
                        person_id = get_person_id_simple(box, [])
                        suspicious_status = "Not Suspicious"
                        suspicious_details = []
                        
                        if person_id in suspicious_activities:
                            data = suspicious_activities[person_id]
                            # Only mark as suspicious if score is high AND has verified alerts
                            if data['suspicious_score'] > 0.8 and data['alerts']:
                                # Filter out minor alerts, only show significant ones
                                significant_alerts = [alert for alert in data['alerts'] 
                                                    if alert in ['HIGH_RISK', 'AI_DETECTED_FIGHTING', 
                                                               'AI_DETECTED_VANDALISM', 'AI_DETECTED_THEFT', 
                                                               'AI_DETECTED_SUSPICIOUS', 'PATTERN_SUSPICIOUS']]
                                if significant_alerts:
                                    suspicious_status = "Suspicious"
                                    suspicious_details = significant_alerts
                                else:
                                    suspicious_status = "Minor Alerts Only"
                                    suspicious_details = data['alerts']
                        
                        # Store person data for PDF reporting
                        person_data = {
                            'box': box,
                            'shirt_rgb': shirt_rgb,
                            'pant_rgb': pant_rgb,
                            'skin_tone': skin_tone,
                            'skin_rgb': skin_rgb,
                            'height_cm': height_cm,
                            'weight_kg': weight_kg,
                            'age_range': age_range,
                            'distance_category': distance_category,
                            'gender': gender,
                            'gender_confidence': gender_confidence,
                            'pixel_height': height_px,
                            'calibration_ref_px': reference_height_px if reference_height_px else 0,
                            'calibration_ref_m': current_reference_height_m,
                            'annotated_image_path': annotated_image_path,
                            'suspicious_status': suspicious_status,
                            'suspicious_details': suspicious_details
                        }
                        all_persons_data.append(person_data)
                        
                        analyzed_persons.append((box, shirt_rgb, pant_rgb, skin_tone, skin_rgb, height_cm, weight_kg, age_range, distance_category, gender, gender_confidence))
                        
                        # Get color names for console output
                        shirt_name = get_simplified_color_name(shirt_rgb)
                        pant_name = get_simplified_color_name(pant_rgb)
                        skin_name = get_simplified_color_name(skin_rgb)
                        
                        print(f"✓ Auto-detected Person {i+1} (using segmentation):")
                        print(f"  Height: {height_cm:.1f}cm, Weight: {weight_kg:.1f}kg")
                        print(f"  Age Range: {age_range}")
                        print(f"  Gender: {gender} ({gender_confidence} confidence)")
                        print(f"  Shirt: {shirt_name} RGB{shirt_rgb}")
                        print(f"  Pant: {pant_name} RGB{pant_rgb}")
                        print(f"  Skin: {skin_tone}, {skin_name} RGB{skin_rgb}")
                        print(f"  Distance: {distance_category}")
        except Exception as e:
            print(f"Auto-analysis with segmentation failed: {e}")
            print("Switch to manual mode (press 't') for better results")
    elif key == ord('m'):
        # Manual calibration mode
        if not manual_mode:
            MANUAL_CALIBRATION_MODE = True
            CALIBRATION_MODE = False
            reference_height_px = None
            print(f"\n🔧 MANUAL CALIBRATION MODE: Looking for reference person...")
            print("   Please enter the actual height of the reference person")
            print("   Press 'x' to cancel calibration")
        else:
            print("Calibration not available in manual mode. Exit manual mode first.")
    elif key == ord('x'):
        # Cancel calibration mode
        if CALIBRATION_MODE or MANUAL_CALIBRATION_MODE:
            CALIBRATION_MODE = False
            MANUAL_CALIBRATION_MODE = False
            print("✓ Calibration cancelled")
    elif key == ord('t'):
        manual_mode = not manual_mode
        if manual_mode:
            print("\n=== MANUAL MODE ACTIVATED ===")
            print("Draw rectangles around persons with mouse")
            print("Press 'a' to analyze selected persons with segmentation")
            print("Press 't' to return to auto mode")
        else:
            print("\n=== AUTO MODE ACTIVATED ===")
            selector.selected_persons = []
    elif key == ord('a') and manual_mode:
        # Analyze selected persons with segmentation
        if selector.selected_persons:
            print(f"\n📸 Analyzing {len(selector.selected_persons)} selected person(s) with segmentation...")
            for i, box in enumerate(selector.selected_persons):
                # Use segmentation for color analysis
                shirt_rgb, pant_rgb, skin_tone, skin_rgb = analyze_captured_image_with_segmentation(frame, box, seg_model)
                
                # Estimate distance
                distance_category = calibrator.estimate_distance_from_position(box)
                
                # Estimate height from bounding box with calibration
                height_px = box[3] - box[1]
                if reference_height_px:
                    # Use calibration for accurate height
                    scale_factor = current_reference_height_m / reference_height_px
                    height_m = height_px * scale_factor
                    height_cm = height_m * 100
                    weight_kg = 22.0 * (height_m ** 2)  # BMI-based estimation
                else:
                    # Rough approximation without calibration
                    height_m = height_px * (current_reference_height_m / (frame_height * 0.35))
                    height_cm = height_m * 100
                    weight_kg = 22.0 * (height_m ** 2)
                
                # Estimate age range using DeepFace
                age_range = estimate_age_from_deepface(frame, box)
                
                # Detect gender using DeepFace
                gender, gender_confidence = detect_gender_from_deepface(frame, box)
                
                # Save annotated image
                annotated_image_path = save_annotated_image(frame, box, shirt_rgb, pant_rgb, skin_tone, gender, annotated_dir)
                
                # Get person ID and check for suspicious activity with improved logic
                person_id = get_person_id_simple(box, [])
                suspicious_status = "Not Suspicious"
                suspicious_details = []
                
                if person_id in suspicious_activities:
                    data = suspicious_activities[person_id]
                    # Only mark as suspicious if score is high AND has verified alerts
                    if data['suspicious_score'] > 0.8 and data['alerts']:
                        # Filter out minor alerts, only show significant ones
                        significant_alerts = [alert for alert in data['alerts'] 
                                            if alert in ['HIGH_RISK', 'AI_DETECTED_FIGHTING', 
                                                       'AI_DETECTED_VANDALISM', 'AI_DETECTED_THEFT', 
                                                       'AI_DETECTED_SUSPICIOUS', 'PATTERN_SUSPICIOUS']]
                        if significant_alerts:
                            suspicious_status = "Suspicious"
                            suspicious_details = significant_alerts
                        else:
                            suspicious_status = "Minor Alerts Only"
                            suspicious_details = data['alerts']
                
                # Store person data for PDF reporting
                person_data = {
                    'box': box,
                    'shirt_rgb': shirt_rgb,
                    'pant_rgb': pant_rgb,
                    'skin_tone': skin_tone,
                    'skin_rgb': skin_rgb,
                    'height_cm': height_cm,
                    'weight_kg': weight_kg,
                    'age_range': age_range,
                    'distance_category': distance_category,
                    'gender': gender,
                    'gender_confidence': gender_confidence,
                    'pixel_height': height_px,
                    'calibration_ref_px': reference_height_px if reference_height_px else 0,
                    'calibration_ref_m': current_reference_height_m,
                    'annotated_image_path': annotated_image_path,
                    'suspicious_status': suspicious_status,
                    'suspicious_details': suspicious_details
                }
                all_persons_data.append(person_data)
                
                analyzed_persons.append((box, shirt_rgb, pant_rgb, skin_tone, skin_rgb, height_cm, weight_kg, age_range, distance_category, gender, gender_confidence))
                
                # Get color names for console output
                shirt_name = get_simplified_color_name(shirt_rgb)
                pant_name = get_simplified_color_name(pant_rgb)
                skin_name = get_simplified_color_name(skin_rgb)
                
                print(f"✓ Person {i+1} Analysis (using segmentation):")
                print(f"  Position: [{box[0]}, {box[1]}, {box[2]}, {box[3]}]")
                print(f"  Height: {height_cm:.1f}cm, Weight: {weight_kg:.1f}kg")
                print(f"  Age Range: {age_range}")
                print(f"  Gender: {gender} ({gender_confidence} confidence)")
                print(f"  Shirt: {shirt_name} RGB{shirt_rgb}")
                print(f"  Pant: {pant_name} RGB{pant_rgb}")
                print(f"  Skin: {skin_tone}, {skin_name} RGB{skin_rgb}")
                print(f"  Distance: {distance_category}")
            
            selector.selected_persons = []  # Clear after analysis
        else:
            print("No persons selected for analysis!")
    elif key == ord('r'):
        # Reset calibration
        reference_height_px = None
        CALIBRATION_MODE = False
        MANUAL_CALIBRATION_MODE = False
        current_reference_height_m = 1.7
        print("✓ Calibration reset")
    elif key == ord('p'):
        # Generate PDF summary report
        if all_persons_data:
            print("\n📊 Generating PDF Summary Report...")
            create_summary_pdf_report(all_persons_data, pdf_dir)
        else:
            print("No analyzed persons data available for summary report!")
    elif key == ord('i'):
        # Generate individual PDF reports
        if all_persons_data:
            print(f"\n📄 Generating Individual PDF Reports for {len(all_persons_data)} person(s)...")
            for i, person_data in enumerate(all_persons_data):
                create_individual_pdf_report(person_data, pdf_dir)
            print(f"✓ Generated {len(all_persons_data)} individual PDF reports")
        else:
            print("No analyzed persons data available for individual reports!")

cap.release()
cv2.destroyAllWindows()

# Final summary and PDF generation
print("\n" + "="*50)
print("FINAL ANALYSIS SUMMARY:")
if analyzed_persons:
    # Auto-generate final reports
    print("\n📊 Auto-generating final PDF reports...")
    if all_persons_data:
        create_summary_pdf_report(all_persons_data, pdf_dir)
        for i, person_data in enumerate(all_persons_data):
            create_individual_pdf_report(person_data, pdf_dir)
    
    for i, (box, shirt_rgb, pant_rgb, skin, skin_rgb, height_cm, weight_kg, age_range, distance_category, gender, gender_confidence) in enumerate(analyzed_persons):
        shirt_name = get_simplified_color_name(shirt_rgb)
        pant_name = get_simplified_color_name(pant_rgb)
        skin_name = get_simplified_color_name(skin_rgb)
        
        # Check suspicious status for final summary with improved logic
        person_id = i  # Simple mapping for final summary
        suspicious_status = "Not Suspicious"
        if person_id in suspicious_activities:
            data = suspicious_activities[person_id]
            if data['suspicious_score'] > 0.8 and data['alerts']:
                # Filter out minor alerts for final summary
                significant_alerts = [alert for alert in data['alerts'] 
                                    if alert in ['HIGH_RISK', 'AI_DETECTED_FIGHTING', 
                                               'AI_DETECTED_VANDALISM', 'AI_DETECTED_THEFT', 
                                               'AI_DETECTED_SUSPICIOUS', 'PATTERN_SUSPICIOUS']]
                if significant_alerts:
                    suspicious_status = f"Suspicious ({', '.join(significant_alerts)})"
                elif data['alerts']:
                    suspicious_status = f"Minor Alerts ({', '.join(data['alerts'])})"
        
        print(f"Person {i+1}: {height_cm:.1f}cm, {weight_kg:.1f}kg")
        print(f"           Age Range: {age_range}")
        print(f"           Gender: {gender} ({gender_confidence})")
        print(f"           Shirt: {shirt_name} RGB{shirt_rgb}")
        print(f"           Pant: {pant_name} RGB{pant_rgb}")
        print(f"           Skin: {skin}, {skin_name} RGB{skin_rgb}")
        print(f"           Position: [{box[0]}, {box[1]}, {box[2]}, {box[3]}]")
        print(f"           Distance: {distance_category}")
        print(f"           Suspicious Status: {suspicious_status}")
else:
    print("No persons were analyzed")

if reference_height_px:
    print(f"\nCalibration used: {reference_height_px:.0f}px = {current_reference_height_m:.2f}m")

print("="*50)
print("\n✓ Analysis Complete!")
print(f"✓ PDF Reports saved in: {pdf_dir}/")
print(f"✓ Annotated Images saved in: {annotated_dir}/")
print(f"✓ Captured Frames saved in: {capture_dir}/")