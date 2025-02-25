import os
import json
import base64
import logging
import argparse
import time
import traceback
from io import BytesIO

from dotenv import load_dotenv
from PIL import Image
import openai

# PDF / ReportLab
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.platypus import Image as RLImage  # Avoid conflict with PIL.Image

# ============ Logging Configuration ============
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# ============ Load Environment Variables ============
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ Missing OPENAI_API_KEY in .env file.")
openai.api_key = OPENAI_API_KEY

# ============ Constants ============
MAX_IMAGE_DIMENSION = 1024       # Max dimension for PIL resizing (px)
PDF_IMG_MAX_WIDTH = 200          # Max width for images in PDF (points)
MAX_RETRIES = 3                  # OpenAI API retry times
RETRY_DELAY = 3                  # seconds delay between retries
MODEL_NAME = "o1"                # e.g. "o1", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"

# ============ Initialize Styles ============
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='Heading3_EN', parent=styles['Heading3'], fontName='Helvetica', fontSize=14))
styles.add(ParagraphStyle(name='Normal_EN', parent=styles['Normal'], fontName='Helvetica', fontSize=10))

# ============ FoodAnalyzer ============
class FoodAnalyzer:
    """
    1) Read and resize image in-memory (longest side <= MAX_IMAGE_DIMENSION)
    2) Convert image to Base64 data URI
    3) Use GPT model to analyze and return structured JSON
    """

    def __init__(self,
                 max_image_dimension: int = MAX_IMAGE_DIMENSION,
                 max_retries: int = MAX_RETRIES,
                 retry_delay: int = RETRY_DELAY,
                 model: str = MODEL_NAME):
        self.max_image_dimension = max_image_dimension
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.model = model
        # Create the new openai.OpenAI() client
        self.client = openai.OpenAI()

    def resize_image_in_memory(self, image_path: str) -> bytes:
        """
        Read and resize the image so that its longest side <= max_image_dimension.
        Use high-quality resampling (LANCZOS).
        Return the JPEG bytes, or None if error.
        """
        try:
            with Image.open(image_path) as img:
                img.thumbnail((self.max_image_dimension, self.max_image_dimension), Image.LANCZOS)
                buf = BytesIO()
                # Save as JPEG with relatively high quality
                img.save(buf, format="JPEG", quality=95)
                return buf.getvalue()
        except Exception as e:
            logging.error(f"Failed to resize image: {image_path} - {str(e)}")
            return None

    def analyze_image_with_openai(self, image_bytes: bytes) -> dict:
        """
        Analyze the in-memory JPEG image using the GPT model.
        We want a strict JSON format, e.g.:
          {
            "dish_name": "...",
            "nutrition_assessment": "...",
            "cuisine_category": "..."
          }
        Return {} if fails after multiple retries.
        """
        if not image_bytes:
            logging.error("No image data to analyze.")
            return {}

        # Convert to base64
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        # Construct the messages with a robust prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a world-renowned culinary and nutrition expert. "
                            "Please carefully examine the attached food image and provide the following details:\n"
                            "1) dish_name: a concise name of the dish\n"
                            "2) nutrition_assessment: a short statement about its nutritional aspects\n"
                            "3) cuisine_category: the style or region of cuisine (e.g., Italian, Mediterranean, etc.)\n\n"
                            "Return your answer in a strict JSON format, with no extra text or explanation:\n"
                            "{\n"
                            '  "dish_name": "...",\n'
                            '  "nutrition_assessment": "...",\n'
                            '  "cuisine_category": "..." \n'
                            "}\n"
                            "If you are unsure, make your best guess. "
                            "Do not include any additional fields or commentary."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ]

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages
                )
                result_text = response.choices[0].message.content.strip()
                logging.info(f"[OpenAI] Attempt {attempt+1} success. Response:\n{result_text}")
                parsed = json.loads(result_text)
                # Check if all required keys exist
                if all(k in parsed for k in ["dish_name", "nutrition_assessment", "cuisine_category"]):
                    return parsed
                else:
                    logging.warning(f"JSON missing required fields: {parsed}")
            except Exception as e:
                logging.error(f"[OpenAI] Attempt {attempt+1} exception: {str(e)}")
                logging.error(traceback.format_exc())
            time.sleep(self.retry_delay)

        logging.error("Exceeded max retry attempts. No valid result.")
        return {}

# ============ PDF Generation ============
def export_results_as_pdf(results: list, pdf_path: str) -> None:
    """
    Generate a PDF report containing:
      - Original image filename
      - Embedded (resized) image
      - dish_name, nutrition_assessment, cuisine_category
    """
    if not results:
        logging.warning("No results. PDF not generated.")
        return

    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    story = []

    # Title
    story.append(Paragraph("Food Recognition Report", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Generated by OpenAI Vision, including images and expert commentary.", styles["Normal_EN"]))
    story.append(Spacer(1, 20))

    for item in results:
        # Image filename heading
        filename = item.get('image', 'Unknown')
        story.append(Paragraph(f"File name: {filename}", styles["Heading3_EN"]))
        story.append(Spacer(1, 6))

        # Prepare image for PDF (second pass of resizing to avoid large images in PDF)
        image_bytes = item.get("resized_image_bytes", None)
        if image_bytes:
            try:
                with Image.open(BytesIO(image_bytes)) as img:
                    # Limit width to PDF_IMG_MAX_WIDTH, keep aspect ratio
                    if img.width > PDF_IMG_MAX_WIDTH:
                        w_percent = PDF_IMG_MAX_WIDTH / float(img.width)
                        new_height = int(img.height * w_percent)
                        img = img.resize((PDF_IMG_MAX_WIDTH, new_height), Image.LANCZOS)
                    buf = BytesIO()
                    img.save(buf, format="JPEG", quality=90)
                    buf.seek(0)
                    pdf_img = RLImage(buf, width=img.width, height=img.height)
            except Exception as e:
                logging.error(f"Error loading image for PDF: {e}")
                pdf_img = Paragraph("Image not available", styles["Normal_EN"])
        else:
            pdf_img = Paragraph("No image data", styles["Normal_EN"])

        # Build a small table for the GPT analysis
        dish_name = item.get("dish_name", "Unknown")
        nutrition = item.get("nutrition_assessment", "No info")
        cuisine = item.get("cuisine_category", "Unclear")

        details_data = [
            ["Dish Name", dish_name],
            ["Nutrition Assessment", nutrition],
            ["Cuisine Category", cuisine]
        ]
        details_table = Table(details_data, colWidths=[120, 360])
        details_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica')
        ]))

        # Layout: Image on top, table below
        story.append(pdf_img)
        story.append(Spacer(1, 10))
        story.append(details_table)
        story.append(Spacer(1, 20))

    try:
        doc.build(story)
        logging.info(f"PDF generated: {pdf_path}")
    except Exception as e:
        logging.error(f"Error generating PDF: {e}")

# ============ Main Processing ============
def process_images(image_dir: str, output_json: str, output_pdf: str) -> None:
    """
    1) For each .jpg in image_dir (skipping duplicates by filename):
       - Resize in memory
       - Send to OpenAI for analysis
    2) Write textual results to JSON (excluding image bytes)
    3) Generate PDF with images & analysis
    """
    analyzer = FoodAnalyzer()
    results = []

    if not os.path.exists(image_dir):
        logging.error(f"Directory not found: {image_dir}")
        return

    processed_filenames = set()  # to skip duplicates by filename

    for filename in os.listdir(image_dir):
        if filename.lower().endswith('.jpg'):
            if filename in processed_filenames:
                logging.warning(f"Skipping duplicate filename: {filename}")
                continue
            processed_filenames.add(filename)

            image_path = os.path.join(image_dir, filename)
            logging.info(f"Processing image: {image_path}")

            resized_bytes = analyzer.resize_image_in_memory(image_path)
            if not resized_bytes:
                logging.error(f"Skipping image due to resize failure: {image_path}")
                continue

            analysis = analyzer.analyze_image_with_openai(resized_bytes)
            if analysis:
                # Store both the textual analysis and the image bytes for PDF usage
                results.append({
                    "image": filename,
                    "dish_name": analysis.get("dish_name", "Unknown"),
                    "nutrition_assessment": analysis.get("nutrition_assessment", "No info"),
                    "cuisine_category": analysis.get("cuisine_category", "Unclear"),
                    "resized_image_bytes": resized_bytes
                })
                logging.info(f"Success: {filename}")
            else:
                logging.error(f"Failed analysis: {filename}")

    # Output JSON
    if results:
        # Build a separate list for JSON output, excluding image bytes
        json_results = []
        for item in results:
            json_results.append({
                "image": item["image"],
                "dish_name": item["dish_name"],
                "nutrition_assessment": item["nutrition_assessment"],
                "cuisine_category": item["cuisine_category"]
            })

        try:
            with open(output_json, "w", encoding="utf-8") as json_file:
                json.dump(json_results, json_file, indent=4, ensure_ascii=False)
            logging.info(f"JSON saved: {output_json}")
        except Exception as e:
            logging.error(f"Error writing JSON: {e}")

        # Generate PDF
        export_results_as_pdf(results, output_pdf)
    else:
        logging.warning("No valid results. Skipping JSON/PDF generation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch image recognition and PDF report generation (OpenAI Vision)"
    )
    parser.add_argument("--dir", type=str, default="imglib", help="Image directory (default: imglib)")
    parser.add_argument("--json", type=str, default="output.json", help="Output JSON filename (default: output.json)")
    parser.add_argument("--pdf", type=str, default="report.pdf", help="Output PDF filename (default: report.pdf)")
    args = parser.parse_args()
    process_images(args.dir, args.json, args.pdf)
