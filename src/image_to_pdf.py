__author__ = "Adriano Fonseca"
__email__ = "a.fonseca@ccom.unh.edu"
__version__ = "1.0.0"




import os
from pathlib import Path
from PIL import Image
import math
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import logging
import argparse

class HistogramPDFGenerator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Calculate grid layout (3x3 for 9 images per page)
        self.rows = 3
        self.cols = 3
        self.images_per_page = self.rows * self.cols  # Will be 9
        
        # Calculate image size to fit on letter page with margins
        self.page_width, self.page_height = letter
        self.margin = 20  # 20 points margin
        
        # Calculate available space for images
        self.usable_width = self.page_width - (2 * self.margin)
        self.usable_height = self.page_height - (2 * self.margin)
        
        # Calculate individual image size
        self.img_width = self.usable_width / self.cols
        self.img_height = self.usable_height / self.rows

    def create_pdf(self, input_dir: Path, output_file: Path):
        """Create PDF from all PNG files in input directory."""
        try:
            # Get all PNG files
            png_files = sorted(list(input_dir.glob("*.png")))
            if not png_files:
                self.logger.error(f"No PNG files found in {input_dir}")
                return
            
            self.logger.info(f"Found {len(png_files)} PNG files")
            
            # Create PDF
            c = canvas.Canvas(str(output_file), pagesize=letter)
            
            # Process images in groups of 9 (images_per_page)
            for i in range(0, len(png_files), self.images_per_page):
                page_images = png_files[i:i + self.images_per_page]
                self._add_page(c, page_images)
                c.showPage()
            
            c.save()
            self.logger.info(f"Created PDF at {output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to create PDF: {str(e)}")

    def _add_page(self, c, images):
        """Add a page of images to the PDF."""
        for idx, img_path in enumerate(images):
            # Calculate position in grid
            row = idx // self.cols
            col = idx % self.cols
            
            # Calculate position on page
            x = self.margin + (col * self.img_width)
            y = self.page_height - self.margin - ((row + 1) * self.img_height)
            
            # Add image to PDF
            c.drawImage(str(img_path), x, y, self.img_width, self.img_height, 
                       preserveAspectRatio=True)

def main():
    parser = argparse.ArgumentParser(description='Convert histogram PNGs to PDF')
    parser.add_argument('input_dir', type=str, help='Input directory containing PNG files')
    args = parser.parse_args()

    output_file = os.path.join(args.input_dir, 'survey_histograms.pdf')
    
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create PDF
    generator = HistogramPDFGenerator()
    generator.create_pdf(Path(args.input_dir), Path(output_file))

if __name__ == "__main__":
    main() 