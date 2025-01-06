from typing import List, Dict, Tuple
import fitz
from unidecode import unidecode
import re
import pandas as pd
import warnings
import numpy as np
import os
from collections import Counter
warnings.filterwarnings("ignore")


def clean_text(s: str) -> str:
    """
    Cleans and normalizes input text. Removes specific symbols that cause issues with the fitz library,
    allows only alphanumeric characters and some special characters, and normalizes whitespace.

    Args:
        s (str): The text string to be cleaned.

    Returns:
        str: The cleaned and normalized text string.
    """
    # Edge case of layout format where section headers are included inside these symbols,
    # which are not processed correctly by the fitz library.
    # Check if '<', '>', and '/' all appear simultaneously in the string and remove them if found.
    if all(x in s for x in ['<', '>', '/']):
        s = re.sub(r'[<>\/]', '', s)

    # Remove all characters except for alphanumeric, '&', '-', and space
    s = re.sub(r'[^\w& -]', '', s)

    # Normalize whitespace by replacing sequences of whitespace with a single space
    s = re.sub(r'\s+', ' ', s)

    return s.strip()  # Remove leading/trailing whitespace


class ResumeParser:
    def __init__(self, candidate_data: Dict[str, bytes]):
        """
        Initialize the ResumeParser with the given inputs.

        Parameters:
        - candidate_data: Dictionary with candidate names as keys and bytes content of their resume PDFs as values.
        """
        self.candidate_data = candidate_data

        # Define section headers and sub headers as instance variables
        self.section_headers = [
            "Education", "Profile", "Work Experience", "Experience", "Personal Info",
            "Personal Information", "Personal Bio", "Seminars", "Technologies",
            "Licences & Certifications", "Volunteering", "Summary", "Skills", "Projects",
            "Interests", "Working Experience", "Professional Experience", "Certifications",
            "References"
        ]

        self.sub_headers = [
            'Languages', 'Interests', 'Computer skills', 'Affiliations', 'Computing',
            'Computing skills', 'Volunteer', 'Seminars', 'Language', 'Personal skills',
            'Achievements', 'Accomplishments', 'Skills', 'Projects'
        ]

    def store_resumes(self):
        """
        Reads the PDF files and saves them to the specified directory.
        """
        # Ensure the save directory exists
        os.makedirs('Resumes', exist_ok=True)
        for filename, pdf_bytes in self.candidate_data.items():
            # Generate the path to save the file using the original filename
            pdf_file_path = os.path.join('Resumes', filename)

            try:
                # Save the PDF to the specified directory
                with open(pdf_file_path, 'wb') as f:
                    f.write(pdf_bytes)

            except Exception as e:
                print(f"Error processing file '{filename}': {e}")

    def process_pdf(self, file_path: str) -> pd.DataFrame:
        """
        Processes a PDF file to extract and clean text blocks from each page using the fitz library,
        gathering text attributes like font size, type, and coordinates. This function constructs and returns
        a structured DataFrame containing detailed information about each text span.

        Args:
            file_path (str): The file path of the PDF to be processed.

        Returns:
            df (DataFrame): A pandas DataFrame with columns for text, bounding box coordinates, page number,
                            font properties, and text characteristics like whether the text is bold, italic,
                            uppercase, contains digits, and the number of tokens in the text.
        """
        file = fitz.open(file_path)
        output = []
        for page in file:
            output += page.get_text("blocks")

        block_dict = {}
        page_num = 1
        for page in file:
            file_dict = page.get_text('dict')
            block = file_dict['blocks']
            block_dict[page_num] = block
            page_num += 1

        rows = []
        for page_num, blocks in block_dict.items():
            for block in blocks:
                if block['type'] == 0:
                    for line in block['lines']:
                        for span in line['spans']:
                            text = clean_text(unidecode(span['text']))
                            xmin, ymin, xmax, ymax = span['bbox']
                            font_size = span['size']
                            span_font = span['font']
                            color = span['color']
                            is_bold = "bold" in span_font.lower()
                            is_italic = "italic" in span_font.lower()
                            is_upper = re.sub("[\(\[].*?[\)\]]", "", text).isupper()
                            token_count = len(text.split())
                            contains_digits = any(char.isdigit() for char in text)
                            if text.replace(" ", "") != "":
                                rows.append({
                                    'text': text,
                                    'xmin': xmin,
                                    'ymin': ymin,
                                    'xmax': xmax,
                                    'ymax': ymax,
                                    'page': page_num,
                                    'bbox': span['bbox'],
                                    'font_size': font_size,
                                    'font': span_font,
                                    'color': color,
                                    'bold': is_bold,
                                    'italic': is_italic,
                                    'uppercase': is_upper,
                                    'token_count': token_count,
                                    'contains_digits': contains_digits
                                })

        spans = pd.DataFrame(rows)
        return spans

    def classify_headers(self, df) -> dict:
        """
        Classifies headers in a resume by identifying direct and potential headers based on their attributes.
        Direct headers are matched against a provided list of section headers, while potential headers are identified
        based on common attributes derived from direct headers.

        Args:
            df (DataFrame): A DataFrame containing text spans with 'text', 'font_size', 'font', 'color',
                            'bold', 'italic', 'uppercase', 'token_count', 'contains_digits', 'xmin', 'ymin', 'xmax',
                             'ymax', and 'page' columns.
            section_headers (list of str): List of section headers to be identified as direct headers.

        Returns:
            headers(dict): A dictionary of potential headers with their bounding box coordinates and page number.
        """
        # headers that match with the keywords list
        direct_headers = []
        potential_headers = set()
        header_coordinates = {}

        # Process each row in the DataFrame to find direct and keyword-related headers
        for _, attr in df.iterrows():
            nospaces_match = attr['text'].replace(" ", "").lower() in [header.replace(" ", "").lower() for header in
                                                                       self.section_headers]
            if (re.sub(r'[^\w&/-]', ' ', attr['text']).lower() in [header.lower() for header in
                                                                   self.section_headers]) or nospaces_match:
                direct_headers.append(attr)

        # Calculate the most frequent features of the direct headers
        if direct_headers:
            font_sizes = [x['font_size'] for x in direct_headers]
            fonts = [x['font'] for x in direct_headers]
            colors = [x['color'] for x in direct_headers]
            bold_status = [x['bold'] for x in direct_headers]
            italic_status = [x['italic'] for x in direct_headers]
            uppercase_status = [x['uppercase'] for x in direct_headers]

            common_font_size = Counter(font_sizes).most_common(1)[0][0]
            common_font = Counter(fonts).most_common(1)[0][0]
            common_color = Counter(colors).most_common(1)[0][0]
            common_bold = Counter(bold_status).most_common(1)[0][0]
            common_italic = Counter(italic_status).most_common(1)[0][0]
            common_uppercase = Counter(uppercase_status).most_common(1)[0][0]

            # Evaluate each direct header against common attributes and sub_headers list
            # Keep only the direct headers that have almost all the stored attributes
            # (tolerance = 1 different attribute).
            for header in direct_headers:
                # Calculate the number of attributes that do not match the common attributes
                not_common_attributes = sum([
                    header['font_size'] != common_font_size,
                    header['font'] != common_font,
                    header['bold'] != common_bold,
                    header['italic'] != common_italic,
                    header['uppercase'] != common_uppercase,
                    header['color'] != common_color
                ]) > 1

                # Strictly enforcing the uppercase attribute if it is the common case
                if common_uppercase and not header['uppercase']:
                    # Treat uppercase mismatch as a significant deviation if uppercase is the common style
                    continue  # Skip this header for main header classification

                is_sub_header = header['text'] in self.sub_headers and not_common_attributes

                # Store as a direct header if not in subheader list and conforms to attributes, taking strict case
                # enforcement into account
                if not is_sub_header:
                    if header['text'] in header_coordinates.keys():
                        continue  # Avoid reclassifying already processed headers
                    potential_headers.add(header['text'])
                    header_coordinates[header['text']] = (
                        header['xmin'], header['ymin'], header['xmax'], header['ymax'], header['page']
                    )

            # Store headers that are not in the section_headers list and have the same attributes of the direct headers
            for _, attr in df.iterrows():
                if (attr['font_size'] == common_font_size and
                        attr['font'] == common_font and
                        attr['bold'] == common_bold and
                        attr['italic'] == common_italic and
                        attr['uppercase'] == common_uppercase and
                        attr['color'] == common_color and
                        (attr['token_count'] <= 3 or sum(
                            header.lower() in attr['text'].lower().split() for header in
                            self.section_headers) >= 2) and not attr['contains_digits']):
                    if attr['text'] in header_coordinates.keys():
                        continue
                    header_coordinates[attr['text']] = (
                        attr['xmin'], attr['ymin'], attr['xmax'], attr['ymax'], attr['page']
                    )
                    potential_headers.add(attr['text'])

        return {header.lstrip(): header_coordinates[header] for header in potential_headers if
                header.lstrip() and header.lstrip()[0].isupper() and len(header.lstrip()) > 2}

    def adjust_y_coordinates(self, headers, file_path) -> dict:
        """
        Adjusts the y-coordinates of headers to account for multiple pages. This function ensures that the y-coordinates
        are cumulative across all pages in the document.

        Args:
            headers (dict): Dictionary where keys are header text and values are tuples of coordinates and page number.
                            The tuple format is (xmin, ymin, xmax, ymax, page_num).
            file_path (str): The file path of the PDF document.

        Returns:
            adjusted_headers (dict): A dictionary with adjusted y-coordinates for headers.
                                     The format is the same as the input headers dictionary.
        """
        doc = fitz.open(file_path)
        page_height = doc[0].rect.height

        headers_by_page = {}
        for header, coords in headers.items():
            page_num = coords[4]
            if page_num not in headers_by_page:
                headers_by_page[page_num] = {}
            # headers by page along with coordinates
            headers_by_page[page_num][header] = coords

        adjusted_headers = {}
        for page_num, headers in headers_by_page.items():
            for header, coords in headers.items():
                x0, y0, x1, y1 = coords[:4]
                y0_adjusted = y0 + (page_num - 1) * page_height
                y1_adjusted = y1 + (page_num - 1) * page_height
                adjusted_headers[header] = (x0, y0_adjusted, x1, y1_adjusted, page_num)
        return adjusted_headers

    def merge_broken_headers(self, headers_dict, y_threshold=30, x_threshold=10) -> dict:
        """
        Merges headers that are close to each other based on specified x and y thresholds. This is useful when headers
        are broken into multiple parts across a document and need to be consolidated into single entries.

        Args:
            headers_dict (dict): Dictionary where keys are header text and values are tuples of coordinates (xmin, ymin).
            y_threshold (int): The maximum y-coordinate difference allowed for merging headers.
            x_threshold (int): The maximum x-coordinate difference allowed for merging headers.

        Returns:
            merged_headers (dict): A dictionary with the merged headers. The keys are the new merged header texts,
                                   and the values are the coordinates of one of the original headers used in the merge.
        """
        merged_headers = {}
        headers_list = list(headers_dict.keys())
        merged_set = set()  # Keep track of headers that have been merged

        # Iterate through each pair of headers to find ones close enough to merge
        for i, header1 in enumerate(headers_list):
            for j, header2 in enumerate(headers_list):
                if i >= j:  # Avoid comparing the same pair twice or a header with itself
                    continue

                coord1 = headers_dict[header1]
                coord2 = headers_dict[header2]

                # Check if the headers are close enough to be considered the same
                if abs(coord1[1] - coord2[1]) < y_threshold and abs(coord1[0] - coord2[0]) < x_threshold:
                    merged_text = header1 + " " + header2
                    merged_headers[merged_text] = coord1  # Use the coordinates of the first header in the merge
                    merged_set.update([header1, header2])  # Mark these headers as merged

        # Include headers that were not merged into the final output
        for header in headers_list:
            if header not in merged_set:
                merged_headers[header] = headers_dict[header]

        return merged_headers

    def detect_two_column_resume(self, headers, x_threshold=100, y_threshold=50) -> (bool, dict, dict):
        """
        Detects if a resume is in a two-column layout by analyzing the X-coordinates of headers.
        Headers are classified into left and right columns based on their X-coordinates, and the function
        checks for significant deviations and the presence of headers on the same Y-axis to confirm the layout.

        Args:
            headers (dict): Dictionary where keys are header text and values are tuples of coordinates and page number.
                            The tuple format is (xmin, ymin, xmax, ymax, page_num).
            x_threshold (int): Maximum X-coordinate difference to consider for detecting significant deviations.
            y_threshold (int): Maximum Y-coordinate difference to consider for detecting headers on the same line.

        Returns:
            is_two_column (bool): True if a two-column layout is detected, False otherwise.
            left_headers (dict): Headers classified in the left column. Format is the same as the input headers dictionary.
            right_headers (dict): Headers classified in the right column. Format is the same as the input headers dictionary.
        """
        # Extract X-min and X-max coordinates from headers
        x_min_coords = [coords[0] for coords in headers.values()]
        x_max_coords = [coords[2] for coords in headers.values()]

        # Calculate mean X-min and X-max coordinates
        mean_x_min = np.mean(x_min_coords)
        mean_x_max = np.mean(x_max_coords)

        # Check standard deviation of x_min_coords to see if they are almost at the same x and confirm single-column layout
        if np.std(x_min_coords) < 20:
            return False, {}, {}

        # Calculate deviations from mean X-min
        deviations = [abs(x - mean_x_min) for x in x_min_coords]

        # Identify significant deviations
        significant_deviations = [d for d in deviations if d > x_threshold]

        left_headers = {}
        right_headers = {}

        # Classify headers into left and right columns based on X-min coordinate
        for header, coords in headers.items():
            xmin = coords[0]
            if xmin < mean_x_min:
                left_headers[header] = coords
            else:
                right_headers[header] = coords

        # Adjust right headers if their X-max is less than mean X-max
        for header, coords in list(right_headers.items()):
            xmax = coords[2]
            if xmax < mean_x_max:
                left_headers[header] = right_headers.pop(header)

        # Check for two headers on the same Y-axis to confirm two-column layout
        for header1, coords1 in headers.items():
            for header2, coords2 in headers.items():
                if header1 != header2 and abs(coords1[1] - coords2[1]) < y_threshold and abs(
                        coords1[0] - coords2[0]) > 70:
                    return True, left_headers, right_headers

        # Determine if the layout is two-column based on headers classification and deviations
        is_two_column = len(left_headers) > 0 and len(right_headers) > 0 and len(significant_deviations) > 0
        return is_two_column, left_headers, right_headers

    def extract_spans(self, page_num, rect, doc) -> List[dict]:
        """
        Extracts text spans from a specific area (rectangle) of a given page in a PDF document.
        Adjusts the y-coordinates for multipage resumes to ensure accurate span placement.

        Args:
            page_num (int): The page number to extract spans from.
            rect (fitz.Rect): The rectangular area on the page to extract spans from.
            doc (fitz.Document): The PDF document object.

        Returns:
            list: A list of dictionaries, each representing a span with its attributes
        """
        spans = []
        page = doc.load_page(page_num)
        text_page = page.get_text("dict", clip=rect)

        for block in text_page.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if fitz.Rect(span["bbox"]).intersects(rect):
                        text = span["text"]
                        xmin, ymin, xmax, ymax = span["bbox"]
                        font_size = span.get("size")
                        span_font = span.get("font")
                        color = span.get("color")
                        is_bold = "bold" in span_font.lower()
                        is_italic = "italic" in span_font.lower()
                        is_upper = text.isupper()
                        token_count = len(text.split())
                        contains_digits = any(char.isdigit() for char in text)
                        # Adjust y-coordinates for multipage resumes
                        ymin_adjusted = ymin + (page_num - 1) * doc[0].rect.height
                        ymax_adjusted = ymax + (page_num - 1) * doc[0].rect.height

                        spans.append({
                            'text': text,
                            'xmin': xmin,
                            'ymin': ymin_adjusted,
                            'xmax': xmax,
                            'ymax': ymax_adjusted,
                            'page': page_num,
                            'bbox': (xmin, ymin_adjusted, xmax, ymax_adjusted),
                            'font_size': font_size,
                            'font': span_font,
                            'color': color,
                            'bold': is_bold,
                            'italic': is_italic,
                            'uppercase': is_upper,
                            'token_count': token_count,
                            'contains_digits': contains_digits
                        })
        return spans

    def extract_text_between_headers(self, pdf_path, adjusted_headers) -> pd.DataFrame:
        """
        Extracts text between headers to break down the resume into sections. This function handles both single-column
        and two-column layouts by analyzing header positions and extracting text spans accordingly.

        Args:
            pdf_path (str): The file path of the PDF document.
            adjusted_headers (dict): Dictionary where keys are header text and values are tuples of coordinates and page number.
                                     The tuple format is (xmin, ymin, xmax, ymax, page_num).

        Returns:
            df (DataFrame): A DataFrame with extracted section text, headers, page numbers, and text spans.
        """
        doc = fitz.open(pdf_path)
        page_height = doc[0].rect.height

        extracted_data = []
        sorted_headers = sorted(adjusted_headers.items(), key=lambda item: (item[1][1], item[1][0]))

        # Detect if the resume is in a two-column layout
        is_two_column, left_headers, right_headers = self.detect_two_column_resume(adjusted_headers)

        if is_two_column:
            # where the second column starts
            column_boundary = min([coords[0] for header, coords in right_headers.items()])
            first_right_header_y = min([coords[1] for header, coords in right_headers.items()])

            first_left_header_y = min([coords[1] for header, coords in left_headers.items()])
            if first_right_header_y - first_left_header_y > 150:
                boundary_start_y = first_right_header_y
            else:
                boundary_start_y = 0
            # Separate headers into left and right columns
            left_col_headers = {header: coords for header, coords in adjusted_headers.items() if
                                coords[0] < column_boundary}
            right_col_headers = {header: coords for header, coords in adjusted_headers.items() if
                                 coords[0] >= column_boundary}

            def extract_column_text(headers, left_edge, right_edge, column_label, start_y=0):
                sorted_headers = sorted(headers.items(), key=lambda item: item[1][1])

                for i, (header, (x0, y0, x1, y1, _)) in enumerate(sorted_headers):
                    if y0 >= boundary_start_y:
                        if i < len(sorted_headers) - 1:
                            next_y0 = sorted_headers[i + 1][1][1]
                        else:
                            next_y0 = page_height * len(doc)

                        rect = fitz.Rect(left_edge, max(y1, start_y), right_edge, next_y0)
                        text = ""
                        spans = []

                        # Each time the text of the next page is added
                        # In the case of a rectangle that must cover two pages
                        for page_num in range(int(y0 // page_height),
                                              min(len(doc), int(next_y0 // page_height) + 1)):
                            y0_page = y0 - page_num * page_height
                            y1_page = next_y0 - page_num * page_height
                            page_rect = fitz.Rect(left_edge, max(0, y0_page), right_edge, min(page_height, y1_page))
                            page_text = doc.load_page(page_num).get_text("text", clip=page_rect).strip()
                            text += " " + page_text
                            spans += self.extract_spans(page_num, page_rect, doc)

                        if text.strip():
                            extracted_data.append({
                                'header': header,
                                'section text': text.strip(),
                                'page': int(y0 // page_height) + 1,
                                'rect': rect,
                                'column': column_label,
                                'spans': spans
                            })

            extract_column_text(left_col_headers, 0, column_boundary, 'left', 0)
            extract_column_text(right_col_headers, column_boundary, doc[0].rect.width, 'right', boundary_start_y)

        else:
            # Process a single-column layout
            for i, (header, (x0, y0, x1, y1, _)) in enumerate(sorted_headers):
                if i < len(sorted_headers) - 1:
                    next_y0 = sorted_headers[i + 1][1][1]
                else:
                    next_y0 = page_height * len(doc)

                rect = fitz.Rect(0, y1, doc[0].rect.x1, next_y0)
                text = ""
                spans = []
                for page_num in range(int(y0 // page_height), min(len(doc), int(next_y0 // page_height) + 1)):
                    y0_page = y0 - page_num * page_height
                    y1_page = next_y0 - page_num * page_height
                    page_rect = fitz.Rect(0, max(0, y0_page), doc[0].rect.x1, min(page_height, y1_page))
                    page_text = doc.load_page(page_num).get_text("text", clip=page_rect).strip()
                    text += " " + page_text
                    spans += self.extract_spans(page_num, page_rect, doc)

                if text.strip():
                    extracted_data.append({
                        'header': header,
                        'section text': text.strip(),
                        'page': int(y0 // page_height) + 1,
                        'rect': rect,
                        'column': 'single',
                        'spans': spans
                    })

        df = pd.DataFrame(extracted_data)
        return df

    def main(self) -> pd.DataFrame:
        """
        Processes each PDF file in the given root directory, extracts data, and saves the results to a CSV file.

        Returns:
            None: This function does not return any value but writes out to 'sections.csv' and handles exceptions.

        Raises:
            Exception: Catches and prints exceptions related to file processing errors.
        """
        self.store_resumes()
        all_dfs = []
        resume_number = 1

        for subdir, dirs, files in os.walk('Resumes'):
            for file in files:
                file_path = os.path.join(subdir, file)
                if file_path.endswith('.pdf'):
                    try:
                        # Process the PDF and extract sections
                        df_spans = self.process_pdf(file_path)
                        headers = self.classify_headers(df_spans)
                        adjusted_headers = self.adjust_y_coordinates(headers, file_path)
                        adjusted_headers = self.merge_broken_headers(adjusted_headers)
                        df = self.extract_text_between_headers(file_path, adjusted_headers)
                        # Add a column to indicate the resume index
                        df['index'] = resume_number
                        df['filename'] = file_path.split('\\')[-1]
                        all_dfs.append(df[['index', 'filename', 'header', 'section text']])
                        print(f"Processed file: {file_path}")
                        resume_number += 1
                    except Exception as e:
                        print(f"Error processing file {file}: {str(e)}\n")

        return pd.concat(all_dfs, ignore_index=True)

