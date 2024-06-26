import fitz
import cv2
import numpy as np
import pytesseract


def detect_block(image_bytes: bytearray, Full_width: int, Full_height: int, lang: str):
    image_np_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_np_array, cv2.IMREAD_COLOR)

    # Getting OCR results from already using pytesseract
    ocr_result = pytesseract.image_to_data(
        image, output_type=pytesseract.Output.DICT, lang=lang
    )

    # Initialize the current paragraph information variables
    current_paragraph_text = ""
    current_block_num = 0
    last_block_left = 0
    last_block_right = 0
    last_block_height = 0

    # Initialize the box coordinates of the paragraph
    paragraph_bbox = {
        "left": float("inf"),
        "top": float("inf"),
        "right": 0,
        "bottom": 0,
    }
    Threshold_width = 0.06 * Full_width
    Threshold_height = 0.006 * Full_height

    # Merge paragraphs to get text blocks
    text_rect = []
    for i in range(len(ocr_result["text"])):
        text = ocr_result["text"][i].strip()
        block_num = ocr_result["block_num"][i]
        left = ocr_result["left"][i]
        top = ocr_result["top"][i]
        width = ocr_result["width"][i]
        height = ocr_result["height"][i]

        # Check text block
        if block_num != current_block_num or (
            abs(left - last_block_right) > Threshold_width
            and abs(height - last_block_height) > Threshold_height
            and abs(left - last_block_left) > Threshold_width
        ):
            # Append paragraph to text block
            if current_paragraph_text != "":
                x0 = paragraph_bbox["left"]
                y0 = paragraph_bbox["top"]
                x1 = paragraph_bbox["right"]
                y1 = paragraph_bbox["bottom"]

                text_rect.append([current_paragraph_text, (x0, y0, x1, y1)])

                # Reset paragraph info
                current_paragraph_text = ""
                paragraph_bbox = {
                    "left": float("inf"),
                    "top": float("inf"),
                    "right": 0,
                    "bottom": 0,
                }

            current_block_num = block_num
        if text:
            current_paragraph_text += text + " "

            # Update right and height of the last block
            last_block_left = paragraph_bbox["left"]
            last_block_right = left + width
            last_block_height = height

            # Update coordinate of text block
            paragraph_bbox["left"] = min(paragraph_bbox["left"], left)
            paragraph_bbox["top"] = min(paragraph_bbox["top"], top)
            paragraph_bbox["right"] = max(paragraph_bbox["right"], left + width)
            paragraph_bbox["bottom"] = max(paragraph_bbox["bottom"], top + height)

    # Last paragraph process
    if current_paragraph_text:
        # Append paragraph to text block
        x0 = paragraph_bbox["left"]
        y0 = paragraph_bbox["top"]
        x1 = paragraph_bbox["right"]
        y1 = paragraph_bbox["bottom"]

        text_rect.append([current_paragraph_text, (x0, y0, x1, y1)])

    text_list = [item[0] for item in text_rect]  # Texts of OCR results
    block_list = [item[1] for item in text_rect]  # Coordinates of OCR results

    return text_list, block_list


def pdf_ocr(doc: fitz.Document, page: fitz.Page, lang="eng"):
    img_list = page.get_images()  # Get all images of a page
    rect_list = []
    text_list = []

    # OCR for each image
    for img in img_list:
        img_rect = page.get_image_bbox(img[-2])  # Get image rect
        xref = img[0]  # Reference of image in pdf
        width = img[2]
        height = img[3]
        base_image = doc.extract_image(xref)  # Extract image from pdf using xref
        image_bytes = base_image["image"]  # Byte array of image

        # Detect blocks of pdf page
        texts, boxes = detect_block(image_bytes, width, height, lang)

        # Change coordinates to fit pdf page size
        for i, (x0, y0, x1, y1) in enumerate(boxes):
            p = img_rect.width / width
            rect_list.append(
                fitz.Rect(
                    img_rect.x0 + x0 * p,
                    img_rect.y0 + y0 * p,
                    img_rect.x0 + x1 * p,
                    img_rect.y0 + y1 * p + 2,
                )
            )
            text_list.append(texts[i])
    return rect_list, text_list


def main(input_file, output_file, start_page=0, end_page=-1):
    doc = fitz.open(input_file)

    if end_page == -1 or doc.page_count <= end_page:
        end_page = doc.page_count - 1

    for page_num in range(start_page, end_page + 1):
        print(f"Processing page {page_num + 1} of {doc.page_count}...")
        page = doc.load_page(page_num)
        rect_list, text_list = pdf_ocr(doc, page, "eng")
        # Clear image blocks and insert text here
        for index, rect in enumerate(rect_list):
            page.add_redact_annot(rect)
            page.apply_redactions()
            page.insert_htmlbox(
                rect,
                text_list[index],
                css="* {font-family:AdobeSongStd-Light;font-size:50px;}",
            )

    # Save output file
    doc.ez_save(output_file, garbage=4, deflate=True)
    doc.close()


if __name__ == "__main__":
    main("input.pdf", "output.pdf", 0, 2)
