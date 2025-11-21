# from paddleocr import PaddleOCR
#
# # ocr = PaddleOCR(
# #     use_doc_orientation_classify=False, # Disables document orientation classification model via this parameter
# #     use_doc_unwarping=False, # Disables text image rectification model via this parameter
# #     use_textline_orientation=False, # Disables text line orientation classification model via this parameter
# # )
# # ocr = PaddleOCR(lang="en") # Uses English model by specifying language parameter
# # ocr = PaddleOCR(ocr_version="PP-OCRv4") # Uses other PP-OCR versions via version parameter
# # ocr = PaddleOCR(device="gpu") # Enables GPU acceleration for model inference via device parameter
# ocr = PaddleOCR(
#     text_detection_model_name="PP-OCRv5_mobile_det",
#     text_recognition_model_name="PP-OCRv5_mobile_rec",
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_textline_orientation=False,
# ) # Switch to PP-OCRv5_mobile models
# result = ocr.predict("/Users/ibk5106/Desktop/test_2.png")
# for res in result:
#     res.print()
#     res.save_to_img("output2")
#     res.save_to_json("output2")


from docling.document_converter import DocumentConverter

source = "/Users/ibk5106/Desktop/test_2.png"  # document per local path or URL
converter = DocumentConverter()
result = converter.convert(source)
print(result.document.export_to_markdown())  # output: "## Docling Technical Report[...]"