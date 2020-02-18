# image_to_table_ocrparser

The script is used to parse tables in an image or a PDF file. 

The algorithm is based on image processing and OCR (Optical Character Recognition) techniques*.

*notes: 'pytesseract' package and Google Tesseract-Ocr Engine need to be installed. 

<br><br>
For images:
- (1) image data need to be put in the './samples/images/' folder
- (2) results will be saved in './results/result_images/' folder, with table boundaries drawn on the original image

<br><br>
For pdfs: 
- (1) pdf data need to be put in the './samples/scanned/' folder
- (2) results will be saved in './results/result_scanned/' folder, with table boundaries drawn on the original pdf

<br><br>
To parse tables in an image or a PDF file, run:
- python img_to_table.py

The result is in the format of: 
- result[page_n] = [table1, table2, ...]. Each table is in the format of dataframe. 


<br><br>
Notes about automatic table detection
- (1) if you do not want to automatically detect tables by frcnn, please provide table boundaries as './results/preresult.txt' (use the format as shown), and indicate 'has_predict = True' in 'img_to_table.py'
- (2) if you want to automatically detect tables by frcnn, please
    - (a) save the Config file as './pretrained_frcnn/config.pickle'
    - (b) save the pretrained model as './pretrained_frcnn/model_frcnn.hdf5'
    - (c) install keras 2.2.4, by pip install --user keras==2.2.4. 
    - (d) indicate 'has_predict = False' in 'img_to_table.py'

Please go to 'object detection' repo for detailed instruction. 
