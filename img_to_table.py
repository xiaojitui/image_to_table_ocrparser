#!/usr/bin/env python
# coding: utf-8

# In[1]:


import utils import ocr_table
import frcnn import frcnn_predict
import pandas as pd


# In[2]:


#if .jpg direcly put into samples
#if .pdf but scanned, generate imgs first

def parse_image(infile, config_filename, model_path, has_predict = True):

    if infile.endswith('jpg'):
        test_path = './samples/images/' 
        out_path = './results/result_images/'

    if infile.endswith('pdf'):
        out_img_path = './samples/scanned/'
        ocr_table.pdf_to_img(infile, out_img_path, zoom = 2)
        test_path = './samples/scanned/' 
        out_path = './results/result_scanned/'




    ### predict
    #config_filename = './keras-frcnn/config.pickle'
    #model_path = './keras-frcnn/model_frcnn.hdf5'

	if not has_predict:
		frcnn_predict.run_prediction(config_filename, model_path, test_path, out_path)

    ### parse
    img_path = test_path
    # where predicted boundary image is in
    result_path = out_path

    preresult = pd.read_csv(result_path + 'preresults.txt', names = ['img_name', 'prediction'])
    preresult = preresult[preresult['prediction'] != '[]']


    file_record = {}

    file_n = preresult.shape[0]

    for i in range(file_n):
        imgname = preresult.iloc[i, 0]
        imgfile = img_path + imgname

        imgboxes = ocr_table.get_boundary(preresult, imgname) # tol = 20
        tables = ocr_table.ocr_table(imgfile, imgboxes, ratio = 2)
        table_results = []
        if tables != {}:
            for ele in tables:
                if len(tables[ele]) != 0:
                    table_results.append(tables[ele][0])
        #else:
            #table_results = []

        file_record[imgname] = {}
        file_record[imgname]['imgboxes'] = imgboxes
        file_record[imgname]['tables'] = table_results


    alltables = {}
    for imgname in file_record:
        page_n = imgname.split('.')[0].split('_')[-1]
        page_n = int(page_n)
        if file_record[imgname]['tables'] != []:
            alltables[page_n] = file_record[imgname]['tables']

        
    return alltables, file_record


# In[3]:


def visualize_result(img_path, file_record, pick_page = 0):
   # pick a imgfile to visualize
    #pick_page = list(alltables.keys())[0]
    pick_img = 'page_' + str(pick_page) + '.jpg'

    ocr_table.show_boundary(img_path + pick_img, file_record[pick_img]['imgboxes']) #img_size = 10)


# In[ ]:





# In[ ]:





# In[4]:


## test

'''
###infile = './samples/images/img4.jpg'
infile = './samples/image1.pdf'
config_filename = './keras-frcnn/config.pickle'
model_path = './keras-frcnn/model_frcnn.hdf5'
alltables, file_record = parse_image(infile, config_filename, model_path)


# In[13]:


img_path = './samples/scanned/'
visualize_result(img_path, file_record, 1)
'''

# In[ ]:





# In[ ]:





# In[ ]:


if __name__ == '__main__':
    ###infile = './samples/images/img4.jpg'
    infile = './samples/scanned/image1.pdf'
    config_filename = './pretrained_frcnn/config.pickle'
    model_path = './pretrained_frcnn/model_frcnn.hdf5'
    alltables, file_record = parse_image(infile, config_filename, model_path, has_predict = True)
    print(alltables)


# In[ ]:




