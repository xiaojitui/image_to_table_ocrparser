#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import os
import re
import glob

import pytesseract
import PIL
from PIL import Image
import cv2

from . import img_utils
import copy
# pip install PyMuPDF
import fitz
import pdb
from datetime import datetime
from shutil import copyfile
import sys
#sys.path.insert(0, './ocr_table')
from .img_utils import pdf_to_img
import pickle

# In[ ]:


# define function to get prediction
def getprediction(preresult, imgname):
    pred = preresult[preresult['img_name'] == imgname]['prediction'].values[0]
    index = []
    
    for match in re.finditer('table', pred):
        index.append([match.start(), match.end()])
    
    results = []
    for i in range(len(index)-1):
        cur = pred[index[i][1]+4: index[i+1][0]-6]
        cur = cur.split(',')
        cur = [int(k) for k in cur]
        results.append(cur)
        
    cur = pred[index[-1][1]+4: -3]
    cur = cur.split(',')
    cur = [int(k) for k in cur]
    results.append(cur)
    
    return results
    


# In[ ]:


def preprocess(imgboxes, tol = 10):
    imgboxes_clean = []
    for box in imgboxes:
        x1 = max(box[0] - tol, 10)
        y1 = max(box[1] - tol, 10)
        x2 = box[2] + tol
        y2 = box[3] + tol
        
        imgboxes_clean.append([x1, y1, x2, y2])
    return imgboxes_clean



def postprocess(imgfile):
    image = cv2.imread(imgfile)
    result = image.copy()
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,1))
    remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(result, [c], -1, (255,255,255), 2)

    # Remove vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
    remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(result, [c], -1, (255,255,255), 2)
    return result



# In[ ]:


# try to increase resolution 

# pip install PyMuPDF
#import fitz

def get_img_ocr(imgfile, config_number = 3, ratio = 2):
    #img = Image.open(imgfile)
    #custom_config = r'--oem 3 --psm 6'
    #####img = cv2.imread(imgfile)
    
    # default is 3: fully automatic
    # support 1-12
    # may use 11: Sparse text. Find as much text as possible in no particular order.
    # may use 12: Sparse text with OSD.
    # may use 6: Assume a single uniform block of text
    custom_config =  r'--oem 3 --psm ' + str(config_number)
    
    img = postprocess(imgfile)
    
    height, width = img.shape[:2]
    
    img_1 = cv2.resize(img,(int(ratio*width), int(ratio*height)), interpolation = cv2.INTER_CUBIC)
    img_array = np.array(img_1)
    H, W, _ = img_array.shape
    text = pytesseract.image_to_string(img_1, config = custom_config)
    bbox = pytesseract.image_to_boxes(img_1, config = custom_config) # (x1, y1, x2, y2)
    #img.close()
    ######imgname = path + '/scanned.png'
    ######pix.writePNG(imgname)
    return bbox, text, H, W


# In[ ]:


def get_char(bbox, H, imgbox, tol = 10):
    chars = []
    for k in bbox.split('\n'):
        char = {}
        char['text'] = k.split()[0]
        char['x0'] = int(k.split()[1])
        char['bottom'] = H - int(k.split()[2])
        char['x1'] = int(k.split()[3])
        char['top'] = H - int(k.split()[4])
        
        if char['x0']>=imgbox[0]-tol and char['x1']<=imgbox[2]+tol and char['top']>=imgbox[1]-tol and char['bottom']<=imgbox[3]+tol:
            chars.append(char)
    return chars


# In[2]:


def groupchars(allwords, row_tol = 1.5, col_tol = 4):
    
    allwords_clean = []
    
    for words in allwords:
        
        if words == []:
            allwords_clean.append([])
            continue
                
        # group words
        words_clean = []
        i = 0
        while i < len(words):

            x0 = words[i]['x0']  # need some tolerence?
            x1 = words[i]['x1']
            text = words[i]['text']
            top = words[i]['top']
            bottom = words[i]['bottom']
            row = words[i]['row']

            j = i + 1
            
            while j < len(words) and words[j]['row'] == words[j-1]['row'] and                 abs(words[j]['x0'] - words[j-1]['x1']) <= col_tol:
                x0 = min(x0, words[j-1]['x0'])
                x1 = max(x1, words[j]['x1'])
                
                #if words[j]['text'] != ' ' and words[j]['text'] != '  ':
                text = text + '' + words[j]['text']
                #else: 
                    #text = text + ''
                j = j + 1
            
            # do some cleaning for $ sign
            #if words[j-1]['text'].strip() == '$':
                #x1 = min(x1, words[j-2]['x1'])
                #text = text.strip().strip('$').strip()
            
            words_clean.append({'x0': int(x0), 'x1': int(x1), 'top': int(top), 'bottom': int(bottom), 
                                 'text': text, 'row': int(row)})
            i = j
        
        words_clean = sorted(words_clean, key = lambda x: [x['row'], x['x0']])
        allwords_clean.append(words_clean)
    
    return allwords_clean


# In[3]:


# post-processing functions: find rows and group words
# row_tol: used to consider font variance in a row
# col_tol: if two words are within col_tol, connect them. 

def groupwords(allwords, row_tol = 1.5, col_tol = 6):
    
    allwords_clean = []
    
    for words in allwords:
        
        if words == []:
            allwords_clean.append([])
            continue
        
        #group words
        words_clean = []
        i = 0
        while i < len(words):

            x0 = words[i]['x0']  # need some tolerence?
            x1 = words[i]['x1']
            text = words[i]['text']
            top = words[i]['top']
            bottom = words[i]['bottom']
            row = words[i]['row']

            j = i + 1
            
            while j < len(words) and words[j]['row'] == words[j-1]['row'] and                 abs(words[j]['x0'] - words[j-1]['x1']) <= col_tol:
                x0 = min(x0, words[j-1]['x0'])
                x1 = max(x1, words[j]['x1'])
                text = text + ' ' + words[j]['text']
                j = j + 1
            
            # do some cleaning for $ sign
            #if words[j-1]['text'].strip() == '$':
                #x1 = min(x1, words[j-2]['x1'])
                #text = text.strip().strip('$').strip()
            
            words_clean.append({'x0': int(x0), 'x1': int(x1), 'top': int(top), 'bottom': int(bottom), 
                                 'text': text, 'row': int(row)})
            i = j
        
        #words_clean = sorted(words_clean, key = lambda x: [x['row'], x['x0']])
        allwords_clean.append(words_clean)
    
    return allwords_clean


# In[4]:


def find_rows(allwords, row_tol = 1.5):
    
    for words in allwords:
        
        if words == []:
            allwords_clean.append([])
            continue
        
        # find rows first
        rows = []
        new_row = {}
        new_row['top'] = words[0]['top']
        new_row['bottom'] = words[0]['bottom']
        rows.append(new_row)
        
        for i in range(1, len(words)):
            in_row_record = 0
            for ele in rows:
                row_top = ele['top']
                row_bottom = ele['bottom']
                if abs(words[i]['top'] - row_top) <= row_tol or                     abs(words[i]['bottom'] - row_bottom) <= row_tol:
                        in_row_record = 1
                        break 
            if in_row_record == 0:
                new_row = {}
                new_row['top'] = words[i]['top']
                new_row['bottom'] = words[i]['bottom']
                rows.append(new_row)
    
    sorted(rows, key = lambda x: x['top']) 
    
    return rows


# In[5]:


def assign_rows(allwords, rows, row_tol = 1.5):
     
    for words in allwords:
        
        if words == []:
            allwords_clean.append([])
            continue
            
        for word in words:
            for i in range(len(rows)):
                row_top = rows[i]['top']
                row_bottom = rows[i]['bottom']
                if abs(word['top'] - row_top) <= row_tol or                     abs(word['bottom'] - row_bottom) <= row_tol:
                    word['row'] = i
                    break
                    
        words = sorted(words, key = lambda x: [x['row'], x['x0']])
                    
    return allwords


# In[6]:


def finalclean(alltables):
    
    alltables_clean = []
    page_tracker_clean = []
    
    for table in alltables:
        table_1 = copy.deepcopy(table)
        
        ### clean row and col 
        row_clean = []
        for i in range(table_1.shape[0]):
            row_ele = 0
            for j in range(table_1.shape[1]):
                if table_1.iloc[i, j].strip() != '':
                    row_ele += 1
            if row_ele == 1:
                row_clean.append(i)
    
        if row_clean != []:
            table_1.drop(row_clean, axis = 0, inplace = True)
            table_1.index = np.arange(len(table_1))

        ### cols
        col_clean = []
        for j in range(table_1.shape[1]):
            if pd.isnull(table_1.iloc[:, j]).all():
                col_clean.append(j)

        if col_clean != []:
            table_1.drop(col_clean, axis = 1, inplace = True)
            table_1.columns = np.arange(table_1.shape[1])

        table_new = largetable.convert2int(table_1)
        
        ### correct int values
        for i in range(table_new.shape[0]):
            for j in range(table_new.shape[1]):
                if type(table_new.iloc[i, j]) == str:
                    eles = table_new.iloc[i, j].strip().split()
                    if np.all([k.isdigit() for k in eles]):
                        new_ele = ''.join(k for k in eles)
                        try:
                            new_ele = int(new_ele)
                            table_new.iloc[i, j] = new_ele
                        except:
                            try: 
                                new_ele = float(new_ele)
                                table_new.iloc[i, j] = new_ele
                            except ValueError:
                                continue
                                
        alltables_clean.append(table_new)
            
    return alltables_clean


# In[7]:


## post-processing: get dataframe 

def preparedf_img_single(allwords, page, W, col_seperators):
    
    alltables = []
    
    #print('Processing table in this area:  ', table_edge, '...', end = ' ')
        
    #cur_table_edge = [0, table_edge[0], int(pages[page].width), table_edge[1]]
    col_use = [0] + col_seperators + [W]
        
        
    # find word in this row and col 
    #picked = [word for word in allwords]
    
    picked = []
    for word in allwords:
        for i in range(len(col_use)-1):
            if col_use[i] + 2  <= 0.5*(word['x0'] + word['x1']) <= col_use[i+1] + 2:
                word['col'] = i              
                            
        picked.append(word)

        
    # col correction:
    for i in range(1, len(picked)):
            
        if picked[i]['row'] == picked[i-1]['row'] and picked[i]['col'] <= picked[i-1]['col']:
    
            if picked[i]['col'] == 0 or picked[i]['col'] == len(col_seperators):
                picked[i]['text'] = picked[i-1]['text'] + ' ' + picked[i]['text']
                
            if 2<=i<= len(picked)-2 and picked[i-1]['col'] == picked[i-2]['col'] + 1                 and picked[i]['col']+1 == picked[i+1]['col']:
                picked[i]['text'] = picked[i-1]['text'] + ' ' + picked[i]['text']
                    
            if 2<=i<= len(picked)-2 and picked[i-1]['col'] == picked[i-2]['col'] + 1                 and picked[i]['col']+1 < picked[i+1]['col']:
                picked[i]['col'] += 1
                 
            # add this: 
            if 2<=i<= len(picked)-2 and picked[i-1]['col'] == picked[i-2]['col'] + 1                 and picked[i]['col']+1 > picked[i+1]['col']:
                picked[i]['text'] = picked[i-1]['text'] + ' ' + picked[i]['text']
            # add end
                
            if 2<=i<= len(picked)-2 and picked[i-1]['col'] > picked[i-2]['col'] + 1                 and picked[i]['col']+1 == picked[i+1]['col']:
                picked[i-1]['col'] -= 1
                    
            if 2<=i<= len(picked)-2 and picked[i-1]['col'] > picked[i-2]['col'] + 1                 and picked[i]['col']+1 < picked[i+1]['col']:
                picked[i]['col'] += 1
                    
            # add this: 
            if 2<=i<= len(picked)-2 and picked[i-1]['col'] < picked[i-2]['col'] + 1                 and picked[i]['col']+1 == picked[i+1]['col']:
                picked[i]['text'] = picked[i-1]['text'] + ' ' + picked[i]['text']
            # add end
             
    # find row and col numbers
    delta = picked[0]['row']

    for ele in picked:
        ele['row'] = ele['row']- delta

    maxrow = 0
    maxcol = 0
    for ele in picked:
        maxrow = max(maxrow, ele['row'])
        maxcol = max(maxcol, ele['col'])
            
        
    # fill the dataframe
    table_df = pd.DataFrame('', index=np.arange(maxrow+1), columns=np.arange(maxcol+1))
    for ele in picked: 
        i = ele['row']
        j = ele['col']
        table_df.iloc[i, j] = ele['text']
            
        
        
    # merge cols
    for i in range(table_df.shape[0]):
        for j in range(1, table_df.shape[1]-1):
            cur_ele = []
            next_ele = []
            for ele in picked:
                if ele['row'] == i and ele['col'] == j:
                    cur_ele = ele
                if ele['row'] == i and ele['col'] == j+1:
                    next_ele = ele
                if cur_ele !=[] and next_ele !=[]:
                    break
                        
            if cur_ele !=[] and next_ele !=[]:
                cur_left, cur_right = cur_ele['x0'], cur_ele['x1']
                next_left, next_right = next_ele['x0'], next_ele['x1']
                    
                if cur_right > next_left:
                    table_df.iloc[i, j] = table_df.iloc[i, j] + ' ' + table_df.iloc[i, j+1]
                    table_df.iloc[i, j+1] = ''
        
        
    # find each's edges and median line, for fine tuning 
    col_data = []
    for i in range(table_df.shape[1]):
        col_p_left = []
        col_p_right = []
        col_p_median = []
        for ele in picked:
            if ele['col'] == i:
                col_p_left.append(ele['x0'])
                col_p_right.append(ele['x1'])
                col_p_median.append((ele['x0']+ele['x1'])/2)
        if col_p_left != [] and col_p_right !=[]:
            ##### line 11122
            col_p = [np.median(col_p_left), np.median(col_p_right), np.median(col_p_median)]
            ##### col_p = [np.min(col_p_left), np.median(col_p_right), np.median(col_p_median)]
            #col_m = np.median(col_p_median)
        else: 
            col_p = []
            #col_m = []
        col_data.append(col_p)
            
    #print(col_data[0])
        
        
    # merge close cols and drop empty cols
    first_drop = []
    
    for i in range(table_df.shape[1]-1):
        if col_data[i] != [] and col_data[i+1] != []:
            if col_data[i+1][0] < col_data[i][2] < col_data[i+1][1]:
            #if col_data[i+1][0] < col_data[i][0] and col_data[i+1][1] > col_data[i][1]  or \
                    #col_data[i+1][0] > col_data[i][0] and col_data[i+1][1] < col_data[i][1]:
                first_drop.append(i)
                table_df.iloc[:, i+1] = table_df.iloc[:, i] + ' ' + table_df.iloc[:, i+1]
                
    #print(first_drop)
    table_df.drop(first_drop, axis = 1, inplace = True)    
    table_df.columns = range(table_df.shape[1])    
        
        
    # further clean the df
    second_drop = []
        
    for i in range(table_df.shape[1]):
        #table_df[i] = [' '.join(table_df[i][k].split('\n')) for k in range(table_df.shape[0])]
        if table_df[i].isnull().all() or np.all( table_df[i] == ''):
            second_drop.append(i)
    table_df.drop(second_drop, axis = 1, inplace = True)
    table_df.columns = range(table_df.shape[1])
        
       
    
    # clean rows
    row_drop = []
    for i in range(table_df.shape[0]):
        if table_df.iloc[i, :].isnull().all() or np.all(table_df.iloc[i, :] == ''):
            row_drop.append(i)
        
    table_df.drop(row_drop, axis = 0, inplace = True)
    table_df.index = range(table_df.shape[0])
    
    # save the results
    alltables.append(table_df)
        
    #print('Done')
        
    return alltables


# In[8]:



### ratio = 2
def ocr_table(imgfile, imgboxes, config_number = 3, ratio = 1.2): 
    
    try:
        bbox, text, H, W = get_img_ocr(imgfile, config_number, ratio)
    except:
        print('cannot do ocr...')

    allresults = {}
    for i in range(len(imgboxes)):
        #print(i)
        imgbox = imgboxes[i]

        imgbox = [k*ratio for k in imgbox]
        allresults[i] = {}


        allwords = []
        try:
            chars = get_char(bbox, H, imgbox, tol = 10)
            rows = find_rows([chars], row_tol = 15)
            chars = assign_rows([chars], rows, row_tol = 15)
            chars = groupchars(chars, row_tol = 15, col_tol = 10) #col_tol = 8
            # chars[0] = largetable.removesign(chars[0])
            words = groupwords(chars, row_tol = 6, col_tol = 15) #col_tol = 20
            allwords.append(words[0])
        except:
            print('no words parsed in the bbox...')

        try:
            allwords_df = copy.deepcopy(allwords) # 
            #allwords_df = img_utils.groupwords(allwords_df)
            table_edges, _, page_positions = img_utils.find_page_structure(allwords_df, 0, tol = 5, min_sep = 1)
            #table_edges = [table_edges[0][-1], table_edges[-1][-1]]
            
            if len(table_edges) >1:
                table_edges = [table_edges[0][-1]-1, table_edges[-1][-1]+1]
            elif len(table_edges) == 0:
                table_edges = [page_positions.row.min(), page_positions.row.max()]
            else:
                table_edges = [table_edges[0][0]-1, table_edges[0][-1]+1] ##table_edges[0]+1

        
            #table_edges = [imgbox[1], imgbox[3]]
            page_positions = img_utils.check_connect_row(page_positions)

            col_break_clean, col_n, _, ele_n = img_utils.find_table_structure(allwords_df[0], page_positions, table_edges)

            allresults[i]['ele_n'] = ele_n
            allresults[i]['col_n'] = col_n
            allresults[i]['col_sep'] = col_break_clean
            #allresults[page]['table_edge'] = table_edges
            allresults[i]['page_position'] = page_positions

            col_sep = [col_break_clean[0] - 50] + col_break_clean + [col_break_clean[-1] + 50]

            curtable = img_utils.preparedf_img_single(allwords_df[0], 0, W, col_sep)
            if curtable != []:
                curtable[0] = img_utils.clean_single_header_footnote(curtable[0])
                curtable[0] = img_utils.merge_row_df(curtable[0], allresults[i]['page_position'])
            allresults[i]['table'] = [curtable[0]]
        except:
            continue

    alltables = {}

    for k in allresults:
        if 'table' in allresults[k]:
            alltables[k] = allresults[k]['table']
        else:
            alltables[k] = []
    
    return alltables


# In[9]:


# convert to the format for camelot

def tocamelot(boxes, pageinfo, pdftoimgr=[1, 1]):
    results = []
    for box in boxes:
        #box = [int(k/pdftoimgr) for k in box]
        x1 = str(int(box[0]/pdftoimgr[0]))
        y1 = str(pageinfo[3] - int(box[1]/pdftoimgr[1]))
        x2 = str(int(box[2]/pdftoimgr[0]))
        y2 = str(pageinfo[3] - int(box[3]/pdftoimgr[1]))
        results.append([x1 + ',' + y1+',' +x2+',' + y2])
    return results


# In[ ]:

def get_boundary(preresult, imgname, tol = 20):
    #imgname = preresult.iloc[i, 0]
    imgboxes = getprediction(preresult, imgname)
    imgboxes = preprocess(imgboxes, tol = tol)
    
    return imgboxes


def show_boundary(imgfile, imgboxes, img_size = 10):

    img = plt.imread(imgfile)

    fig, ax = plt.subplots(1, 1, figsize = (img_size, img_size))
    ax.imshow(img)

    for box in imgboxes:
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]

        rect = Rectangle((x1,y1),x2-x1,y2-y1, fill=None, linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    plt.show()
    



# In[ ]:

# pip install PyMuPDF
#import fitz


    

def pdf_boundry_img(imgfile, table_id, box, img_size = 10):
    userfolder = os.path.split(os.path.dirname(imgfile))[0]
    img = plt.imread(imgfile) 
    fig, ax = plt.subplots(1, 1, figsize = (img_size, img_size))
    ax.imshow(img)

    x1 = float(box[0])
    y1 = float(box[1]-10)
    x2 = float(box[2])
    y2 = float(box[3]+10)

    rect = Rectangle((x1,y1),x2-x1,y2-y1, fill=None, linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    ax.axis('off')
    fig.savefig(os.path.join(userfolder , table_id + '.jpg'), bbox_inches='tight',  pad_inches = 0)
    plt.cla()
    plt.clf()
    plt.close()
    return (img.shape[0], img.shape[1])

def save_to_excel(savefile, parsed_results, show_boundries,filename, bbox_all):
    tables = []
    userfolder = os.path.split(os.path.dirname(savefile))[0]
    
    if show_boundries == 'yes':
        doc = fitz.open(filename) 
        with pd.ExcelWriter(savefile) as writer:
            for page_n in parsed_results:
                page_n = int(page_n)
                imgname = os.path.join(userfolder, 'scanned', 'p%d.jpg'%page_n)
                if not os.path.exists(imgname):
                    page = doc.loadPage(page_n) #number of page
                    mat = fitz.Matrix(1, 1)
                    pix = page.getPixmap(matrix = mat, alpha = False)  
                    pix.writePNG(imgname)
                
                if parsed_results[page_n] != []:
                    for idx in range(len(parsed_results[page_n])):
                        table_id = 'p' + str(page_n + 1) + '_id_' + str(idx + 1)
                        t0 = datetime.now()
                        imgheight, imgwidth = pdf_boundry_img(imgname, table_id, bbox_all[page_n][idx])
                        print('get pdf boundry img for %s in'%table_id, datetime.now() - t0)
                        table = parsed_results[page_n][idx]
                        table = table.applymap(lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x)
                        table.to_excel(writer, sheet_name=table_id)
                        table.columns = table.columns.astype(str)
                        tables.append({'table_id': table_id, 'table_html': table.to_html(index=False, index_names=False), 'imgheight': imgheight, 'imgwidth': imgwidth, 'remove': False})  
               
        doc.close()
        
    elif show_boundries == 'ocr':
        with pd.ExcelWriter(savefile) as writer:
            for page_n in parsed_results:    
                if parsed_results[page_n] != []:
                    for idx in range(len(parsed_results[page_n])):
                        table = parsed_results[page_n][idx]
                      
                        if table is None:
                            continue
                        table = table.applymap(lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x)
                        table_id = 'p' + str(page_n + 1) + '_id_' + str(idx + 1)
                        img = plt.imread(os.path.join(userfolder, 'scanned', 'p%d.jpg'%page_n))
                        imgheight, imgwidth = img.shape[0], img.shape[1]
                        copyfile(os.path.join(userfolder, 'result_scanned', '%s.jpg'%table_id), os.path.join(userfolder, table_id + '.jpg'))
                        table.to_excel(writer, sheet_name=table_id)
                        table.columns = table.columns.astype(str)
                        tables.append({'table_id': table_id, 'table_html': table.to_html(index=False, index_names=False), 'imgheight': imgheight, 'imgwidth': imgwidth, 'remove': False})          

        
    return tables

def save_to_excel_singleid(savefile, filename, parsed_result, table_id, bbox):
    userfolder = os.path.split(os.path.dirname(savefile))[0]
    if table_id.startswith('new_'):
        page = int(table_id.split('_')[1][1:]) - 1
    else:
        page = int(table_id.split('_')[0][1:]) - 1
    table = list(parsed_result.values())[0][0]
    table = table.applymap(lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x)
    if not os.path.exists(savefile):
        oldtables = [table_id]
    else:
        oldtables = pd.read_excel(savefile, index_col = 0, sheet_name = None)
    imgname = os.path.join(userfolder, 'scanned', 'p%d.jpg'%page)
    pdf_boundry_img(imgname, table_id, bbox)
    with pd.ExcelWriter(savefile) as writer:
        for tab in oldtables:
            if tab != table_id:
                oldtables[tab].to_excel(writer, sheet_name=tab)
            if tab == table_id:
                table.to_excel(writer, sheet_name=tab) 
                table.columns = table.columns.astype(str)
                response = table.to_html(index=False, index_names=False)  
        if table_id not in oldtables:
            table.to_excel(writer, sheet_name=table_id) 
            table.columns = table.columns.astype(str)
            response = table.to_html(index=False, index_names=False)  
            
    return response

# In[ ]:

def get_table_ocr(img_path, result_path):
    preresult = pd.read_csv(os.path.join(result_path, 'preresults.txt'), names = ['img_name', 'prediction'])
    preresult = preresult[preresult['prediction'] != '[]']
    preresult.index = np.arange(len(preresult))
    file_record = {}
    file_n = preresult.shape[0]
    for i in range(file_n):
        imgname = preresult.iloc[i, 0]
        imgfile = os.path.join(img_path, imgname)

        imgboxes = get_boundary(preresult, imgname) # tol = 20
        tables = ocr_table(imgfile, imgboxes, ratio = 2)
  
        table_results = []
        if tables != {}:
            for ele in tables:
                if len(tables[ele]) != 0:
                    table_results.append(tables[ele][0])
                else:
                    table_results.append(None)
        #else:
            #table_results = []

        file_record[imgname] = {}
        file_record[imgname]['imgboxes'] = imgboxes
        file_record[imgname]['tables'] = table_results
    alltables = {}
    for imgname in file_record:
        page_n = os.path.basename(imgname).split('.')[0][1:]
        page_n = int(page_n)
#         if file_record[imgname]['tables'] != []:
        alltables[page_n] = file_record[imgname]['tables']
    return alltables

if __name__ == "__main__":
    savefile = "./result/sample_tables.xlsx"
    filename = "./result/temp_file.pdf"
    with open('./result/test_result.pkl', 'rb') as f:
        parsed_result = pickle.load(f)
    table_id = "p1_id_1"
    bbox = [50, 200, 400, 600]
    save_to_excel_singleid(savefile, filename, parsed_result, table_id, bbox)


