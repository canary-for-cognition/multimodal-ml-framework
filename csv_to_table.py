#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
# import pylatex
from pylatex import Document, Section, Tabular, Math, Axis, Subsection
import pandas as pd
import sys
import os


def main():
    pm = u"\u00B1"
    filename = sys.argv[1]
    results = pd.read_csv(filename+'.csv')
    cols = results.columns
    task_fusion = ((results.loc[results['settings']=='overall']).loc[results['model']!='DummyClassifier']).sort_values('model')
    reading = ((results.loc[results['settings']=='Reading']).loc[results['model']!='DummyClassifier']).sort_values('model')
    cookie = ((results.loc[results['settings']=='CookieTheft']).loc[results['model']!='DummyClassifier']).sort_values('model')
    memory = ((results.loc[results['settings']=='Memory']).loc[results['model']!='DummyClassifier']).sort_values('model')
    pupil = ((results.loc[results['settings']=='PupilCalib']).loc[results['model']!='DummyClassifier']).sort_values('model')

    ET_basic = ((results.loc[results['settings']=='ET_basic']).loc[results['model']!='DummyClassifier']).sort_values('model')
    Eye = ((results.loc[results['settings']=='Eye']).loc[results['model']!='DummyClassifier']).sort_values('model')
    Language = ((results.loc[results['settings']=='Language']).loc[results['model']!='DummyClassifier']).sort_values('model')
    Eye_Reading = ((results.loc[results['settings']=='Eye_Reading']).loc[results['model']!='DummyClassifier']).sort_values('model')
    NLP_Reading = ((results.loc[results['settings']=='NLP_Reading']).loc[results['model']!='DummyClassifier']).sort_values('model')
    TextAudio = ((results.loc[results['settings']=='Text+Audio']).loc[results['model']!='DummyClassifier']).sort_values('model')
    
    task_fusion = np.array(task_fusion.dropna()).astype('str')
    reading = np.array(reading.dropna()).astype('str')
    cookie = np.array(cookie.dropna()).astype('str')
    memory = np.array(memory.dropna()).astype('str')
    pupil = np.array(pupil.dropna()).astype('str')

    ET_basic = np.array(ET_basic.dropna()).astype('str')
    Eye = np.array(Eye.dropna()).astype('str')
    Language = np.array(Language.dropna()).astype('str')
    Eye_Reading = np.array(Eye_Reading.dropna()).astype('str')
    NLP_Reading = np.array(NLP_Reading.dropna()).astype('str')
    TextAudio = np.array(TextAudio.dropna()).astype('str')
    
    abc = np.array((task_fusion, reading, cookie, memory, pupil, ET_basic, Eye, Language, Eye_Reading, NLP_Reading, TextAudio))
    
    for i in range(len(abc)):
        for j in range(len(abc[i])):
            if abc[i][j][1] == 'RandomForest':
                abc[i][j][1] = 'RF'
            elif abc[i][j][1] == 'GausNaiveBayes':
                abc[i][j][1] = 'GNB'
            elif abc[i][j][1] == 'LogReg':
                abc[i][j][1] = 'LR' 
    
    geometry_options = {"tmargin": "1cm", "lmargin": "1cm"}
    doc = Document(geometry_options=geometry_options)

    # for overall task_fusion_result
    with doc.create(Section('Results')):
        for i in range(len(abc)): 
            overall = abc[i]
            with doc.create(Subsection(overall[0][0])):
                with doc.create(Tabular('c c c c c c c c')) as table:
                    table.add_hline()
                    table.add_row(('Algo', 'N', 'AUC', 'F1', 'Accuracy', 'Precision', 'Recall', 'Specificity'))
                    table.add_hline()
                    for i in range(len(overall)):
                        table.add_row((overall[i][1], '162', 
                                       overall[i][3] + pm + overall[i][12],      # roc
                                       overall[i][4] + pm + overall[i][9],      # f1
                                       overall[i][2] + pm + overall[i][8],      # acc
                                       overall[i][5] + pm + overall[i][10],      # prec
                                       overall[i][6] + pm + overall[i][11],      # rec
                                       overall[i][7] + pm + overall[i][13]))     # spec

    doc.generate_pdf(filename, clean_tex=False, compiler='pdflatex')


main()
