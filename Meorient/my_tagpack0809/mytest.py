#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 17:03:18 2019

@author: heimi
"""

import pandas as pd
import numpy as np

df_y=pd.DataFrame([[1],[2],[1],[4],[2]],columns=['y'])



y_classes = df_y.idxmax(1, skipna=False)


import pandas as pd
import numpy as np

# Create a pd.series that represents the categorical class of each one-hot encoded row
y_classes = df_y.idxmax(1, skipna=False)

from sklearn.preprocessing import LabelEncoder

# Instantiate the label encoder
le = LabelEncoder()

y_classes=df_y['y'].tolist()
# Fit the label encoder to our label series
le.fit(list(y_classes))

# Create integer based labels Series
y_integers = le.transform(list(y_classes))

# Create dict of labels : integer representation
labels_and_integers = dict(zip(y_classes, y_integers))

from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
sample_weights = compute_sample_weight('balanced', y_integers)

class_weights_dict = dict(zip(le.transform(list(le.classes_)), class_weights))











