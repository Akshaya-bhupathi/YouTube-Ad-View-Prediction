**Data** **Description**

The file train.csv contains metrics and other details of about 15000 youtube videos. The metrics include number of views,likes,dislikes,comments and apart from that published date,duration and category are also included. The train.csv file also contains the metric number of adviews which is our target variable for prediction


Importing Libraries


```python
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
```

Importing data


```python
path = "" # put path of your folder of your data if it's not in the same folder
data_train = pd.read_csv(path + "youtubead_train.csv")
```


```python
data_train.head()
```





  <div id="df-c6bbfb18-9555-43f5-863f-87f29f2b4701">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>vidid</th>
      <th>adview</th>
      <th>views</th>
      <th>likes</th>
      <th>dislikes</th>
      <th>comment</th>
      <th>published</th>
      <th>duration</th>
      <th>category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>VID_18655</td>
      <td>40</td>
      <td>1031602</td>
      <td>8523</td>
      <td>363</td>
      <td>1095</td>
      <td>9/14/2016</td>
      <td>PT7M37S</td>
      <td>F</td>
    </tr>
    <tr>
      <th>1</th>
      <td>VID_14135</td>
      <td>2</td>
      <td>1707</td>
      <td>56</td>
      <td>2</td>
      <td>6</td>
      <td>10/1/2016</td>
      <td>PT9M30S</td>
      <td>D</td>
    </tr>
    <tr>
      <th>2</th>
      <td>VID_2187</td>
      <td>1</td>
      <td>2023</td>
      <td>25</td>
      <td>0</td>
      <td>2</td>
      <td>7/2/2016</td>
      <td>PT2M16S</td>
      <td>C</td>
    </tr>
    <tr>
      <th>3</th>
      <td>VID_23096</td>
      <td>6</td>
      <td>620860</td>
      <td>777</td>
      <td>161</td>
      <td>153</td>
      <td>7/27/2016</td>
      <td>PT4M22S</td>
      <td>H</td>
    </tr>
    <tr>
      <th>4</th>
      <td>VID_10175</td>
      <td>1</td>
      <td>666</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>6/29/2016</td>
      <td>PT31S</td>
      <td>D</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-c6bbfb18-9555-43f5-863f-87f29f2b4701')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-c6bbfb18-9555-43f5-863f-87f29f2b4701 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-c6bbfb18-9555-43f5-863f-87f29f2b4701');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
data_train.shape
```




    (14999, 9)



Assigning each category a number for Category feature


```python
category = {'A': 1, 'B': 2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'H':8}
data_train["category"] = data_train["category"].map(category)
data_train.head()
```





  <div id="df-313766aa-3253-455e-916f-db802c4f4ebc">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>vidid</th>
      <th>adview</th>
      <th>views</th>
      <th>likes</th>
      <th>dislikes</th>
      <th>comment</th>
      <th>published</th>
      <th>duration</th>
      <th>category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>VID_18655</td>
      <td>40</td>
      <td>1031602</td>
      <td>8523</td>
      <td>363</td>
      <td>1095</td>
      <td>9/14/2016</td>
      <td>PT7M37S</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>VID_14135</td>
      <td>2</td>
      <td>1707</td>
      <td>56</td>
      <td>2</td>
      <td>6</td>
      <td>10/1/2016</td>
      <td>PT9M30S</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>VID_2187</td>
      <td>1</td>
      <td>2023</td>
      <td>25</td>
      <td>0</td>
      <td>2</td>
      <td>7/2/2016</td>
      <td>PT2M16S</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>VID_23096</td>
      <td>6</td>
      <td>620860</td>
      <td>777</td>
      <td>161</td>
      <td>153</td>
      <td>7/27/2016</td>
      <td>PT4M22S</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>VID_10175</td>
      <td>1</td>
      <td>666</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>6/29/2016</td>
      <td>PT31S</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-313766aa-3253-455e-916f-db802c4f4ebc')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-313766aa-3253-455e-916f-db802c4f4ebc button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-313766aa-3253-455e-916f-db802c4f4ebc');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




Removing character "F" present in data


```python
data_train = data_train[data_train.views!='F']
data_train = data_train[data_train.likes!='F']
data_train = data_train[data_train.dislikes!='F']
data_train = data_train[data_train.comment!='F']
```

Convert values to integers for views, likes, dislikes, comments and adview


```python
data_train["views"] = pd.to_numeric(data_train["views"])
data_train["comment"] = pd.to_numeric(data_train["comment"])
data_train["likes"] = pd.to_numeric(data_train["likes"])
data_train["dislikes"] = pd.to_numeric(data_train["dislikes"])
data_train["adview"] = pd.to_numeric(data_train["adview"])
```


```python
column_vidid = data_train['vidid']
```

Encoding features like category, Duration, Vidid


```python
from sklearn.preprocessing import LabelEncoder
data_train['duration'] = LabelEncoder().fit_transform(data_train['duration'])
data_train['vidid'] = LabelEncoder().fit_transform(data_train['vidid'])
data_train['published'] = LabelEncoder().fit_transform(data_train['published'])
```


```python
data_train.head()
```





  <div id="df-b6be104e-c677-4e20-a06b-16821d4499e9">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>vidid</th>
      <th>adview</th>
      <th>views</th>
      <th>likes</th>
      <th>dislikes</th>
      <th>comment</th>
      <th>published</th>
      <th>duration</th>
      <th>category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5912</td>
      <td>40</td>
      <td>1031602</td>
      <td>8523</td>
      <td>363</td>
      <td>1095</td>
      <td>2235</td>
      <td>2925</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2741</td>
      <td>2</td>
      <td>1707</td>
      <td>56</td>
      <td>2</td>
      <td>6</td>
      <td>207</td>
      <td>3040</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8138</td>
      <td>1</td>
      <td>2023</td>
      <td>25</td>
      <td>0</td>
      <td>2</td>
      <td>1905</td>
      <td>1863</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9005</td>
      <td>6</td>
      <td>620860</td>
      <td>777</td>
      <td>161</td>
      <td>153</td>
      <td>1952</td>
      <td>2546</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>122</td>
      <td>1</td>
      <td>666</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1783</td>
      <td>1963</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-b6be104e-c677-4e20-a06b-16821d4499e9')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-b6be104e-c677-4e20-a06b-16821d4499e9 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-b6be104e-c677-4e20-a06b-16821d4499e9');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




Convert Time_in_sec for duration


```python
import datetime
import time
```


```python
def checki(x):
    y = x[2:]
    h = ''
    m = '' 
    s = ''
    nm = ''
    p = ['H', 'M', 'S']
    for i in y:
        if i not in p:
            nm+=i
        else:
            if(i=="H"):
                h = nm
                nm = ''
            elif(i=="H"):
                m = nm
                nm = ''
            else:
                s = nm
                nm = ''
    if(h==''):
        h = '00'
    if(m == ''):
        m = '00'
    if(s==''):
        s = '00'
    bp = h+':'+m+':'+s
    return bp
```


```python
train = pd.read_csv("youtubead_train.csv")
mp = pd.read_csv(path + "youtubead_train.csv")["duration"]
time = mp.apply(checki)
```


```python
def func_sec(time_string):
    h, m, s = time_string.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)
```


```python
time1 = time.apply(func_sec)
```


```python
data_train["duration"] = time1
data_train.head()
```





  <div id="df-4228795b-347e-4acb-9dd0-2423cffbc906">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>vidid</th>
      <th>adview</th>
      <th>views</th>
      <th>likes</th>
      <th>dislikes</th>
      <th>comment</th>
      <th>published</th>
      <th>duration</th>
      <th>category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5912</td>
      <td>40</td>
      <td>1031602</td>
      <td>8523</td>
      <td>363</td>
      <td>1095</td>
      <td>2235</td>
      <td>37</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2741</td>
      <td>2</td>
      <td>1707</td>
      <td>56</td>
      <td>2</td>
      <td>6</td>
      <td>207</td>
      <td>30</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8138</td>
      <td>1</td>
      <td>2023</td>
      <td>25</td>
      <td>0</td>
      <td>2</td>
      <td>1905</td>
      <td>16</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9005</td>
      <td>6</td>
      <td>620860</td>
      <td>777</td>
      <td>161</td>
      <td>153</td>
      <td>1952</td>
      <td>22</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>122</td>
      <td>1</td>
      <td>666</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1783</td>
      <td>31</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-4228795b-347e-4acb-9dd0-2423cffbc906')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-4228795b-347e-4acb-9dd0-2423cffbc906 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-4228795b-347e-4acb-9dd0-2423cffbc906');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




# VISUALISATION
Individual plots
     


```python
plt.hist(data_train["category"])
plt.show()
plt.plot(data_train["adview"])
plt.show()
```


    
![png](output_26_0.png)
    



    
![png](output_26_1.png)
    


Remove videos with adview greater than 2000000 as outlier


```python
data_train = data_train[data_train["adview"]<2000000]
```


```python
# Heatmap
import seaborn as sns
```


```python
f, ax = plt.subplots(figsize=(10,8))
corr = data_train.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax, annot=True)
plt.show()
```


    
![png](output_30_0.png)
    


split data


```python
Y_train = pd.DataFrame(data_train.iloc[:, 1].values, columns = ['target'])
data_train = data_train.drop(["adview"],axis=1)
data_train = data_train.drop(["vidid"],axis=1)
data_train.head()
```





  <div id="df-c78934e8-0834-47d6-853b-3411574d19e5">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>views</th>
      <th>likes</th>
      <th>dislikes</th>
      <th>comment</th>
      <th>published</th>
      <th>duration</th>
      <th>category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1031602</td>
      <td>8523</td>
      <td>363</td>
      <td>1095</td>
      <td>2235</td>
      <td>37</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1707</td>
      <td>56</td>
      <td>2</td>
      <td>6</td>
      <td>207</td>
      <td>30</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023</td>
      <td>25</td>
      <td>0</td>
      <td>2</td>
      <td>1905</td>
      <td>16</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>620860</td>
      <td>777</td>
      <td>161</td>
      <td>153</td>
      <td>1952</td>
      <td>22</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>666</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1783</td>
      <td>31</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-c78934e8-0834-47d6-853b-3411574d19e5')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-c78934e8-0834-47d6-853b-3411574d19e5 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-c78934e8-0834-47d6-853b-3411574d19e5');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_train, Y_train, test_size=0.2, random_state = 42)
```


```python
X_train.shape
```




    (11708, 7)



Normalise Data


```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
```

Evaluation Metrics


```python
from sklearn import metrics
def print_error(X_test, y_test, model_name):
    prediction = model_name.predict(X_test)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, prediction))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, prediction))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))
```

Linear Regression


```python
from sklearn import linear_model
linear_regression = linear_model.LinearRegression()
linear_regression.fit(X_train, y_train)
print_error(X_test, y_test, linear_regression)
```

    Mean Absolute Error: 3486.914249706325
    Mean Squared Error: 838702210.0737183
    Root Mean Squared Error: 28960.35583472203
    

Desicion Tree Regressor


```python
from sklearn.tree import DecisionTreeRegressor
decision_tree = DecisionTreeRegressor()
decision_tree.fit(X_train, y_train)
print_error(X_test, y_test, decision_tree)
```

    Mean Absolute Error: 4602.836407103825
    Mean Squared Error: 2798499198.7940574
    Root Mean Squared Error: 52900.8430820725
    

Random Forest Regressor


```python
from sklearn.ensemble import RandomForestRegressor
n_estimators = 200
max_depth = 25
min_samples_split = 15
min_samples_leaf = 2
random_forest = RandomForestRegressor(n_estimators = n_estimators,  max_depth = max_depth, min_samples_split = min_samples_split, min_samples_leaf= min_samples_leaf)
random_forest.fit(X_train, y_train)
print_error(X_test, y_test, random_forest)
```

    Mean Absolute Error: 3872.1309079517523
    Mean Squared Error: 778294869.6792512
    Root Mean Squared Error: 27897.936656305807
    

Support Vector Regressor


```python
from sklearn.svm import SVR
supportvector_regressor = SVR()
supportvector_regressor.fit(X_train,y_train)
print_error(X_test, y_test, linear_regression)
```

    Mean Absolute Error: 3486.914249706325
    Mean Squared Error: 838702210.0737183
    Root Mean Squared Error: 28960.35583472203
    

Artificial Neural Network


```python
import keras
from keras.layers import Dense
```


```python
ann = keras.models.Sequential([Dense(6, activation="relu",
                               input_shape=X_train.shape[1:]),
                               Dense(6,activation="relu"),
                               Dense(1)])
```


```python
import keras.optimizers
opt = "adam"
loss = keras.losses.mean_squared_error
ann.compile(optimizer=opt, loss=loss, metrics=["mean_absolute_error"])
```


```python
history = ann.fit(X_train,y_train,epochs=100)
```

    Epoch 1/100
    366/366 [==============================] - 1s 1ms/step - loss: 767407552.0000 - mean_absolute_error: 1694.2950
    Epoch 2/100
    366/366 [==============================] - 1s 1ms/step - loss: 767406080.0000 - mean_absolute_error: 1693.9208
    Epoch 3/100
    366/366 [==============================] - 1s 1ms/step - loss: 767400960.0000 - mean_absolute_error: 1693.4783
    Epoch 4/100
    366/366 [==============================] - 1s 1ms/step - loss: 767387712.0000 - mean_absolute_error: 1694.8165
    Epoch 5/100
    366/366 [==============================] - 0s 1ms/step - loss: 767362752.0000 - mean_absolute_error: 1698.6813
    Epoch 6/100
    366/366 [==============================] - 1s 1ms/step - loss: 767331584.0000 - mean_absolute_error: 1706.5372
    Epoch 7/100
    366/366 [==============================] - 0s 1ms/step - loss: 767296896.0000 - mean_absolute_error: 1714.7937
    Epoch 8/100
    366/366 [==============================] - 0s 1ms/step - loss: 767251072.0000 - mean_absolute_error: 1726.0453
    Epoch 9/100
    366/366 [==============================] - 1s 1ms/step - loss: 767196672.0000 - mean_absolute_error: 1740.6859
    Epoch 10/100
    366/366 [==============================] - 0s 1ms/step - loss: 767133952.0000 - mean_absolute_error: 1756.5933
    Epoch 11/100
    366/366 [==============================] - 1s 1ms/step - loss: 767065920.0000 - mean_absolute_error: 1775.6776
    Epoch 12/100
    366/366 [==============================] - 1s 1ms/step - loss: 766993728.0000 - mean_absolute_error: 1796.1136
    Epoch 13/100
    366/366 [==============================] - 1s 1ms/step - loss: 766914112.0000 - mean_absolute_error: 1818.3434
    Epoch 14/100
    366/366 [==============================] - 1s 1ms/step - loss: 766822272.0000 - mean_absolute_error: 1843.8729
    Epoch 15/100
    366/366 [==============================] - 1s 1ms/step - loss: 766719552.0000 - mean_absolute_error: 1876.2374
    Epoch 16/100
    366/366 [==============================] - 1s 1ms/step - loss: 766613632.0000 - mean_absolute_error: 1911.7988
    Epoch 17/100
    366/366 [==============================] - 1s 1ms/step - loss: 766497472.0000 - mean_absolute_error: 1945.1051
    Epoch 18/100
    366/366 [==============================] - 1s 1ms/step - loss: 766379392.0000 - mean_absolute_error: 1987.8641
    Epoch 19/100
    366/366 [==============================] - 0s 1ms/step - loss: 766260992.0000 - mean_absolute_error: 2023.9332
    Epoch 20/100
    366/366 [==============================] - 1s 1ms/step - loss: 766144896.0000 - mean_absolute_error: 2069.1877
    Epoch 21/100
    366/366 [==============================] - 1s 1ms/step - loss: 766030912.0000 - mean_absolute_error: 2111.4558
    Epoch 22/100
    366/366 [==============================] - 1s 1ms/step - loss: 765918208.0000 - mean_absolute_error: 2153.6545
    Epoch 23/100
    366/366 [==============================] - 1s 1ms/step - loss: 765810944.0000 - mean_absolute_error: 2198.2397
    Epoch 24/100
    366/366 [==============================] - 1s 1ms/step - loss: 765706048.0000 - mean_absolute_error: 2240.9998
    Epoch 25/100
    366/366 [==============================] - 1s 1ms/step - loss: 765607296.0000 - mean_absolute_error: 2284.5688
    Epoch 26/100
    366/366 [==============================] - 1s 1ms/step - loss: 765512896.0000 - mean_absolute_error: 2331.1726
    Epoch 27/100
    366/366 [==============================] - 1s 1ms/step - loss: 765425344.0000 - mean_absolute_error: 2373.8394
    Epoch 28/100
    366/366 [==============================] - 1s 1ms/step - loss: 765345152.0000 - mean_absolute_error: 2413.5679
    Epoch 29/100
    366/366 [==============================] - 1s 1ms/step - loss: 765268800.0000 - mean_absolute_error: 2464.8958
    Epoch 30/100
    366/366 [==============================] - 1s 1ms/step - loss: 765203456.0000 - mean_absolute_error: 2501.5432
    Epoch 31/100
    366/366 [==============================] - 1s 1ms/step - loss: 765145216.0000 - mean_absolute_error: 2540.1714
    Epoch 32/100
    366/366 [==============================] - 1s 1ms/step - loss: 765091968.0000 - mean_absolute_error: 2568.2539
    Epoch 33/100
    366/366 [==============================] - 1s 1ms/step - loss: 765040000.0000 - mean_absolute_error: 2598.3523
    Epoch 34/100
    366/366 [==============================] - 1s 1ms/step - loss: 764992704.0000 - mean_absolute_error: 2636.6357
    Epoch 35/100
    366/366 [==============================] - 1s 1ms/step - loss: 764951296.0000 - mean_absolute_error: 2670.5247
    Epoch 36/100
    366/366 [==============================] - 0s 1ms/step - loss: 764913600.0000 - mean_absolute_error: 2700.6685
    Epoch 37/100
    366/366 [==============================] - 0s 1ms/step - loss: 764879040.0000 - mean_absolute_error: 2728.5713
    Epoch 38/100
    366/366 [==============================] - 1s 1ms/step - loss: 764847232.0000 - mean_absolute_error: 2756.9204
    Epoch 39/100
    366/366 [==============================] - 0s 1ms/step - loss: 764820928.0000 - mean_absolute_error: 2784.1680
    Epoch 40/100
    366/366 [==============================] - 1s 1ms/step - loss: 764794944.0000 - mean_absolute_error: 2809.4265
    Epoch 41/100
    366/366 [==============================] - 1s 1ms/step - loss: 764774784.0000 - mean_absolute_error: 2838.3176
    Epoch 42/100
    366/366 [==============================] - 1s 1ms/step - loss: 764755072.0000 - mean_absolute_error: 2853.0518
    Epoch 43/100
    366/366 [==============================] - 1s 1ms/step - loss: 764739136.0000 - mean_absolute_error: 2871.5203
    Epoch 44/100
    366/366 [==============================] - 1s 1ms/step - loss: 764720320.0000 - mean_absolute_error: 2891.5076
    Epoch 45/100
    366/366 [==============================] - 1s 1ms/step - loss: 764706752.0000 - mean_absolute_error: 2911.7341
    Epoch 46/100
    366/366 [==============================] - 1s 1ms/step - loss: 764693952.0000 - mean_absolute_error: 2926.6123
    Epoch 47/100
    366/366 [==============================] - 1s 1ms/step - loss: 764683648.0000 - mean_absolute_error: 2947.4124
    Epoch 48/100
    366/366 [==============================] - 0s 1ms/step - loss: 764674752.0000 - mean_absolute_error: 2960.0356
    Epoch 49/100
    366/366 [==============================] - 1s 1ms/step - loss: 764663104.0000 - mean_absolute_error: 2971.7664
    Epoch 50/100
    366/366 [==============================] - 1s 1ms/step - loss: 764656064.0000 - mean_absolute_error: 2996.0986
    Epoch 51/100
    366/366 [==============================] - 1s 1ms/step - loss: 764647744.0000 - mean_absolute_error: 3002.5134
    Epoch 52/100
    366/366 [==============================] - 1s 1ms/step - loss: 764641600.0000 - mean_absolute_error: 3017.2822
    Epoch 53/100
    366/366 [==============================] - 1s 1ms/step - loss: 764634432.0000 - mean_absolute_error: 3025.1958
    Epoch 54/100
    366/366 [==============================] - 1s 1ms/step - loss: 764629824.0000 - mean_absolute_error: 3029.5549
    Epoch 55/100
    366/366 [==============================] - 1s 1ms/step - loss: 764623744.0000 - mean_absolute_error: 3050.9177
    Epoch 56/100
    366/366 [==============================] - 1s 1ms/step - loss: 764619584.0000 - mean_absolute_error: 3046.7629
    Epoch 57/100
    366/366 [==============================] - 1s 1ms/step - loss: 764613952.0000 - mean_absolute_error: 3058.2864
    Epoch 58/100
    366/366 [==============================] - 1s 1ms/step - loss: 764611200.0000 - mean_absolute_error: 3063.1951
    Epoch 59/100
    366/366 [==============================] - 1s 1ms/step - loss: 764605888.0000 - mean_absolute_error: 3074.2852
    Epoch 60/100
    366/366 [==============================] - 0s 1ms/step - loss: 764601792.0000 - mean_absolute_error: 3087.3296
    Epoch 61/100
    366/366 [==============================] - 1s 1ms/step - loss: 764596928.0000 - mean_absolute_error: 3089.9504
    Epoch 62/100
    366/366 [==============================] - 0s 1ms/step - loss: 764593536.0000 - mean_absolute_error: 3093.0930
    Epoch 63/100
    366/366 [==============================] - 1s 1ms/step - loss: 764591680.0000 - mean_absolute_error: 3096.5696
    Epoch 64/100
    366/366 [==============================] - 0s 1ms/step - loss: 764587200.0000 - mean_absolute_error: 3106.0640
    Epoch 65/100
    366/366 [==============================] - 1s 1ms/step - loss: 764584640.0000 - mean_absolute_error: 3108.3484
    Epoch 66/100
    366/366 [==============================] - 0s 1ms/step - loss: 764581376.0000 - mean_absolute_error: 3110.0134
    Epoch 67/100
    366/366 [==============================] - 1s 1ms/step - loss: 764578304.0000 - mean_absolute_error: 3117.1890
    Epoch 68/100
    366/366 [==============================] - 1s 1ms/step - loss: 764576128.0000 - mean_absolute_error: 3129.9172
    Epoch 69/100
    366/366 [==============================] - 1s 1ms/step - loss: 764573440.0000 - mean_absolute_error: 3126.4800
    Epoch 70/100
    366/366 [==============================] - 1s 1ms/step - loss: 764572416.0000 - mean_absolute_error: 3124.2119
    Epoch 71/100
    366/366 [==============================] - 1s 1ms/step - loss: 764567744.0000 - mean_absolute_error: 3131.1934
    Epoch 72/100
    366/366 [==============================] - 1s 1ms/step - loss: 764564800.0000 - mean_absolute_error: 3137.0024
    Epoch 73/100
    366/366 [==============================] - 1s 1ms/step - loss: 764563328.0000 - mean_absolute_error: 3138.9644
    Epoch 74/100
    366/366 [==============================] - 1s 1ms/step - loss: 764559616.0000 - mean_absolute_error: 3139.8938
    Epoch 75/100
    366/366 [==============================] - 1s 1ms/step - loss: 764558336.0000 - mean_absolute_error: 3133.8762
    Epoch 76/100
    366/366 [==============================] - 1s 1ms/step - loss: 764555584.0000 - mean_absolute_error: 3143.4695
    Epoch 77/100
    366/366 [==============================] - 1s 1ms/step - loss: 764553152.0000 - mean_absolute_error: 3151.2898
    Epoch 78/100
    366/366 [==============================] - 1s 1ms/step - loss: 764551552.0000 - mean_absolute_error: 3143.3123
    Epoch 79/100
    366/366 [==============================] - 1s 1ms/step - loss: 764549376.0000 - mean_absolute_error: 3145.8743
    Epoch 80/100
    366/366 [==============================] - 1s 1ms/step - loss: 764545536.0000 - mean_absolute_error: 3143.2712
    Epoch 81/100
    366/366 [==============================] - 1s 1ms/step - loss: 764545152.0000 - mean_absolute_error: 3155.8210
    Epoch 82/100
    366/366 [==============================] - 1s 1ms/step - loss: 764541440.0000 - mean_absolute_error: 3157.3337
    Epoch 83/100
    366/366 [==============================] - 1s 1ms/step - loss: 764540160.0000 - mean_absolute_error: 3146.4963
    Epoch 84/100
    366/366 [==============================] - 1s 1ms/step - loss: 764538112.0000 - mean_absolute_error: 3147.6106
    Epoch 85/100
    366/366 [==============================] - 1s 1ms/step - loss: 764535488.0000 - mean_absolute_error: 3161.7742
    Epoch 86/100
    366/366 [==============================] - 1s 2ms/step - loss: 764532352.0000 - mean_absolute_error: 3155.4639
    Epoch 87/100
    366/366 [==============================] - 1s 1ms/step - loss: 764529024.0000 - mean_absolute_error: 3155.9751
    Epoch 88/100
    366/366 [==============================] - 1s 1ms/step - loss: 764528960.0000 - mean_absolute_error: 3155.8193
    Epoch 89/100
    366/366 [==============================] - 1s 1ms/step - loss: 764527424.0000 - mean_absolute_error: 3165.8860
    Epoch 90/100
    366/366 [==============================] - 1s 1ms/step - loss: 764523584.0000 - mean_absolute_error: 3161.4348
    Epoch 91/100
    366/366 [==============================] - 1s 1ms/step - loss: 764523200.0000 - mean_absolute_error: 3167.5679
    Epoch 92/100
    366/366 [==============================] - 1s 1ms/step - loss: 764519104.0000 - mean_absolute_error: 3160.3689
    Epoch 93/100
    366/366 [==============================] - 1s 1ms/step - loss: 764516480.0000 - mean_absolute_error: 3164.5122
    Epoch 94/100
    366/366 [==============================] - 1s 1ms/step - loss: 764518208.0000 - mean_absolute_error: 3159.7715
    Epoch 95/100
    366/366 [==============================] - 1s 1ms/step - loss: 764513216.0000 - mean_absolute_error: 3166.8760
    Epoch 96/100
    366/366 [==============================] - 1s 1ms/step - loss: 764511040.0000 - mean_absolute_error: 3160.7615
    Epoch 97/100
    366/366 [==============================] - 1s 1ms/step - loss: 764506816.0000 - mean_absolute_error: 3169.1443
    Epoch 98/100
    366/366 [==============================] - 1s 1ms/step - loss: 764505664.0000 - mean_absolute_error: 3166.5818
    Epoch 99/100
    366/366 [==============================] - 1s 1ms/step - loss: 764503680.0000 - mean_absolute_error: 3163.3989
    Epoch 100/100
    366/366 [==============================] - 1s 1ms/step - loss: 764500928.0000 - mean_absolute_error: 3170.1589
    


```python
ann.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense (Dense)               (None, 6)                 48        
                                                                     
     dense_1 (Dense)             (None, 6)                 42        
                                                                     
     dense_2 (Dense)             (None, 1)                 7         
                                                                     
    =================================================================
    Total params: 97
    Trainable params: 97
    Non-trainable params: 0
    _________________________________________________________________
    


```python
print_error(X_test,y_test,ann)
```

    Mean Absolute Error: 3170.3323490762973
    Mean Squared Error: 831255992.5307742
    Root Mean Squared Error: 28831.510410153234
    

Saving Scikitlearn models


```python
import joblib
joblib.dump(decision_tree,"decisiontree_youtubeadview.pkl")
```




    ['decisiontree_youtubeadview.pkl']



Saving Keras Atrificial Neural Network model


```python
ann.save("ann_youtubeadview.h5")
```


```python
prediction = decision_tree.predict(X_test)
```


```python
prediction=pd.DataFrame(prediction)
prediction.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2928 entries, 0 to 2927
    Data columns (total 1 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   0       2928 non-null   float64
    dtypes: float64(1)
    memory usage: 23.0 KB
    


```python
prediction = prediction.rename(columns={0: "Adview"})
```


```python
prediction.head()
```





  <div id="df-f185e037-335c-4c63-960d-65c5d223fcac">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Adview</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>83.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-f185e037-335c-4c63-960d-65c5d223fcac')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-f185e037-335c-4c63-960d-65c5d223fcac button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-f185e037-335c-4c63-960d-65c5d223fcac');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
prediction.to_csv('predictions.csv')
```
