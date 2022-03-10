#load the libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from sksurv.datasets import load_gbsg2
from sksurv.preprocessing import OneHotEncoder
from sksurv.ensemble import RandomSurvivalForest

from collections import Counter
import pandas_profiling as pp
from sklearn.model_selection import train_test_split
from sklearn.ensemble._forest import RandomForestClassifier
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

## HEADER
st.set_page_config(layout="wide")

st.markdown("""
<style>
.big-font {
    font-size:25px !important; }
.texto {
    font-size:13px;
    color:#FFF }
#texto {
    background-color: #ff4b4b;
    padding:5px;
    border-radius:5px;
}
.subtitulo {
    font-size:17px;
}
#subtitulo {
    margin-bottom:0px;
    z-index: 100;
}
</style>
""", unsafe_allow_html=True)

st.markdown(f'''
    <style>
        section[data-testid="stSidebar"] .css-ng1t4o {{width: 17rem;}}
        section[data-testid="stSidebar"] .css-1d391kg {{width: 17rem;}}
    </style>
''',unsafe_allow_html=True)

st.markdown('<b class="big-font">Predicting post-operative evolution of patients with Chronic Rhinosinusitis</b>', unsafe_allow_html=True)


col1,col2,col3 = st.columns([1,1,2]) 

@st.cache(allow_output_mutation=True)
def model():
    df = pd.read_csv("data7.csv") 
    df2 = df[['BIC', 'BIOFILME', 'ASMA','AERD', 'POLIPO', 'CULTURA', 'STAPHYLO', 'LM', 'EOSINOFILICO']]  # Features
    df3 = OneHotEncoder().fit_transform(df2)
    Xt= df3
    feature_names = Xt.columns.tolist()

    y = []
    for x in df.intervalo:
        if x > 8:
            a = False
        else:
            a = True
            
        y.append((a, x))

    dt = np.dtype([('cens', np.bool), ('time', np.float64, 1)])
    y = np.array(y, dtype=dt)

    X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=0.3, random_state=12)

    rsf = RandomSurvivalForest()
    rsf.fit(X_train, y_train)
    rsf.score(X_test, y_test)

    return rsf, feature_names

def predict():
    data = pd.read_csv("data7.csv") 

    X=data[['BIC', 'BIOFILME', 'ASMA','AERD', 'POLIPO', 'CULTURA', 'STAPHYLO', 'LM', 'EOSINOFILICO']]  # Features
    y=data['recidiva']  # Labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)
    
    ## LOGISTIC REGRESSION
    lr = LogisticRegression()
    model = lr.fit(X_train, y_train)
    lr_predict = lr.predict(X_test)
    lr_conf_matrix = confusion_matrix(y_test, lr_predict)
    lr_acc_score = accuracy_score(y_test, lr_predict)
    
    ## RANDOM FOREST
    rf = RandomForestClassifier(n_estimators=30, random_state=12,max_depth=1)
    rf.fit(X_train,y_train)
    rf_predicted = rf.predict(X_test)
    rf_conf_matrix = confusion_matrix(y_test, rf_predicted)
    rf_acc_score = accuracy_score(y_test, rf_predicted)

    return lr

with col1:
    st.markdown("Pacient 1 Clinical variables associated")
    lm =              st.slider("Lund-Mackay Score", 0, 20, 10, key=1)
    bf =         st.checkbox('Bacterial Biofilm', key=1)
    bic =         st.checkbox('Intracellular Bacteria', key=1)
    aerd =            st.checkbox('AERD', key=1)
    asma =            st.checkbox('Asthma', key=1)
    cultura =         st.checkbox('Bacterial Culture', key=1)
    polipo =         st.checkbox('Nasal Polyps', key=1)
    eosinofilico =        st.checkbox('Eosinophilic Polyps', key=1)
    staphylo =        st.checkbox('Staphylococcus aureus', key=1)

with col2:
    st.markdown("Pacient 2 Clinical variables associated")
    lm2 =              st.slider("Lund-Mackay Score", 0, 20, 10)
    bf2 =         st.checkbox('Bacterial Biofilm')
    bic2 =         st.checkbox('Intracellular Bacteria')
    aerd2 =            st.checkbox('AERD')
    asma2 =            st.checkbox('Asthma')
    cultura2 =         st.checkbox('Bacterial Culture')
    polipo2 =         st.checkbox('Nasal Polyps')
    eosinofilico2 =        st.checkbox('Eosinophilic Polyps')
    staphylo2 =        st.checkbox('Staphylococcus aureus')
    predizer = st.button('Predict patient outcome')

with col3:
    st.info("Kaplan Meier Curves")

if bf:
    bf = 1
else:
    bf =  0

if bic:
    bic = 1
else:
    bic =  0

if aerd:
    aerd = 1
else:
    aerd =  0

if asma:
    asma = 1
else:
    asma =  0

if cultura:
    cultura = 1
else:
    cultura =  0

if polipo:
    polipo = 1
else:
    polipo = 0

if eosinofilico:
    eosinofilico = 1
else:
    eosinofilico =  0

if staphylo:
    staphylo = 1
else:
    staphylo =  0

if lm >= 15:
    lm12 = 1
else:
    lm12 = 0

if bf2:
    bf2 = 1
else:
    bf2 =  0

if bic2:
    bic2 = 1
else:
    bic2 =  0

if aerd2:
    aerd2 = 1
else:
    aerd2 =  0

if asma2:
    asma2 = 1
else:
    asma2 =  0

if cultura2:
    cultura2 = 1
else:
    cultura2 =  0

if polipo2:
    polipo2 = 1
else:
    polipo2 = 0

if eosinofilico2:
    eosinofilico2 = 1
else:
    eosinofilico2 =  0

if staphylo2:
    staphylo2 = 1
else:
    staphylo2 =  0

if lm2 >= 15:
    lm22 = 1
else:
    lm22 = 0

if (predizer):
    texto = "Hello World"
    rsf, feature_names = model()
    rf = predict()
   
    paciente = [[bic, bf, asma, aerd, polipo, cultura, staphylo, lm, eosinofilico]]
    pct = pd.DataFrame(paciente, columns = feature_names)

    paciente2 = [[bic2, bf2, asma2, aerd2, polipo2, cultura2, staphylo2, lm2, eosinofilico2]]
    pct2 = pd.DataFrame(paciente2, columns = feature_names)

    with col3:
        st.subheader(" ")
    
        surv = rsf.predict_survival_function(pct, return_array=True)
        for i, s in enumerate(surv):
            plt.step(rsf.event_times_, s, where="post", label="Pacient 1")
            prob = s

        surv2 = rsf.predict_survival_function(pct2, return_array=True)

        for i, s in enumerate(surv2):
            plt.step(rsf.event_times_, s, where="post", label="Pacient 2")

        plt.ylabel("Survival probability")
        plt.xlabel("Time in years")
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        from io import BytesIO

        #st.text(prob[-1])
        #fig, ax = plt.subplots(figsize=(5, 5))
        #st.pyplot(fig = plt, use_container_width=True)
        fig = plt
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf, width = 500)
