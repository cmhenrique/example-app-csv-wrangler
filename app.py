#load the libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from sksurv.datasets import load_gbsg2
from sksurv.preprocessing import OneHotEncoder
from sksurv.ensemble.forest import RandomSurvivalForest

from collections import Counter
#import pandas_profiling as pp
from sklearn.model_selection import train_test_split
from sklearn.ensemble._forest import RandomForestClassifier
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import pickle5 as pkl
from joblib import dump, load

## HEADER
from PIL import Image

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

st.markdown('<b class="big-font">Análise da influência de fatores clínicos do doador/receptor e circunstanciais na sobrevida do transplante renal no curto e longo prazo</b>', unsafe_allow_html=True)


col1,col2,col3 = st.columns([1,1,2]) 

@st.cache(allow_output_mutation=True)
def model():
    with open('ICURO_data_y.pkl', 'rb') as f:
        data_y = pkl.load(f)
    with open('ICURO_data_x.pkl', 'rb') as f:
        data_x = pkl.load(f)  

    random_state = 12
    feature_names = data_x.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.25, random_state=random_state)

    rsf = RandomSurvivalForest(n_estimators=1000,
                               min_samples_split=10,
                               min_samples_leaf=20,
                               max_features="sqrt",
                               n_jobs=-1,
                               random_state=random_state)
    rsf.fit(X_train, y_train)
    rsf.score(X_test, y_test)

    return rsf, feature_names

with st.expander("Dados gerais do transplante"):
    colAd,colBd,colDRd = st.columns([1,1,1])
    with colAd:
        tif = st.number_input('Tempo de Isquemia Fria', step = 1, key=7)
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        tipo_doador = st.radio('Tipo de Doador', ('Vivo', 'Cadaver'), key=8)
    with colBd:
        idade_d = st.number_input('Idade do Doador', step = 1, key=9)
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        sexo_d = st.radio('Sexo do Receptor', ('Masculino', 'Feminino'), key=10)
    with colDRd:
        idade_r = st.number_input('Idade do Receptor', step = 1,key=11)
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        sexo_r = st.radio('Sexo do Receptor', ('Masculino', 'Feminino'), key=12)

with st.expander("Dados de HLA do Doador"):
    colAd,colBd,colDRd = st.columns([1,1,1])
    with colAd:
        HLAa1_d = st.number_input('HLA A1', step = 1, key=1)
        HLAa2_d = st.number_input('HLA A2', step = 1, key=2)
    with colBd:
        HLAb1_d = st.number_input('HLA B1', step = 1, key=3)
        HLAb2_d = st.number_input('HLA B2', step = 1, key=4)
    with colDRd:
        HLAdr1_d = st.number_input('HLA DR1', step = 1,key=5)
        HLAdr2_d = st.number_input('HLA DR2', step = 1, key=6)

with st.expander("Dados de HLA do Receptor"):
    colAr,colBr,colDRr = st.columns([1,1,1])
    with colAr:
        HLAa1_r = st.number_input('HLA A1', step = 1)
        HLAa2_r = st.number_input('HLA A2', step = 1)
    with colBr:
        HLAb1_r = st.number_input('HLA B1', step = 1)
        HLAb2_r = st.number_input('HLA B2', step = 1)
    with colDRr:
        HLAdr1_r = st.number_input('HLA DR1', step = 1)
        HLAdr2_r = st.number_input('HLA DR2', step = 1)

if (st.button('Predizer sobrevida após transplante')):
    rsf, feature_names = model()

    ## Compatibilidades
    # Para cada HLA, temos 4 combinações para checar
    
    match = 0
    mismatch = 0
    
    #### COMPATIBILIDADE HLAa ####
    if HLAa1_r == HLAa1_d:
        comp_HLAa1a1 = 0
    else:
        comp_HLAa1a1 = 1
        
    if HLAa1_r == HLAa2_d:
        comp_HLAa1a2 = 0
    else:
        comp_HLAa1a2 = 1 
        
    if HLAa2_r == HLAa1_d:
        comp_HLAa2a1 = 0
    else:
        comp_HLAa2a1 = 1
        
    if HLAa2_r == HLAa2_d:
        comp_HLAa2a2 = 0
    else:
        comp_HLAa2a2 = 1
    
     #### COMPATIBILIDADE HLAb ####
    if HLAb1_r == HLAb1_d:
        comp_HLAb1b1 = 0
    else:
        comp_HLAb1b1 = 1
        
    if HLAb1_r == HLAb2_d:
        comp_HLAb1b2 = 0
    else:
        comp_HLAb1b2 = 1 
        
    if HLAb2_r == HLAb1_d:
        comp_HLAb2b1 = 0
    else:
        comp_HLAb2b1 = 1
    
    if HLAb2_r == HLAb2_d:
        comp_HLAb2b2 = 0
    else:
        comp_HLAb2b2 = 1
        
    #### COMPATIBILIDADE HLAdr ####
    if HLAdr1_r == HLAdr1_d:
        comp_HLAdr1dr1 = 0
    else:
        comp_HLAdr1dr1 = 1
        
    if HLAdr1_r == HLAdr2_d:
        comp_HLAdr1dr2 = 0
    else:
        comp_HLAdr1dr2 = 1 
        
    if HLAdr2_r == HLAdr1_d:
        comp_HLAdr2dr1 = 0
    else:
        comp_HLAdr2dr1 = 1
        
    if HLAdr2_r == HLAdr2_d:
        comp_HLAdr2dr2 = 0
    else:
        comp_HLAdr2dr2 = 1
    
    #MATCH
    l_a = [HLAa1_d, HLAa2_d]
    l_b = [HLAb1_d, HLAb2_d]
    l_dr = [HLAdr1_d, HLAdr2_d]

    if HLAa1_r in l_a and pd.isnull(HLAa1_r) == False and pd.isnull(HLAa1_d) == False:
        match += 1

    if HLAa2_r in l_a and pd.isnull(HLAa2_r) == False and pd.isnull(HLAa2_d) == False:
        match += 1

    if HLAb1_r in l_b and pd.isnull(HLAb1_r) == False and pd.isnull(HLAb1_d) == False:
        match += 1

    if HLAb2_r in l_b and pd.isnull(HLAb2_r) == False and pd.isnull(HLAb2_d) == False:
        match += 1

    if HLAdr1_r in l_dr and pd.isnull(HLAdr1_r) == False and pd.isnull(HLAdr1_d) == False:
        match += 1

    if HLAdr2_r in l_dr and pd.isnull(HLAdr2_r) == False and pd.isnull(HLAdr2_d) == False:
        match += 1
    
    #MISMATCH
    l_a = [HLAa1_d, HLAa2_d]
    l_b = [HLAb1_d, HLAb2_d]
    l_dr = [HLAdr1_d, HLAdr2_d]

    if HLAa1_r not in l_a and pd.isnull(HLAa1_r) == False and pd.isnull(HLAa1_d) == False:
        mismatch += 1

    if HLAa2_r not in l_a and pd.isnull(HLAa2_r) == False and pd.isnull(HLAa2_d) == False:
        mismatch += 1

    if HLAb1_r not in l_b and pd.isnull(HLAb1_r) == False and pd.isnull(HLAb1_d) == False:
        mismatch += 1

    if HLAb2_r not in l_b and pd.isnull(HLAb2_r) == False and pd.isnull(HLAb2_d) == False:
        mismatch += 1

    if HLAdr1_r not in l_dr and pd.isnull(HLAdr1_r) == False and pd.isnull(HLAdr1_d) == False:
        mismatch += 1

    if HLAdr2_r not in l_dr and pd.isnull(HLAdr2_r) == False and pd.isnull(HLAdr2_d) == False:
        mismatch += 1

    if sexo_r == "Masculino":
        sexo_r = 1
    else:
        sexo_r = 0
        
    if sexo_d == "Masculino":
        sexo_d = 1
    else:
        sexo_d = 0 

    if tipo_doador == "Cadaver":
        tipo_doador = 0
    else:
        tipo_doador = 1

    paciente = [
        idade_r,
        idade_d,
        sexo_r,
        sexo_d,
        tif,
        tipo_doador,
        comp_HLAa1a1, 
        comp_HLAa1a2,
        comp_HLAa2a1,
        comp_HLAa2a2,
        comp_HLAb1b1,
        comp_HLAb1b2,
        comp_HLAb2b1,
        comp_HLAb2b2,
        comp_HLAdr1dr1, 
        comp_HLAdr1dr2,
        comp_HLAdr2dr1,
        comp_HLAdr2dr2,
        match,
        mismatch]

    pct = pd.DataFrame([paciente], columns = feature_names)


    surv = rsf.predict_survival_function(pct, return_array=True)

    col1,col2 = st.columns([1,1]) 

    for i, s in enumerate(surv):
        plt.step(rsf.event_times_/365, s, where="post", label=str(i))
        prob = s
        ano = (rsf.event_times_/365).astype(int)

        plt.ylabel("Survival probability")
        plt.xlabel("Time in years")
        plt.legend('Teste')
        plt.tight_layout()
        plt.grid(True)
        from io import BytesIO

        #st.text(prob[-1])
        #fig, ax = plt.subplots(figsize=(5, 5))
        #st.pyplot(fig = plt, use_container_width=True)
        fig = plt
        buf = BytesIO()
        fig.savefig(buf, format="png")

        with col1:
            st.image(buf, width = 500)

        with col2:
            st.subheader('Probabilidade de sobrevida do enxerto após transplante renal')
            index5 = np.where(ano==5)[0][0]
            index10 = np.where(ano==10)[0][0]
            index20 = np.where(ano==20)[0][0]

            p5 = "{0:.0%}".format(s[index5])
            p10 = "{0:.0%}".format(s[index10])
            p20 = "{0:.0%}".format(s[index20])

            st.metric(label="5 anos", value=p5)
            st.metric(label="10 anos", value=p10)
            st.metric(label="20 anos", value=p20)
