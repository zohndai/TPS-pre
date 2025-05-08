import streamlit as st
import pandas as pd
#import numpy as np
from PIL import Image



st.set_page_config(
    page_title="Welcome to TP-Transformer",    
    page_icon="💧",        
    layout="wide",                
    initial_sidebar_state="auto" 
)

visitor = pd.read_csv("visi_num.txt")
#st.write(visitor)
visi_num = visitor['num'][0]

if 'visitor_count' not in st.session_state:
	st.session_state.visitor_count = int(visi_num)
if 'session_initialized' not in st.session_state:
	st.session_state.session_initialized = True
st.session_state.visitor_count += 1
visitor['num'][0] += 1
st.metric(label=f'👀page views: {visi_num}', value='')
#st.write(f'Visitor Number: {st.session_state.visitor_count}')
visitor.to_csv("visi_num.txt", index=False)


TEXT1 = """
        <body style='text-align: justify; color: black;'>
        <p> The TP-Transformer platform is powered by advanced machine learning models to assist users in predicting the transformation products of aqueous organic pollutants in chemical oxidation processes. TP-Transformer
	is now capable of predicting both the degradation products and pathways of organic pollutants. It utilizes SMILES notation to represent chemical structures.   
        </p>The TP-Transformer is built on a Transformer architecture. It accepts pollutant SMILES, oxidative species, and reaction conditions (e.g., pH) as inputs, 
	and outputs the SMILES of the degradation products. This model can predict not only degradation intermediates but also complete degradation pathways. The 
 prediction of degradation pathways is achieved through an iterative process, where the degradation product predicted by TP-Transformer is used as input for subsequent 
 predictions. This process continues until the model predicts CO<sub>2</sub> or when the predicted chemicals remain unchanged (i.e., non-degradable), indicating the formation of 
 the final degradation products (Figure 1).
	  <p> 
	  </p>
        </body>         
        """


if "show_animation" not in st.session_state:
    st.session_state.show_animation = True
st.header('Welcome to TP-Transformer!')
st.markdown(f'{TEXT1}', unsafe_allow_html=True)
#st.image(Image.open('Fig1.jpg'), caption = 'Figure 1. The comparison between binary MF (B-MF) and count-based MF (C-MF) when representing 1-Decanol, 1-Nonanol and 1-Ocatal')
#col1= st.columns([1])
st.image(Image.open('predic.jpg'), width=800, caption = 'Figure 1. The workflow that TP-Transformer makes predictions')
#col2.image(Image.open('Fig2.jpg'), caption = 'Figure 2. The performance enhancment C-MF brings for each dataset')


if "has_snowed" not in st.session_state:
    st.snow()
    st.session_state["has_snowed"] = True

#try:
#	visitor = pd.read_csv("visi_num.txt", header=None)
#	visi_num = visitor[0][0]
#except:
#	visi_num = 0

#with open("visi_num.txt", 'w') as f:
#	f.write(str(visi_num))
#	f.close()
	
