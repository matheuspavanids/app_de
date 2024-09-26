import streamlit as st
import lightgbm as lgb
import pickle
import numpy as np

st.title('Previsão do Tipo de Escola (Pública ou Privada)')

#Carregar o modelo e o escalonador
@st.cache_resource
def load_model():
    #model = lgb.Booster(model_file='modelo_lgb.txt')
    with open('modelo.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

def predict(input_data):
    input_array = np.array([input_data])
    #input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_array)
    predicted_class = 'Pública' if prediction[0] < 0.5 else 'Privada'
    return predicted_class, prediction[0]

st.header('Insira as notas do aluno')

def input_nota(nome_nota):
    return st.number_input(
        nome_nota,
        min_value=0.0,
        max_value=1000.0,
        value=500.0,
        step=1.0,
        format="%.2f"
    )

nu_nota_mt = input_nota('Nota em Matemática')
nu_nota_lc = input_nota('Nota em Linguagens e Códigos')
nu_nota_ch = input_nota('Nota em Ciências Humanas')
nu_nota_cn = input_nota('Nota em Ciências da Natureza')
nu_nota_redacao = input_nota('Nota da Redação')

if st.button('Prever Tipo de Escola'):
    try:
        input_data = [
            nu_nota_mt,
            nu_nota_lc,
            nu_nota_ch,
            nu_nota_cn,
            nu_nota_redacao
        ]
        resultado, probabilidade = predict(input_data)
        st.success(f'**Tipo de Escola Previsto:** {resultado}')
        st.info(f'**Probabilidade de ser Privada:** {probabilidade:.2%}')
    except Exception as e:
        st.error(f'Ocorreu um erro durante a previsão: {e}')