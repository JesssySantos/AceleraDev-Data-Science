import streamlit as st
import pandas as pd
import base64
import seaborn as sns
import matplotlib.pyplot as plt

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
    return href

def basic_info(df):
    st.write("Dataframe de",df.shape[0],"Linhas e", df.shape[1], "colunas")
    slider_st = st.slider('Quantidade de linhas', 0, df.shape[0], 5)
    st.dataframe(df.head(slider_st))
    st.subheader('Tipos das colunas: ')
    st.write(df.dtypes)
    st.subheader('Missing Values por coluna:')
    st.write(df.isna().sum())

def replacing_with(df, measure, cols):

    df_out = df.copy()

    for c in cols:
        str_success = "Valores da coluna " + c + " substituídos com sucesso!"
        if measure == 'Média' and df[c].dtype != 'object':
            df_out[c] = df[c].fillna(df[c].mean())
            st.success(str_success)
        elif measure == 'Mediana' and df[c].dtype != 'object':
            df_out[c] = df[c].fillna(df[c].median())
            st.success(str_success)
        elif measure == 'Moda':
            df_out[c] = df[c].fillna(df[c].mode())
            st.success(str_success)
        else:
            str_error = "ERRO: Colunas em texto como '" + c + "' não podem ser preenchidas com " + measure + "!"
            st.error(str_error)

    return df_out

def input_data(df):
    exploracao = pd.DataFrame({'nomes': df.columns, 'tipos': df.dtypes, 'NA #': df.isna().sum(),
                               'NA %': (df.isna().sum() / df.shape[0]) * 100})
    st.markdown('** Tabela com coluna e percentual de dados faltantes : **')
    st.table(exploracao[exploracao['NA #'] != 0][['tipos', 'NA %']])
    st.subheader('Tratamento de Missing Values:')
    colunas = st.multiselect('Selecione as colunas que deseja popular:', df.columns.tolist())
    st.write(df[colunas].dtypes)
    select_method = st.radio('Escolha um metodo abaixo :', ('Média','Moda','Mediana','Default'))
    st.markdown('Você selecionou : ' + str(select_method))

    if select_method == 'Default':
        usr_input = st.text_input("Digite aqui o valor desejado",0)
        if usr_input.isdigit() and '.' in usr_input:
            usr_input = float(usr_input)
        else:
            usr_input = int(usr_input)

    if st.button("APLICAR"):

        if select_method == 'Default':
            df_inputado = df.copy()
            for c in colunas:
                if (df[c].dtype == 'object' and  type(usr_input) == str) or (df[c].dtype != 'object' and  type(usr_input) != str):
                    df_inputado = df[c].fillna(usr_input)
                    st.success("Dados inputados com sucesso!")
                else:
                    st.error("Tipos incompativeis!")
        else:
            df_inputado = replacing_with(df, select_method, colunas)

        if 'series' in str(type(df_inputado)):
            df_inputado = df_inputado.to_frame()

        exploracao_inputado = pd.DataFrame({'nomes': df_inputado.columns, 'tipos': df_inputado.dtypes, 'NA #': df_inputado.isna().sum(),
                                            'NA %': (df_inputado.isna().sum() / df_inputado.shape[0]) * 100})
        st.table(exploracao_inputado[exploracao_inputado['NA #'] != 0][['tipos', 'NA %']])
        st.subheader('Faça o download dos dados tratados: ')
        st.markdown(get_table_download_link(df_inputado), unsafe_allow_html=True)

def data_visualization(df):
    aux_df = pd.Series(index=df.columns.to_list())

    for col in aux_df.index.to_list():
        aux_df[col] = int(len(df[col].value_counts().index))

    aux_df.rename_axis('EITA')
    st.write(aux_df)
    var_principal = st.selectbox('Selecione a coluna para visualizar:',df.columns.to_list())
    tipo_grafico = st.radio('Escolha um metodo abaixo :', ('Gráfico de Barras', 'Gráfico de Distribuição'))
    st.markdown('Você selecionou : ' + str(tipo_grafico))

def main():

    #Barra Lateral Fixa
    st.sidebar.image('logo.png', use_column_width=True)
    st.sidebar.subheader('AceleraDev Data Science')
    st.sidebar.subheader('Semana 2 - Pré-processamento de Dados em Python')
    st.sidebar.markdown('Esta aplicação foi desenvolvida para fins didáticos durante o programa AceleraDev '
                        'da Codenation iniciado em Junho/2020.')
    st.sidebar.subheader('Desenvolvido por Jéssica Santos:(https://github.com/jesssysantos)')

    #Main Page
    st.title('Exploring Data App')
    st.subheader('Medidas Estatísticas e Gráficos diversos de seus dados')
    file = st.file_uploader('Escolha a base de dados que deseja analisar (.csv)', type = 'csv')

    if file is not None:

        df = pd.read_csv(file)
        copia_df = df.copy()
        df = copia_df.copy()

        if st.button('Resetar alterações'):
            df = copia_df.copy()

        option = st.selectbox(
            "Selecione a operação desejada:",
            ("Informações Básicas", "Tratar Dados Faltantes",
             "Medidas Descritivas", "Correlação", "Visualização dos Dados")
        )

        st.subheader('Analisando os dados')

        if option == "Informações Básicas":
            basic_info(df)
        elif option == "Tratar Dados Faltantes":
            input_data(df)
        elif option == "Visualização dos Dados":
            data_visualization(df)

if __name__ == '__main__':
	main()