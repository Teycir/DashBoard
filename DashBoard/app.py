# streamlit run C:\Users\HP\source\repos\DashBoard\DashBoard\app.py


# -*- coding: utf-8 -*-
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import cross_validate
from lightgbm import LGBMClassifier
import plotly.express as px
import pickle
from my_functions.functions_cached import * # personnal functions pkg and module
#######################################################################################
# No warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

#######################################################################################
# To run this code, type in terminal at the file path: 
# streamlit run app.py
#######################################################################################
# Stating graphical parameters
COLOR_BR_r = ['#00CC96', '#EF553B'] #['dodgerblue', 'indianred']
COLOR_BR = ['indianred', 'dodgerblue']
#######################################################################################
# Managing data import/export

pathabsolutedir = os.path.dirname(os.path.abspath(__file__))
PATH_INPUT = pathabsolutedir+"/input/"
FILENAME_TRAIN = PATH_INPUT+'application_train_sample.csv' # sample of train set for online version 25MB
FILENAME_TEST = PATH_INPUT+'application_test.csv'
FILENAME_MODEL = pathabsolutedir+'/optimized_model.sav'
data_processed = pd.read_csv( pathabsolutedir +'/input/data_processed.csv', index_col='SK_ID_CURR')
data_original_le = pd.read_csv( pathabsolutedir +'/input/data_original_le.csv', index_col='SK_ID_CURR')
features_desc = pd.read_csv(pathabsolutedir  +  "/input/features_descriptions.csv", index_col=0)

#######################################################################################
# Setting layout & navigation pane
st.set_page_config(page_title="Dashboard Pret a depenser", # Must be 1st st statement
                   page_icon="❁",
                   initial_sidebar_state="expanded")
surrogate_model_lgbm = joblib.load(pathabsolutedir +'/input/surrogate_model_lgbm.pkl')
df_train = get_data(FILENAME_TRAIN) # load trainset data in a df
df_test = get_data(FILENAME_TEST) # load testset (unlabeled) data in a df
buttonAdmin = st.button('📄 Adminisitrateur')
sb = st.sidebar # add a side bar 
sb.image('https://user.oc-static.com/upload/2019/02/25/15510866018677_logo%20projet%20fintech.png', width=280)



np.random.seed(13) # one major change is that client is directly asked as input since sidebar
label_test = df_test['SK_ID_CURR'].sample(50).sort_values()
radio = sb.radio('', ['Client ID aléatoire', 'Saisir client ID'])
if radio == 'Client ID aléatoire': # Choice choose preselected seed13 or a known client ID
        input_client = sb.selectbox('Selectionner client ID', label_test)
if radio == 'Saisir client ID':
        input_client = int(sb.text_input('Saisir client ID', value=147254))
sb.markdown('**Navigation**')
rad = sb.radio('', ['🏠 Home', 
    '📉 Prédiction détaillée',
    '🔎 Exploration des données client',
    '🌐 Features globales',
    '✦ Description de features'])


#else:
#    sb.markdown('**Navigation**')
#    rad = sb.radio('', ['🏠 Home'])
# defining containers of the app
header = st.container()
dataset = st.container()
eda = st.container()
model_training = st.container()
model_predict = st.container()

#######################################################################################
# Implementing containers
#######################################################################################

if rad == '🏠 Home': # with this we choose which container to display on the screen
    with header:
        a,z,e,r,t = st.columns(5) #OOP style 
      

        st.title("Bienvenu au Dashboard! \n ----")
        st.header("Predisez la solvabilité des clients")


        col1, col2 = st.columns(2)
        col1.markdown(f'** ID Client: {input_client}**')

        if col2.button('Predire !'):
            # this time we need all outputs of preprocessing                    
            X_train_sc, X_test_sc, feat_list = preprocess(df_train, df_test)
            y_train = df_train['TARGET']
            # calling pretrained model from pickle file (.sav)
            try: 
                model = pickle.load(open(FILENAME_MODEL, 'rb'))
            except:
                raise "Il faut entrainer le modèle d'abord."
            # finding client row index in testset
            idx = df_test.SK_ID_CURR[df_test.SK_ID_CURR == input_client].index
            client = X_test_sc[idx, :] # for then slicing preprocessed test data
            
            y_prob = model.predict_proba(client) # predicting proba
            y_prob = [y_prob.flatten()[0], y_prob.flatten()[1]] #misalignement of format
            # importance of features extracted using scikit learn: pred_contrib=True
            imp_feat = model.predict_proba(X_test_sc[idx, :], pred_contrib=True).flatten()
            imp = pd.DataFrame([feat_list, imp_feat]).T.sort_values(by=1, ascending=False).head(20)

            col1, col2 = st.columns(2)
            # adapting message wether client's pos or neg
            if y_prob[1] < y_prob[0]:
                st.header("Prêt accordé  :sunglasses:")
               
            else:
               st.header("Prêt non accordé  :disappointed:")
            



#######################################################################################


if rad ==  '📉 Prédiction détaillée': 
    with model_predict:
        st.header("**Prediction de la solvabilité du client.** \n ----")

        col1, col2 = st.columns(2)
        col1.markdown(f'** ID Client: {input_client}**')

        if col2.button('Predire !'):
            # this time we need all outputs of preprocessing                    
            X_train_sc, X_test_sc, feat_list = preprocess(df_train, df_test)
            y_train = df_train['TARGET']
            # calling pretrained model from pickle file (.sav)
            try: 
                model = pickle.load(open(FILENAME_MODEL, 'rb'))
            except:
                raise "Il faut entrainer le modèle d'abord."
            # finding client row index in testset
            idx = df_test.SK_ID_CURR[df_test.SK_ID_CURR == input_client].index
            client = X_test_sc[idx, :] # for then slicing preprocessed test data
            
            y_prob = model.predict_proba(client) # predicting proba
            y_prob = [y_prob.flatten()[0], y_prob.flatten()[1]] #misalignement of format
            # importance of features extracted using scikit learn: pred_contrib=True
            imp_feat = model.predict_proba(X_test_sc[idx, :], pred_contrib=True).flatten()
            imp = pd.DataFrame([feat_list, imp_feat]).T.sort_values(by=1, ascending=False).head(20)

            col1, col2 = st.columns(2)
            # adapting message wether client's pos or neg
            if y_prob[1] < y_prob[0]:
                col1.subheader(f"**Probabilité de payer.**")
            else:
                col1.subheader(f"**Probabilité de défaut de paiement.**")
            # plotting pie plot for proba, finding good h x w was a bit tough
            fig = px.pie(values=y_prob, names=[0,1], color=[0,1], color_discrete_sequence=COLOR_BR_r, 
            width=230, height=230)
            fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            col1.plotly_chart(fig, use_container_width=True)

            import plotly.graph_objects as go

            fig = go.Figure(go.Indicator(
            domain = {'x': [0, 1], 'y': [0, 1]},
            value = y_prob[0]*100,
            mode = "gauge+number",
            title = {'text': "Probabilité de payer du client vs seuil"},
            gauge = {'axis': {'range': [None, 100]},
                     'steps' : [
                        
                         {'range': [0, 100], 'color': "gray"}],
                     'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 47}}))

            
            st.plotly_chart(fig, use_container_width=True)

            col2.subheader("**Graphe client.**")
            # plotting radar chart
            columns = (imp.head(5)[0].values) # recovering top5 most important features as... tuples, why did I do that???
            df_test_sc = pd.DataFrame(X_test_sc, columns=feat_list)
            # I wanted to plot average that's why I made a df, but I think it's useless now 
            # since it was a bit difficult and I drop this idea. Instead, I kept scaled version of data
            # so average should be zero and 1 = +1 sigma (StandardScaler)
            client_radar = df_test_sc.loc[idx,columns].T.reset_index()
            client_radar = client_radar.rename(columns={"index":"theta", idx.values[0] :'r'})

            fig = px.line_polar(client_radar, 
                                theta='theta', 
                                r='r', 
                                log_r=False, 
                                line_close=True,
                                color_discrete_sequence=['indianred'],
                                width=250,
                                height=250,
                                )
            fig.update_traces(fill='toself')
            fig.update_layout(margin=dict(l=50, r=50, t=50, b=10))  
            col2.plotly_chart(fig, use_container_width=True)

            st.subheader("**Importance des features à la décision.**")
            # then plotting feature importance, but for readibility slicing absissa labels using:
            labels = [(i[:7] + '...'+i[-7:]) if len(i) > 17 else i for i in imp[0]]
            fig = px.bar(   imp.head(10),
                            x=0,
                            y=1,
                            width=300,
                            height=300,
                            color=range(10),
                            color_continuous_scale='OrRd_r',
                            orientation='v')
            fig.update(layout_coloraxis_showscale=False)
            fig.update_xaxes(title='')
            fig.update_layout(xaxis = dict(
                            tickmode = 'array',
                            tickvals = [i for i in range(20)],
                            ticktext = labels))
            fig.update_yaxes(title='Importance relative')
            fig.update_yaxes(showticklabels=False)
            fig.update_layout(margin=dict(l=20, r=20, t=10, b=10))                
            st.plotly_chart(fig, use_container_width=True)

            # one-hot-encoded columns added a "_string" as lower case to col names
            # thus checking if the col name is full upper case if a good test to 
            # check whether the col is num or cat (I want only 6 most num feats here)
            num_plots=[]
            i=0
            while (i in range(len(imp))) and (len(num_plots) < 7):
                if imp.iloc[i,0] == imp.iloc[i,0].upper():
                    num_plots.append(imp.iloc[i,0])
                i+=1


           

            st.subheader("Classement client sur d'importantes features.")      
            col1, col2, col3 = st.columns(3)
            col1.plotly_chart(histogram(df_train, x=num_plots[0], client=[df_test, input_client]), use_container_width=True)
            col2.plotly_chart(histogram(df_train, x=num_plots[1], client=[df_test, input_client]), use_container_width=True)
            col3.plotly_chart(histogram(df_train, x=num_plots[2], client=[df_test, input_client]), use_container_width=True)

            col1, col2, col3 = st.columns(3)
            col1.plotly_chart(histogram(df_train, x=num_plots[3], client=[df_test, input_client]), use_container_width=True)
            col2.plotly_chart(histogram(df_train, x=num_plots[4], client=[df_test, input_client]), use_container_width=True)
            col3.plotly_chart(histogram(df_train, x=num_plots[5], client=[df_test, input_client]), use_container_width=True)

      
           




#######################################################################################
if rad ==  '🔎 Exploration des données client': 
    with eda:
        st.header("**Données Client.** \n ----")
        # retrieving whole row of client from sidebar input ID
        client_data = df_test[df_test.SK_ID_CURR == input_client]
        client_data = client_data.dropna(axis=1) # avoiding bugs

        st.subheader(f"**Client ID: {input_client}.**")
        # plotting features from train set, with client's data as dashed line (client!=None in func)
        st.subheader("Classement client dans certaines features.")      
        col1, col2, col3 = st.columns(3)
        col1.plotly_chart(histogram(df_train, x='CODE_GENDER', client=[df_test, input_client]), use_container_width=True)
        col2.plotly_chart(histogram(df_train, x='EXT_SOURCE_1', client=[df_test, input_client]), use_container_width=True)
        col3.plotly_chart(histogram(df_train, x='EXT_SOURCE_2', client=[df_test, input_client]), use_container_width=True)

        st.subheader("Features numériques.")
        col1, col2, col3 = st.columns(3)
        num_col = client_data.select_dtypes(include=np.number).columns.sort_values()
        input1 = col1.selectbox('Premier graphe', num_col)
        input2 = col2.selectbox('Second graphe', num_col[1:])
        input3 = col3.selectbox('Troisième graphe', num_col[2:])

        st.subheader("Features catégorielles.")
        col4, col5, col6 = st.columns(3)
        cat_col = client_data.select_dtypes(exclude=np.number).columns.sort_values()
        input4 = col4.selectbox('Premier graphe', cat_col[1:])
        input5 = col5.selectbox('Second graphe', cat_col[2:])
        input6 = col6.selectbox('Troisième graphe', cat_col[3:])

        button = st.button('Afficher! ')
        if button:
            col1.plotly_chart(histogram(df_train, x=input1, legend=False, client=[df_test, input_client]),use_container_width=True)
            col2.plotly_chart(histogram(df_train, x=input2, legend=False, client=[df_test, input_client]),use_container_width=True)
            col3.plotly_chart(histogram(df_train, x=input3, legend=False, client=[df_test, input_client]),use_container_width=True)
            col4.plotly_chart(histogram(df_train, x=input4, legend=False, client=[df_test, input_client]),use_container_width=True)
            col5.plotly_chart(histogram(df_train, x=input5, legend=False, client=[df_test, input_client]),use_container_width=True)
            col6.plotly_chart(histogram(df_train, x=input6, legend=False, client=[df_test, input_client]),use_container_width=True)
        
        st.subheader("Plus d'information sur ce client.")
        col1, col2 = st.columns(2)
        info = col1.selectbox('Quelle info?', client_data.columns.sort_values())     
        info_print = client_data[info].to_numpy()[0]

        col1.subheader(info_print)
        

#######################################################################################

if rad ==  '🌐 Features globales':
        @st.cache
        def get_features_importance():
        # convert data to pd.Series
            features_imp = pd.Series(surrogate_model_lgbm.feature_importances_, index=data_original_le.columns).sort_values(ascending=False)
            return features_imp

        st.header('INTERPRETATION AU NIVEAU DE LA POPULATION GLOBALE')
        # Get features importance (surrogate model, cached)
        # get the features' importance
        features_imp = get_features_importance()

        # initialization
        sum_fi = 0
        labels = []
        frequencies = []

        # get the labels and frequencies of 10 most important features
        for feat_name, feat_imp in features_imp[:9].iteritems():
            labels.append(feat_name)
            frequencies.append(feat_imp)
            sum_fi += feat_imp

        # complete the FI of other features
        labels.append("AUTRES FEATURES…")
        frequencies.append(1 - sum_fi)

        # Set up the axe
        _, ax = plt.subplots()
        ax.axis("equal")
        ax.pie(frequencies)
        ax.set_title("Importance")
        ax.legend(
            labels,
            loc='center left',
            bbox_to_anchor=(0.7, 0.5),
        )
        # Plot the pie-plot of features importance
        st.pyplot()
      

#######################################################################################
if rad ==  '✦ Description de features':
        st.header('DESCRIPTION DU SENS DES FEATURES')
        st.dataframe(data=features_desc, height=500) 



#######################################################################################
if buttonAdmin: 
    with model_training:
        st.header("**Entrainement.** \n ----")
        st.markdown("Utilisation de LightGBM Classifier (Microsoft).")

        _, col2, _ = st.columns(3)
        col2.image('https://raw.githubusercontent.com/microsoft/LightGBM/master/docs/logo/LightGBM_logo_black_text_tiny.png')     
        # preprocess = home-made func, with 3 outputs (X_train_sc, X_test_sc, feat_list)
        X_train_sc, _, _ = preprocess(df_train, df_test)
        y_train = df_train['TARGET']
        
        col1, col2 = st.columns(2)
        col1.subheader("**Tuning des meilleurs hyperparamètres.**")
        # sliders for hyperprams of LightGBM classifier
        n_estimators = col1.slider("Nombres d' arbres", value=300, min_value=200, max_value=1000)
        num_leaves = col1.slider("Nombre de feuilles", value=10, min_value=5, max_value=100)
        lr = col1.select_slider("Taux d'apprentissage", options=[1e-4, 1e-3, 1e-2, 1e-1, 1e0], value=1e-1)
        scale_pos_weight = col1.select_slider("Poids des positives (>10 hautement reommandé)",\
            options=[1e-1, 1e0, 1e1, 2e1, 5e1, 1e2], value=1e1) # as alternative for log sliders
        reg_alpha = col1.slider("L1 terme de régularisation", value=0, min_value=0, max_value=100)
        reg_lambda = col1.slider("L2 terme de régularisation", value=0,  min_value=0, max_value=100)
        checkbox = col1.checkbox("Exporter modèle 🥒🥒🥒") # export or not model checkbox

        if col1.button('Fit en utilisant la cross-validation!'):
            col2.subheader('**Validation.**')
            st.spinner('Fitting...') # not working...
            model = LGBMClassifier(max_depth=-1,
                                    random_state=13,
                                    silent=True,
                                    metric='none',
                                    n_jobs=-1,
                                    n_estimators=n_estimators,
                                    num_leaves=num_leaves,
                                    learning_rate=lr,
                                    scale_pos_weight=scale_pos_weight,
                                    reg_alpha=reg_alpha,
                                    reg_lambda=reg_lambda
                                )

            scoring = ['roc_auc','precision','recall','f1']
            x_val = cross_validate(model, X_train_sc, y_train, cv=3, scoring=scoring)
            # putting output of Xval for easier aggregations
            time, unk, auc, precision, recall, f1 = pd.DataFrame(x_val).mean(axis=0)
            d_time, d_unk, d_auc, d_precision, d_recall, d_f1 = pd.DataFrame(x_val).std(axis=0)

            col2.subheader('Fit temps moyen (s)')
            col2.write(f'{time:.0f} ± {d_time:.0f}')
            col2.subheader('AUC-score')
            col2.write(f'{auc:.0%} ± {d_auc:.0%}')
            col2.subheader('Recall')
            col2.write(f'{recall:.0%} ± {d_recall:.0%}')
            col2.subheader('Precision')
            col2.write(f'{precision:.0%} ± {d_precision:.0%}')
            col2.subheader('f1-score')
            col2.write(f'{f1:.0%} ± {d_f1:.0%}')

            if checkbox: # export with pickle
                model.fit(X_train_sc, y_train)
                pickle.dump(model, open(FILENAME_MODEL, 'wb'))
                st.header('**Export réussi!**')
                st.balloons()


#######################################################################################

if __name__ == "__main__":
    print("Script runned directly")
else:
    print("Script called by other")