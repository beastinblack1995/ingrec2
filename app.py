from flask import Flask, request, render_template
from numpy import random

import pandas as pd

from keras.models import Sequential
from keras.layers import Dense

import keras
print('as')

# Create Flask object to run
app = Flask(__name__,template_folder= 'templates' )


df = pd.read_csv('new_ing_csv.csv')


Yencoded_Extracts = pd.get_dummies(df['Extracts'])
Yencoded_All_Age_Suitable_Ing = pd.get_dummies(df['All_Age_Suitable_Ing'])
Yencoded_Anti_Acne_Moisturizer = pd.get_dummies(df['Anti-Acne Moisturizer'])
Yencoded_Antioxidant_Anti_Aging_Moisturizer = pd.get_dummies(df['Antioxidant+Anti-Aging Moisturizer'])
Yencoded_F_Skin_ID_Soothing = pd.get_dummies(df['F_Skin_ID+Soothing'])
Yencoded_M_Antioxidant_Occlusive = pd.get_dummies(df['M_Antioxidant+Occlusive'])
Yencoded_EA_Anti_Age_Skin_ID_Cell_Commute = pd.get_dummies(df['EA_Anti_Age+Skin_ID+Cell_Commute'])
Yencoded_IA_Antioxidant = pd.get_dummies(df['IA_Antioxidant'])
Yencoded_YA_Occusive = pd.get_dummies(df['YA_Occusive'])
Yencoded_BR_Anti_Acne = pd.get_dummies(df['BR_Anti_Acne'])
Yencoded_WR_Skin_ID_Occusive = pd.get_dummies(df['WR_Skin_ID+Occusive'])
Yencoded_DRY_Dry = pd.get_dummies(df['DRY_Dry'])
Yencoded_TROPICAL_Antioxidant_Humectant = pd.get_dummies(df['TROPICAL_Antioxidant+Humectant'])
Yencoded_TEMPERATE_Humectant = pd.get_dummies(df['TEMPERATE_Humectant'])
Yencoded_CONTINENAL_Emolient = pd.get_dummies(df['CONTINENAL_Emolient'])
Yencoded_POLAR_Humectants_Occlusive_Emollients = pd.get_dummies(df['POLAR_Humectants+Occlusive+Emollients'])



Yencoded_anti_acne_serum = pd.get_dummies(pd.read_csv('ancti_acne_serum_ing.csv')['Anti-Acne'])
Yencoded_anti_aging_serum = pd.get_dummies(pd.read_csv('ancti_aging_serum_ing.csv')['Anti-Aging'])
Yencoded_brightning_serum = pd.get_dummies(pd.read_csv('brightning_serum_ing.csv')['Brightning'])

Yencoded_anti_acne_serum_shefee = pd.get_dummies(pd.read_csv('ancti_acne_serum_ing_shefee.csv')['Anti-Acne'])
Yencoded_anti_aging_serum_shefee = pd.get_dummies(pd.read_csv('ancti_aging_serum_ing_shefee.csv')['Anti-Aging'])
Yencoded_brightning_serum_shefee = pd.get_dummies(pd.read_csv('brightning_serum_ing_shefee.csv')['Brightning'])


Yencoded_anti_acne_serum_akmal = pd.get_dummies(pd.read_csv('anti_acne_serum_ing_akmal.csv')['Anti-Acne'])
Yencoded_anti_aging_serum_akmal = pd.get_dummies(pd.read_csv('anti_aging_serum_ing_akmal.csv')['Anti-Aging'])
Yencoded_brightning_serum_akmal = pd.get_dummies(pd.read_csv('anti_bightning_serum_ing_akmal.csv')['Brightning'])



Y_list_serum = [Yencoded_anti_acne_serum,Yencoded_anti_aging_serum,Yencoded_brightning_serum]
Y_list_serum_shefee = [Yencoded_anti_acne_serum_shefee,Yencoded_anti_aging_serum_shefee,Yencoded_brightning_serum_shefee]
Y_list_serum_akmal = [Yencoded_anti_acne_serum_akmal ,Yencoded_anti_aging_serum_akmal ,Yencoded_brightning_serum_akmal ]




Yencoded_Extracts_model = keras.models.load_model('Yencoded_Extracts_model')
Yencoded_All_Age_Suitable_Ing_model = keras.models.load_model('Yencoded_All_Age_Suitable_Ing_model')
Yencoded_Anti_Acne_Moisturizer_model = keras.models.load_model('Yencoded_Anti_Acne_Moisturizer_model')
Yencoded_Antioxidant_Anti_Aging_Moisturizer_model =keras.models.load_model('Yencoded_Antioxidant_Anti_Aging_Moisturizer_model')
Yencoded_F_Skin_ID_Soothing_model = keras.models.load_model('Yencoded_F_Skin_ID_Soothing_model')
Yencoded_M_Antioxidant_Occlusive_model  = keras.models.load_model('Yencoded_M_Antioxidant_Occlusive_model')
Yencoded_EA_Anti_Age_Skin_ID_Cell_Commute_model  = keras.models.load_model('Yencoded_EA_Anti_Age_Skin_ID_Cell_Commute_model')
Yencoded_IA_Antioxidant_model  = keras.models.load_model('Yencoded_IA_Antioxidant_model')
Yencoded_YA_Occusive_model  = keras.models.load_model('Yencoded_YA_Occusive_model')
Yencoded_BR_Anti_Acne_model  = keras.models.load_model('Yencoded_BR_Anti_Acne_model')
Yencoded_WR_Skin_ID_Occusive_model  = keras.models.load_model('Yencoded_WR_Skin_ID_Occusive_model')
Yencoded_DRY_Dry_model  = keras.models.load_model('Yencoded_DRY_Dry_model')
Yencoded_TROPICAL_Antioxidant_Humectant_model  = keras.models.load_model('Yencoded_TROPICAL_Antioxidant_Humectant_model')
Yencoded_TEMPERATE_Humectant_model  = keras.models.load_model('Yencoded_TEMPERATE_Humectant_model')
Yencoded_CONTINENAL_Emolient_model  = keras.models.load_model('Yencoded_CONTINENAL_Emolient_model')
Yencoded_POLAR_Humectants_Occlusive_Emollients_model  = keras.models.load_model('Yencoded_POLAR_Humectants_Occlusive_Emollients_model')

Yencoded_anti_acne_serum_model = keras.models.load_model('Yencoded_anti_acne_model')
Yencoded_anti_aging_serum_model = keras.models.load_model('Yencoded_anti_aging_model')
Yencoded_brightning_serum_model= keras.models.load_model('Yencoded_anti_brightning_model')

Yencoded_anti_acne_shefee_model = keras.models.load_model('Yencoded_anti_acne_shefee_model')
Yencoded_anti_aging_shefee_model = keras.models.load_model('Yencoded_anti_aging_shefee_model')
Yencoded_brightning_shefee_model= keras.models.load_model('Yencoded_anti_brightning_shefee_model')




Yencoded_anti_acne_akmal_model = keras.models.load_model('Yencoded_Anti Acne_model_akmal')
Yencoded_anti_aging_akmal_model = keras.models.load_model('Yencoded_Anti aging_model_akmal')
Yencoded_brightning_akmal_model= keras.models.load_model('Yencoded_Brightning_model_akmal')



st = ['Yencoded_Extracts_model','Yencoded_All_Age_Suitable_Ing_model','Yencoded_Anti_Acne_Moisturizer_model','Yencoded_Antioxidant_Anti_Aging_Moisturizer_model','Yencoded_F_Skin_ID_Soothing_model','Yencoded_M_Antioxidant_Occlusive_model','Yencoded_EA_Anti_Age_Skin_ID_Cell_Commute_model','Yencoded_IA_Antioxidant_model','Yencoded_YA_Occusive_model','Yencoded_BR_Anti_Acne_model','Yencoded_WR_Skin_ID_Occusive_model','Yencoded_DRY_Dry_model','Yencoded_TROPICAL_Antioxidant_Humectant_model','Yencoded_TEMPERATE_Humectant_model','Yencoded_CONTINENAL_Emolient_model','Yencoded_POLAR_Humectants_Occlusive_Emollients_model'
]



modellist = [Yencoded_Extracts_model,Yencoded_All_Age_Suitable_Ing_model,Yencoded_Anti_Acne_Moisturizer_model,Yencoded_Antioxidant_Anti_Aging_Moisturizer_model,Yencoded_F_Skin_ID_Soothing_model,Yencoded_M_Antioxidant_Occlusive_model,Yencoded_EA_Anti_Age_Skin_ID_Cell_Commute_model,Yencoded_IA_Antioxidant_model,Yencoded_YA_Occusive_model,Yencoded_BR_Anti_Acne_model,Yencoded_WR_Skin_ID_Occusive_model,Yencoded_DRY_Dry_model,Yencoded_TROPICAL_Antioxidant_Humectant_model,Yencoded_TEMPERATE_Humectant_model,Yencoded_CONTINENAL_Emolient_model,Yencoded_POLAR_Humectants_Occlusive_Emollients_model]

serum_model = [Yencoded_anti_acne_serum_model,Yencoded_anti_aging_serum_model,Yencoded_brightning_serum_model]
serum_model_shefee = [Yencoded_anti_acne_shefee_model,Yencoded_anti_aging_shefee_model,Yencoded_brightning_shefee_model]
serum_model_akmal = [Yencoded_anti_acne_akmal_model ,Yencoded_anti_aging_akmal_model ,Yencoded_brightning_akmal_model]
serum_model_name = ['Yencoded_anti_acne_serum_model','Yencoded_anti_aging_serum_model','Yencoded_brightning_serum_model']

Ylist = [Yencoded_Extracts,
Yencoded_All_Age_Suitable_Ing,
Yencoded_Anti_Acne_Moisturizer,
Yencoded_Antioxidant_Anti_Aging_Moisturizer,
Yencoded_F_Skin_ID_Soothing ,
Yencoded_M_Antioxidant_Occlusive,
Yencoded_EA_Anti_Age_Skin_ID_Cell_Commute,
Yencoded_IA_Antioxidant,
Yencoded_YA_Occusive,
Yencoded_BR_Anti_Acne,
Yencoded_WR_Skin_ID_Occusive,
Yencoded_DRY_Dry ,
Yencoded_TROPICAL_Antioxidant_Humectant,
Yencoded_TEMPERATE_Humectant ,
Yencoded_CONTINENAL_Emolient,
Yencoded_POLAR_Humectants_Occlusive_Emollients ]
print(len(st),len(Ylist),len(modellist))

def arrr(predictionss):
   
    prod = list([0] * len(list(predictionss)))
    maxm = max(predictionss)

    indd = list(predictionss).index(maxm)
    return indd


def givlis(dictionin):
    X = df[['Age', 'SkinType', 'SkinTone', 'SkinConcerns', 'Gender',
           'Race', 'Climate']]
    Xencoded = pd.get_dummies(X)
    X = Xencoded


    X_fu = list(X.columns)

    listofzeros = [0] * len(X_fu)
    listofzeros[X_fu.index('Age'+'_'+dictionin['Age'])] = 1
    listofzeros[X_fu.index('SkinType'+'_'+dictionin['SkinType'])] = 1
    listofzeros[X_fu.index('SkinTone'+'_'+dictionin['SkinTone'])] = 1
    listofzeros[X_fu.index('SkinConcerns'+'_'+dictionin['SkinConcerns'])] = 1
    listofzeros[X_fu.index('Gender'+'_'+dictionin['Gender'])] = 1
    listofzeros[X_fu.index('Race'+'_'+dictionin['Race'])] = 1
    listofzeros[X_fu.index('Climate'+'_'+dictionin['Climate'])] = 1
    return listofzeros

def modreccomenderz(custdetails,modelz,Ymod):
    Y_col = list(Ymod.columns)
    #print(Y_col)
    predictionn = (modelz.predict([givlis(custdetails)]))[0]

    maxm = max(predictionn)

    indd = list(predictionn).index(maxm)

def givlis_serum(dictionin,concern):
    if concern == 'Anti_Acne_serum_ingrdients':
        # Acne_serum = pd.read_csv('serum_anti_acne_cosmly.csv')
       
        # X = Acne_serum[['Age', 'SkinType', 'SkinConcerns', 'SkinTyone']]
        spare_age = '25-34'
        spare_SkinType = 'Combination'
        spare_SkinConcerns = 'Acne'
        spare_SkinTyone = 'Dark'
       
        # print(spare_age,spare_SkinType,spare_SkinConcerns,spare_SkinTyone)
       
        # Xencoded = pd.get_dummies(X)
        # X = Xencoded
        # X_fu = list(X.columns)

        X_fu =  ['Age_13-17', 'Age_18-24', 'Age_25-34', 'Age_35-44', 'Age_45-54',
           'Age_55-120', 'SkinType_Combination', 'SkinType_Dry', 'SkinType_Normal',
           'SkinType_Oily', 'SkinTyone_Dark', 'SkinTyone_Deep', 'SkinTyone_Ebony',
           'SkinTyone_Fair', 'SkinTyone_Light', 'SkinTyone_Medium',
           'SkinTyone_Olive', 'SkinTyone_Porcelain', 'SkinTyone_Tan',
           'SkinConcerns_Acne', 'SkinConcerns_Aging', 'SkinConcerns_Blackheads',
           'SkinConcerns_Calluses', 'SkinConcerns_Cellulite',
           'SkinConcerns_Cuticles', 'SkinConcerns_Dark circles',
           'SkinConcerns_Dullness', 'SkinConcerns_Pores', 'SkinConcerns_Puffiness',
           'SkinConcerns_Redness', 'SkinConcerns_Sensitivity',
           'SkinConcerns_Stretch marks', 'SkinConcerns_Sun damage',
           'SkinConcerns_Uneven skin tones']
        if 'Age'+'_'+dictionin['Age'] not in X_fu:
            dictionin['Age'] = spare_age
       
        if 'SkinType'+'_'+dictionin['SkinType'] not in X_fu:
            dictionin['SkinType'] = spare_SkinType

        if 'SkinConcerns'+'_'+dictionin['SkinConcerns'] not in X_fu:
            dictionin['SkinConcerns'] = spare_SkinConcerns
           
        if 'SkinTyone'+'_'+dictionin['SkinTone'] not in X_fu:
            dictionin['SkinTone'] = spare_SkinTyone                
           

        listofzeros = [0] * len(X_fu)
        listofzeros[X_fu.index('Age'+'_'+dictionin['Age'])] = 1
        listofzeros[X_fu.index('SkinType'+'_'+dictionin['SkinType'])] = 1
        listofzeros[X_fu.index('SkinTyone'+'_'+dictionin['SkinTone'])] = 1
        listofzeros[X_fu.index('SkinConcerns'+'_'+dictionin['SkinConcerns'])] = 1
        return listofzeros
    if concern == 'Anti_Aging_serum_ingrdients':
        #Aging_serum = pd.read_csv('serum_anti_aging_cosmly.csv')
        #X = Aging_serum[['Age', 'SkinType', 'SkinConcerns', 'SkinTyone']]
        spare_age = '25-34'
        spare_SkinType = 'Combination'
        spare_SkinConcerns = 'Aging'
        spare_SkinTyone = 'Dark'
       
       
       
        # Xencoded = pd.get_dummies(X)
        # X = Xencoded
        # X_fu = list(X.columns)
        X_fu = ['Age_13-17', 'Age_18-24', 'Age_25-34', 'Age_35-44', 'Age_45-54',
           'Age_55-120', 'SkinType_Combination', 'SkinType_Dry', 'SkinType_Normal',
           'SkinType_Oily', 'SkinTyone_Dark', 'SkinTyone_Deep', 'SkinTyone_Ebony',
           'SkinTyone_Fair', 'SkinTyone_Light', 'SkinTyone_Medium',
           'SkinTyone_Olive', 'SkinTyone_Porcelain', 'SkinTyone_Tan',
           'SkinConcerns_Acne', 'SkinConcerns_Aging', 'SkinConcerns_Blackheads',
           'SkinConcerns_Calluses', 'SkinConcerns_Cellulite',
           'SkinConcerns_Cuticles', 'SkinConcerns_Dark circles',
           'SkinConcerns_Dullness', 'SkinConcerns_Pores', 'SkinConcerns_Puffiness',
           'SkinConcerns_Redness', 'SkinConcerns_Sensitivity',
           'SkinConcerns_Stretch marks', 'SkinConcerns_Sun damage',
           'SkinConcerns_Uneven skin tones']
       
        if 'Age'+'_'+dictionin['Age'] not in X_fu:
            dictionin['Age'] = spare_age
       
        if 'SkinType'+'_'+dictionin['SkinType'] not in X_fu:
            dictionin['SkinType'] = spare_SkinType

        if 'SkinConcerns'+'_'+dictionin['SkinConcerns'] not in X_fu:
            dictionin['SkinConcerns'] = spare_SkinConcerns
           
        if 'SkinTyone'+'_'+dictionin['SkinTone'] not in X_fu:
            dictionin['SkinTone'] = spare_SkinTyone    

        listofzeros = [0] * len(X_fu)
        listofzeros[X_fu.index('Age'+'_'+dictionin['Age'])] = 1
        listofzeros[X_fu.index('SkinType'+'_'+dictionin['SkinType'])] = 1
        listofzeros[X_fu.index('SkinTyone'+'_'+dictionin['SkinTone'])] = 1
        listofzeros[X_fu.index('SkinConcerns'+'_'+dictionin['SkinConcerns'])] = 1
        return listofzeros
    if concern == 'Brightning_serum_ingrdients':
        #Brightning_serum_ = pd.read_csv('serum_anti_brightning_cosmly.csv')
       
        #X = Brightning_serum_[['Age', 'SkinType', 'SkinConcerns', 'SkinTyone']]
        spare_age = '25-34'
        spare_SkinType = 'Combination'
        spare_SkinConcerns = 'Dullness'
        spare_SkinTyone = 'Dark'
       
       
       
        # Xencoded = pd.get_dummies(X)
        # X = Xencoded
        # X_fu = list(X.columns)
        X_fu = ['Age_13-17', 'Age_18-24', 'Age_25-34', 'Age_35-44', 'Age_45-54',
           'Age_55-120', 'SkinType_Combination', 'SkinType_Dry', 'SkinType_Normal',
           'SkinType_Oily', 'SkinTyone_Dark', 'SkinTyone_Deep', 'SkinTyone_Ebony',
           'SkinTyone_Fair', 'SkinTyone_Light', 'SkinTyone_Medium',
           'SkinTyone_Olive', 'SkinTyone_Porcelain', 'SkinTyone_Tan',
           'SkinConcerns_Acne', 'SkinConcerns_Aging', 'SkinConcerns_Blackheads',
           'SkinConcerns_Calluses', 'SkinConcerns_Cellulite',
           'SkinConcerns_Cuticles', 'SkinConcerns_Dark circles',
           'SkinConcerns_Dullness', 'SkinConcerns_Pores', 'SkinConcerns_Puffiness',
           'SkinConcerns_Redness', 'SkinConcerns_Sensitivity',
           'SkinConcerns_Stretch marks', 'SkinConcerns_Sun damage',
           'SkinConcerns_Uneven skin tones']  
       
        if 'Age'+'_'+dictionin['Age'] not in X_fu:
            dictionin['Age'] = spare_age
       
        if 'SkinType'+'_'+dictionin['SkinType'] not in X_fu:
            dictionin['SkinType'] = spare_SkinType

        if 'SkinConcerns'+'_'+dictionin['SkinConcerns'] not in X_fu:
            dictionin['SkinConcerns'] = spare_SkinConcerns
           
        if 'SkinTyone'+'_'+dictionin['SkinTone'] not in X_fu:
            dictionin['SkinTone'] = spare_SkinTyone  

        listofzeros = [0] * len(X_fu)
        listofzeros[X_fu.index('Age'+'_'+dictionin['Age'])] = 1
        listofzeros[X_fu.index('SkinType'+'_'+dictionin['SkinType'])] = 1
        listofzeros[X_fu.index('SkinTyone'+'_'+dictionin['SkinTone'])] = 1
        listofzeros[X_fu.index('SkinConcerns'+'_'+dictionin['SkinConcerns'])] = 1
        return listofzeros  
       
       
       
       
def givlis_serum_shefee(dictionin,concern):
    if concern == 'Anti_Acne_serum_ingrdients':
        # Acne_serum = pd.read_csv('serum_anti_acne_cosmly.csv')
       
        # X = Acne_serum[['Age', 'SkinType', 'SkinConcerns', 'SkinTyone']]
        spare_age = '25-34'
        spare_SkinType = 'Combination'
        spare_SkinConcerns = 'Acne'
        spare_SkinTyone = 'Dark'
       
        # print(spare_age,spare_SkinType,spare_SkinConcerns,spare_SkinTyone)
       
        # Xencoded = pd.get_dummies(X)
        # X = Xencoded
        # X_fu = list(X.columns)

        X_fu =  ['Age_13-17', 'Age_18-24', 'Age_25-34', 'Age_35-44', 'Age_45-54',
           'Age_55-120', 'SkinType_Combination', 'SkinType_Dry', 'SkinType_Normal',
           'SkinType_Oily', 'SkinTyone_Dark', 'SkinTyone_Deep', 'SkinTyone_Ebony',
           'SkinTyone_Fair', 'SkinTyone_Light', 'SkinTyone_Medium',
           'SkinTyone_Olive', 'SkinTyone_Porcelain', 'SkinTyone_Tan',
           'SkinConcerns_Acne', 'SkinConcerns_Aging', 'SkinConcerns_Blackheads',
           'SkinConcerns_Calluses', 'SkinConcerns_Cellulite',
           'SkinConcerns_Cuticles', 'SkinConcerns_Dark circles',
           'SkinConcerns_Dullness', 'SkinConcerns_Pores', 'SkinConcerns_Puffiness',
           'SkinConcerns_Redness', 'SkinConcerns_Sensitivity',
           'SkinConcerns_Stretch marks', 'SkinConcerns_Sun damage',
           'SkinConcerns_Uneven skin tones']
        if 'Age'+'_'+dictionin['Age'] not in X_fu:
            dictionin['Age'] = spare_age
       
        if 'SkinType'+'_'+dictionin['SkinType'] not in X_fu:
            dictionin['SkinType'] = spare_SkinType

        if 'SkinConcerns'+'_'+dictionin['SkinConcerns'] not in X_fu:
            dictionin['SkinConcerns'] = spare_SkinConcerns
           
        if 'SkinTyone'+'_'+dictionin['SkinTone'] not in X_fu:
            dictionin['SkinTone'] = spare_SkinTyone                
           

        listofzeros = [0] * len(X_fu)
        listofzeros[X_fu.index('Age'+'_'+dictionin['Age'])] = 1
        listofzeros[X_fu.index('SkinType'+'_'+dictionin['SkinType'])] = 1
        listofzeros[X_fu.index('SkinTyone'+'_'+dictionin['SkinTone'])] = 1
        listofzeros[X_fu.index('SkinConcerns'+'_'+dictionin['SkinConcerns'])] = 1
        return listofzeros

    if concern == 'Anti_Aging_serum_ingrdients':
        #Aging_serum = pd.read_csv('serum_anti_aging_cosmly.csv')
        #X = Aging_serum[['Age', 'SkinType', 'SkinConcerns', 'SkinTyone']]
        spare_age = '25-34'
        spare_SkinType = 'Combination'
        spare_SkinConcerns = 'Aging'
        spare_SkinTyone = 'Dark'
       
       
       
        # Xencoded = pd.get_dummies(X)
        # X = Xencoded
        # X_fu = list(X.columns)
        X_fu = ['Age_13-17', 'Age_18-24', 'Age_25-34', 'Age_35-44', 'Age_45-54',
           'Age_55-120', 'SkinType_Combination', 'SkinType_Dry', 'SkinType_Normal',
           'SkinType_Oily', 'SkinTyone_Dark', 'SkinTyone_Deep', 'SkinTyone_Ebony',
           'SkinTyone_Fair', 'SkinTyone_Light', 'SkinTyone_Medium',
           'SkinTyone_Olive', 'SkinTyone_Porcelain', 'SkinTyone_Tan',
           'SkinConcerns_Acne', 'SkinConcerns_Aging', 'SkinConcerns_Blackheads',
           'SkinConcerns_Calluses', 'SkinConcerns_Cellulite',
           'SkinConcerns_Cuticles', 'SkinConcerns_Dark circles',
           'SkinConcerns_Dullness', 'SkinConcerns_Pores', 'SkinConcerns_Puffiness',
           'SkinConcerns_Redness', 'SkinConcerns_Sensitivity',
           'SkinConcerns_Stretch marks', 'SkinConcerns_Sun damage',
           'SkinConcerns_Uneven skin tones']
       
        if 'Age'+'_'+dictionin['Age'] not in X_fu:
            dictionin['Age'] = spare_age
       
        if 'SkinType'+'_'+dictionin['SkinType'] not in X_fu:
            dictionin['SkinType'] = spare_SkinType

        if 'SkinConcerns'+'_'+dictionin['SkinConcerns'] not in X_fu:
            dictionin['SkinConcerns'] = spare_SkinConcerns
           
        if 'SkinTyone'+'_'+dictionin['SkinTone'] not in X_fu:
            dictionin['SkinTone'] = spare_SkinTyone    

        listofzeros = [0] * len(X_fu)
        listofzeros[X_fu.index('Age'+'_'+dictionin['Age'])] = 1
        listofzeros[X_fu.index('SkinType'+'_'+dictionin['SkinType'])] = 1
        listofzeros[X_fu.index('SkinTyone'+'_'+dictionin['SkinTone'])] = 1
        listofzeros[X_fu.index('SkinConcerns'+'_'+dictionin['SkinConcerns'])] = 1
        return listofzeros
    if concern == 'Brightning_serum_ingrdients':
        #Brightning_serum_ = pd.read_csv('serum_anti_brightning_cosmly.csv')
       
        #X = Brightning_serum_[['Age', 'SkinType', 'SkinConcerns', 'SkinTyone']]
        spare_age = '25-34'
        spare_SkinType = 'Combination'
        spare_SkinConcerns = 'Dullness'
        spare_SkinTyone = 'Dark'
       
       
       
        # Xencoded = pd.get_dummies(X)
        # X = Xencoded
        # X_fu = list(X.columns)
        X_fu = ['Age_13-17', 'Age_18-24', 'Age_25-34', 'Age_35-44', 'Age_45-54',
           'Age_55-120', 'SkinType_Combination', 'SkinType_Dry', 'SkinType_Normal',
           'SkinType_Oily', 'SkinTyone_Dark', 'SkinTyone_Deep', 'SkinTyone_Ebony',
           'SkinTyone_Fair', 'SkinTyone_Light', 'SkinTyone_Medium',
           'SkinTyone_Olive', 'SkinTyone_Porcelain', 'SkinTyone_Tan',
           'SkinConcerns_Acne', 'SkinConcerns_Aging', 'SkinConcerns_Blackheads',
           'SkinConcerns_Calluses', 'SkinConcerns_Cellulite',
           'SkinConcerns_Cuticles', 'SkinConcerns_Dark circles',
           'SkinConcerns_Dullness', 'SkinConcerns_Pores', 'SkinConcerns_Puffiness',
           'SkinConcerns_Redness', 'SkinConcerns_Sensitivity',
           'SkinConcerns_Stretch marks', 'SkinConcerns_Sun damage',
           'SkinConcerns_Uneven skin tones']
       
        if 'Age'+'_'+dictionin['Age'] not in X_fu:
            dictionin['Age'] = spare_age
       
        if 'SkinType'+'_'+dictionin['SkinType'] not in X_fu:
            dictionin['SkinType'] = spare_SkinType

        if 'SkinConcerns'+'_'+dictionin['SkinConcerns'] not in X_fu:
            dictionin['SkinConcerns'] = spare_SkinConcerns
           
        if 'SkinTyone'+'_'+dictionin['SkinTone'] not in X_fu:
            dictionin['SkinTone'] = spare_SkinTyone  

        listofzeros = [0] * len(X_fu)
        listofzeros[X_fu.index('Age'+'_'+dictionin['Age'])] = 1
        listofzeros[X_fu.index('SkinType'+'_'+dictionin['SkinType'])] = 1
        listofzeros[X_fu.index('SkinTyone'+'_'+dictionin['SkinTone'])] = 1
        listofzeros[X_fu.index('SkinConcerns'+'_'+dictionin['SkinConcerns'])] = 1
        return listofzeros  





def givlis_serum_akmal(dictionin,concern):
    X_fu = ['White Race', 'Black Race', 'Elder Age', 'Intermediate Age',
       'Young Age', 'Tropical', 'Dry', 'Temperate', 'Continental', 'Polar']

    listofzeros = [0] * len(X_fu)
    if   (dictionin['Age'] == '13-17') or (dictionin['Age'] == '18-24')  or (dictionin['Age'] == '25-34'):
        
        listofzeros[X_fu.index('Young Age')] = 1
    if   (dictionin['Age'] == '35-44') or (dictionin['Age'] == '45-54')  :       
        listofzeros[X_fu.index('Intermediate Age')] = 1
    if   (dictionin['Age'] == '55-120') :
        listofzeros[X_fu.index('Elder Age')] = 1
    if   (dictionin['Race'] == 'White') :
        listofzeros[X_fu.index('White Race')] = 1
    if   (dictionin['Race'] == 'Black') :
        listofzeros[X_fu.index('Black Race')] = 1        
    if   (dictionin['Climate'] == 'Tropical') :
        listofzeros[X_fu.index('Tropical')] = 1
    if   (dictionin['Climate'] == 'Dry') :
        listofzeros[X_fu.index('Dry')] = 1
    if   (dictionin['Climate'] == 'Temperate') :
        listofzeros[X_fu.index('Temperate')] = 1        
    if   (dictionin['Climate'] ==  'Continental') :
        listofzeros[X_fu.index( 'Continental')] = 1           
    if   (dictionin['Climate'] == 'Polar') :
        listofzeros[X_fu.index( 'Polar')] = 1         
    return listofzeros



























def modreccomender(custtdetails):
    dc = {}
    final = {}
    for modind in range(len(modellist)):

        predictionn = (modellist[modind].predict([givlis(custtdetails)]))[0]
        maxm = max(predictionn)
       

        indd = list(predictionn).index(maxm)

        dc[st[modind]] = (Ylist[modind].columns)[indd]
    final['Extracts'] = dc['Yencoded_Extracts_model']
       
    if   (custtdetails['Age'] == '13-17') or (custtdetails['Age'] == '18-24')  or (custtdetails['Age'] == '25-34'):
       
         final[f"| {custtdetails['Age']} - Occusive"] = dc['Yencoded_YA_Occusive_model'] + "|"
           
    if   (custtdetails['Age'] == '35-44') or (custtdetails['Age'] == '45-54')  :
       
         final[f"| {custtdetails['Age']} - Antioxidant"] = dc['Yencoded_IA_Antioxidant_model']  + "|"
           
           
    if   (custtdetails['Age'] == '55-120') :
       
         final[f"| {custtdetails['Age']} - Skin_Identical and Cell_Commute"] = dc['Yencoded_EA_Anti_Age_Skin_ID_Cell_Commute_model']  + "|"
           
           
           
           
           
    if   (custtdetails['SkinConcerns'] == 'Acne'):
       
         final[f"| {custtdetails['SkinConcerns']} - Anti-Acne"] = dc['Yencoded_Anti_Acne_Moisturizer_model'] + "|"
           
           
    if   (custtdetails['SkinConcerns'] == 'Aging'):
       
         final[f"| {custtdetails['SkinConcerns']} - Anti-Aging and Antioxidant"] = dc['Yencoded_Antioxidant_Anti_Aging_Moisturizer_model']  + "|"
           
           
           
    if   (custtdetails['Climate'] == 'Continental')  :
       
         final[f"| {custtdetails['Climate']} - Emolient"] = dc['Yencoded_CONTINENAL_Emolient_model']  + "|"
           
           
    if   (custtdetails['Climate'] == 'Polar') :
       
         final[f"| {custtdetails['Climate']} - Humectants Occlusive and Emollients"] = dc['Yencoded_POLAR_Humectants_Occlusive_Emollients_model']    + "|"          
           
    if   (custtdetails['Climate'] == 'Tropical') :
       
         final[f"| {custtdetails['Climate']} - Antioxidant and Humectant"] = dc['Yencoded_TROPICAL_Antioxidant_Humectant_model']        + "|"        
       
    if   (custtdetails['Climate'] == 'Dry') :
       
         final[f"| {custtdetails['Climate']} - Skin_Identical and Occusive"] = dc['Yencoded_DRY_Dry_model']             + "|"
           
    if   (custtdetails['Climate'] == 'Temperate') :
       
         final[f"| {custtdetails['Climate']} - Humectant"] = dc['Yencoded_TEMPERATE_Humectant_model']            + "|"  
           
           
           
           
           
           
           
    if   (custtdetails['Gender'] == 'Male') :
       
         final[f"| {custtdetails['Gender']} - Antioxidant and Occlusive"] = dc['Yencoded_M_Antioxidant_Occlusive_model']       + "|"      
           
    if   (custtdetails['Gender'] == 'Female') :
       
         final[f"| {custtdetails['Gender']} -  Skin_Identical and Soothing"] = dc['Yencoded_F_Skin_ID_Soothing_model']     + "|"          
           
    if   (custtdetails['Race'] == 'Black'):
       
         final[f"| {custtdetails['Race']} - Anti-Acne"] = dc['Yencoded_BR_Anti_Acne_model']  + "|"
    if   (custtdetails['Race'] == 'White')  :
       
         final[f"| {custtdetails['Race']} - Skin_Identical and Occusive "] = dc['Yencoded_WR_Skin_ID_Occusive_model']    + "|"  
           

 
   
   

    return  final






def modreccomender_serum(custdetails):
    dc = {}
    dcx = {}
    dcxx = {}
    final = {}
    model_concern = ['Anti_Acne_serum_ingrdients','Anti_Aging_serum_ingrdients','Brightning_serum_ingrdients']
    for modind in range(len(serum_model)):

        predictionn = (serum_model[modind].predict([givlis_serum(custdetails,model_concern[modind])]))[0]
        #maxm = max(predictionn)
        sorted_prediction = sorted( list(predictionn),reverse = True)
        maxm = sorted_prediction[random.randint(3)]
        
       

        indd = list(predictionn).index(maxm)

        dc[model_concern[modind]] = (Y_list_serum[modind].columns)[indd]
       
       
       
       
       
       
       
        predictionnx = (serum_model_shefee[modind].predict([givlis_serum_shefee(custdetails,model_concern[modind])]))[0]
        #maxmx = max(predictionnx)
        sorted_prediction = sorted( list(predictionnx),reverse = True)
        maxmx = sorted_prediction[random.randint(3)]       

        inddx = list(predictionnx).index(maxmx)

        dcx[model_concern[modind]] = (Y_list_serum_shefee[modind].columns)[inddx]    



    


        predictionnxx = (serum_model_akmal[modind].predict([givlis_serum_akmal(custdetails,model_concern[modind])]))[0]
        #maxmxx = max(predictionnxx)
        sorted_prediction = sorted( list(predictionnxx),reverse = True)
        maxmxx = sorted_prediction[random.randint(3)]       

        inddxx = list(predictionnxx).index(maxmxx)

        dcxx[model_concern[modind]] = (Y_list_serum_akmal[modind].columns)[inddxx]               
       
       
       
        final[model_concern[modind]] = dc[model_concern[modind]] + ',' + dcx[model_concern[modind]] + ',' + dcxx[model_concern[modind]]
       
       
   
           

    return  final


 
 
 
 
 
 
 
 
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    product =    str(request.form.get('Product'))
   
    SkinConcerns = str(request.form.get('SkinConcerns'))

    Age = str(request.form.get('Age'))

    SkinType = str(request.form.get('SkinType'))

    Gender = str(request.form.get('Gender'))

    SkinTone = str(request.form.get('SkinTyone'))

    Race = str(request.form.get('Race'))
   
    Climate = str(request.form.get('Climate'))  


               

 
    out = {}
   
   
    if product == 'Moisturizer':
   
        custdetails = {'SkinConcerns':SkinConcerns,'Age':Age,'SkinType':SkinType,'SkinTone':SkinTone,'Gender':Gender,
                  'Race':Race,'Climate':Climate}
 
        out = modreccomender(custdetails)
    if product == 'Serum':
        custdetails = {'SkinConcerns':SkinConcerns,'Age':Age,'SkinType':SkinType,'SkinTone':SkinTone,'Gender':Gender,
                  'Race':Race,'Climate':Climate}
 
        out = modreccomender_serum(custdetails)        
       
       
    print(out)

    return render_template('index.html', prediction_text= out)



   
   
if __name__ == "__main__":
    app.run()
