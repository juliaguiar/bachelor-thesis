import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict

typeViolence = defaultdict(int)

secondSexualViolence = defaultdict(int)
secondPhysicalViolence = defaultdict(int)
secondPsychologicalViolence = defaultdict(int)
secondMoralViolence = defaultdict(int)
secondPatrimonialViolence = defaultdict(int)

df_Keyword = pd.read_csv("csv/v2/datasetReportsKeyword.csv")
df_hashtagName = pd.read_csv("csv/v2/datasetReportsHashtagName.csv")

def firstAndSecondViolence(firstViolence, secondViolence):
    if firstViolence == "Violência Sexual":
        secondSexualViolence[secondViolence] +=1
    elif firstViolence == "Violência Física":
        secondPhysicalViolence[secondViolence] += 1
    elif firstViolence == "Violência Psicológica":
        secondPsychologicalViolence[secondViolence] += 1
    elif firstViolence == "Violência Moral":
        secondMoralViolence[secondViolence] += 1
    else:
        secondPatrimonialViolence[secondViolence] += 1

def plot(df, title, image):
    ordered_df = df.sort_values(by='value')
    my_range = range(1, len(df.index) + 1)

    plt.hlines(y=my_range, xmin=0, xmax=ordered_df['value'], color='skyblue')
    plt.plot(ordered_df['value'], my_range, "o")

    plt.yticks(my_range, ordered_df['type'])
    plt.title(title, loc='right', color='skyblue')
    plt.xlabel('Quantidade de Relatos')
    plt.savefig('img/'+image+'.png', bbox_inches='tight')
    plt.show()

for index, row in df_Keyword.iterrows():
    text = row[3]
    for ch in ["'", "[", "]"]:
        text = text.replace(ch, "")
    listViolence = text.split(', ')
    typeViolence[listViolence[0]] += 1
    firstAndSecondViolence(listViolence[0], listViolence[1])

for index, row in df_hashtagName.iterrows():
    text = row[3]
    for ch in ["'", "[", "]"]:
        text = text.replace(ch, "")
    listViolence = text.split(', ')
    typeViolence[listViolence[0]] += 1
    firstAndSecondViolence(listViolence[0], listViolence[1])

df_typeViolence = pd.DataFrame({'type': [x for x, y in typeViolence.items()], 'value': list(typeViolence.values())})

df_secondSexualViolence = pd.DataFrame({'type': [x for x, y in secondSexualViolence.items()], 'value': list(secondSexualViolence.values())})
df_secondPhysicalViolence = pd.DataFrame({'type': [x for x, y in secondPhysicalViolence.items()], 'value': list(secondPhysicalViolence.values())})
df_secondPsychologicalViolence = pd.DataFrame({'type': [x for x, y in secondPsychologicalViolence.items()], 'value': list(secondPsychologicalViolence.values())})
df_secondMoralViolence = pd.DataFrame({'type': [x for x, y in secondMoralViolence.items()], 'value': list(secondMoralViolence.values())})
df_secondPatrimonialViolence = pd.DataFrame({'type': [x for x, y in secondPatrimonialViolence.items()], 'value': list(secondPatrimonialViolence.values())})

plot(df_typeViolence, "Classificação de Violência", "classificacaoViolencia")

plot(df_secondSexualViolence, "Segundo tipo quando primeiro é Violência Sexual", "segundaClassificacaoViolenciaSexual")
plot(df_secondPhysicalViolence, "Segundo tipo quando primeiro é Violência Física", "segundaClassificacaoViolenciaFisica")
plot(df_secondPsychologicalViolence, "Segundo tipo quando primeiro é Violência Psicológica", "segundaClassificacaoViolenciaPsicologica")
plot(df_secondMoralViolence, "Segundo tipo quando primeiro é Violência Moral", "segundaClassificacaoViolenciaMoral")
plot(df_secondPatrimonialViolence, "Segundo tipo quando primeiro é Violência Patrimonial", "segundaClassificacaoViolenciaPatrimonial")