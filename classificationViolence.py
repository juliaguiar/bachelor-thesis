import spacy
import tweepy
import pandas as pd
from myTweetTokenizer import MyTweetTokenizer
from violence import Violence

#dataframe dos relatos de busca por HASHTAG com suas classificações
df_hashtagName = pd.DataFrame(columns=["data", "post", "tipoDeViolencia", "localizacaoUsuario", "localizacaoPost", "descricaoPerfil", "seguidores", "perfilVerificado"])

#dataframe dos relatos de busca por KEYWORD com suas classificações
df_Keyword = pd.DataFrame(columns=["data", "post", "tipoDeViolencia", "localizacaoUsuario", "localizacaoPost", "descricaoPerfil", "seguidores", "perfilVerificado"])


#Dicionários reduzidos de palavras dos tipos de violência
dictionaryKeywordsSexualViolence = ["estuprar", "abusar", "assediar", "agarrar"]
dictionaryKeywordsPhysicalViolence = ["bater", "machucar", "espancar", "empurrar"]
dictionaryKeywordsPsychologicalViolence = ["ameaçar", "humilhar", "xingar", "ofender"]
dictionaryKeywordsMoralViolence = ["acusar", "chantagear", "ofender", "caluniar"]
dictionaryKeywordsPatrimonialViolence = ["quebrar", "destruir", "pegar", "roubar"]

#pesos para as palavras dos dicionários
weightsForKeywordsSexualViolence = [4, 3, 2, 1]
weightsForKeywordsPhysicalViolence = [3, 2, 4, 1]
weightsForKeywordsPsychologicalViolence = [4, 2, 1, 3]
weightsForKeywordsMoralViolence = [2, 4, 1, 3]
weightsForKeywordsPatrimonialViolence = [2, 4, 1, 3]

#variáveis de autentitacação para API Twitter

consumer_key = [YOUR_CONSUMER_KEY]
consumer_secret = [YOUR_CONSUMER_SECRET]
token_key = [YOUR_TOKEN_KEY]
token_secret = [YOUR_TOKEN_SECRET]

def accessApiTwitter():

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(token_key, token_secret)

    api = tweepy.API(auth,wait_on_rate_limit=True)

    return api

def verifyCoordinates(coord):

    if coord is None:
        coordVerified = "Tweet sem localização!"
    else:
        coordVerified = coord

    return coordVerified

def searchHashtagReportOnTwitter(hashtagName, dateSince, accessPackageApiTwitter):
    global df_hashtagName

    for tweet in tweepy.Cursor(accessPackageApiTwitter.search,
                               q=hashtagName,
                               lang="pt",
                               since=dateSince,
                               tweet_mode="extended").items(10):
        user = tweet.user
        coordinates = tweet.coordinates
        locationPost = verifyCoordinates(coordinates)
        df_hashtagName = df_hashtagName.append({"data": tweet.created_at, "post": tweet.full_text, "localizacaoUsuario": user.location, "localizacaoPost": locationPost, "descricaoPerfil": user.description, "seguidores": user.followers_count, "perfilVerificado": user.verified}, ignore_index=True)

def searchKeywordReportOnTwitter(keyword, dateSince, accessPackageApiTwitter):
    global df_Keyword

    for tweet in tweepy.Cursor(accessPackageApiTwitter.search,
                               q=keyword,
                               lang="pt",
                               since=dateSince,
                               tweet_mode="extended").items(10):
        user = tweet.user
        coordinates = tweet.coordinates
        locationPost = verifyCoordinates(coordinates)
        df_Keyword = df_Keyword.append({"data": tweet.created_at, "post": tweet.full_text, "localizacaoUsuario": user.location, "localizacaoPost": locationPost, "descricaoPerfil": user.description, "seguidores": user.followers_count, "perfilVerificado": user.verified}, ignore_index=True)

def reportTokenization(tweet):
    doc = nlp(tweet)
    return doc

def selectOnlyVerbsReport(doc):
    list = []
    for token in doc:
        if token.pos_ == 'VERB':
            list.append(token)

    return list

def sumOfList(list):
    sum = 0
    for num in list:
        sum += num

    return sum

def createNumericRepresentation(list, dictionary, weight):
    listNumericRepresentation = []
    nlp = spacy.load("pt_core_news_sm")

    for index, word in enumerate(dictionary):
        doc = nlp(word)
        sum = 0
        for token in list:
            sum += doc.similarity(token)
        mean = sum / len(list)

        listNumericRepresentation.append(mean*weight[index])

    return listNumericRepresentation

def typeAnalysis(list):

    listViolence = []

    listNumRepresSexualViolence = createNumericRepresentation(list, dictionaryKeywordsSexualViolence, weightsForKeywordsSexualViolence)
    listViolence.append(Violence(-sumOfList(listNumRepresSexualViolence), 'Violência Sexual'))
    
    listNumRepresPhysicalViolence = createNumericRepresentation(list, dictionaryKeywordsPhysicalViolence, weightsForKeywordsPhysicalViolence)
    listViolence.append(Violence(-sumOfList(listNumRepresPhysicalViolence), 'Violência Física'))

    listNumRepresPsychologicalViolence = createNumericRepresentation(list, dictionaryKeywordsPsychologicalViolence, weightsForKeywordsPsychologicalViolence)
    listViolence.append(Violence(-sumOfList(listNumRepresPsychologicalViolence), 'Violência Psicológica'))

    listNumRepresMoralViolence = createNumericRepresentation(list, dictionaryKeywordsMoralViolence, weightsForKeywordsMoralViolence)
    listViolence.append(Violence(-sumOfList(listNumRepresMoralViolence), 'Violência Moral'))

    listNumRepresPatrimonialViolence = createNumericRepresentation(list, dictionaryKeywordsPatrimonialViolence, weightsForKeywordsPatrimonialViolence)
    listViolence.append(Violence(-sumOfList(listNumRepresPatrimonialViolence), 'Violência Patrimonial'))

    typeViolenceSorted = sorted(listViolence, key=Violence.get_value)

    return typeViolenceSorted

#seleção, gravação e pre-processamento dos dados

accessPackageApiTwitter = accessApiTwitter()

#busca por HASHTAG
searchHashtagReportOnTwitter("#MeuExAbusivo -filter:retweets", "2020-01-10", accessPackageApiTwitter)

#busca por KEYWORD
searchKeywordReportOnTwitter("violência contra mulher -filter:retweets", "2020-01-10", accessPackageApiTwitter)

#personalização do Tokenizador
nlp = spacy.load("pt_core_news_sm")
nlp.tokenizer = MyTweetTokenizer(nlp.vocab)

#tratamento e classificação dos dados

#da busca POR HASHTAG
for index, row in df_hashtagName.iterrows():
    doc = reportTokenization(row[1])
    listVerbsOfReport = selectOnlyVerbsReport(doc)
    valueAndTypeViolenceSorted = typeAnalysis(listVerbsOfReport)
    typeViolenceSorted = [v.get_type() for v in valueAndTypeViolenceSorted]
    df_hashtagName.at[index, 'tipoDeViolencia'] = typeViolenceSorted

df_hashtagName.to_csv('csv/v2/datasetReportsHashtagName.csv')

#da busca POR KEYWORD
for index, row in df_Keyword.iterrows():
    doc = reportTokenization(row[1])
    listVerbsOfReport = selectOnlyVerbsReport(doc)
    valueAndTypeViolenceSorted = typeAnalysis(listVerbsOfReport)
    typeViolenceSorted = [v.get_type() for v in valueAndTypeViolenceSorted]
    df_Keyword.at[index, 'tipoDeViolencia'] = typeViolenceSorted

df_Keyword.to_csv('csv/v2/datasetReportsKeyword.csv')













