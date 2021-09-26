from collections import defaultdict
import spacy
from spacy.tokens import Doc
from gensim import corpora
from gensim import models
import pandas as pd
import tweepy
from myTweetTokenizer import MyTweetTokenizer
from gensim import similarities
from violence import Violence

#dataframe dos relatos de busca por HASHTAG com suas classificações
df_hashtagName = pd.DataFrame(columns=["data", "post", "tipoDeViolencia", "localizacaoUsuario", "localizacaoPost", "descricaoPerfil", "seguidores", "perfilVerificado"])

#dataframe dos relatos de busca por KEYWORD com suas classificações
df_Keyword = pd.DataFrame(columns=["data", "post", "tipoDeViolencia", "localizacaoUsuario", "localizacaoPost", "descricaoPerfil", "seguidores", "perfilVerificado"])

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

def createListFromAFile(filePath):
    archive = open(filePath, encoding='utf-8')

    list = []

    for line in archive:
        line = line.strip()
        list.append(line)

    archive.close()

    return list

def preprocessing(typeViolence):
    documents = createListFromAFile("corpus/documents"+typeViolence+".txt")

    documentsLem = []

    nlp = spacy.load('pt_core_news_sm')
    for document in documents:
        doc = nlp(document)

        docLem = []
        for t in doc:
            if t.text.lower() not in stoplist:
                docLem.append(t.lemma_)

        documentsLem.append(docLem)

    texts = [
        [word for word in document]
        for document in documentsLem
    ]

    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [[token for token in text if frequency[token] > 1] for text in texts]

    return texts

def reportTokenization(tweet):
    doc = nlp(tweet)
    return doc

def meanOfList(list):
    sum = 0
    for num in list:
        sum += num

    mean = sum / len(list)

    return mean

def processReport(doc, dictionary, lsi, index):
    vec_bow = dictionary.doc2bow([t.text.lower() for t in doc])
    vec_lsi = lsi[vec_bow]
    sims = index[vec_lsi]

    return sims

def typeAnalysis(doc):
    listViolence = []

    simsSexualViolence = processReport(doc, dictionarySexualViolence, lsiSexualViolence, indexSexualViolence)
    listViolence.append(Violence(-meanOfList(list(simsSexualViolence)), 'Violência Sexual'))

    simsPhysicalViolence = processReport(doc, dictionaryPhysicalViolence, lsiPhysicalViolence, indexPhysicalViolence)
    listViolence.append(Violence(-meanOfList(list(simsPhysicalViolence)), 'Violência Física'))

    simsPsychologicalViolence = processReport(doc, dictionaryPsychologicalViolence, lsiPsychologicalViolence, indexPsychologicalViolence)
    listViolence.append(Violence(-meanOfList(list(simsPsychologicalViolence)), 'Violência Psicológica'))

    simsMoralViolence = processReport(doc, dictionaryMoralViolence, lsiMoralViolence, indexMoralViolence)
    listViolence.append(Violence(-meanOfList(list(simsMoralViolence)), 'Violência Moral'))

    simsPatrimonialViolence = processReport(doc, dictionaryPatrimonialViolence, lsiPatrimonialViolence, indexPatrimonialViolence)
    listViolence.append(Violence(-meanOfList(list(simsPatrimonialViolence)), 'Violência Patrimonial'))

    typeViolenceSorted = sorted(listViolence, key=Violence.get_value)

    return typeViolenceSorted

#seleção, gravação e pre-processamento dos dados de Twitter

accessPackageApiTwitter = accessApiTwitter()

#busca por HASHTAG
searchHashtagReportOnTwitter("#MeuExAbusivo -filter:retweets", "2020-01-10", accessPackageApiTwitter)

#busca por KEYWORD
searchKeywordReportOnTwitter("violência contra mulher -filter:retweets", "2020-01-10", accessPackageApiTwitter)

stopwords = createListFromAFile("txt/stopwords.txt")
stoplist = set(stopwords)

#Pre-processamento do Corpus
textsSexualViolence = preprocessing("SexualViolence")
textsPhysicalViolence = preprocessing("PhysicalViolence")
textsPsychologicalViolence = preprocessing("PsychologicalViolence")
textsMoralViolence = preprocessing("MoralViolence")
textsPatrimonialViolence = preprocessing("PatrimonialViolence")

#Dictionary - defines the vocabulary
dictionarySexualViolence = corpora.Dictionary(textsSexualViolence)
dictionaryPhysicalViolence = corpora.Dictionary(textsPhysicalViolence)
dictionaryPsychologicalViolence = corpora.Dictionary(textsPsychologicalViolence)
dictionaryMoralViolence = corpora.Dictionary(textsMoralViolence)
dictionaryPatrimonialViolence = corpora.Dictionary(textsPatrimonialViolence)

#to vectorize
corpusSexualViolence = [dictionarySexualViolence.doc2bow(text) for text in textsSexualViolence]
corpusPhysicalViolence = [dictionaryPhysicalViolence.doc2bow(text) for text in textsPhysicalViolence]
corpusPsychologicalViolence = [dictionaryPsychologicalViolence.doc2bow(text) for text in textsPsychologicalViolence]
corpusMoralViolence = [dictionaryMoralViolence.doc2bow(text) for text in textsMoralViolence]
corpusPatrimonialViolence = [dictionaryPatrimonialViolence.doc2bow(text) for text in textsPatrimonialViolence]

#Model
lsiSexualViolence = models.LsiModel(corpusSexualViolence, id2word=dictionarySexualViolence, num_topics=2)
lsiPhysicalViolence = models.LsiModel(corpusPhysicalViolence, id2word=dictionaryPhysicalViolence, num_topics=2)
lsiPsychologicalViolence = models.LsiModel(corpusPsychologicalViolence, id2word=dictionaryPsychologicalViolence, num_topics=2)
lsiMoralViolence = models.LsiModel(corpusMoralViolence, id2word=dictionaryMoralViolence, num_topics=2)
lsiPatrimonialViolence = models.LsiModel(corpusPatrimonialViolence, id2word=dictionaryPatrimonialViolence, num_topics=2)

#indexação em Matriz de Similaridade
indexSexualViolence = similarities.MatrixSimilarity(lsiSexualViolence[corpusSexualViolence])
indexPhysicalViolence = similarities.MatrixSimilarity(lsiPhysicalViolence[corpusPhysicalViolence])
indexPsychologicalViolence = similarities.MatrixSimilarity(lsiPsychologicalViolence[corpusPsychologicalViolence])
indexMoralViolence = similarities.MatrixSimilarity(lsiMoralViolence[corpusMoralViolence])
indexPatrimonialViolence = similarities.MatrixSimilarity(lsiPatrimonialViolence[corpusPatrimonialViolence])

#personalização do Tokenizador
nlp = spacy.load("pt_core_news_sm")
nlp.tokenizer = MyTweetTokenizer(nlp.vocab)

#tratamento e classificação dos dados

#da busca POR HASHTAG
for index, row in df_hashtagName.iterrows():
    doc = reportTokenization(row[1])
    valueAndTypeViolenceSorted = typeAnalysis(doc)
    typeViolenceSorted = [v.get_type() for v in valueAndTypeViolenceSorted]
    df_hashtagName.at[index, 'tipoDeViolencia'] = typeViolenceSorted

df_hashtagName.to_csv('csv/datasetReportsHashtagName.csv')

#da busca POR KEYWORD
for index, row in df_Keyword.iterrows():
    doc = reportTokenization(row[1])
    valueAndTypeViolenceSorted = typeAnalysis(doc)
    typeViolenceSorted = [v.get_type() for v in valueAndTypeViolenceSorted]
    df_Keyword.at[index, 'tipoDeViolencia'] = typeViolenceSorted

df_Keyword.to_csv('csv/datasetReportsKeyword.csv')
