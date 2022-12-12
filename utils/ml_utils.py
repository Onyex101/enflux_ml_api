import pickle
import spacy
from lime.lime_text import LimeTextExplainer
from sklearn.preprocessing import normalize
import pandas as pd
import numpy as np
from io import BytesIO
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=False)

nlp = spacy.load("en_core_web_sm")

xt = 0.030321932902136105

# with open('ml_model/tfidf_vectorizer.pkl', 'rb') as f:
#     vectorizer = pickle.load(f)
# with open('ml_model/count_vectorizer.pkl', 'rb') as f:
#     cv = pickle.load(f)
# with open('ml_model/model.pkl', 'rb') as f:
#     clf = pickle.load(f)
#     labels = clf.classes_


def preprocess(text):
    doc = nlp(text)
    tag_pair = []
    text = []
    for token in doc:
        if token.is_alpha and len(token.lemma_) > 1:
            tag_pair.append([token.lemma_, token.pos_])
            text.append(token.lemma_)
    return ' '.join(text), tag_pair


def flatten(term_doc_matrix):
    return [term_doc_matrix[i].toarray().tolist()[0] for i in range(term_doc_matrix.shape[0])]


def termdoc_weighting(termdoc, features, tag_pair, xt):
    result = []
    for i in range(len(termdoc)):
        holder = termdoc[i]
        holder_pair = tag_pair[i]
        temp = []
        for j in range(len(holder)):
            if holder[j] != 0:
                term = features[j]
                weight = 1
                for k in range(len(holder_pair)):
                    if holder_pair[k][0] == term:
                        if holder_pair[k][1] == 'VERB':
                            weight = xt + 3
                        elif holder_pair[k][1] == 'ADJ' or holder_pair[k][1] == 'NOUN':
                            weight = xt + 1
                        else:
                            weight = 1
                temp.append(holder[j] * weight)
            else:
                temp.append(holder[j])
        result.append(temp)
    return result


def count_tfidf(termdoc_weighted, features, df):
    result = []
    for i in range(len(termdoc_weighted)):
        holder = termdoc_weighted[i]
        temp = []
        denom = sum(termdoc_weighted[i])
        for j in range(len(holder)):
            if holder[j] != 0:
                term = features[j]
                idf = df.loc[df['word'] == term].idf
                temp.append((holder[j]/denom) * (float(idf)))
            else:
                temp.append(holder[j])
        result.append(temp)
    return result


def pred_LIME_prob(text, Probability=True):
    df_pred = pd.DataFrame(text, columns=['question'])
    df_pred['text'], df_pred['tag_pair'] = zip(
        *df_pred['question'].parallel_apply(preprocess))
    X_pair = df_pred['tag_pair']
    X_vec = df_pred['text']
    skl_tf_idf_vectorized = vectorizer.transform(X_vec)
    df_idf = pd.DataFrame(vectorizer.idf_, columns=['idf'])
    df_idf['word'] = vectorizer.get_feature_names_out()
    tfidf_def = flatten(skl_tf_idf_vectorized)
    df_tfidf = pd.DataFrame(tfidf_def)
    X_def = cv.transform(X_vec)
    features_def = cv.get_feature_names_out()
    termdoc_def = X_def.toarray()
    termdoc_weighted_def = termdoc_weighting(
        termdoc_def, features_def, X_pair, xt)
    ctfidf_def = count_tfidf(termdoc_weighted_def, features_def, df_idf)
    tfposidf_def = normalize(ctfidf_def, norm='l2').tolist()
    text_tdf = pd.DataFrame(tfposidf_def)
    return clf.predict_proba(text_tdf) if Probability else clf.predict(text_tdf)


def get_prediction(text):
    text = extract_sentence(text)
    return pred_LIME_prob([text], Probability=False)[0]


def load_LIME(text):
    text = extract_sentence(text)
    explainer = LimeTextExplainer(class_names=labels, verbose=True)
    explanation = explainer.explain_instance(
        text, classifier_fn=pred_LIME_prob, num_features=6, top_labels=1, num_samples=50)
    return explanation.as_html()


def extract_sentence(text):
  doc = nlp(text.lower())
  return str(list(doc.sents)[-1])


def predict_file(file):
    df = pd.read_excel(file, engine="openpyxl")
    df['text'] = df['item_text'].parallel_apply(extract_sentence)
    df["predicted_level"] = pred_LIME_prob(
        df['text'].tolist(), Probability=False)
    # Creating output and writer (pandas excel writer)
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(excel_writer=writer, index=False,
                    sheet_name='Sheet1', header=True)
    return buffer
