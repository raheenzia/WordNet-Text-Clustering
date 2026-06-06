import nltk
from collections import Counter
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

for _path, _pkg in [
    ('tokenizers/punkt',                          'punkt'),
    ('tokenizers/punkt_tab',                      'punkt_tab'),
    ('taggers/averaged_perceptron_tagger',         'averaged_perceptron_tagger'),
    ('taggers/averaged_perceptron_tagger_eng',     'averaged_perceptron_tagger_eng'),
    ('corpora/wordnet',                            'wordnet'),
]:
    try:
        nltk.data.find(_path)
    except LookupError:
        nltk.download(_pkg, quiet=True)

_lemmatizer = WordNetLemmatizer()

def _pos_tag(text):
    tokens = nltk.word_tokenize(text.lower())
    return nltk.pos_tag(tokens)

def _to_wn_pos(tag):
    if tag.startswith('N'): return wn.NOUN
    if tag.startswith('V'): return wn.VERB
    if tag.startswith('J'): return wn.ADJ
    return None


def _remove_stopwords(tagged):
    kept = []
    for word, tag in tagged:
        wn_pos = _to_wn_pos(tag)
        if wn_pos is None:     continue
        if len(word) < 3:      continue
        if not word.isalpha(): continue
        kept.append((word, wn_pos))
    return kept

def _lemmatize(filtered):
    return [(_lemmatizer.lemmatize(word, pos=wn_pos), wn_pos)
            for word, wn_pos in filtered]

# Recursively collect hypernym ID
def _get_hypernyms(synset, depth):
    if depth == 0:
        return []
    result = []
    for hyper in synset.hypernyms():
        result.append(f"hyp_{hyper.pos()}_{hyper.offset()}")
        result += _get_hypernyms(hyper, -1 if depth == -1 else depth - 1)
    return result


def _enrich(stemmed, config):
    use_pos        = config['use_pos']
    use_synonyms   = config['use_synonyms']
    hypernym_depth = config['hypernym_depth']

    terms = []
    for stem, wn_pos in stemmed:
        terms.append(f"{stem}_{wn_pos}" if use_pos else stem)

        if use_synonyms:
            for ss in wn.synsets(stem, pos=wn_pos):
                terms.append(f"syn_{ss.pos()}_{ss.offset()}")
                if hypernym_depth != 0:
                    terms += _get_hypernyms(ss, hypernym_depth)

    return terms

def _prune(tokenized_docs, threshold):
    doc_freq = Counter()
    for doc in tokenized_docs:
        doc_freq.update(set(doc))

    valid  = {t for t, f in doc_freq.items() if f >= threshold}
    pruned = [[t for t in doc if t in valid] for doc in tokenized_docs]

    print(f"    pruning: {len(doc_freq)} → {len(valid)} terms "
          f"(threshold={threshold})")
    return pruned

def _tfidf(tokenized_docs):
    text_docs  = [" ".join(doc) if doc else "empty" for doc in tokenized_docs]
    vectorizer = TfidfVectorizer(
        analyzer='word',
        token_pattern=r'\S+',
        min_df=1,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(text_docs)
    print(f"    tfidf matrix: {X.shape[0]} docs × {X.shape[1]} terms")
    return X

def build_document_vectors(docs, config, pruning_threshold):
    tokenized = []
    for i, doc in enumerate(docs):
        tagged   = _pos_tag(doc)             # Step 1
        filtered = _remove_stopwords(tagged) # Step 2
        stemmed  = _lemmatize(filtered)      # Step 3
        enriched = _enrich(stemmed, config)  # Step 4
        tokenized.append(enriched)
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(docs)} docs tokenized...")

    pruned = _prune(tokenized, pruning_threshold)  # Step 5
    X      = _tfidf(pruned)                        # Step 6
    return X
