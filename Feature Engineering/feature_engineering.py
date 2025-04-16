import os
import re
import time
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union
from scipy import stats

import nltk
import numpy as np
import pandas as pd
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer

try:
    from nrclex import NRCLex
except ImportError:
    exit()

from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning)

FEATURES_X_CSV = 'optimized_features_X.csv'
TARGET_Y_CSV = 'target_y.csv'

INTENSIFIER_KEYWORDS = {
    "absolutely", "completely", "entirely", "extremely", "fucking", "fully", "goddamn", "highly", "hugely",
    "incredibly", "insanely", "literally", "massive", "perfectly", "positively", "purely", "remarkably",
    "strikingly", "stupendously", "terrifically", "thoroughly", "totally", "unbelievably", "utterly", "very",
    "awfully", "certainly", "clearly", "considerably", "damn", "deadly", "decidedly", "deeply", "distinctly",
    "eminently", "enormously", "especially", "exceptionally", "extraordinarily", "fairly", "frightfully",
    "greatly", "hella", "immensely", "indeed", "intensely", "jolly", "mightily", "most", "noticeably",
    "particularly", "peculiarly", "plenty", "pretty", "quite", "radically", "rather", "real", "really",
    "significantly", "so", "somewhat", "strongly", "substantially", "super", "supremely", "surely", "terribly",
    "truly", "unusually", "vastly", "almost", "barely", "effectively", "essentially", "fundamentally", "hardly",
    "just", "largely", "mainly", "merely", "minimally", "mostly", "nearly", "nominally", "only", "partially",
    "partly", "practically", "primarily", "principally", "relatively", "roughly", "scarcely", "simply", "slightly",
    "technically", "virtually", "always", "constantly", "definitely", "every", "never", "undoubtedly",
    "unquestionably", "bloody", "crazy", "damn", "freaking", "mad", "wicked"
}
GENERIC_PERSON_SINGLE_WORDS = {
    "man", "woman", "person", "guy", "gal", "dude", "chap", "bloke", "individual", "adult", "someone", "somebody",
    "anyone", "anybody", "everyone", "everybody", "nobody", "mom", "dad", "mother", "father", "parent", "parents",
    "son", "daughter", "child", "children", "kid", "kids", "baby", "infant", "toddler", "grandma", "grandpa",
    "grandmother", "grandfather", "grandparent", "husband", "wife", "spouse", "brother", "sister", "sibling",
    "uncle", "aunt", "cousin", "friend", "buddy", "pal", "acquaintance", "stranger", "teen", "teenager", "youth",
    "senior", "retiree", "coworker", "colleague", "boss", "employee", "staffer", "worker", "customer", "client",
    "patient", "student", "teacher", "driver", "pedestrian", "bystander", "resident", "neighbor"
}
GENERIC_PERSON_PHRASES_REGEX_PATTERNS = [
    r'\barea\s+(man|woman|dad|mom|resident|teenager?|youth|official|couple|child|business\s+owner|teacher)\b',
    r'\blocal\s+(man|woman|resident|teenager?|youth|official|couple|child|business\s+owner|teacher|mom|dad)\b',
    r'\b(city|town|county|nearby|neighborhood)\s+(resident|man|woman)\b', r'\bsenior\s+citizen\b',
    r'\belderly\s+(man|woman|couple|person)\b', r'\bmiddle-aged\s+(man|woman|person)\b',
    r'\byoung\s+(child|man|woman|person)\b',
    r'\b(office|store|shop|fast\s+food|restaurant|government|city|state|county|hospital|health\s+care)\s+(worker|employee)\b',
    r'\b(delivery|bus|taxi|truck)\s+(driver|person)\b',
    r'\b(unnamed|anonymous)\s+(official|source|person|employee|worker)\b', r'\b(pet|car|home)\s*owner\b',
    r'\beyewitness(?:es)?\b', r'\bconcerned\s+(citizen|resident|parent|person)\b',
    r'\bmember\s+of\s+the\s+public\b', r'\b(injured|missing)\s+(man|woman|person|teen|child)\b',
    r'\bmarried\s+couple\b', r'\bgroup\s+of\s+(friends|teens|youths|people|students|workers)\b',
    r'\bfamily\s+members?\b', r'\b(several|some|few|many)\s+people\b', r'\bno\s+one\b',
]
GENERIC_PERSON_REGEX_COMPILED = [re.compile(pattern, re.IGNORECASE) for pattern in GENERIC_PERSON_PHRASES_REGEX_PATTERNS]
MUNDANE_ACTION_VERBS = {
    "use", "enter", "announce", "learn", "stare", "hand", "visit", "watch", "find", "get", "take", "make", "keep",
    "report", "say", "tell", "ask", "talk", "speak", "call", "greet", "wave", "nod", "write", "read", "email",
    "text", "post", "tweet", "see", "hear", "feel", "smell", "taste", "think", "believe", "know", "realize",
    "consider", "remember", "forget", "notice", "wonder", "decide", "plan", "hope", "wish", "expect", "assume",
    "guess", "go", "come", "leave", "arrive", "stay", "sit", "stand", "lie", "walk", "run", "move", "turn", "carry",
    "hold", "put", "place", "set", "give", "bring", "send", "eat", "drink", "sleep", "wake", "breathe", "look",
    "point", "reach", "touch", "open", "close", "start", "stop", "continue", "wait", "try", "help", "have", "own",
    "need", "want", "like", "love", "hate", "prefer", "seem", "appear", "become", "remain", "work", "play", "study",
    "shop", "buy", "sell", "pay", "drive", "ride", "cook", "clean", "wash", "dress", "undress", "sit", "add", "begin",
    "change", "check", "choose", "finish", "happen", "include", "let", "listen", "live", "lose", "mean", "meet",
    "offer", "order", "pass", "pull", "push", "raise", "receive", "return", "serve", "show", "spend", "suggest", "wear"
}

nlp: Optional[spacy.language.Language] = None
vader_analyzer: Optional[SentimentIntensityAnalyzer] = None

def setup_nlp_resources() -> None:
    global nlp, vader_analyzer
    spacy_model_name = "en_core_web_lg"
    try:
        nlp = spacy.load(spacy_model_name)
        if not nlp.vocab.vectors.size:
             pass
        else:
             pass
    except OSError:
        try:
            spacy.cli.download(spacy_model_name)
            nlp = spacy.load(spacy_model_name)
            if not nlp.vocab.vectors.size:
                pass
            else:
                pass
        except Exception as e:
            exit()

    try:
        vader_analyzer = SentimentIntensityAnalyzer()
        _ = vader_analyzer.polarity_scores("test")
    except LookupError:
        try:
            nltk.download('vader_lexicon')
            vader_analyzer = SentimentIntensityAnalyzer()
        except Exception as e:
            exit()
    except Exception as e:
        exit()

def get_lexical_basic_features(doc: spacy.tokens.doc.Doc) -> Dict[str, Union[int, float]]:
    features: Dict[str, Union[int, float]] = {}
    non_space_tokens = [token for token in doc if not token.is_space]
    num_tokens = len(non_space_tokens)

    if num_tokens == 0:
        return {
            'feat_char_count': 0, 'feat_word_count': 0, 'feat_avg_word_length': 0.0,
            'feat_question_mark_flag': 0, 'feat_exclamation_mark_count': 0,
            'feat_quote_count': 0, 'feat_punct_count': 0, 'feat_punct_ratio': 0.0,
            'feat_all_caps_count': 0, 'feat_all_caps_ratio': 0.0,
            'feat_first_person_pron_count': 0, 'feat_second_person_pron_count': 0
        }

    features['feat_char_count'] = len(doc.text)
    features['feat_word_count'] = num_tokens
    features['feat_avg_word_length'] = np.mean([len(t.text) for t in non_space_tokens]) if num_tokens > 0 else 0.0
    features['feat_question_mark_flag'] = 1 if '?' in doc.text else 0
    features['feat_exclamation_mark_count'] = doc.text.count('!')
    features['feat_quote_count'] = sum(1 for token in doc if token.is_quote)
    features['feat_punct_count'] = sum(1 for token in doc if token.is_punct)
    features['feat_punct_ratio'] = features['feat_punct_count'] / num_tokens if num_tokens > 0 else 0.0

    all_caps_count = sum(1 for token in non_space_tokens if token.text.isupper() and len(token.text) > 1)
    features['feat_all_caps_count'] = all_caps_count
    features['feat_all_caps_ratio'] = all_caps_count / num_tokens if num_tokens > 0 else 0.0

    first_person_pronouns = {'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
    second_person_pronouns = {'you', 'your', 'yours'}
    features['feat_first_person_pron_count'] = sum(1 for token in doc if token.lemma_.lower() in first_person_pronouns)
    features['feat_second_person_pron_count'] = sum(1 for token in doc if token.lemma_.lower() in second_person_pronouns)

    return features


def get_pos_features(doc: spacy.tokens.doc.Doc) -> Dict[str, float]:
    pos_counts = Counter(token.pos_ for token in doc if not token.is_punct and not token.is_space)
    num_valid_tokens = sum(pos_counts.values())

    if num_valid_tokens == 0:
        return {
            'feat_noun_ratio': 0.0, 'feat_verb_ratio': 0.0, 'feat_adj_ratio': 0.0,
            'feat_adv_ratio': 0.0, 'feat_propn_ratio': 0.0, 'feat_pron_ratio': 0.0,
            'feat_det_ratio': 0.0
        }

    features: Dict[str, float] = {}
    target_pos = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN', 'PRON', 'DET']
    for pos in target_pos:
        features[f'feat_{pos.lower()}_ratio'] = pos_counts.get(pos, 0) / num_valid_tokens

    return features


def get_sentiment_emotion_features(headline_text: str, doc: spacy.tokens.doc.Doc, sentiment_threshold: float = 0.2) -> Dict[str, Union[int, float]]:
    features: Dict[str, Union[int, float]] = {}
    if vader_analyzer is None:
        raise RuntimeError("VADER analyzer not initialized. Call setup_nlp_resources() first.")

    vader_scores = vader_analyzer.polarity_scores(headline_text)
    overall_compound = vader_scores['compound']
    features['feat_sentiment_vader_compound'] = overall_compound
    features['feat_sentiment_vader_pos'] = vader_scores['pos']
    features['feat_sentiment_vader_neg'] = vader_scores['neg']
    features['feat_sentiment_vader_neu'] = vader_scores['neu']

    sentence_sentiments = []
    try:
        sentence_sentiments = [vader_analyzer.polarity_scores(sent.text)['compound'] for sent in doc.sents]
    except ValueError as e:
        sentence_sentiments = [overall_compound]

    num_sentences = len(sentence_sentiments)

    default_sentiment_dyn_features = {
        'feat_sentiment_variance': 0.0, 'feat_sentiment_range': 0.0,
        'feat_sentiment_oscillation_flag': 0, 'feat_sentiment_pos_peak_count': 0,
        'feat_sentiment_neg_trough_count': 0, 'feat_sentiment_max_deviation_from_overall': 0.0,
        'feat_sentiment_deviation_variance': 0.0, 'feat_sentiment_peak_prominence': 0.0,
        'feat_sentiment_trough_prominence': 0.0
    }
    features.update(default_sentiment_dyn_features)


    if num_sentences > 0 and len(sentence_sentiments) > 0:
        sent_array = np.array(sentence_sentiments)

        if sent_array.size > 0:
            min_sent, max_sent = np.min(sent_array), np.max(sent_array)
            features['feat_sentiment_range'] = max_sent - min_sent

            deviations = np.abs(sent_array - overall_compound)
            if deviations.size > 0:
                features['feat_sentiment_max_deviation_from_overall'] = np.max(deviations)

            pos_peaks_mask = sent_array > sentiment_threshold
            neg_troughs_mask = sent_array < -sentiment_threshold
            features['feat_sentiment_pos_peak_count'] = int(np.sum(pos_peaks_mask))
            features['feat_sentiment_neg_trough_count'] = int(np.sum(neg_troughs_mask))

            if num_sentences > 1:
                features['feat_sentiment_variance'] = np.var(sent_array)
                if deviations.size > 1:
                    features['feat_sentiment_deviation_variance'] = np.var(deviations)
                else:
                    features['feat_sentiment_deviation_variance'] = 0.0

                if np.any(pos_peaks_mask) and np.any(neg_troughs_mask):
                    features['feat_sentiment_oscillation_flag'] = 1

                if num_sentences >= 2:
                    sorted_sentiments = np.sort(sent_array)
                    features['feat_sentiment_peak_prominence'] = sorted_sentiments[-1] - sorted_sentiments[-2]
                    features['feat_sentiment_trough_prominence'] = sorted_sentiments[1] - sorted_sentiments[0]
                else:
                     features['feat_sentiment_peak_prominence'] = 0.0
                     features['feat_sentiment_trough_prominence'] = 0.0

    emotions = ['fear', 'anger', 'anticipation', 'trust', 'surprise', 'sadness', 'disgust', 'joy']
    for emotion in emotions:
        features[f'feat_emotion_{emotion}_freq'] = 0.0

    try:
        if headline_text and headline_text.strip():
            nrc = NRCLex(headline_text)
            emotion_freqs = nrc.affect_frequencies
            emotion_scores = {e: emotion_freqs.get(e, 0.0) for e in emotions}
            total_affect_words = sum(emotion_freqs.values())

            if total_affect_words > 0:
                for emotion in emotions:
                    features[f'feat_emotion_{emotion}_freq'] = emotion_scores.get(emotion, 0.0) / total_affect_words
    except Exception as e:
        pass

    return features


def get_exaggeration_features(doc: spacy.tokens.doc.Doc, absurd_num_threshold: int = 1_000_000) -> Dict[str, Union[int, float]]:
    features: Dict[str, Union[int, float]] = {
        'feat_intensifier_count': 0, 'feat_number_count': 0, 'feat_superlative_count': 0,
        'feat_comparative_count': 0, 'feat_intensifier_flag': 0, 'feat_contains_number_flag': 0,
        'feat_absurd_number_flag': 0, 'feat_intensifier_ratio': 0.0
    }
    non_space_tokens = [token for token in doc if not token.is_space]
    num_tokens = len(non_space_tokens)
    if num_tokens == 0:
        return features

    intensifier_count = 0
    number_count = 0
    superlative_count = 0
    comparative_count = 0
    max_number_val = 0.0

    for token in doc:
        if not token.is_space:
            if token.lemma_.lower() in INTENSIFIER_KEYWORDS:
                intensifier_count += 1

            if token.like_num:
                number_count += 1
                try:
                    num_text = re.sub(r'[,\$%]', '', token.text)
                    current_num_val = float(num_text)
                    max_number_val = max(abs(current_num_val), max_number_val)
                except ValueError:
                    pass

            if token.tag_ in ['JJS', 'RBS']:
                superlative_count += 1
            elif token.tag_ in ['JJR', 'RBR']:
                comparative_count += 1

    features['feat_intensifier_count'] = intensifier_count
    features['feat_number_count'] = number_count
    features['feat_superlative_count'] = superlative_count
    features['feat_comparative_count'] = comparative_count

    if intensifier_count > 0:
        features['feat_intensifier_flag'] = 1
    if number_count > 0:
        features['feat_contains_number_flag'] = 1
        if max_number_val >= absurd_num_threshold:
            features['feat_absurd_number_flag'] = 1

    features['feat_intensifier_ratio'] = intensifier_count / num_tokens if num_tokens > 0 else 0.0

    return features


def get_mundanity_ner_features(headline_text: str, doc: spacy.tokens.doc.Doc) -> Dict[str, Union[int, float]]:
    features: Dict[str, Union[int, float]] = {
        'feat_generic_person_term_count': 0, 'feat_mundane_action_verb_count': 0,
        'feat_generic_person_term_flag': 0, 'feat_mundane_action_verb_flag': 0,
        'feat_mundane_combo_flag': 0, 'feat_ner_total_count': 0, 'feat_ner_person_count': 0,
        'feat_ner_org_count': 0, 'feat_ner_gpe_count': 0, 'feat_ner_norp_count': 0,
        'feat_ner_fac_count': 0, 'feat_ner_loc_count': 0, 'feat_ner_product_count': 0,
        'feat_ner_event_count': 0, 'feat_ner_avg_char_length': 0.0
    }

    generic_person_count = 0
    found_generic_person = False

    for token in doc:
        if token.lemma_.lower() in GENERIC_PERSON_SINGLE_WORDS:
            generic_person_count += 1
            found_generic_person = True

    if headline_text and headline_text.strip():
        for pattern in GENERIC_PERSON_REGEX_COMPILED:
            try:
                matches = pattern.findall(headline_text)
                if matches:
                    generic_person_count += len(matches)
                    found_generic_person = True
            except Exception as regex_error:
                pass


    features['feat_generic_person_term_count'] = generic_person_count
    if found_generic_person:
        features['feat_generic_person_term_flag'] = 1

    mundane_verb_count = 0
    found_mundane_verb = False
    for token in doc:
        if token.pos_ == 'VERB' and token.lemma_ in MUNDANE_ACTION_VERBS:
             mundane_verb_count += 1
             found_mundane_verb = True

    features['feat_mundane_action_verb_count'] = mundane_verb_count
    if found_mundane_verb:
        features['feat_mundane_action_verb_flag'] = 1

    if features['feat_generic_person_term_flag'] == 1 and features['feat_mundane_action_verb_flag'] == 1:
        features['feat_mundane_combo_flag'] = 1

    entities = doc.ents
    features['feat_ner_total_count'] = len(entities)

    if entities:
        entity_labels = Counter(ent.label_ for ent in entities)
        features['feat_ner_person_count'] = entity_labels.get('PERSON', 0)
        features['feat_ner_org_count'] = entity_labels.get('ORG', 0)
        features['feat_ner_gpe_count'] = entity_labels.get('GPE', 0)
        features['feat_ner_norp_count'] = entity_labels.get('NORP', 0)
        features['feat_ner_fac_count'] = entity_labels.get('FAC', 0)
        features['feat_ner_loc_count'] = entity_labels.get('LOC', 0)
        features['feat_ner_product_count'] = entity_labels.get('PRODUCT', 0)
        features['feat_ner_event_count'] = entity_labels.get('EVENT', 0)

        entity_lengths = [len(ent.text) for ent in entities if ent.text]
        if entity_lengths:
             features['feat_ner_avg_char_length'] = np.mean(entity_lengths)
        else:
             features['feat_ner_avg_char_length'] = 0.0

    return features


def get_structure_incongruity_features(doc: spacy.tokens.doc.Doc, distance_threshold: float = 0.75) -> Dict[str, Union[int, float]]:
    features: Dict[str, Union[int, float]] = {
        'feat_nc_max_semantic_distance': 0.0, 'feat_nc_avg_semantic_distance': 0.0,
        'feat_nc_semantic_distance_gt_threshold_flag': 0, 'feat_dep_max_tree_depth': 0,
        'feat_dep_avg_distance': 0.0, 'feat_dep_nsubj_count': 0, 'feat_dep_dobj_count': 0,
        'feat_dep_amod_count': 0, 'feat_dep_advcl_count': 0, 'feat_dep_pobj_count': 0,
        'feat_dep_nsubj_ratio': 0.0, 'feat_dep_dobj_ratio': 0.0
    }

    noun_chunks = [chunk for chunk in doc.noun_chunks if chunk.has_vector and np.any(chunk.vector)]

    if len(noun_chunks) >= 2:
        distances = []
        for i in range(len(noun_chunks)):
            for j in range(i + 1, len(noun_chunks)):
                vec1 = noun_chunks[i].vector
                vec2 = noun_chunks[j].vector
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                if norm1 > 0 and norm2 > 0:
                    similarity = np.dot(vec1, vec2) / (norm1 * norm2)
                    similarity = np.clip(similarity, -1.0, 1.0)
                    distance = 1.0 - similarity
                    distances.append(distance)

        if distances:
            max_dist = max(distances)
            features['feat_nc_max_semantic_distance'] = max_dist
            features['feat_nc_avg_semantic_distance'] = np.mean(distances)
            if max_dist > distance_threshold:
                features['feat_nc_semantic_distance_gt_threshold_flag'] = 1

    max_depth = 0
    dep_distances = []
    dep_counts = Counter()
    num_valid_tokens_for_ratio = 0

    for token in doc:
        is_valid_token = not token.is_punct and not token.is_space

        if is_valid_token:
            num_valid_tokens_for_ratio += 1

            current_depth = 0
            current = token
            for _ in range(len(doc)):
                if current.head == current:
                    break
                current_depth += 1
                current = current.head
            max_depth = max(current_depth, max_depth)

            if token.dep_ != 'ROOT':
                 dep_distances.append(abs(token.i - token.head.i))

        dep_counts[token.dep_.lower()] += 1

    features['feat_dep_max_tree_depth'] = max_depth
    if dep_distances:
        features['feat_dep_avg_distance'] = np.mean(dep_distances)
    else:
        features['feat_dep_avg_distance'] = 0.0

    features['feat_dep_nsubj_count'] = dep_counts.get('nsubj', 0) + dep_counts.get('nsubjpass', 0)
    features['feat_dep_dobj_count'] = dep_counts.get('dobj', 0)
    features['feat_dep_amod_count'] = dep_counts.get('amod', 0)
    features['feat_dep_advcl_count'] = dep_counts.get('advcl', 0)
    features['feat_dep_pobj_count'] = dep_counts.get('pobj', 0)

    if num_valid_tokens_for_ratio > 0:
        features['feat_dep_nsubj_ratio'] = features['feat_dep_nsubj_count'] / num_valid_tokens_for_ratio
        features['feat_dep_dobj_ratio'] = features['feat_dep_dobj_count'] / num_valid_tokens_for_ratio
    else:
        features['feat_dep_nsubj_ratio'] = 0.0
        features['feat_dep_dobj_ratio'] = 0.0

    return features


def get_doc_vector_features(doc: spacy.tokens.doc.Doc) -> Dict[str, float]:
    features: Dict[str, float] = {}
    if nlp is None or not nlp.vocab.vectors.size or not doc.has_vector:
        expected_dim = 300
        if nlp and nlp.vocab.vectors_length > 0:
             expected_dim = nlp.vocab.vectors_length
        else:
             pass

        for i in range(expected_dim):
            features[f'feat_doc_vec_{i}'] = 0.0
        return features

    vector = doc.vector
    vector_dim = nlp.vocab.vectors_length

    if not np.any(vector):
        for i in range(vector_dim):
             features[f'feat_doc_vec_{i}'] = 0.0
    elif vector.shape[0] != vector_dim:
         for i in range(vector_dim):
            features[f'feat_doc_vec_{i}'] = 0.0
    else:
        for i in range(vector_dim):
            features[f'feat_doc_vec_{i}'] = float(vector[i])

    return features


def get_interaction_features(features_dict: Dict[str, Union[int, float]]) -> Dict[str, float]:
    interactions: Dict[str, float] = {}
    def safe_get(key: str, default: float = 0.0) -> float:
        val = features_dict.get(key)
        return float(val) if val is not None else default

    epsilon = 1e-6

    interactions['feat_interact_mundane_combo_x_intensifier_ratio'] = (
        safe_get('feat_mundane_combo_flag') * safe_get('feat_intensifier_ratio')
    )
    interactions['feat_interact_generic_person_flag_x_superlative_count'] = (
        safe_get('feat_generic_person_term_flag') * safe_get('feat_superlative_count')
    )

    pos_emotion_freq_sum = sum(safe_get(f'feat_emotion_{e}_freq')
                               for e in ['joy', 'trust', 'anticipation'])
    interactions['feat_interact_vader_neg_x_pos_emotion_freq'] = (
        safe_get('feat_sentiment_vader_neg') * pos_emotion_freq_sum
    )

    neg_emotion_freq_sum = sum(safe_get(f'feat_emotion_{e}_freq')
                               for e in ['anger', 'fear', 'sadness', 'disgust'])
    interactions['feat_interact_vader_pos_x_neg_emotion_freq'] = (
        safe_get('feat_sentiment_vader_pos') * neg_emotion_freq_sum
    )

    interactions['feat_interact_sentiment_range_x_intensifier_ratio'] = (
        safe_get('feat_sentiment_range') * safe_get('feat_intensifier_ratio')
    )
    interactions['feat_interact_oscillation_flag_x_surprise_freq'] = (
        safe_get('feat_sentiment_oscillation_flag') * safe_get('feat_emotion_surprise_freq')
    )

    interactions['feat_interact_nc_max_distance_x_ner_avg_len'] = (
        safe_get('feat_nc_max_semantic_distance') * safe_get('feat_ner_avg_char_length')
    )
    interactions['feat_interact_nc_max_distance_x_dep_max_depth'] = (
        safe_get('feat_nc_max_semantic_distance') * safe_get('feat_dep_max_tree_depth')
    )

    interactions['feat_interact_pos_neg_emotion_ratio'] = (
        pos_emotion_freq_sum / (neg_emotion_freq_sum + epsilon)
    )
    interactions['feat_interact_vader_compound_x_sentiment_range'] = (
        safe_get('feat_sentiment_vader_compound') * safe_get('feat_sentiment_range')
    )

    interactions['feat_interact_absurd_num_flag_x_generic_person_flag'] = (
        safe_get('feat_absurd_number_flag') * safe_get('feat_generic_person_term_flag')
    )

    interactions['feat_interact_punct_ratio_x_abs_vader_compound'] = (
        safe_get('feat_punct_ratio') * abs(safe_get('feat_sentiment_vader_compound'))
    )

    interactions['feat_interact_all_caps_ratio_x_exclamation_count'] = (
        safe_get('feat_all_caps_ratio') * safe_get('feat_exclamation_mark_count')
    )

    return interactions


def extract_all_optimized_features(headline: str) -> Optional[Dict[str, Union[int, float]]]:
    if not headline or not isinstance(headline, str) or not headline.strip():
        return None
    if nlp is None or vader_analyzer is None:
        raise RuntimeError("NLP resources (spaCy/VADER) not initialized. Call setup_nlp_resources() first.")

    try:
        doc = nlp(headline)
        all_features: Dict[str, Union[int, float]] = {}

        all_features.update(get_lexical_basic_features(doc))
        all_features.update(get_pos_features(doc))
        all_features.update(get_sentiment_emotion_features(headline, doc))
        all_features.update(get_exaggeration_features(doc))
        all_features.update(get_mundanity_ner_features(headline, doc))
        all_features.update(get_structure_incongruity_features(doc))
        all_features.update(get_doc_vector_features(doc))
        all_features.update(get_interaction_features(all_features))

        for key, value in all_features.items():
             if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                 all_features[key] = 0.0

        return all_features

    except Exception as e:
        return None


def load_raw_data(raw_dataset_path: str) -> pd.DataFrame:
    if not os.path.exists(raw_dataset_path):
         raise FileNotFoundError(f"Raw dataset file not found at {raw_dataset_path}")

    try:
        if raw_dataset_path.lower().endswith('.csv'):
            df = pd.read_csv(raw_dataset_path)
        elif raw_dataset_path.lower().endswith(('.json', '.jsonl')):
            try:
                df = pd.read_json(raw_dataset_path, lines=True)
            except ValueError:
                df = pd.read_json(raw_dataset_path)
        else:
            raise ValueError("Unsupported file format for raw data. Please use .csv or .json/.jsonl")

        required_cols = ['is_sarcastic', 'headline']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Raw dataset must contain columns: {required_cols}. Found: {df.columns.tolist()}")

        return df

    except Exception as e:
        raise


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    initial_rows = len(df)

    df.dropna(subset=['headline', 'is_sarcastic'], inplace=True)
    df['headline'] = df['headline'].astype(str).str.strip()

    try:
        df['is_sarcastic'] = pd.to_numeric(df['is_sarcastic'], errors='coerce')
        df.dropna(subset=['is_sarcastic'], inplace=True)
        df['is_sarcastic'] = df['is_sarcastic'].astype(int)
        if not df['is_sarcastic'].isin([0, 1]).all():
             warnings.warn(f"Target variable 'is_sarcastic' contains values other than 0 or 1 after cleaning.", UserWarning)
    except Exception as e:
        raise ValueError(f"Column 'is_sarcastic' could not be reliably converted to integer 0 or 1. Error: {e}")

    df = df[df['headline'].str.len() > 0]

    rows_after_cleaning = len(df)

    if rows_after_cleaning == 0:
        raise ValueError("No valid headlines/targets remaining after cleaning process.")

    df = df.reset_index(drop=True)
    return df


def extract_features_and_impute(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    start_time = time.time()
    setup_nlp_resources()

    if TQDM_AVAILABLE:
        feature_dicts_series = df['headline'].progress_apply(extract_all_optimized_features)
    else:
        feature_dicts_series = df['headline'].apply(extract_all_optimized_features)

    valid_indices = feature_dicts_series.dropna().index
    feature_list = feature_dicts_series.loc[valid_indices].tolist()

    df_filtered = df.loc[valid_indices].reset_index(drop=True)

    if not feature_list:
        raise ValueError("Feature extraction returned no valid results. Check feature functions or input data.")

    features_df = pd.DataFrame(feature_list)
    features_df = features_df.reindex(sorted(features_df.columns), axis=1)

    y = df_filtered['is_sarcastic']

    if len(features_df) != len(y):
         raise ValueError(f"Feature rows ({len(features_df)}) and target rows ({len(y)}) count mismatch after filtering failed extractions.")

    end_time = time.time()

    nan_count_before = features_df.isnull().sum().sum()
    inf_count_before = np.isinf(features_df.replace([np.inf, -np.inf], np.nan)).sum().sum()

    features_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(features_df)

    X = pd.DataFrame(X_imputed, columns=features_df.columns)

    nan_count_after = X.isnull().sum().sum()
    inf_count_after = np.isinf(X.replace([np.inf, -np.inf], np.nan)).sum().sum()
    if nan_count_after > 0 or inf_count_after > 0:
        pass

    return X, y


def save_features_target(X: pd.DataFrame, y: pd.Series, x_path: str, y_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(x_path), exist_ok=True)
        os.makedirs(os.path.dirname(y_path), exist_ok=True)
        X.to_csv(x_path, index=False)
        y.to_frame(name='is_sarcastic').to_csv(y_path, index=False)
    except IOError as e:
        raise


def load_features_target(x_path: str, y_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"Required feature/target CSV files not found at {x_path} or {y_path}")

    try:
        X = pd.read_csv(x_path)
        y_df = pd.read_csv(y_path)

        if 'is_sarcastic' in y_df.columns:
             y = y_df['is_sarcastic']
        elif len(y_df.columns) == 1:
            y = y_df[y_df.columns[0]]
        else:
            raise ValueError(f"Loaded target CSV '{y_path}' must have the 'is_sarcastic' column or exactly one column. Found columns: {y_df.columns.tolist()}")

        if len(X) != len(y):
            raise ValueError(f"Row count mismatch between loaded features ({len(X)}) and target ({len(y)}). Files may be corrupted or from different runs.")

        return X, y
    except Exception as e:
        raise


def get_data(raw_dataset_path: str, force_extract: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    if not force_extract and os.path.exists(FEATURES_X_CSV) and os.path.exists(TARGET_Y_CSV):
        try:
            X, y = load_features_target(FEATURES_X_CSV, TARGET_Y_CSV)
            return X, y
        except (FileNotFoundError, ValueError, Exception) as e:
            force_extract = True

    try:
        df_raw = load_raw_data(raw_dataset_path)
        df_cleaned = clean_data(df_raw)
        X, y = extract_features_and_impute(df_cleaned)

        try:
            save_features_target(X, y, FEATURES_X_CSV, TARGET_Y_CSV)
        except IOError as e:
            pass

        return X, y

    except (FileNotFoundError, ValueError, RuntimeError, Exception) as e:
        raise


def apply_pca(X_train_scaled: np.ndarray, X_test_scaled: np.ndarray, n_components: Union[int, float, None]) -> Tuple[np.ndarray, np.ndarray, PCA]:
    start_time = time.time()

    pca_model = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca_model.fit_transform(X_train_scaled)
    X_test_pca = pca_model.transform(X_test_scaled)

    end_time = time.time()

    return X_train_pca, X_test_pca, pca_model