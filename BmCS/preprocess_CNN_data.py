from tensorflow.keras.preprocessing.sequence import pad_sequences
import math
import numpy as np
from nltk.tokenize import word_tokenize


TITLE_MAX_WORDS = 64
ABSTRACT_MAX_WORDS = 448
MIN_PUB_YEAR = 1809 
MIN_YEAR_COMPLETED = 1965
MODEL_MAX_YEAR = 2021
TIME_PERIOD_SIZE = 5
UNKNOWN_JOURNAL_INDEX = 0
UNKNOWN_WORD_INDEX = 1
PADDING_INDEX = 0


def get_batch_data(citations, journal_ids_path, word_indices_path):

    journal_index_lookup = _create_lookup(journal_ids_path)
    word_indices_lookup = _create_lookup(word_indices_path)

    pmids, titles, abstracts, pub_years, year_completed, journal_indices = _extract_data(citations, journal_index_lookup)

    title_input = _vectorize_batch_text(word_indices_lookup, titles, TITLE_MAX_WORDS)
    abstract_input = _vectorize_batch_text(word_indices_lookup, abstracts, ABSTRACT_MAX_WORDS)

    num_pub_year_time_periods = _num_time_periods(MIN_PUB_YEAR)
    num_year_completed_time_periods = _num_time_periods(MIN_YEAR_COMPLETED)
    pub_year_input =       _create_year_input(pub_years     , MIN_PUB_YEAR      , num_pub_year_time_periods      )
    year_completed_input = _create_year_input(year_completed, MIN_YEAR_COMPLETED, num_year_completed_time_periods)

    journal_input = np.array(journal_indices, dtype=np.int32).reshape(-1, 1)

    batch_x = { 'title_input': title_input, 'abstract_input': abstract_input, 'pub_year_input': pub_year_input, 'year_completed_input': year_completed_input, 'journal_input': journal_input}
    return batch_x


def _create_lookup(path):
    lookup = {}
    with open(path, 'rt', encoding='utf8') as file:
        for line in file:
            id, value = line.split('\t')
            lookup[value.strip()] = int(id)
    return lookup


def _create_year_input(year_data, min_year, num_time_periods):
    year_data = np.array(year_data, dtype=np.int32).reshape(-1, 1)
    year_data = np.clip(year_data, a_min=min_year, a_max=MODEL_MAX_YEAR)
    year_indices = MODEL_MAX_YEAR - year_data
    year_indices = np.floor_divide(year_indices, TIME_PERIOD_SIZE)
    year_indices = num_time_periods - year_indices - 1
    year_input = _to_time_period_input(year_indices, num_time_periods)
    return year_input


def _extract_data(citations, journal_index_lookup):
    pmids, titles, abstracts, pub_years, year_completed, journal_indices = [], [], [], [], [], []
    for citation in citations:
        pmids.append(citation['pmid'])
        titles.append(citation['title'].lower())
        abstracts.append(citation['abstract'].lower())
        pub_year = citation['pub_year']
        if pub_year is None:
            pub_year = MODEL_MAX_YEAR
        if pub_year > MODEL_MAX_YEAR:
            pub_year = MODEL_MAX_YEAR
        if pub_year < MIN_PUB_YEAR:
            pub_year = MIN_PUB_YEAR
        pub_years.append(pub_year)
        year_completed.append(MODEL_MAX_YEAR)
        journal_index = UNKNOWN_JOURNAL_INDEX
        journal_nlmid = citation['journal_nlmid']
        if journal_nlmid in journal_index_lookup:
           journal_index = journal_index_lookup[journal_nlmid]
        journal_indices.append(journal_index)
    return pmids, titles, abstracts, pub_years, year_completed, journal_indices


def _num_time_periods(min_year):
    num_years = (MODEL_MAX_YEAR - min_year) + 1
    num_periods = num_years / TIME_PERIOD_SIZE
    num_periods = math.ceil(num_periods)
    return num_periods


def _to_time_period_input(year_indices, num_time_periods):
    batch_size = year_indices.shape[0]
    batch_indices = np.zeros([batch_size, num_time_periods], np.int32)
    batch_indices[np.arange(batch_size)] = np.arange(num_time_periods)
    year_indices_rep = np.repeat(year_indices, num_time_periods, axis=1)
    time_period_input = batch_indices <= year_indices_rep
    time_period_input = time_period_input.astype(np.int32)
    return time_period_input


def _vectorize_batch_text(word_index_lookup, batch_text, max_words):
    batch_words = [word_tokenize(text) for text in batch_text]
    batch_word_indices = [[word_index_lookup[word] if word in word_index_lookup else UNKNOWN_WORD_INDEX for word in words] for words in batch_words]
    vectorized_text = pad_sequences(batch_word_indices, maxlen=max_words, dtype='int32', padding='post', truncating='post', value=PADDING_INDEX)
    return vectorized_text
