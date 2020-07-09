#Dataset keys
essay_id_key = "essay_id"
essay_set_key = "essay_set"
essay_key = "essay"
domain1_score_key = "domain1_score"

#Feature keys
feature_keys = {
    "char_count_key": "char_count",
    "word_count_key": "word_count",
    "diff_words_count_key": "diff_words_count",
    "word_count_root_key": "word_count_root",
    "sen_count_key": "sen_count",
    "avg_word_len_key": "avg_word_len",
    "avg_sen_len_key": "avg_sen_len",
    "l5_word_count_key": "l5_word_count",
    "l6_word_count_key": "l6_word_count",
    "l7_word_count_key": "l7_word_count",
    "l8_word_count_key": "l8_word_count",
}

#Extra features
extra_feature_keys = {
    "spelling_error_count_key": "spelling_error_count",
    "stopwords_count_key": "stopwords_count",
    "small_sentences_count_key": "small_sentence_count", #sentences less than len 4
    "punctuations_count_key": "punctuations_count",
    "verbs_count_key": "verbs_count",
    "adverbs_count_key": "adverbs_count",
    "nouns_count_key": "nouns_count",
    "adjectives_count_key": "adjective_count",
}

sentences_key = "sentences"
word_count_vector_key = "word_count_vector"
feature_keys["sentenc_key"] = sentences_key
feature_keys["word_count_vector_key"] = word_count_vector_key

similarity_labels = ["para_1_sim", "para_2_sim", "para_3_sim", "para_4_sim", "whole_prompt_sim"]
worvec_similarity_labels = ["wv_para_1_sim", "wv_para_2_sim", "wv_para_3_sim", "wv_para_4_sim", "wv_whole_prompt_sim"]

feature_keys_list = list(feature_keys.values())
extra_feature_keys_list = list(extra_feature_keys.values())
all_feature_keys_list = feature_keys_list + extra_feature_keys_list



