from requirements_base import *
from requirements_key import *

tagger = PerceptronTagger()
spell = SpellChecker()
spell.word_frequency.load_words(["PERSON", "ORGANIZATION", "LOCATION", "DATE", "TIME", "MONEY", "PERCENT", "CAPS"])
stemmer = PorterStemmer()
max_idx = [0, 0, 0, 0]
word_to_idx = [{}, {}, {}, {}]
idx_to_word = [{}, {}, {}, {}]

# Process words of essays
def sentence_to_word_list(sentence, ignore_stopwords, set_no):
    global max_idx
    global word_to_idx
    global idx_to_word
            
    sen_char_count = 0
    sen_word_count = 0
    l5_sen_word_count = 0
    l6_sen_word_count = 0
    l7_sen_word_count = 0
    l8_sen_word_count = 0    
    sen_diff_words = set()
    ### Extra Features ###
    sen_verbs_count = 0
    sen_adverbs_count = 0
    sen_nouns_count = 0
    sen_adjectives_count = 0
    sen_spelling_error_count = 0
    sen_stopwords_count = 0
    is_small_sentence = 0

    stops = set(stopwords.words("english"))
    all_words = sentence.lower().split()


    kept_words = []

    if len(all_words) <= 4: is_small_sentence = 1

    misspelled = spell.unknown(all_words)
    sen_spelling_error_count = len(misspelled)
    
    for word in all_words:
        
        sen_char_count += len(word)
        sen_word_count += 1
        word_len = len(word)
        if word_len > 5:
            l5_sen_word_count += 1
        if word_len > 6:
            l6_sen_word_count += 1
        if word_len > 7:
            l7_sen_word_count += 1
        if word_len > 8:
            l8_sen_word_count += 1

        sen_diff_words.add(word)
        kept_words.append(word)
        
        isStopword = word in stops
        process_word = (ignore_stopwords and not isStopword) or (not ignore_stopwords) 
        if process_word:
            set_idx = set_no - 3
            if set_idx >= 0 and set_idx <= 3:
                stem_word = stemmer.stem(word)
                if not stem_word in word_to_idx[set_idx]:    
                    word_to_idx[set_idx][stem_word] = max_idx[set_idx]
                    idx_to_word[set_idx][max_idx[set_idx]] = stem_word
                    max_idx[set_idx] += 1

        if isStopword: sen_stopwords_count += 1

    features = {
         feature_keys["char_count_key"]: sen_char_count,
         feature_keys["word_count_key"]: sen_word_count,
         feature_keys["l5_word_count_key"]: l5_sen_word_count,
         feature_keys["l6_word_count_key"]: l6_sen_word_count,
         feature_keys["l7_word_count_key"]: l7_sen_word_count,
         feature_keys["l8_word_count_key"]: l8_sen_word_count,
         feature_keys["diff_words_count_key"]: sen_diff_words
    }

    extra_features = {
        extra_feature_keys["small_sentences_count_key"]: is_small_sentence,
        extra_feature_keys["spelling_error_count_key"]: sen_spelling_error_count,
        extra_feature_keys["stopwords_count_key"]: sen_stopwords_count,
        extra_feature_keys["verbs_count_key"]: sen_verbs_count,
        extra_feature_keys["adverbs_count_key"]: sen_adverbs_count,
        extra_feature_keys["nouns_count_key"]: sen_nouns_count,
        extra_feature_keys["adjectives_count_key"]: sen_adjectives_count,
    }

    return (kept_words, features, extra_features)


# Process sentences of essays
def essay_to_sentences(essay, set_no, ignore_stopwords=False):
    sentences = sent_tokenize(essay.strip())
    split_sentences = []
    split_words = []
    
    char_count = 0
    word_count = 0
    diff_words = set()
    word_count_root = 0
    sen_count = 0
    avg_word_len = 0
    avg_sen_len = 0
    l5_word_count = 0
    l6_word_count = 0
    l7_word_count = 0
    l8_word_count = 0    
    ### Extra Features ###
    spelling_error_count = 0
    stopwords_count = 0
    small_sentences_count = 0
    punctuation_count = 0
    grammer_error_count = 0
    small_sentences_count = 0
    verbs_count = 0
    adverbs_count = 0
    nouns_count = 0
    adjectives_count = 0
 
    all_words = word_tokenize(essay)
    pos_count = Counter([tag for word, tag in tagger.tag(all_words)])
    # print("Tag count: ", count)

    nouns_count = pos_count.get("NN", 0) + pos_count.get("NNS", 0) + pos_count.get("NNP", 0) + pos_count.get("NNPS", 0)
    adjectives_count = pos_count.get("JJ", 0) + pos_count.get("JJR", 0) #+ pos_count.get("JJS", 0)
    verbs_count = pos_count.get("VB", 0) + pos_count.get("VBG", 0) + pos_count.get("VBN", 0) + pos_count.get("VBP", 0) + pos_count.get("VBZ", 0)# + pos_count.get('VBD', 0)
    adverbs_count = pos_count.get("RB", 0) + pos_count.get("RBR", 0) + pos_count.get("RBS", 0)

    punctuations = ['?', '.', ':', ';', '!']
    for punctuation in punctuations:
        punctuation_count += essay.count(punctuation)
    
    for sentence in sentences:
        if len(sentence) > 0:
            sentence = re.sub("[^a-zA-Z]", " ", sentence)
            kept_words, features, extra_features = sentence_to_word_list(sentence, ignore_stopwords, set_no)
            split_sentences.append(kept_words)
            split_words.extend(kept_words)
            
            sen_count +=1
            char_count += features[feature_keys["char_count_key"]]
            word_count += features[feature_keys["word_count_key"]]
            l5_word_count += features[feature_keys["l5_word_count_key"]]
            l6_word_count += features[feature_keys["l6_word_count_key"]]
            l7_word_count += features[feature_keys["l7_word_count_key"]]
            l8_word_count += features[feature_keys["l8_word_count_key"]]
            diff_words = diff_words|features[feature_keys["diff_words_count_key"]]
            ### Extra Features ###
            spelling_error_count += extra_features[extra_feature_keys["spelling_error_count_key"]]
            stopwords_count += extra_features[extra_feature_keys["stopwords_count_key"]]
            small_sentences_count += extra_features[extra_feature_keys["small_sentences_count_key"]]
          
    word_count_root = word_count ** (1/4)
    avg_word_len = char_count / word_count
    avg_sen_len = word_count / sen_count
    
    features = {
        feature_keys["char_count_key"]: char_count,
        feature_keys["word_count_key"]: word_count,
        feature_keys["diff_words_count_key"]: len(diff_words),
        feature_keys["word_count_root_key"]: word_count_root,
        feature_keys["sen_count_key"]: sen_count,
        feature_keys["avg_word_len_key"]: avg_word_len,
        feature_keys["avg_sen_len_key"]: avg_sen_len,
        feature_keys["l5_word_count_key"]: l5_word_count,
        feature_keys["l6_word_count_key"]: l6_word_count,
        feature_keys["l7_word_count_key"]: l7_word_count,
        feature_keys["l8_word_count_key"]: l8_word_count
    }

    extra_features = {
        extra_feature_keys["spelling_error_count_key"]: spelling_error_count,
        extra_feature_keys["stopwords_count_key"]: stopwords_count,
        extra_feature_keys["small_sentences_count_key"]: small_sentences_count,
        extra_feature_keys["punctuations_count_key"]: punctuation_count,
        extra_feature_keys["verbs_count_key"]: verbs_count,
        extra_feature_keys["adverbs_count_key"]: adverbs_count,
        extra_feature_keys["nouns_count_key"]: nouns_count,
        extra_feature_keys["adjectives_count_key"]: adjectives_count 
    }

    return (split_words, split_sentences, features, extra_features)

# Generate the average word vector representation of a word list
def get_avg_word_vec(words, model, wv_size):
    wordvec = np.zeros((wv_size,),dtype="float32")
    word_count = 0.
    index2word_set = set(model.wv.index2word)
    for word in words:
        if word in index2word_set: 
            word_count += 1.
            wordvec = np.add(wordvec, model[word])        
    wordvec = np.divide(wordvec, word_count)
    return wordvec

# Generate the word vector representation for paragraphs from essay prompt
def get_prompt_word_vecs(prompt, model, wv_size):
    whole_prompt_words = []
    vectors = []
    for para in prompt:
        sentences = sent_tokenize(para.strip())
        para_words = []
        for sentence in sentences:
            if len(sentence) > 0:
                sentence = re.sub("[^a-zA-Z]", " ", sentence)
                words = sentence.lower().split()
                para_words.extend(words)
                whole_prompt_words.extend(words)
        vectors.append(get_avg_word_vec(para_words, model, wv_size))
    vectors.append(get_avg_word_vec(whole_prompt_words, model, wv_size))
    return vectors

# Generate the word count vector for a word list
def get_word_count_vector(words, set_no):
    set_idx = set_no - 3
    if set_idx >= 0 and set_idx <= 3: 
        word_count_vector = np.zeros((max_idx[set_idx]+1,))
        for word in words:
            word = stemmer.stem(word)
            if word in word_to_idx[set_idx]:
                word_count_vector[word_to_idx[set_idx][word]] += 1
        return word_count_vector
    
# Generate the word count vector for paragraphs of essay prompts
def get_prompt_vectors(prompt, set_no):
    whole_prompt_words = []
    vectors = []
    for para in prompt:
        sentences = sent_tokenize(para.strip())
        para_words = []
        for sentence in sentences:
            if len(sentence) > 0:
                sentence = re.sub("[^a-zA-Z]", " ", sentence)
                words = sentence.lower().split()
                para_words.extend(words)
                whole_prompt_words.extend(words)
        vectors.append(get_word_count_vector(para_words, set_no))
    vectors.append(get_word_count_vector(whole_prompt_words, set_no))
    return vectors