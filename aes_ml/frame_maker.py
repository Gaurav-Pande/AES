from requirements_base import *
from requirements_feature import *
from requirements_key import *
from prompts_reader import set_prompts

prompt_vectors = []

def makeDataFrame(data):
    global prompt_vectors
    all_features = {}
    all_scores = {}
    essay_words = []
    log_frequency = 3000

    print("Generating Style Features")
    for row in range(len(data)):
        if (row+1) % log_frequency == 0 and row != 0: print("Processed ", (row+1), " essays of", len(data))
        essay_data = data.iloc[row]
        essay = essay_data[essay_key]
        essay_id = essay_data.name
        set_no = essay_data[essay_set_key]
        essay_score = essay_data[domain1_score_key]
        words, _, features, extra_features = essay_to_sentences(essay, set_no=set_no)
        essay_words.append(words)
        combined_features = {}
        combined_features.update(features)
        combined_features.update(extra_features)
        combined_features[sentences_key] = words
        all_features[essay_id] = combined_features
        all_scores[essay_id] = essay_score
        
    for set_idx, set_prompt in enumerate(set_prompts):
        set_no = set_idx + 3
        vectors = get_prompt_vectors(set_prompt, set_no)
        prompt_vectors.append(vectors)
    
    print("Generating Similarity Measures")
    for row in range(len(data)):
        if (row+1) % log_frequency == 0 and row != 0: print("Processed ", (row+1), " essays of", len(data))
        essay_data = data.iloc[row]
        essay_id = essay_data.name
        set_no = essay_data[essay_set_key]
        set_idx = set_no - 3

        if (0 > set_idx or 3 < set_idx):
            continue
        else: 
            essay_vector = get_word_count_vector(essay_words[row], set_no)
            all_features[essay_id][word_count_vector_key] = essay_vector
            prompt_vecs = prompt_vectors[set_idx]
            prompt_vecs = np.stack(prompt_vecs, axis=0)
            all_vecs = np.vstack([prompt_vecs, essay_vector])
            
            tfd = TruncatedSVD(6, random_state=1)
            all_vecs = tfd.fit_transform(all_vecs)
            
            for i in range(len(prompt_vecs)):
                sim = 1 - cosine(all_vecs[i], all_vecs[all_vecs.shape[0]-1])
                all_features[essay_id][similarity_labels[i]] = sim    
        
    X = pd.DataFrame.from_dict(all_features, orient="index")
    y = pd.DataFrame.from_dict(all_scores, orient="index")

    return(X, y)

def split_in_sets(data, base_min_scores, base_max_scores):
    essay_sets = []
    min_scores = []
    max_scores = []
    grade = [8, 10, 10, 10, 8, 10, 7, 10]
    prompt_types = ["PNR", "SDR"]
    set_type_id = [0, 0, 1, 1, 1, 1, 0, 0]
    print("Set", "Count", "Attributes", "Score Range", "Grade", "Type", sep="\t")
    for s in range(1,9):
        essay_set = data[data["essay_set"] == s]
        essay_set.dropna(axis=1, inplace=True)
        n, d = essay_set.shape
        set_scores = essay_set["domain1_score"]
        mis, mas = base_min_scores[s-1], base_max_scores[s-1]
        print(s, n, d, end="", sep="\t")
        print("\t\t[", mis, ",", mas, "]", sep='', end="")
        print("\t\t", grade[s-1], "th\t", prompt_types[set_type_id[s-1]], sep="")
        min_scores.append(set_scores.min())
        max_scores.append(set_scores.max())
        essay_sets.append(essay_set)
    print("\nPNR: Persuasive / Narrative  / Expository \t SDR: Source Dependent Responses")
    return (essay_sets, min_scores, max_scores)
