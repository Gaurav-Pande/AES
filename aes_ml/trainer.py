from requirements_base import *
from requirements_feature import *
from requirements_key import *
from requirements_frame import *
from prompts_reader import set_prompts


def get_all_classifiers():
    linr = LinearRegression()
    svm = SVC(kernel="linear", C=0.025)
    knn = KNeighborsClassifier(10)
    return [linr, svm, knn]

def create_word_vecs(X, wv_model, wv_size, essay_wordvecs):
    essay_vectors = {}
    for idx in X.index.values:
        essay_wordvec = get_avg_word_vec(X.loc[idx][sentences_key], wv_model, wv_size)
        essay_wordvecs[idx] = essay_wordvec
        essay_vectors[idx] = {"wv_"+str(ind): vec for ind, vec in enumerate(essay_wordvec)}
    X_vec = pd.DataFrame.from_dict(essay_vectors, orient="index")
    return X_vec, essay_wordvecs 

def create_sim_from_word_vecs(X, data, wv_model, wv_size, essay_wordvecs):
    essay_vec_sim = {}
    for idx in X.index.values:
        essay_words = X.loc[idx][sentences_key]
        set_no = data.loc[idx][essay_set_key]
        set_idx = set_no - 3
        prompt = set_prompts[set_idx]
        prompt_word_vecs = get_prompt_word_vecs(prompt, wv_model, wv_size) 
        essay_wordvec = []
        if idx in essay_wordvecs:
            essay_wordvec = essay_wordvecs[idx]
        else:
            essay_wordvec = get_avg_word_vec(essay_words, wv_model, wv_size)

        essay_vec_sim[idx] = { worvec_similarity_labels[ind] : (1 - cosine(vec, essay_wordvec)) for ind, vec in enumerate(prompt_word_vecs)}
        
    X_vec_sim = pd.DataFrame.from_dict(essay_vec_sim, orient="index")
    return X_vec_sim

def evaluate(X, y, data=None, model = LinearRegression(), plot=False, wordvec=False, wv_size=300, min_count=30, context=10, sample=0, lsa=False, wordvec_sim=False):
  
    X = X.dropna(axis=1, inplace=False)

    X, X_unseen, y, y_unseen = train_test_split(X, y, test_size=0.1, random_state=1)
    kf = KFold(n_splits=5, shuffle=True)
    cv = kf.split(X)
    results = []
 
    start = time()
    for traincv, testcv in cv:
            X_test, X_train, y_test, y_train = X.iloc[testcv], X.iloc[traincv], y.iloc[testcv], y.iloc[traincv]
            
            if wordvec or wordvec_sim:
                num_workers = 4
                sentences = X_train[sentences_key]
                wv_model = word2vec.Word2Vec(sentences, workers=num_workers, size=wv_size, min_count=min_count, window=context, sample=sample)
                wv_model.init_sims(replace=True)
                
                essay_wordvecs = {}
                
                if wordvec:

                    X_train_vec, essay_wordvecs = create_word_vecs(X_train, wv_model, wv_size, essay_wordvecs)
                    X_test_vec, essay_wordvecs = create_word_vecs(X_test, wv_model, wv_size, essay_wordvecs)     
                    X_train = X_train.join(X_train_vec)
                    X_test = X_test.join(X_test_vec)
                
                if wordvec_sim:
                   
                    X_train_vec_sim = create_sim_from_word_vecs(X_train, data, wv_model, wv_size, essay_wordvecs)                    
                    X_test_vec_sim = create_sim_from_word_vecs(X_test, data, wv_model, wv_size, essay_wordvecs)         
                    X_train = X_train.join(X_train_vec_sim)
                    X_test = X_test.join(X_test_vec_sim)
                    
                    
            if word_count_vector_key in X_train.columns:
                X_train.drop([word_count_vector_key], axis=1, inplace=True)
                X_test.drop([word_count_vector_key], axis=1, inplace=True)
            
            if not lsa:
                if set(similarity_labels).issubset(X_train.columns):
                    X_train.drop(similarity_labels, axis=1, inplace=True)
                    X_test.drop(similarity_labels, axis=1, inplace=True)

            X_train.drop([sentences_key], axis=1, inplace=True)
            X_test.drop([sentences_key], axis=1, inplace=True)
                       
            model.fit(X_train, y_train.values.ravel())
            
            y_pred = model.predict(X_test)
            y_pred = y_pred.reshape(-1)
            y_pred = np.around(y_pred, decimals=0).astype(int)
            y_test = [item for sublist in y_test.values for item in sublist]
            
            result = kappa(y_test, y_pred, labels=None, weights='quadratic')
            results.append(result)
            
    end = time()
    
    print("[", round((end-start)/60, 3), " mins", end=" ] ")
    
    if plot:
        y_unseen_pred = model.predict(X_unseen)
        y_unseen_pred = y_unseen_pred.reshape(-1)
        y_unseen_pred = np.around(y_unseen_pred, decimals=0).astype(int)
        
        labels = set(y_unseen_pred)
        ncol = int(len(labels)/2.5)
        plt.figure(figsize=(8, 8))
        scatter = plt.scatter(X_unseen.index, y_unseen, c=y_unseen_pred)
        handles = scatter.legend_elements()[0]
        plt.legend(handles=handles, labels=labels, ncol=ncol, loc="lower right")
        plt.ylabel("Score")
        plt.xlabel("Essay ID")
        model_name = type(model).__name__
        plt.title("Model: " + model_name)

        dateTimeObj = datetime.now()
        timestampStr = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S.%f)")
        plt.savefig("../figs/" + model_name + "-" +timestampStr + ".png")
                
    return round(np.array(results).mean(), 4), model