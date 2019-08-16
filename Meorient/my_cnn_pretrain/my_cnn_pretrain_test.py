
import gensim

word_vectors=gensim.models.KeyedVectors.load_word2vec_format("../data/word2vec/GoogleNews-vectors-negative300.bin", binary=True)  # C bin format 

#
#model.most_similar(['nvidia'])
#

MBEDDING_DIM=300
vocabulary_size=min(len(word_index)+1,NUM_WORDS)
embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))
for word, i in word_index.items():
    if i>=NUM_WORDS:
        continue
    try:
        embedding_vector = word_vectors[word]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        embedding_matrix[i]=np.random.normal(0,np.sqrt(0.25),EMBEDDING_DIM)





#model = gensim.models.Word2Vec(sentences, min_count=5,size=tag_args.VOC_DIM,window=2,workers=8,iter=10)
data_model_dir=os.path.join('../data/temp' ,cnn.MODEL_ID) 
model_path=os.path.join(data_model_dir,'word2vec.model')
model.save(model_path)


word2vec.model
