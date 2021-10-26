import nltk
from nltk.tag import hmm
from nltk.stem import PorterStemmer
from nltk.corpus.reader import TaggedCorpusReader 
from nltk.corpus.reader import PlaintextCorpusReader
from sklearn import metrics

# tagged_corpus = TaggedCorpusReader(".","guj_entertainment.txt")
#
# tagged_sentences = tagged_corpus.tagged_sents()
# tagged_words = tagged_corpus.tagged_words()

tagged_corpus1 = TaggedCorpusReader(".","Data/guj_entertainment.txt")
tagged_sentences1 = tagged_corpus1.tagged_sents()
tagged_word1 = tagged_corpus1.tagged_words()

tagged_corpus2 = TaggedCorpusReader(".","Data/guj_entertainment_set1.txt")
tagged_sentences2 = tagged_corpus2.tagged_sents()
tagged_word2 = tagged_corpus2.tagged_words()

tagged_corpus3 = TaggedCorpusReader(".","Data/guj_entertainment_set5.txt")
tagged_sentences3 = tagged_corpus3.tagged_sents()
tagged_word3 = tagged_corpus3.tagged_words()

tagged_corpus4 = TaggedCorpusReader(".","Data/guj_entertainment_set6.txt")
tagged_sentences4 = tagged_corpus4.tagged_sents()
tagged_word4 = tagged_corpus4.tagged_words()

tagged_corpus5 = TaggedCorpusReader(".","Data/guj_entertainment_set7.txt")
tagged_sentences5 = tagged_corpus5.tagged_sents()
tagged_word5 = tagged_corpus5.tagged_words()

tagged_corpus6 = TaggedCorpusReader(".","Data/guj_entertainment_set8.txt")
tagged_sentences6 = tagged_corpus6.tagged_sents()
tagged_word6 = tagged_corpus6.tagged_words()

tagged_corpus7 = TaggedCorpusReader(".","Data/guj_entertainment_set9.txt")
tagged_sentences7 = tagged_corpus7.tagged_sents()
tagged_word7 = tagged_corpus7.tagged_words()

tagged_corpus8 = TaggedCorpusReader(".","Data/guj_entertainment_set10.txt")
tagged_sentences8 = tagged_corpus8.tagged_sents()
tagged_word8 = tagged_corpus8.tagged_words()

tagged_corpus9 = TaggedCorpusReader(".","Data/guj_entertainment_set11.txt")
tagged_sentences9 = tagged_corpus9.tagged_sents()
tagged_word9 = tagged_corpus9.tagged_words()

tagged_corpus10 = TaggedCorpusReader(".","Data/guj_entertainment_set12.txt")
tagged_sentences10 = tagged_corpus10.tagged_sents()
tagged_word10 = tagged_corpus10.tagged_words()

tagged_corpus12 = TaggedCorpusReader(".","Data/guj_entertainment_set13.txt")
tagged_sentences11 = tagged_corpus12.tagged_sents()
tagged_word11 = tagged_corpus12.tagged_words()

tagged_corpus13 = TaggedCorpusReader(".","Data/guj_entertainment_set14.txt")
tagged_sentences12 = tagged_corpus13.tagged_sents()
tagged_word12 = tagged_corpus13.tagged_words()

tagged_corpus14 = TaggedCorpusReader(".","Data/guj_entertainment_set15.txt")
tagged_sentences13 = tagged_corpus14.tagged_sents()
tagged_word13 = tagged_corpus14.tagged_words()

tagged_corpus15 = TaggedCorpusReader(".","Data/guj_entertainment_set16.txt")
tagged_sentences14 = tagged_corpus15.tagged_sents()
tagged_word14 = tagged_corpus15.tagged_words()

tagged_corpus16 = TaggedCorpusReader(".","Data/guj_entertainment_set17.txt")
tagged_sentences15 = tagged_corpus16.tagged_sents()
tagged_word15 = tagged_corpus16.tagged_words()

tagged_corpus17 = TaggedCorpusReader(".","Data/guj_entertainment_set18.txt")
tagged_sentences16 = tagged_corpus17.tagged_sents()
tagged_word16 = tagged_corpus17.tagged_words()

tagged_corpus18 = TaggedCorpusReader(".","Data/guj_entertainment_set19.txt")
tagged_sentences17 = tagged_corpus18.tagged_sents()
tagged_word17 = tagged_corpus18.tagged_words()

tagged_corpus19 = TaggedCorpusReader(".","Data/guj_entertainment_set20.txt")
tagged_sentences18 = tagged_corpus19.tagged_sents()
tagged_word18 = tagged_corpus19.tagged_words()

tagged_sentences = tagged_sentences1 + tagged_sentences2 + tagged_sentences3 + tagged_sentences4 + tagged_sentences5 + tagged_sentences6 + tagged_sentences7 + tagged_sentences8 + tagged_sentences9 + tagged_sentences10 + tagged_sentences11 + tagged_sentences12 + tagged_sentences13 + tagged_sentences14 + tagged_sentences15 + tagged_sentences16 + tagged_sentences17 + tagged_sentences18
tagged_words = tagged_word1 + tagged_word2 + tagged_word3 + tagged_word4 + tagged_word5 + tagged_word6 + tagged_word7 + tagged_word8 + tagged_word9 + tagged_word10 + tagged_word11 + tagged_word12 + tagged_word13 + tagged_word14 + tagged_word15 + tagged_word16 + tagged_word17 + tagged_word18

# tagged_sentences = tagged_corpus.tagged_sents()
# tagged_words = tagged_corpus.tagged_words()

# print (tagged_sentences[4])
# print (tagged_words[0])

print("No. of tagged sentences for training: " , len(tagged_sentences))
print("No. of tagged words for training: " , len(tagged_words))

tag_fd = nltk.FreqDist(tag for (word, tag) in tagged_words)
tag_fd.most_common(4)

split = int(len(tagged_sentences) * 0.6)

train_sents = tagged_sentences[:split]
dev_sents = tagged_sentences[split:]


#Training and testing unigram tagger
unigram_tagger = nltk.UnigramTagger(train_sents)
print("Unigram accuracy:", unigram_tagger.evaluate(dev_sents)*100)


#Using backoff tagger for bi-tri tagger
bigram_tagger = nltk.BigramTagger(train_sents, backoff=unigram_tagger)
print("Bigram accuracy:",bigram_tagger.evaluate(dev_sents)*100)
trigram_tagger = nltk.TrigramTagger(train_sents, backoff=bigram_tagger)
print("Trigram accuracy:",trigram_tagger.evaluate(dev_sents)*100)


#Training and tagging HMM using nltk trainer
hmm_trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = hmm_trainer.train_supervised(train_sents)
print("HMM train accuracy:", hmm_tagger.evaluate(train_sents)*100)
print("HMM dev accuracy:", hmm_tagger.evaluate(dev_sents)*100)

#Using tags for given corpus to evaluate different metrics for different tagging model
uni_dev_tagged_sents = unigram_tagger.tag_sents([[word for word,tag in sentence] for sentence in dev_sents])
standard = [str(tag) for sentence in dev_sents for token,tag in sentence]
uni_predicted = [str(tag) for sentence in uni_dev_tagged_sents for token,tag in sentence]
print(" Unigram Accuracy :", metrics.accuracy_score(standard,uni_predicted))
print(" Unigram Precision:", metrics.precision_score(standard,uni_predicted,average='weighted'))
print(" Unigram Recall   :", metrics.recall_score(standard,uni_predicted,average='weighted'))
print(" Unigram F1-Score :", metrics.f1_score(standard,uni_predicted,average='weighted'))

tri_dev_tagged_sents = trigram_tagger.tag_sents([[word for word,tag in sentence] for sentence in dev_sents])
tri_predicted = [str(tag) for sentence in tri_dev_tagged_sents for token,tag in sentence]
print(" Trigram Accuracy :", metrics.accuracy_score(standard,tri_predicted))
print(" Trigram Precision:", metrics.precision_score(standard,tri_predicted,average='weighted'))
print(" Trigram Recall   :", metrics.recall_score(standard,tri_predicted,average='weighted'))
print(" Trigram F1-Score :", metrics.f1_score(standard,tri_predicted,average='weighted'))

hmm_dev_tagged_sents = hmm_tagger.tag_sents([[word for word,tag in sentence] for sentence in dev_sents])
hmm_predicted = [str(tag) for sentence in hmm_dev_tagged_sents for token,tag in sentence]
print(" Hmm Accuracy :", metrics.accuracy_score(standard,hmm_predicted))
print(" Hmm Precision:", metrics.precision_score(standard,hmm_predicted,average='weighted'))
print(" Hmm Recall   :", metrics.recall_score(standard,hmm_predicted,average='weighted'))
print(" Hmm F1-Score :", metrics.f1_score(standard,hmm_predicted,average='weighted'))



