import nltk
from nltk.tag import hmm
from nltk.stem import PorterStemmer
from nltk.corpus.reader import TaggedCorpusReader
from nltk.corpus.reader import PlaintextCorpusReader
from sklearn import metrics

tagged_corpus = TaggedCorpusReader(".","Data/guj_economy_set1.txt")
#
tagged_sentences = tagged_corpus.tagged_sents()
tagged_words = tagged_corpus.tagged_words()

# tagged_sentences = tagged_corpus.tagged_sents()
# tagged_words = tagged_corpus.tagged_words()

print (tagged_sentences[4])
print (tagged_words[0])

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



