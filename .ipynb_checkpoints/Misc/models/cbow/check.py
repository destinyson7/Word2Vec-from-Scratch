import json
import operator
import numpy as np
import matplotlib.pyplot as plt

file = open('epoch1.json','r')

data = json.load(file)

word2index = data['word2index']
index2word = data['index2word']
costs = data['cost']
W_output = np.array(data['W_output'])

# print(word2index)

def word_sim(word, top_n):
		
	w1_index = word2index[word]
	v_w1 = W_output[w1_index]

		# CYCLE THROUGH VOCAB
	word_sim = {}

	for i in range(len(word2index)):
		v_w2 = W_output[i]
		theta_num = np.dot(v_w1, v_w2)
		theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
		theta = theta_num / theta_den

		word = index2word[str(i)]
		word_sim[word] = theta

	words_sorted = sorted(word_sim.items(), key=operator.itemgetter(1), reverse=True)
	ret = []
	
	for word, sim in words_sorted[:top_n+1]:
		ret.append(word)
		
	# print(i)

	return ret

Words = ["quickly","throw","awesome","hard","computer"]

# for word in Words:

# 	top_words = word_sim(word, 11)
# 	keys = list(top_words)
# 	# keys.append(word)
# 	visualizeWords = keys
# 	visualizeIdx = [word2index[word] for word in visualizeWords]
# 	visualizeVecs = W_output[visualizeIdx, :]
# 	temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
# 	covariance = 1.0 / len(visualizeIdx) * temp.T.dot(temp)
# 	U,S,V = np.linalg.svd(covariance)
# 	coord = temp.dot(U[:,0:2])

# 	for i in range(len(visualizeWords)):
# 		plt.text(coord[i,0], coord[i,1], visualizeWords[i],
# 			bbox=dict(facecolor='green', alpha=0.1))

# 	plt.xlim((np.min(coord[:,0]), np.max(coord[:,0])))
# 	plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))
# 	plt.title('Top 10 word vectors for ' + word)
# 	plt.show()

# print(word_sim("quickly", 11))
# print()
# print()
# print(word_sim("throw", 11))
# print()
# print()
# print(word_sim("awesome", 11))
# print()
# print()
# print(word_sim("hard", 11))
# print()
# print()
print(word_sim("camera", 11))
print()
print()
  



