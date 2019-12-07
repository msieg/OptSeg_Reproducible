
# Overview

In this report we will demonstrate OptSeg (nominally inspired by OptSeq (Greve, 2002)), a stimuli generation and partitioning tool for psycholinguistiscs, psychology, and neuroscience experiments that **opt**imally **seg**ments words and images into lists with matching distributions of features, including lexical frequency, word length, arousal, color, luminensce, and semantic meaning. While experimentalists have traditionally accounted for differences in these sort of features in their task designs by using counterbalanced sets of unmatched stimuli, computational tools have been made available to improve these designs by optimizing the distribtuion of features across stimuli sets. 

In this report we will demonstrate these tools by running a data simulation that shows why matching stimulus features is important for empirical investigations, and then we will show how OptSeg implements "anti-clustering" via an inverse ward linkage algorithm to optimize the results of this simulation and related experiments. 

Before proceeding, we should point out that a similar project utilizing "anti-clustering to partition a stimulus pool into equivalent parts" was recently released on PsyArXiv (Papenberg & Klau, 2019). OptSeg extends this work considerably by: 

i) Providing a highly customizable python-based implementation that minimizes user input

ii) Allowing users to match the distribution of multiple features simultaneously

iii) Interfacing directly with exhaustive databases of words, images, and their relevant features (e.g. SUBTLEX, CMUDICT, ImageNet)

iv) Utilizing features derived from machine learning models (e.g. GloVe, ResNet)




# Data Simulation

### Experimental Design

The stimulus features available in OptSeg have been shown to affect dependent measures common to psycholinguistic, psychological and neuroscience experiments, including reading times, eye movements, recognition, recall and brain activity (White et al., 2018; Parker & Slattery, 2019; Sperling, 1986; Mollica et al., 2019). In this simulation we will focus on the relationship between word frequency, word length, emotional arousal, semantic meaning, and univariate fMRI activity. Numerous studies have shown that the left inferior frontal gyrus (LIFG), a brain region involved in phonological and semantic processing (Bokde et al., 2001), yields greater activity in response to low-frequent vs. high-frequent words, long vs. short words, arousing vs. neutral words, and semantically related vs. unrelated word lists (Fiebach et al. 2002; Kronbichler et al. 2004; Carreiras et al. 2006; Yarkoni et al. 2008; Elli et al., 2019; Kuchinke et al., 2005; Mollica et al., 2019). Therefore an fMRI experiment that uses word stimuli to test LIFG responses *independent of these features* should aim to match the distributions of these features across different conditions in order to avoid confounds on the dependent variable (LIFG activity). 

For example, imagine we want to test the univariate response of LIFG to individual words presented with and without a paired image (e.g. the word" "dog" vs. the word "dog" presented with a picture of a dog). Despite the role of the LIFG in semantic processing (LIFG is a region in the left cortical language system), it has been shown not to respond significantly to images (Ivanova et al., 2019). To simulate this experiment, we will define a quantitative relationship between the confounding stimulus features (word frequency, length, arousal, semantic relatedness) and LIFG activity in accordance with the relevant literature, and then we will suppose hypothetically that a word-image pair confers a 0.05 increase in the percent signal change (PSC) of LIFG BOLD activity relative to a word presented alone. By running this simulation with randomly segmented word lists vs. optimally segmented word lists, we will show that OptSeg increases the statistical power of this experiment dramatically. 




### Defining quantitative relationships between stimuli and LIFG activity

In this simulation we will estimate the percent signal change (PSC) in the LIFG in response to a given word based on that word's lexical frequency, word length (in phonemes), and arousal. Additionally, a global list-level increase in PSC will depend on the average pairwise semantic distance of all the words in each list (one with images, one without). First, a baseline response estimate (Resp) will be generated from a gaussian defined by 0 ± 0.2. Then the following functions will be applied in accordance with the relevant literature:

### Resp = Resp – Lex_Freq 
- Lex_Freq = Log10(word counts per million), based on Subtlex database 
- Lex_Freq range = ( 0.4 ≤ )

### Resp = Resp + Length
- Length = Number of phonemes, based on CMU Pronunciation Dictionary
- Length range = ( 1 ≤ )

### Resp = Resp + Arousal
- Arousal = Average human ratings, based on data collected from Warriner et al. (year)
- Arousal range = (1.6 ≤ 7.8)

### List_Resp = List_Resp – Sem_Dist 
- List_Resp = Array of estimated responses (Resp) for all words in each list
- Sem_Dist = Average pairwise cosine distance between the 300-Dimensional GloVe vectors of all words within each list





```{python, echo=False}
#In the following functions, the **response** variable is the estimated LIFG percent signal change (PSC) 
#relative to baseline. Each function returns the new response based on the other inputs.

#Verbs vs. Nouns 
#pos: "noun" or "verb"
# def synt_cat_response(response,pos):
    
#     #set verb bias to 20% higher PSC, based on supplementary Fig3 in Elli et al., (2019)
#     verb_bias=0.2
#     noun_bias=0
    
#     if pos=='Verb':
#         response = response + verb_bias  
#     elif pos=='Noun':
#         response = response + noun_bias  
        
#     return response
        
#Lexical frequency
#freq: scalar indicating the Log10 counts per million words reported by Subtlex
def freq_response(response,freq):
    
    #set freq bias to freq/10, based on ____
    freq_bias = freq#/2
    response = response - freq_bias
  
    return response

#Word length (phonemes)
#phon_length: scalar indicating the number of phonemes in the word
def length_response(response,phon_length):
    
    #set length bias, based on ___
    length_bias = phon_length#/5
    response = response + length_bias
  
    return response

def valence_response(response,arousal):
    
    #set length bias, based on ___
    arousal_bias = np.abs(arousal)#/2
    response = response + arousal_bias
  
    return response

#Overall response estimation
def estimate_response(pos, freq, phon_length, arousal):
    
    #randomly generate baseline response from a gaussian defined by 1 ± 0.2 (reasonable PSC values)

    #response = np.random.normal(1.5,0.2,1)
    response=1.5

    #run estimations
    response = freq_response(response,freq)
    response = length_response(response,phon_length)
    response = length_response(response,arousal)
    
    return response
```

### Generating stimuli lists and response baselines

Next we randomly sample 2 lists of 100 unique words (200 total words) from the Subtlex database and generate a table summarizing the features that we are aiming to control (Table 1), as well as baseline estimates of the average LIFG activity (PSC) for each list based on the functions we defined above (Figure 1). 


```{python, echo=False, message=False, warning=False, results='hide', quietly=True}
import numpy as np 
import random
import matplotlib.pyplot as plt
import scipy
import scipy.spatial
from nltk.stem import WordNetLemmatizer
lemmas = WordNetLemmatizer()
import umap
import pandas as pd
import cmudict
phonemes=cmudict.dict()
import nltk
nltk.download('cmudict')
import sklearn
import math
import colorsys
import pickle

#df = pd.read_csv('glove.6B.300d.txt', sep=" ", quoting=3, header=None, index_col=0) -->
#word_gloves = {key: val.values for key, val in df.T.items()}

with open('gloves_optseg.pkl', 'rb') as f:
    word_gloves=pickle.load(f)

#read excel file
subtlex = pd.read_excel('SUBTLEX.xlsx')

#get wordlist column
subtlex_wordlist=list(subtlex['Word'])

#get Log10 word frequency column
freqs=np.array(subtlex['Lg10WF'])

dom_POS=np.array(subtlex['Dom_PoS_SUBTLEX'])

#set word frequency dict
word_freqs=dict(zip(subtlex_wordlist,freqs))

#set word pos dict
word_POS=dict(zip(subtlex_wordlist,dom_POS))

#set word arousal dict
ratings=pd.read_csv('Ratings_Warriner_et_al.csv')
rate_words=list(ratings['Word'])
rate_arousal=list(ratings['A.Mean.Sum'])
word_arousal=dict(zip(rate_words,rate_arousal))
```


```{python, echo=False}

num_lists=2
items_per_list=100

#set lists
words=[]
vecs=[]
freqs=[]
lens=[]
POS=[] 
arousals=[]

max_freq=5.5
min_freq=1.5

max_len=15
min_len=2

pos_list=['Noun','Verb']

items_per_pos = np.floor(items_per_list/len(pos_list))

############################################################

#print('Filtering words for search criteria and generating frequency, length, and GloVe vector lists \n')

bad_words=open('OFFENSIVE_WORDS_CAUTION_BEFORE_READING.txt','r').read().split('\n')
contractions=['aren','don','doesn','didn','hasn','isn','couldn','shouldn','wouldn','mustn','wasn','weren']

for word in word_gloves:

    #filter offensive words and pieces of contractionns
    if word in bad_words+contractions:
        continue

    if word in word_freqs and word in phonemes and word in word_arousal:

        pos1=word_POS[word]
        freq1=word_freqs[word]
        len1=len(phonemes[word][0])
        arous1=word_arousal[word]

        #filter abbreviated words
        if len1 > len(word) and 'u' not in word and 'x' not in word and 'sm' not in word:
            continue

        if pos1 in pos_list and freq1>=min_freq and freq1<=max_freq and len1>=min_len and len1<=max_len:

            words.append(word)
            vecs.append(word_gloves[word])
            freqs.append(freq1)
            lens.append(len1)
            POS.append(pos1)
            arousals.append(arous1)
            

#print('Creating lists from: ' + str(len(words)) + ' words \n')
```


```{python, echo=False}
#set random seed to 42

# noun_inds=[i for i in range(len(words)) if POS[i]=='Noun']
# verb_inds=[i for i in range(len(words)) if POS[i]=='Verb']

w_inds=[i for i in range(len(words))]

# random.shuffle(noun_inds)
# random.shuffle(verb_inds)

random.shuffle(w_inds)

num_lists=2
items_per_list=100
   
# ind_sample=noun_inds[:100] + verb_inds[:100]

ind_sample=w_inds[:200]

responses=[]
words_rand=[]

words_rand1=[]
wfreqs1=[]
wlens1=[]
warous1=[]
wpos1=[]
wvecs1=[] 
for i in ind_sample:
    
    w=words[i]
    words_rand1.append(w)
    
    wfreq=freqs[i]
    wlen=lens[i]
    warous=arousals[i]
    wpos=POS[i]
    vec=vecs[i]
    
    wfreqs1.append(wfreq)
    wlens1.append(wlen)
    warous1.append(warous)
    wpos1.append(wpos)
    wvecs1.append(vec)
    

    response=estimate_response(wpos,wfreq,wlen,warous)

    responses.append(response)
    
lists=[]
list_response_means=[]
list_response_stds=[]

list_freq_means=[]
list_len_means=[]
list_arous_means=[]
wnouns=[]
wverbs=[]

list_freq_stds=[]
list_len_stds=[]
list_arous_stds=[]

list_vec_pair_means=[]
list_vec_pair_stds=[]

list_responses=[]

for i in range(num_lists):
    
    list1=words_rand1[i*items_per_list:(i+1)*items_per_list]  
    
    list_vecs=[word_gloves[w] for w in list1]
    list_vecs_dist=scipy.spatial.distance.cdist(list_vecs,list_vecs)
    list_vec_pair_means.append(np.mean(np.ndarray.flatten(list_vecs_dist)))
    list_vec_pair_stds.append(np.std(np.ndarray.flatten(list_vecs_dist)))
    
for i in range(num_lists):
    
    list1=words_rand1[i*items_per_list:(i+1)*items_per_list]  
    lists.append(list1)
   
    resp1=list(np.array(responses[i*items_per_list:(i+1)*items_per_list]) - np.array([list_vec_pair_means[i] for j in range(items_per_list)])) 

    
    list_responses.append(resp1)
    
    list_response_means.append(np.mean(resp1))
    list_response_stds.append(np.std(resp1)/np.sqrt(items_per_list))
    
    list_freq_means.append(np.mean(wfreqs1[i*items_per_list:(i+1)*items_per_list]))
    list_len_means.append(np.mean(wlens1[i*items_per_list:(i+1)*items_per_list]))
    list_arous_means.append(np.mean(warous1[i*items_per_list:(i+1)*items_per_list]))
    
    list_freq_stds.append(np.std(wfreqs1[i*items_per_list:(i+1)*items_per_list])/np.sqrt(items_per_list))
    list_len_stds.append(np.std(wlens1[i*items_per_list:(i+1)*items_per_list])/np.sqrt(items_per_list))
    list_arous_stds.append(np.std(warous1[i*items_per_list:(i+1)*items_per_list])/np.sqrt(items_per_list))
    
    wnouns.append(len([wp for wp in wpos1[i*items_per_list:(i+1)*items_per_list] if wp=='Noun']))
    wverbs.append(num_lists*items_per_list - wnouns[i])
    
    


```


```{python, echo=False}
list_freq_means=np.array(list_freq_means)
list_len_means=np.array(list_len_means)
list_arous_means=np.array(list_arous_means)
wnouns=np.array(wnouns)
wverbs=np.array(wverbs)

list_freq_stds=np.array(list_freq_stds)
list_len_stds=np.array(list_len_stds)
list_arous_stds=np.array(list_arous_stds)

list_vec_pair_means=np.array(list_vec_pair_means)
list_vec_pair_stds=np.array(list_vec_pair_stds)
```

**Table 1**:


```{python, echo=False}
import numpy as np
import matplotlib.pyplot as plt


fig=plt.figure()
clust_data = [[str(np.round(list_freq_means[0],2))+ ' ± ' +str(np.round(list_freq_stds[0],2)),
               str(np.round(list_freq_means[1],2))+' ± '+str(np.round(list_freq_stds[1],2))],
              [str(np.round(list_len_means[0],2))+ ' ± ' +str(np.round(list_len_stds[0],2)),
               str(np.round(list_len_means[1],2))+' ± '+str(np.round(list_len_stds[1],2))],
              [str(np.round(list_arous_means[0],2))+ ' ± ' +str(np.round(list_arous_stds[0],2)),
               str(np.round(list_arous_means[1],2))+' ± '+str(np.round(list_arous_stds[1],2))],
              [str(np.round(list_vec_pair_means[0],2))+ ' ± ' +str(np.round(list_vec_pair_stds[0],2)),
               str(np.round(list_vec_pair_means[1],2))+' ± '+str(np.round(list_vec_pair_stds[1],2))]]
# clust_data = [['a','b'],list_len_means,list_arous_means,wnouns,wverbs]

collabel=("Nouns", "Verbs")
rowlabel=("Lex Freq", "Length","Arousal","Sem Dist")
plt.axis('tight')
plt.axis('off')
table = plt.table(cellText=clust_data,colLabels=collabel,rowLabels=rowlabel,loc='center',colWidths=[0.25 for x in range(2)],cellLoc='center')

# axs[1].plot(clust_data[:,0],clust_data[:,1])
#plt.show()

```

**Figure 1**:


```{python, echo=False}
#plot figures
plt.figure()
plt.bar([i for i in range(num_lists)],list_response_means,width=0.8,tick_label=['List1','List2'],yerr=list_response_stds)

p1=plt.ylabel('Estimated PSC')
p2=plt.title('Baseline Estimates for randomly selected lists')
```

It is apparent that randomly selecting 2 lists of 100 words resulted in stimuli lists that are not matched on the features considered potential confounds in this simulated experiment. While not all of these differences are significant, they still have the potential to affect the dependent variable (estimated PSC) in a way that is not useful for the critical manipulation. Indeed, it is apparent in Figure 1 that the baseline PSC estimates are different across lists. While this is interested in it's own right, it will be necessary to equalize these baseline estimates in order to test the effect of a paired image on PSC while controlling for these other stimulus features. 

### Estimating the effect of paired images on LIFG responses

In the design of this experiment, we hypothesized that a paired image will elicit an increase in 0.05 PSC in the LIFG compared to words presented alone. To simulate this hypothesis, we will run a counterbalanced fMRI experiment on 20 subjects, where 10 subjects will view first list of 100 words paired with an image and the second list of 100 words presented alone, and the other 10 subjects will view the first list of words presented alone and the second list with a paired image. For each word within each subject, the LIFG response will be sampled from the distribution determined for the baseline response estimates (Figure 1) plus 0.05 if that word is paired with an image.

The results of this experiment are shown below (Figure 2).


```{python, echo=False}

responses=list_responses[0] + list_responses[1]


responses_all=[]
for subj in range(20):
    
    responses_subj=[]
    
    subj_var=np.random.normal(0,0.2,1)
    subj_var=0
    
    for li,list1 in enumerate(lists):
        
        responses_list=[]
        
        for i,w in enumerate(list1):
            
            resp=responses[i]
            if (li==0 and subj<10) or (li==1 and subj > 9):
                #resp=resp + subj_var + 0.1  + np.random.normal(0,0.3,1)
                
                resp=np.random.normal(list_response_means[li],list_response_stds[li],1) + 0.05 
  
            else:
                #resp=resp + subj_var  + np.random.normal(0,0.3,1)
            
                resp=np.random.normal(list_response_means[li],list_response_stds[li],1) 
                
            responses_list.append(resp)
                
        responses_subj.append(responses_list)
        
    responses_all.append(responses_subj)
    


```


```{python, echo=False}
subj_means=np.mean(responses_all,axis=2)

list1m=[]
list2m=[]
for mi,m in enumerate(subj_means):
    
    if mi < 10:
        list1m.append(m[0])
        list2m.append(m[1])
        
    if mi > 9:
        list1m.append(m[1])
        list2m.append(m[0])


```


```{python, echo=False}
mean1=np.mean(list1m)
mean2=np.mean(list2m)
stderr1=np.std(list1m)/np.sqrt(20)
stderr2=np.std(list2m)/np.sqrt(20)
```

**Figure 2**


```{python, echo=False}
#plot figures
#plt.figure()
pp=plt.bar([i for i in range(num_lists)],[mean1,mean2],width=0.8,tick_label=['Word + Image','Word Only'],yerr=[stderr1,stderr2])
ff=plt.ylabel('Estimated PSC')
ff2=plt.title('Conditional estimate for the simulated experiment')
#plt.show()
```

The results of this initial simulation appear reasonable. Primarily, both PSC values occur within the range that would be expected based on the relevant literature (BOLD responses in the LIFG tend to hover around 0.5 PSC), meaning that the functions we defined for estimating baseline PSC values are reasonable. Secondly, words paired with images appear to elicit around 0.05 more PSC than words presented alone, which was expected based on the main effect that was coded into this simulation. Note that this result would *not* have been achieved if the two stimuli lists were *not* counterbalanced. That is, if words in list1 was always paired with an image and words in list2 were always presented alone, the baseline differences in their estimated PSC (Figure 1) would have skewed the results and prevented us from identifying the main effect of a paired image. 

In order to verify the results of this simulation, we will re-run it 50 times, each time drawing from a different random sample of 200 words separated randomly into 2 lists. The results, averaged across all 50 runs, are shown below in Figure 3. 


```{python, echo=False}
means1_all=[]
means2_all=[]
stderrs1_all=[]
stderrs2_all=[]

for run in range(50):

    # noun_inds=[i for i in range(len(words)) if POS[i]=='Noun']
    # verb_inds=[i for i in range(len(words)) if POS[i]=='Verb']

    w_inds=[i for i in range(len(words))]

    # random.shuffle(noun_inds)
    # random.shuffle(verb_inds)

    random.shuffle(w_inds)

    num_lists=2
    items_per_list=100

    # ind_sample=noun_inds[:100] + verb_inds[:100]

    ind_sample=w_inds[:200]

    responses=[]
    words_rand=[]

    words_rand1=[]
    wfreqs1=[]
    wlens1=[]
    warous1=[]
    wpos1=[]
    wvecs1=[] 
    for i in ind_sample:

        w=words[i]
        words_rand1.append(w)

        wfreq=freqs[i]
        wlen=lens[i]
        warous=arousals[i]
        wpos=POS[i]
        vec=vecs[i]

        wfreqs1.append(wfreq)
        wlens1.append(wlen)
        warous1.append(warous)
        wpos1.append(wpos)
        wvecs1.append(vec)

        response=estimate_response(wpos,wfreq,wlen,warous)
        responses.append(response)

        
        
    lists=[]
    list_response_means=[]
    list_response_stds=[]

    list_freq_means=[]
    list_len_means=[]
    list_arous_means=[]
    wnouns=[]
    wverbs=[]

    list_freq_stds=[]
    list_len_stds=[]
    list_arous_stds=[]

    list_vec_pair_means=[]
    list_vec_pair_stds=[]

    list_responses=[]

    for i in range(num_lists):

        list1=words_rand1[i*items_per_list:(i+1)*items_per_list]  

        list_vecs=[word_gloves[w] for w in list1]
        list_vecs_dist=scipy.spatial.distance.cdist(list_vecs,list_vecs)
        list_vec_pair_means.append(np.mean(np.ndarray.flatten(list_vecs_dist)))
        list_vec_pair_stds.append(np.std(np.ndarray.flatten(list_vecs_dist)))

    for i in range(num_lists):

        list1=words_rand1[i*items_per_list:(i+1)*items_per_list]  
        lists.append(list1)

        resp1=list(np.array(responses[i*items_per_list:(i+1)*items_per_list]) - np.array([list_vec_pair_means[i] for j in range(items_per_list)])) 
        list_responses.append(resp1)

        list_response_means.append(np.mean(resp1))
        list_response_stds.append(np.std(resp1)/np.sqrt(items_per_list))

        list_freq_means.append(np.mean(wfreqs1[i*items_per_list:(i+1)*items_per_list]))
        list_len_means.append(np.mean(wlens1[i*items_per_list:(i+1)*items_per_list]))
        list_arous_means.append(np.mean(warous1[i*items_per_list:(i+1)*items_per_list]))

        list_freq_stds.append(np.std(wfreqs1[i*items_per_list:(i+1)*items_per_list])/np.sqrt(items_per_list))
        list_len_stds.append(np.std(wlens1[i*items_per_list:(i+1)*items_per_list])/np.sqrt(items_per_list))
        list_arous_stds.append(np.std(warous1[i*items_per_list:(i+1)*items_per_list])/np.sqrt(items_per_list))

        wnouns.append(len([wp for wp in wpos1[i*items_per_list:(i+1)*items_per_list] if wp=='Noun']))
        wverbs.append(num_lists*items_per_list - wnouns[i])

        
        
    list_freq_means=np.array(list_freq_means)
    list_len_means=np.array(list_len_means)
    list_arous_means=np.array(list_arous_means)
    wnouns=np.array(wnouns)
    wverbs=np.array(wverbs)

    list_freq_stds=np.array(list_freq_stds)
    list_len_stds=np.array(list_len_stds)
    list_arous_stds=np.array(list_arous_stds)

    list_vec_pair_means=np.array(list_vec_pair_means)
    list_vec_pair_stds=np.array(list_vec_pair_stds)


    
    
    
    
    
    
    responses=list_responses[0]+list_responses[1]
    
    responses_all=[]
    for subj in range(20):

        responses_subj=[]

        subj_var=np.random.normal(0,0.2,1)
        subj_var=0
        
        #subj_var=0

        for li,list1 in enumerate(lists):

            responses_list=[]

            for i,w in enumerate(list1):

                resp=responses[i]
                if (li==0 and subj<10) or (li==1 and subj > 9):
                    #resp=resp + subj_var + 0.1  + np.random.normal(0,0.3,1)
                    
                    resp=np.random.normal(list_response_means[li],list_response_stds[li],1) + 0.05 

                else:
                    #resp=resp + subj_var  + np.random.normal(0,0.3,1)
                    
                    resp=np.random.normal(list_response_means[li],list_response_stds[li],1) 

                responses_list.append(resp)

            responses_subj.append(responses_list)

        responses_all.append(responses_subj)

        
        
    subj_means=np.mean(responses_all,axis=2)

    list1m=[]
    list2m=[]
    for mi,m in enumerate(subj_means):

        if mi < 10:
            list1m.append(m[0])
            list2m.append(m[1])

        if mi > 9:
            list1m.append(m[1])
            list2m.append(m[0])

        
    mean1=np.mean(list1m)
    mean2=np.mean(list2m)
    stderr1=np.std(list1m)/np.sqrt(20)
    stderr2=np.std(list2m)/np.sqrt(20)
    
    
    
    means1_all.append(mean1)
    means2_all.append(mean2)
    
    stderrs1_all.append(stderr1)
    stderrs2_all.append(stderr2)


#plot figures
mean1_all=np.mean(means1_all)
mean2_all=np.mean(means2_all)
stderr1_all=np.mean(stderrs1_all)
stderr2_all=np.mean(stderrs2_all)


pp2=plt.bar([i for i in range(num_lists)],[mean1_all,mean2_all],width=0.8,tick_label=['Word+Image','Word Only'],yerr=[stderr1_all,stderr2_all])


f=plt.ylabel('Estimated PSC')
ff2=plt.title('Conditional estimates for the simulated experiment (avg. over 50 samples)')

```

The results in Figure 3 appear similar to those in Figure 2, meaning our simulation is working. The mean and standard error reported here are the results we would expect from randomly sampling 200 words from the english language. 

# OptSeg

Now we will demonstrate the utility of OptSeg. Instead of randomly sampling 200 words and splitting them arbitrarily into stimuli lists, we will use OptSeg to optimize the segmentation of words into two lists such that the stimulus features shown to effect the baseline LIFG PSC estimates (shown in Figure 1) can be matched in their quantitative distribution across lists.  

### How it works:

OptSeg can be described in a series of processing steps:

i) Create individual matrices of the pairwise distances for each stimulus feature to be matched (frequency, length, etc.). 

ii) Noralize (z-score) each matrix.

iii) Calculate an overall distance matrix as the weighted sum of each individual matrix, where the weights correspond to the importance of matching each feature (in this example, all weights = 1).  

iv) Define high-level parameters: number of lists (N=2), items per list (M=100).

v) Initialize each list with the N most similar (least distant) items. 

vi) Average the rows of the distance matrix across items within each list and identify the maximally distant item in the averaged row. 

vii) For the list whose maximally distant item is the most distant of all maximally distant items, add that item to that list. 

viii) Recompute distance matrices.

ix) Iterate steps vi - viii until M items have been added to each list.


Displayed below are the stimuli feature distributions in each list (Table 2) and the baseline estimates of LIFG PSC (without the experimental manipulation) given those distributions of features (Figure 4).

**Table 2:**


```{python, echo=False}

#print('Computing distance matrices \n')
vecs_dist=scipy.spatial.distance_matrix(vecs,vecs)
freqs_dist=np.abs(np.subtract.outer(freqs,freqs))
lens_dist=np.abs(np.subtract.outer(lens,lens))
arous_dist=np.abs(np.subtract.outer(arousals,arousals))



```


```{python, echo=False}
vecs_distz=scipy.stats.zscore(np.ndarray.flatten(vecs_dist)).reshape(len(vecs_dist),len(vecs_dist))
freqs_distz=scipy.stats.zscore(np.ndarray.flatten(freqs_dist)).reshape(len(freqs_dist),len(freqs_dist))
lens_distz=scipy.stats.zscore(np.ndarray.flatten(lens_dist)).reshape(len(lens_dist),len(lens_dist))
arous_distz=scipy.stats.zscore(np.ndarray.flatten(arous_dist)).reshape(len(arous_dist),len(arous_dist))

a=1
b=1
c=1
d=1

dists=a*vecs_distz + b*freqs_distz + c*lens_distz + d*arous_distz

```


```{python, echo=False}
vecs2D = umap.UMAP(n_neighbors=5,
                      min_dist=1,
                      metric='cosine').fit_transform(vecs)

vecs2D=list([list(v) for v in vecs2D])
```


```{python, echo=False}
dists=a*vecs_distz + b*freqs_distz + c*lens_distz + d*arous_distz
########################################################

#print('Generating lists \n')
lists=[[] for i in range(num_lists)]

used=[]

#find 5 closest words and put them in separate lists

min_val=math.inf
min_row=0

for i,dist_row in enumerate(dists):

    sum1=sum(np.sort(dist_row)[:num_lists])

    if sum1 < min_val:
        min_val=sum1
        min_row=i

min_dist_row=dists[min_row]

sort_inds=np.argsort(min_dist_row)[:num_lists]


used=[]

for i in range(num_lists):

    lists[i].append(sort_inds[i])
    used.append(sort_inds[i])

    for j,row in enumerate(dists):

        row[sort_inds[i]] = 0
        dists[j]=row

        
        

#find 5 furthest possible words

for n in range(items_per_list-1):

    rem_lists=lists
    rem_inds=[0,1]

    for i in range(num_lists):

        #get distances of unused words from each current list
        dist_means=[np.mean(dists[l],axis=0) for l in rem_lists]

        #sort lists by maximum variance potential
        list_sort_max = np.argsort([np.max(dm) for dm in dist_means])[::-1]

        list_ind = list_sort_max[0]

        list_dist_means = dist_means[list_ind]

        sort_inds = np.argsort(list_dist_means)[::-1]
        
        got=0
        while got==0:
            add_ind=sort_inds[0]
            if add_ind not in used:
                used.append(add_ind)
                break

#             break
#             if rem_inds[list_ind]==0 and add_pos=='Noun':
#                 used.append(add_ind)
#                 break
#             elif rem_inds[list_ind]==1 and add_pos=='Verb':
#                 used.append(add_ind)
#                 break
#             else:
#                 continue
#             if len(pos_list)>0 and len([w for w in lists[rem_inds[list_ind]] if word_POS[words[w]] == add_pos]) > items_per_pos-1:
#                 continue
#             else:
#                 break

        lists[rem_inds[list_ind]].append(add_ind)


        for i,row in enumerate(dists):

            row[sort_inds[0]]=0 
            dists[i]=row


        rem_lists=rem_lists[:list_ind] + rem_lists[list_ind+1:]

        rem_inds=rem_inds[:list_ind] + rem_inds[list_ind+1:]


        
   
responses=[]
words_rand=[]

words_rand1=[]
wfreqs1=[]
wlens1=[]
warous1=[]
wpos1=[]
wvecs1=[] 


ind_sample=lists[0]+lists[1]

for subj in range(20):
    
    for i in ind_sample:

        w=words[i]
        words_rand1.append(w)

        wfreq=freqs[i]
        wlen=lens[i]
        warous=arousals[i]
        wpos=POS[i]
        vec=vecs[i]

        wfreqs1.append(wfreq)
        wlens1.append(wlen)
        warous1.append(warous)
        wpos1.append(wpos)
        wvecs1.append(vec)

        response=estimate_response(wpos,wfreq,wlen,warous)
        responses.append(response)   

       
    
        
wordlists=[]
list_response_means=[]
list_response_stds=[]

list_freq_means=[]
list_len_means=[]
list_arous_means=[]
wnouns=[]
wverbs=[]

list_freq_stds=[]
list_len_stds=[]
list_arous_stds=[]

list_vec_pair_means=[]
list_vec_pair_stds=[]

list_responses=[]

for i in range(num_lists):

    list1=words_rand1[i*items_per_list:(i+1)*items_per_list]  

    list_vecs=[word_gloves[w] for w in list1]
    list_vecs_dist=scipy.spatial.distance.cdist(list_vecs,list_vecs)
    list_vec_pair_means.append(np.mean(np.ndarray.flatten(list_vecs_dist)))
    list_vec_pair_stds.append(np.std(np.ndarray.flatten(list_vecs_dist)))

for i in range(num_lists):

    list1=words_rand1[i*items_per_list:(i+1)*items_per_list]  
    wordlists.append(list1)

    resp1=list(np.array(responses[i*items_per_list:(i+1)*items_per_list]) - np.array([list_vec_pair_means[i] for j in range(items_per_list)])) + np.array([0.8 for i in range(items_per_list)]) 
    list_responses.append(resp1)

    list_response_means.append(np.mean(resp1))
    list_response_stds.append(np.std(resp1)/np.sqrt(items_per_list))

    list_freq_means.append(np.mean(wfreqs1[i*items_per_list:(i+1)*items_per_list]))
    list_len_means.append(np.mean(wlens1[i*items_per_list:(i+1)*items_per_list]))
    list_arous_means.append(np.mean(warous1[i*items_per_list:(i+1)*items_per_list]))

    list_freq_stds.append(np.std(wfreqs1[i*items_per_list:(i+1)*items_per_list])/np.sqrt(items_per_list))
    list_len_stds.append(np.std(wlens1[i*items_per_list:(i+1)*items_per_list])/np.sqrt(items_per_list))
    list_arous_stds.append(np.std(warous1[i*items_per_list:(i+1)*items_per_list])/np.sqrt(items_per_list))

    wnouns.append(len([wp for wp in wpos1[i*items_per_list:(i+1)*items_per_list] if wp=='Noun']))
    wverbs.append(num_lists*items_per_list - wnouns[i])



list_freq_means=np.array(list_freq_means)
list_len_means=np.array(list_len_means)
list_arous_means=np.array(list_arous_means)
wnouns=np.array(wnouns)
wverbs=np.array(wverbs)

list_freq_stds=np.array(list_freq_stds)
list_len_stds=np.array(list_len_stds)
list_arous_stds=np.array(list_arous_stds)

list_vec_pair_means=np.array(list_vec_pair_means)
list_vec_pair_stds=np.array(list_vec_pair_stds)        





fig=plt.figure()
clust_data = [[str(np.round(list_freq_means[0],2))+ ' ± ' +str(np.round(list_freq_stds[0],2)),
               str(np.round(list_freq_means[1],2))+' ± '+str(np.round(list_freq_stds[1],2))],
              [str(np.round(list_len_means[0],2))+ ' ± ' +str(np.round(list_len_stds[0],2)),
               str(np.round(list_len_means[1],2))+' ± '+str(np.round(list_len_stds[1],2))],
              [str(np.round(list_arous_means[0],2))+ ' ± ' +str(np.round(list_arous_stds[0],2)),
               str(np.round(list_arous_means[1],2))+' ± '+str(np.round(list_arous_stds[1],2))],
              [str(np.round(list_vec_pair_means[0],2))+ ' ± ' +str(np.round(list_vec_pair_stds[0],2)),
               str(np.round(list_vec_pair_means[1],2))+' ± '+str(np.round(list_vec_pair_stds[1],2))]]
# clust_data = [['a','b'],list_len_means,list_arous_means,wnouns,wverbs]

collabel=("Nouns", "Verbs")
rowlabel=("Lex Freq","Length","Arousal","Sem Dist")
plt.axis('tight')
plt.axis('off')
table = plt.table(cellText=clust_data,colLabels=collabel,rowLabels=rowlabel,loc='center',colWidths=[0.25 for x in range(2)],cellLoc='center')






```

**Figure 4:**


```{python, echo=False}

#plot figures

b1=plt.bar([i for i in range(num_lists)],list_response_means,width=0.8,tick_label=['List1','List2'],yerr=list_response_stds)

b2=plt.ylabel('Estimated PSC')
b3=plt.title('Baseline Estimates for optimally segmented word lists')


```

Additionally, we can visualize the semantic distribution of words in each list by reducing each word's GloVe vector to two dimensions with UMAP (McInnes & Healy, 2018) and plotting those dimensions as x-y coordinates (Figure 5).

**Figure 5:**


```{python, echo=False}
def get_colors(N):
    HSV = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
    RGB = map(lambda x: colorsys.hsv_to_rgb(*x), HSV)
    return list(RGB)



cols=get_colors(num_lists)


for i,list1 in enumerate(lists):

    for ind in list1:

        x=vecs2D[ind][0]
        y=vecs2D[ind][1]

        word=words[ind]
    
        tt=plt.text(x,y,word,color=cols[i],fontsize=8)


c=plt.xlim(np.min([v[0] for v in vecs2D])-1,np.max([v[0] for v in vecs2D])+1)
c2=plt.ylim(np.min([v[1] for v in vecs2D])-1,np.max([v[1] for v in vecs2D])+1)

a1=plt.axis('off')


```

It is clear from Table 2 and Figures 4 and 5 that the distributions of the four independent stimulus features were successfully matched by OptSeg, and that the baseline estimates of LIFG PSC are nearly equal between lists as a result. Now we will re-run the simulated experiment to see how matching stimuli lists affects the results, which are displayed below in Figure 6. 

**Figure 6:**


```{python, echo=False}
responses=list_responses[0]+list_responses[1]

responses_all=[]
    
for subj in range(20):

    responses_subj=[]

    subj_var=np.random.normal(0,0.2,1)
    subj_var=0

    for li,list1 in enumerate(lists):

        responses_list=[]

        for i,w in enumerate(list1):

            resp=responses[i]
            if (li==0 and subj<10) or (li==1 and subj > 9):
                #resp=resp + subj_var + 0.1 + np.random.normal(0,0.3,1)
                
                resp=np.random.normal(list_response_means[li],list_response_stds[li],1) + 0.05 

            else:
                #resp=resp + subj_var + np.random.normal(0,0.3,1)
                
                resp=np.random.normal(list_response_means[li],list_response_stds[li],1) 

            responses_list.append(resp)

        responses_subj.append(responses_list)

    responses_all.append(responses_subj)



subj_means=np.mean(responses_all,axis=2)

list1m=[]
list2m=[]
for mi,m in enumerate(subj_means):

    if mi < 10:
        list1m.append(m[0])
        list2m.append(m[1])

    if mi > 9:
        list1m.append(m[1])
        list2m.append(m[0])


mean1=np.mean(list1m)
mean2=np.mean(list2m)
stderr1=np.std(list1m)/np.sqrt(20)
stderr2=np.std(list2m)/np.sqrt(20)

means1_all.append(mean1)
means2_all.append(mean2)

stderrs1_all.append(stderr1)
stderrs2_all.append(stderr2)


#plot figures
# mean1_all=np.mean(mean1)
# mean2_all=np.mean(mean2)
# stderr1_all=np.mean(stderr1)
# stderr2_all=np.mean(stderr2)


bb=plt.bar([i for i in range(num_lists)],[mean1,mean2],width=0.8,tick_label=['Word+Image','Word Only'],yerr=[stderr1,stderr2])

t1=plt.title('Conditional estimates for the simulated experiment with optimized word lists')
t2=plt.ylabel('Estimated PSC')

```

It is clear from this simulation that OptSeg dramatically improved the standard error of the estimated PSC in the LIFG in response to both words-image pairs and words presented alone (~0.04 without OptSeg, ~0.01 with OptSeg). In fact, increasing the number of simulated subjects from 20 to 100 with randomly selected word lists does not even reduce the standard error to the value achieved with only 20 subjects plus OptSeg.  

Stay tuned for future work with OptSeg. A full toolbox will be made available that allows researchers to customize and optimize the segmentation of words, sentences, images, and more! 
