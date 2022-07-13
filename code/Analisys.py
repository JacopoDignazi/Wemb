# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

print("\n\n  {:#^55}".format(" Embedded Words Analysis "))
print()
print()

ME='ANALISYS'


# %%
# ###################################################### SETTINGS

from Wemb_Parameters import *

# CC_number=None
#  ^^^^^ already imported 
print()
if COORDINATE_BLOCKS is True:
    what_to_study=get_switch_analysis()
#     print(what_to_study)
    if what_to_study=='C':
        ANALYST_LOAD_CLUSTER=True
        print("    Coordinated run, step An{}: studying clustering transformed vectors from previous iteration".format(what_to_study))
    elif what_to_study=='M':
        ANALYST_LOAD_CLUSTER=False
        print("    Coordinated run, step An{}: studying embedded word vectors from previous iteration".format(what_to_study))
    else:
        print("    some sort of unwanted error just happened")


# %%
# ##############################  RUNNING OPTIONS ##################################
# <<<<<<<<<<<<<<<<<<<<<<<<  takes them from Wemb_par import   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# # ------------------------ Model and Training options
# min_freq=5
# vec_size=80
# architecture='CBOW'
# # ^^^^^ not yet implemented
# window_size=8
# n_epochs=20

# # ----------------------- Batching Options
# BATCHING=True
# batch_size=500
# # N_batches=1000

# # ----------------------- Clustering options
# ANALYST_LOAD_CLUSTER=False
# CC_number=None

# # ----------------------- S/L settings
# ANALYST_SAVE_IMG=False
# ANALYST_SAVE_TXT=False

# # ----------------------- S/L name
# SL_NAME='google_news'

# ^^^^^^^^^^^^^^^^^ uncomment to manually override imported parameters


# %%
# ############################################ initializing internal params from given ones

def reload():
    global sel_par
    global cls_par 
    sel_par,cls_par=reload_par()
    

def reset_names():
    global sel_par
    sel_par['Vsz']=vec_size
    sel_par['Wsz']=window_size
    sel_par['mfq']=min_freq
    sel_par['bsz']=batch_size
    sel_par['nEp']=n_epochs    
    
    global cls_par
    if CC_number is None:
        cls_par['CC']=sel_par['Vsz']-1
    else:
        cls_par['CC']=CC_number 
reset_names()
# ^^^^^ apply changes only if parameters are manually overwritten

# -------------------------------------------------- save/load settings
SAVE_FIG= ANALYST_SAVE_IMG

# --------------------------------------- names settings
name_load_vec=give_name_to(ME,'LOAD','VEC', sel_par, SL_NAME, cls_par, load_cluster=ANALYST_LOAD_CLUSTER)
name_save_img=give_name_to(ME,'SAVE','IMG', sel_par, SL_NAME, cls_par, load_cluster=ANALYST_LOAD_CLUSTER)
name_save_txt=give_name_to(ME,'SAVE','TXT', sel_par, SL_NAME, cls_par, load_cluster=ANALYST_LOAD_CLUSTER)

# --------------------------------------- pathnames settings
pathname_load_vec=path_load_vectors+name_load_vec
pathname_save_img=path_save_img+name_save_img
pathname_save_txt=path_save_txt+name_save_txt

# ---------------------------- memory save flag
flag_model=False
flag_pair=False
flag_comp=False
flag_rel=False    


# %%
print(name_load_vec)
print(name_save_img)
print(name_save_txt)
# # ^^^^^NAMES CHECKS

# %%
print(pathname_load_vec)
print(pathname_save_img)
print(pathname_save_txt)
# # ^^^^^PATHNAMES CHECKS

# %%
def display_settings():
    print()
    print("  {:#^55}".format(" MODEL INFO "))
    print()
    print("  Using parameters:",using_params)
    print()
    print("  {:-^55}\n".format(" TRAINING'S SETTINGS INFO "))
    print("  Batching document:", True)
    print("  Batch size:", batch_size)
    print("  Number of epochs: ",n_epochs)
    print()
    print("  {:-^55}\n".format(" model parameter "))    
    print("  Minimum frequency of word: ",min_freq)
    print("  Size of vectors: ", vec_size)
    print("  Architecture: ",architecture)
    print("  Size of syntax window: ", window_size)
    print()
    print("  {:#^55}".format(''))
    


# %%
if SL_NAME=='def':
    display_settings()
print()
print("     Loading model's vectors from")
print("          ",name_load_vec+'.vectors')
if SAVE_FIG:
    print()
    print("     At the end of iteration, graph images will be saved in\\as:")
    print("          ",pathname_save_img+'[graph_name].png')
    


# %%
# print()
# print("     Importing spaCy utils...")
# import spacy
# from spacy.lang.en import English
# from spacy.tokenizer import Tokenizer
# nlp = spacy.load('en_core_web_sm')
# tokenizer=nlp.Defaults.create_tokenizer(nlp)


# %%
# ########################################################################################## loading here
# chose the params of model you want to load here
# if nothing will just look for model with preset parameters

if SL_NAME=='google_news':
    loaded_vects = api.load('word2vec-google-news-300')
else:
    loaded_vects=Word2VecKeyedVectors.load(pathname_load_vec+'.vectors')


# ############################################################
# loaded_model=Word2Vec.load(name_load_vec+'.model')
# print("     Loaded model", name_load_vec+'.model')
# print("    ",loaded_model)


print("     Loaded vectors", name_load_vec+'.vectors')
# print("    ",loaded_vects)


# %%
print()
print()
print("   {:_^55}".format(" MODEL INTERPRETATION "))


# %%

# default_model=model
default_vects=loaded_vects
wv=default_vects

PERC_OF_VOCAB=100
n_max_vocab=int(len(default_vects.vocab)/(PERC_OF_VOCAB/100))


# %%

print()
print("     Importing sci-kit utils...")
from sklearn.metrics.pairwise import cosine_similarity

print("     Importing numpy utils...")
import numpy as np
from numpy.linalg import norm

print("     Importing matplot utils...")
import matplotlib.pyplot as plt

# def freq(word):
#     return sorted_dict_f[word]


#return position of the word in the vocab
def word_to_index(word,vects=default_vects):
    if type(word)!= str: 
        print('necessario inserire parola in formato str')
        return
    for i, word_v in enumerate(vects.vocab): 
#     for i, word_v in enumerate(vects): should make a dictionary from model if want it to work
        if word==word_v:
            return i
w2i=word_to_index

def index_to_word(index,vects=default_vects):
    for idx, word in enumerate(vects.vocab):
        if idx==index:
            return word
    else:
        print("     word not found")
    #     return word_in_range(n_max=index, window=1, write=False)[0]
i2w=index_to_word


def w_norm(word):
    return norm(wv[word])

def w_versor(word):
    return wv[word]/w_norm(word)   


#comparing both from string and from vector 
def cos_sim(a,b, vects=default_vects):
    if type(a) is str:
        a=vects[a]
    if type(b) is str:
        b=vects[b]
    return cosine_similarity([a],[b])[0][0]
sim=cos_sim

#top similarity in subset of vocab both from string and from vector
# vects needs to be a dictionary of {'word': (vector)}
# default vectors will be model processed ones; needs to tell if different
# "mode" is not needed but will keep it for now 
def topn_sim(thing, n=10, vects=default_vects, metric_f=cos_sim, N_vocab=n_max_vocab):
#     Sim_list={}
#     if mode=='C':
#         if type(thing) is str:
#             vect=vects[thing]
#         else:
#             vect=thing
#         for el in vects:
#             if type(thing) is str and el==thing:
#                 continue
#             Sim_list[el]=sim(vect,vects[el], N_vocab=n_max_vocab, vects=vects)
#         Sim_list=sorted(Sim_list, reverse=True, key=lambda x:x[1])
#         return Sim_list
            
    if type(thing) is str:
        return vects.similar_by_word(thing,topn=n,restrict_vocab=N_vocab)
    return vects.similar_by_vector(thing,topn=n,restrict_vocab=N_vocab)


#return k-st similar word, both from vector and for word 
# andrebbe ottimizzato per trovare i *meno* simili perché così conta sempre tutti dal primo all'ultimo
def nearest_word(vector_or_word, vects=default_vects, k=1, metric_f=cos_sim, N_vocab=n_max_vocab):
    nearest=topn_sim(vector_or_word, n=k, vects=vects, metric_f=metric_f, N_vocab=N_vocab)
    return nearest[k-1][0]
nw=nearest_word

def operation_vs_expected(vector, word, n=1, vects=default_vects, N_vocab=n_max_vocab):
    print("comparing vectors:", sim(vector,word))
    print("\n comparing", word," with nearest word:")
    for i in range(1,n+1):
        w_v=nearest_word(vector,k=i)
        if n!=1: print("{:3} ".format(i), end='')
        print("{:>20}".format(w_v), "vs", "{:<20}".format(word),":", sim(w_v, word, vects=vects))
        


# %%
# FREQUENCY 


# %%
# FREQUENCY ANALISYS

# freq=[]
# for wrd in wv.vocab:
#     freq.append(processed_sorted_dict_f[wrd])


# %%
# plot frequency


# %%
# 1     ########################################################################  NORMS


# %%
# NORM ANALYSIS
vocab_size=len(default_vects.vocab)

vec_norms=[]
vec_index=[]
stepping=-1

def norm_glob_analisys(vects=default_vects, n_skip=1):
    global vec_norms
    global vec_index
    
    stepping=-1
    for i, x in enumerate(vects.vocab):
        stepping=perc_compl(i,vocab_size,last_step=stepping, step=1,
                            text="     {:>8d}".format(i+1)+" words elaborated;  ")
        if i%n_skip==0:
            vec_index.append(i)
            vec_norms.append(w_norm(x))
        
        
print("     Computing vector's norm:")
norm_glob_analisys()


# %%



# %%
# NORMS MEAN

n_mean=np.mean(vec_norms)
n_std=np.std(vec_norms)

print("     mean of the norms:",n_mean)
print("     standard dev.    :",n_std)

print("\n")


# %%
# BIGGEST NORM INFO
mult_norm=4
max_tell=15


count=0
flag_stop=False
print("   \"far\" words, with bigger norm than {:3.3f}".format(n_mean+mult_norm*n_std),"( mean +",mult_norm,"* st.dev):\n")
for i, word, this_norm in zip(vec_index,wv.vocab,vec_norms):
    if(this_norm>n_mean+mult_norm*n_std):
        count+=1
        if flag_stop: continue
        if count>max_tell:
            print()
            print("      ... and more...")
            flag_stop=True
            continue
        print("   _{:_<3d}_index:__{:_>6d}_______{:_<30}_norm={:_<9.2f}".format(count,i,word,this_norm))
        
#     if i>100000: break
#     if i>143753 and i%12345==0: print(i,'\t', end='\r')
print("\n     Total amount of \"far\" words:", count)


# %%
# 2     ##########################################################################################  NORMS_MEAN 


# %%
# NEAR_NGB_MEAN_NORM 
norms_mean=[]
window=25
l=len(vec_norms)-1
start=0
end=0
stepping=-1
print("     Norms mean Analysis:")
for i in range(l+1):
    stepping=perc_compl(i,l+1,last_step=stepping, step=1,
                        text="     {:>8d}".format(i+1)+" words elaborated;  ") 
    if i-window<0: start=0
    else: start=i-window
    if i+window>l: end=l
    else: end=i+window
    norms_mean.append(np.mean(vec_norms[start:end]))


# %%
# 3     ###############################################################################  TOP(10)SIM MEAN


# %%


def avg_sim_with_topn_sim(word_or_vec,n=10, vects=default_vects,N_vocab=n_max_vocab):
    n_most_similar=topn_sim(word_or_vec,n,vects=vects, N_vocab=N_vocab)
    summ=0
    for word_and_value in n_most_similar:
        summ+=word_and_value[1]
    return summ/n
avgsim=avg_sim_with_topn_sim

N_bins=100
N_topn_words_to_average=10

hist_avgsim=[]
hist_avgsim.append([])
hist_avgsim.append([])

for i in range(N_bins):
    hist_avgsim[0].append(i/N_bins)
    hist_avgsim[1].append(0)
# hist[0] for x-axis, hist[1] for y-axis

# compute on all vocab word but could be sampled or window-selected
N_word_in_model_vocab=len(wv.vocab)
stepping=-1
print()
print("     Processing Avg_sim Histogram:")
wind_start=int(vocab_size/10)
wind_size=1000
for j, word in enumerate(wv.vocab):
    if j<wind_start:continue
    if j>=wind_start+wind_size: break
    stepping=perc_compl(j-wind_start,wind_size,last_step=stepping, step=0.001,
                                     text="     {:>6d}".format(j+1-wind_start)+" words elaborated from model's vocab  ")
    sim_value=avgsim(word,n=N_topn_words_to_average)
    for i in range(N_bins):
        if sim_value <= i/N_bins:
#           in bin 'x' we have value in ]x-1/Nbin,x]
            hist_avgsim[1][i]+=1
            break


# %%
# AVERAGE OF TOP(10)MEAN
# when model performs well 
# mean is around 0.7 and variance  (????)

avgsim=[]
for i,_ in enumerate(hist_avgsim[0]):
    for n in range(hist_avgsim[1][i]):
        avgsim.append(hist_avgsim[0][i])
print("     avg= ",np.mean(avgsim))
print("     std= ",np.std(avgsim))  


# %%
# 4     ###############################################################################  KST WORD MEAN SIMILARITY


# %%
# KST WORD MEAN SIMILARITY

K_max=len(wv.vocab)
# ^^^^^ K up to
N_min=int(vocab_size/5)
# ^^^^^ Index of first word look
sample_size=100
# ^^^^^ How many words to look at
N_max=N_min+sample_size

hist_Kst_sim=[]
for i in range(K_max):
    if i== len(wv.vocab)-1: break
    hist_Kst_sim.append([])
    hist_Kst_sim[i]=0
stepping=-1
print()
print("     Processing Kst_sim Histogram: ")
for n in range(N_min,N_max):
    word=i2w(n)
    stepping=perc_compl(n-N_min,N_max-N_min,last_step=stepping, step=1,
                        text="     {:>6d}".format(n+1-N_min)+" words elaborated from model's vocab  ")
    for K, Kst_word in enumerate(topn_sim(word,n=K_max+1)):
        hist_Kst_sim[K]+=Kst_word[1]
for i,_ in enumerate(hist_Kst_sim):
    hist_Kst_sim[i]/=sample_size


# %%
# KST WORD MEAN SIMILARITY INFO

print("    Vocab words in indexes [{},{}[ :".format(N_min,N_max))
print()
for i in range(11):
    curr=int(i*(K_max/10))
    if curr>=K_max:
        curr=K_max-2
    print("     {:<4d} st word's cos_sim is averagely {:1.5f}".format(curr+1,hist_Kst_sim[curr]) )


# %%
# if I need it here
# KST WORD MEAN SIMILARITY PLOT
# st=0
# en=200000
# graph_name='Average similarity of K-st most similar'

# fig_KSTM, graph_KSTM = plt.subplots()
# graph_KSTM.plot(range(1,K_max+1),hist_Kst_sim)
# graph_KSTM.set(xlabel='K-st most similar', ylabel='Average similarity',
#        title=graph_name+' (in subset of vocab)')
# graph_KSTM.grid()

# this_pathname_save_img=pathname_save_img+' --- '+graph_name+'.png'
# if SAVE_FIG:
#     fig_KSTM.savefig(this_pathname_save_img)
# plt.show()


# %%
# def tell_about(word):
#     index=w2i(word)
#     norm=vec_norms[index]
#     freq=processed_sorted_dict_f[word]
#     print("   ___at_index:{:_>5d}_______{:_<35}_frequency:_{:_<4}_norm={:_<9.2f}\n".format(index,word,freq,norm))
#     pp.pprint(topn_sim(word))


# %%
# FREQ VS NORM
# st=0
# en=20000

# plt.plot(freq[st:en],vec_norms[st:en]) 


# %%
# FREQ VS NEAR_NGB_MEAN_NORM
# # st=0
# en=20000

# plt.plot(freq[st:en],norms_mean[st:en]) 


# %%
# MAX NORM INFO
# index_max=np.argmax(vec_norms)
# tell_about(i2w(index_max))


# %%
# ############################################################################################ TESTING


# %%
if ANALYST_SAVE_TXT:
    this_pathname_save_txt=pathname_save_txt+"Tasks_results.txt"
    print("     Task results will be saved in\\as ")
    print("         ",this_pathname_save_txt)
    file_out=open(this_pathname_save_txt,"w")
else:
    file_out=None


# %%
list_of_words_contxt_recognition=['sound','wave','army','king','achilles','cave','quantum','machine']
list_of_words_gender_recognition=['king','man','son','father','grandfather','husband','boyfriend'] #boyfriend not in voc for perc_data=20
list_of_words_plural_recognition_numerable=['car','house','tree']
list_of_words_plural_recognition_not_numer=['sand','man','some','water']

def_n_ctx=7
def_n_gnd=7
def_n_plr=7
def_n_oth=10

def tell_topn_sim_contxt_recognition(list_of_words,n=5,vects=default_vects, file=file_out):
    for word in list_of_words:
        to_be_evaluated=word
        print(file=file)
        print("     word:", word, file=file)
        print("     most similar:", topn_sim(to_be_evaluated,n=n,vects=vects), file=file)
        
def tell_topn_sim_gender_recognition(l_o_w,n=5,vects=default_vects, file=file_out):
    for word in l_o_w:
        to_be_evaluated=w_versor(word)-w_versor('male')+w_versor('female')
        print(file=file)
        print("     "+word,"- male + female:", file=file)
        print("nearest words:", topn_sim(to_be_evaluated,n=n, vects=vects), file=file)

def tell_topn_sim_plural_recognition(l_o_w_numerable,l_o_w_not_numer,n=5,vects=default_vects, file=file_out):
    print("    {:.^55}".format("Numerable words"), file=file)
    for i, word in enumerate(l_o_w_numerable):
        for j, word_comp in enumerate(l_o_w_numerable):
            if j==i: continue                
            to_be_evaluated=w_versor(word+'s')-w_versor(word)+w_versor(word_comp)
            print(file=file)
            print("     "+word+'s',"-",word,'+',word_comp,":", file=file)
            print("     nearest words:", topn_sim(to_be_evaluated,n=n, vects=vects), file=file)
    print(file=file)
    print("  {:.^55}".format("Not numerable words"), file=file)
    for word in l_o_w_numerable:
        for word_comp in l_o_w_not_numer:
            to_be_evaluated=w_versor(word+'s')-w_versor(word)+w_versor(word_comp)
            print(file=file)
            print("     "+word+'s',"-",word,'+',word_comp,":", file=file)
            print("     nearest words:", topn_sim(to_be_evaluated,n=n, vects=vects), file=file)

def other_tests(n=def_n_oth, file=file_out):
    print("     husband - male vs wife - female :",sim(w_versor('husband')-w_versor('male'), w_versor('wife')-w_versor('female')), file=file)
    print("     king - male vs queen - female :",sim(w_versor('king')-w_versor('male'),w_versor('queen')-w_versor('female')), file=file)
    print("     birds-cars vs bird - car",sim(w_versor('birds')-w_versor('cars'), w_versor('bird')-w_versor('car')), file=file)
    print(file=file)
    print("     france-paris+italy",topn_sim(w_versor('france')-w_versor('paris')+w_versor('italy'),n=def_n_oth,vects=default_vects), file=file)
    
    print(file=file)
    avg_cos_sim=0
    count=0
    bigger_l_o_w_numerable=['dog','cat','chair','hat','king','bed','windmill','rainbow','wall','boat']
    bigger_l_o_w_numerable+=list_of_words_plural_recognition_numerable
    for i, word1 in enumerate(list_of_words_plural_recognition_numerable):
        for j, word2 in enumerate(list_of_words_plural_recognition_numerable):
            if j<= i: continue
            count+=1
            avg_cos_sim+=sim(w_versor(word1+'s')-w_versor(word2+'s'),w_versor(word1)-w_versor(word2))
    avg_cos_sim/=count
    print("     Average result (on sample of {:3d} words) for numerable plural recognition: {:2.4f}".format(len(bigger_l_o_w_numerable), avg_cos_sim), file=file)
    print("     Against average top similarity (between vocab words) of: {:2.4f}".format(hist_Kst_sim[0]), file=file)
    print("     rating on (very) small sample task: {:2.2f}".format(avg_cos_sim/hist_Kst_sim[0]), file=file)

    
def testing(l_o_w_ctx, l_o_w_gnd, l_o_w_plr, l_o_w_plrnn, other=None,
            n_ctx=5,n_gnd=5,n_plr=5,
            vects=default_vects, 
            file=file_out):
    print(file=file)   
    print("    {:#^55}".format(" Testing on some task examples "),file=file)    
    print(file=file)
    print("    {:-^55}".format("context recognition (topn_sim)"), file=file)
    tell_topn_sim_contxt_recognition(l_o_w_ctx,n=n_ctx,vects=vects, file=file)
    print(file=file)
    print("    {:-^55}".format("gender recognition"), file=file)
    tell_topn_sim_gender_recognition(l_o_w_gnd,n=n_gnd,vects=vects, file=file)
    print(file=file)
    print("    {:-^55}".format("plural recognition"),'\n', file=file)
    tell_topn_sim_plural_recognition(l_o_w_plr,l_o_w_plrnn,n=n_gnd,vects=vects, file=file)
    if other != None:
        print(file=file)
        print("    {:-^55}".format(" other tests"), file=file)
        other(file=file)


# %%
# CALLING TESTING FUNCTIONS

testing(list_of_words_contxt_recognition,  
       list_of_words_gender_recognition,  
       list_of_words_plural_recognition_numerable,  
       list_of_words_plural_recognition_not_numer, 
        n_ctx=def_n_ctx, 
        n_gnd=def_n_gnd,
        n_plr=def_n_plr,
       other=other_tests,
       file=file_out)

if file_out is not None:
    testing(list_of_words_contxt_recognition,  
           list_of_words_gender_recognition,  
           list_of_words_plural_recognition_numerable,  
           list_of_words_plural_recognition_not_numer, 
            n_ctx=def_n_ctx, 
            n_gnd=def_n_gnd,
            n_plr=def_n_plr,
           other=other_tests,
           file=None)   


# %%
# # wv.accuracy('questions-words.txt')
# wv.evaluate_word_pairs(datapath('wordsim353.tsv'))
# help(wv.accuracy)


# %%
if ANALYST_SAVE_TXT:
    file_out.close()


# %%
# pathname_save_img='(window=8, epochs=20, n_min=10, vec_size=80)'
if SAVE_FIG:
    print("     Graph pictures will be saved in\\as:")
    print("         ",pathname_save_img+"[graph name].png)")
    


# %%
st=0
en=200000


graph_name='Norm vs vocab Index'

fig_NvI, graph_NvI = plt.subplots()
graph_NvI.plot(vec_index[st:en],vec_norms[st:en])
graph_NvI.set(xlabel='Index in vocabulary', ylabel='Euclidean norm',
       title=graph_name)
graph_NvI.grid()

this_pathname_save_img=pathname_save_img+graph_name+'.png'
if SAVE_FIG:
    fig_NvI.savefig(this_pathname_save_img)


graph_name='Set norms mean vs vocab Index'

fig_NMvI, graph_NMvI = plt.subplots()
graph_NMvI.plot(vec_index[st:en],norms_mean[st:en])
graph_NMvI.set(xlabel='Index in vocabulary', ylabel='Mean of I(index) norm mean ',
       title=graph_name)
graph_NMvI.grid()

this_pathname_save_img=pathname_save_img+graph_name+'.png'
if SAVE_FIG:
    fig_NMvI.savefig(this_pathname_save_img)

    
graph_name='Histogram for mean of Top(N)similar'

fig_TSMvI, graph_TSMvI = plt.subplots()
graph_TSMvI.plot(hist_avgsim[0],hist_avgsim[1])
graph_TSMvI.set(xlabel='mean of Top(N)similar', ylabel='Occurrences',
       title=graph_name)
graph_TSMvI.grid()

this_pathname_save_img=pathname_save_img+graph_name+'.png'
if SAVE_FIG:
    fig_TSMvI.savefig(this_pathname_save_img)

    
graph_name='Average similarity of K-st most similar'

fig_KSTM, graph_KSTM = plt.subplots()
graph_KSTM.plot(range(1,K_max),hist_Kst_sim)
graph_KSTM.set(xlabel='K-st most similar', ylabel='Average similarity',
       title=graph_name+' (in subset of vocab)')
graph_KSTM.grid()

this_pathname_save_img=pathname_save_img+graph_name+'.png'
if SAVE_FIG:
    fig_KSTM.savefig(this_pathname_save_img)
    
    
plt.show()


# %%
# freq_avg=np.mean(freq)
# freq_std=np.std(freq)

# for n, word in enumerate(sorted_dict_f):
#     if sorted_dict_f[word]<freq_avg+1*freq_std:
#         break
#     print(word,end=', ')
# print(word, sorted_dict_f[word])

# topn_sim('p')


# %%
# help(default_vects)
# from gensim.models.keyedvectors import Word2VecKeyedVectors 
# new_vects=Word2VecKeyedVectors(79)


# %%
from numpy import random as rand

wv.init_sims(replace=True)
# ^^^^^^^ NOW ALL WORD VECTOR ARE JUST VERSORS

vocab_list=[]
for word in wv.vocab:
    vocab_list.append(word)


# %%
# flag_pair=False
# flag_comp=False
# flag_rel=False
# ^^^^^ uncomment to reset flags
if flag_pair is False:
    found_pairs=[]
    flag_pair=True
    pair_count=0
if flag_comp is False:
    found_compost=[]
    flag_comp=True
    comp_count=0
if flag_rel is False:
    found_relations=[]
    flag_rel=True
    rel_count=0

max_display=20

def random_word(words=vocab_list[:int((len(vocab_list)/2))]):
    return  rand.choice(words)


# %%
size_sample=10000
sim_trashold=hist_Kst_sim[2]

stepping=-1
for n in range(size_sample):
    to_write="     {:>6} pairs analysed;  {:4d} found   :".format(n+1,len(found_pairs))
    stepping=perc_compl(n,size_sample,stepping,step=0.001, text=to_write)
    i=random_word()
    j=random_word()
    if j==i:
        continue
    sim_val=cos_sim(wv[i],wv[j])
    if sim_val>=sim_trashold:     
        found_pairs.append( ((i,j),sim_val) )


# %%
found_pairs=sorted(found_pairs, key=lambda x: x[1], reverse=True)


# %%
N_sim_found=len(found_pairs)
print("     found ",N_sim_found," top similarities between pairs (above {:.2f}, in sample of {}): \n".format(sim_trashold,n+1))
for h, f_r in enumerate(found_pairs):
    if h>=max_display: break
    print(" {} vs {}".format(f_r[0][0],f_r[0][1]), ":  similarity value: ", f_r[1])
    print()


# %%
size_sample=10000
sim_trashold=hist_Kst_sim[2]

stepping=-1
for n in range(size_sample):
    to_write="     {:>6} compost analysed;  {:4d} found   :".format(n+1,len(found_compost))
    stepping=perc_compl(n,size_sample,stepping,step=0.001, text=to_write)
    i=random_word()
    j=random_word()
    k=random_word()
    if j==i or j==k or k==i:
        continue
    compost=wv[i]-wv[j]+wv[k]
    for sim_obj in topn_sim(compost):
        sim_word=sim_obj[0]
        if sim_word==i or sim_word==j or sim_word==k:
            continue
        sim_val=sim_obj[1]
        if sim_val>=sim_trashold:      
            found_compost.append( ((i,j,k,sim_word),compost,sim_val) )
            break
        else:
            break
        


# %%
found_compost=sorted(found_compost, key=lambda x: x[2], reverse=True)


# %%
N_sim_found=len(found_compost)
print("     found ",N_sim_found," top similarities between compost and word (above {:.2f}, in sample of {}): \n".format(sim_trashold,n+1))
for h, f_r in enumerate(found_compost):
    if h>=max_display: break
    print(" ({}-{}+{}) vs {}".format(f_r[0][0],f_r[0][1],f_r[0][2],f_r[0][3]), ":  similarity value: ", f_r[2])
    print()


# %%
size_sample=10000
sim_trashold=hist_Kst_sim[2]

stepping=-1
for n in range(size_sample):
    to_write="     {:>6} relations analysed;  {:4d} found   :".format(n+1,len(found_relations))
    stepping=perc_compl(n,size_sample,stepping,step=0.001, text=to_write)
    i=random_word()
    j=random_word()
    k=random_word()
    l=random_word()
    if i==j or i==k or i==l or j==k or j==l or k==l:
        continue
    relation_1=wv[i]-wv[j]
    relation_2=wv[k]-wv[l]
    sim_val=cos_sim(relation_1,relation_2)
    if sim_val>=sim_trashold:
        found_relations.append( ((i,j,k,l),(relation_1,relation_2),sim_val) )


# %%
found_relations=sorted(found_relations, key=lambda x: x[2], reverse=True)


# %%
N_sim_found=len(found_relations)
print("     found ",N_sim_found," top similarities between relations (above {:.2f}, in sample of {}): \n".format(sim_trashold,n+1))
for h, f_r in enumerate(found_relations):
    if h>=max_display: break
    print(" ({}-{}) vs ({}-{})".format(f_r[0][0],f_r[0][1],f_r[0][2],f_r[0][3]), "similarity value:", f_r[2])
    print()


# %%



# %%



