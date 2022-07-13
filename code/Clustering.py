# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
print("\n\n  {:#^55}".format(" Word Embedding Clusterization "))
print()
print()

ME='CLUSTERING'

# %%
# ###################################################### SETTINGS

from Wemb_Parameters import *

# CC_number=None
#  ^^^^^ already imported 


# %%
# ##############################  RUNNING OPTIONS ##################################
# <<<<<<<<<<<<<<<<<<<<<<<<  takes them from Wemb_par import   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

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
# SAVING_CLUSTERING=True
# CC_number=None

# ^^^^^^^ uncomment to manually override imported parameters


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

# --------------------------------------- names settings
name_load_vec=give_name_to(ME,'LOAD', 'VEC', sel_par, SL_NAME, cls_par)
name_save_img=give_name_to(ME,'SAVE', 'IMG', sel_par, SL_NAME, cls_par)
name_save_vec=give_name_to(ME,'SAVE', 'VEC', sel_par, SL_NAME, cls_par)

# --------------------------------------- pathnames settings
pathname_load_vec=path_load_vectors+name_load_vec
pathname_save_img=path_save_img+name_save_img
pathname_save_vec=path_save_vectors+name_save_vec

#  ---------------------------- memory save/safety flags
flag_cluster=False
flag_transform=False


# %%
print(name_load_vec)
print(name_save_vec)
print(name_save_img)
# # ^^^^^NAMES CHECKS

# %%
print(pathname_load_vec)
print(pathname_save_vec)
print(pathname_save_img)
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
#     clustering choice display


# %%
display_settings() 
print()
print("     Loading model's vectors from", name_load_vec)
if CLUSTER_SAVE_IMG:
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
# ############################################### loading here
# chose the params of model you want to load here
# if nothing will just look for model with imported parameters
# ^^^^^ def assignation with loaded parameters
if SL_NAME=='google_news':
    loaded_vects = api.load('word2vec-google-news-300')
else:
    loaded_vects=Word2VecKeyedVectors.load(pathname_load_vec+'.vectors')


# ############################################################


# loaded_model=Word2Vec.load(name_load_vec+'.model')
# print("     Loaded model", name_load_vec+'.model')
# print("    ",loaded_model)

print("     Loaded vectors", name_load_vec+'.vectors')
print("    ",loaded_vects)


# %%
print()
print()
print("   {:_^55}".format(" MODEL CLUSTERIZATION "))


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
from sklearn.cluster import KMeans

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
        print("{:>20}".format(w_v), "vs", "{:<20}".format(word),":", sim(w_v, word,vects=vects))
        


# %%
# Making VERSORS for clustering
# ^^^^^ COULD MAKE INLINE TO SAVE MEMORY
MAX_DATA=len(wv.vocab)
# MAX_DATA=100000
versors=[]
clustering_min_freq=0

stepping=-1
for i,word in enumerate(wv.vocab):
#     stepping=perc_compl(i,MAX_DATA,last_step=stepping, step=1)   
    if i>=MAX_DATA:
        print("     MAX_Data reached")
        break
#     if sorted_dict_f[word]<clustering_min_freq:
#         break
    versors.append(w_versor(word))
    


# %%
if flag_cluster==False:
    Cluster_Centers={}


# %%
def cluster_making(vectors, N_cluster, n_wd_dir=10, N_cl_dir='all'):
    kmeans = KMeans(n_clusters=N_cluster)
# fit kmeans object to data
    kmeans.fit(vectors)
    if N_cl_dir=='all':
        n_cluster_to_show=N_cluster
    for i, vec in enumerate(kmeans.cluster_centers_):
        if i==n_cluster_to_show:
            print()
            break
        print(" Cluster{:3}".format(i+1),":  ", end='')
        cluster_el=topn_sim(vec,n=n_wd_dir)
        sum_tip=0
        for i, el in enumerate(cluster_el):
            print(el[0],",",end=' ')
            sum_tip+=el[1]
        print(" ...")
        print("\t\t with average cos_sim: {:2.3f}".format(sum_tip/n_wd_dir))
        print()
    return kmeans.cluster_centers_

def display_cluster_centers(cluster_centers_list, n_wd_dir=10):
    for cluster_centers in cluster_centers_list:
        for i, vec in enumerate(cluster_centers):
            print(" Cluster{:3}".format(i+1),":  ", end='')
            cluster_el=topn_sim(vec,n=n_wd_dir)
            sum_tip=0
            for i, el in enumerate(cluster_el):
                print(el[0],",",end=' ')
                sum_tip+=el[1]
            print(" ...")
            print("\t\t with average cos_sim: {:2.3f}".format(sum_tip/n_wd_dir))
        print()


# %%
# MAKING CLUSTERS
NC_min=cls_par['CC']
# taking this ^^^^^ from imported CC_number
how_many_clusterings=1

for N in range(NC_min,NC_min+how_many_clusterings):
    print(" {:2} of {:2}) {:#^55}".format(N-NC_min+1, how_many_clusterings," Clustering {} ".format(N)))
    Cluster_Centers[str(N)]=cluster_making(versors, N)
    


# %%
# if flag_cluster==True:
#     CCBackup=Cluster_Centers
#     print("     Backup memory of", len(CCBackup),"different clusterization:")
#     print("     CCBackup keys: ",end='')
#     for key in CCBackup:
#         print(key+", ", end='')
    


# %%
CC=Cluster_Centers  #Per comodità


# %%
# taking and keeping biggest clustering 

def get_max_clusterings_dimension(clusterings):
    max_cl_dimension=0
    for key in clusterings:
        if int(key)>max_cl_dimension:
            max_cl_dimension=int(key)
    return max_cl_dimension

max_cl_dimension=get_max_clusterings_dimension(CC)
print("    Biggest clustering has dimension:", max_cl_dimension)

def extract_clustering(clusterings,N):
    Cluster_n={}
    if str(N) not in clusterings:
        print("    clustering of dimension",N,"not found")
        return None
    for n, center in enumerate(clusterings[str(N)]):
        Cluster_n[str(n+1)]=center
    return Cluster_n

C_max= extract_clustering(CC,max_cl_dimension)


# %%
print("     Keeping memory of", len(CC),"different clusterization(s):")
print("     keys: ",end='')
for key in CC:
    print(key+", ", end='')


# %%
# Cluster_Centers: all clustering as {N: (all clustering(N) N centers)} 
# CC=Cluster_Centers
# BackupCC=Cluster_Centers of some iteration

# C_max dictionary as {n: cluster center}


# %%
def tab_CC_sim(centers,idx_cluster=None, tell=False):
    centers_sim_values=[]
    if tell==False:
        for n, center1 in enumerate(centers):
            centers_sim_values.append([])
            for m, center2 in enumerate(centers):
                if m!=n:
                    sim_val=sim(center1,center2)
                    centers_sim_values[n].append(sim_val) 
        return centers_sim_values
#     TELL TABELLA
    print()
    if idx_cluster!= None:
        print("          Clustering(s) ",idx_cluster," similarity matrix:")
    else:
        print("          Clustering",len(centers)," internal similarity matrix")
    print()
    col_max=len(centers)
    if col_max>24:
        col_max=24
    this_idx=0
    count_idx=0
    print("  Vs  ",end='')
    for n in range(col_max):
        count_idx+=1
        if idx_cluster!= None and count_idx>idx_cluster[this_idx]:
            this_idx+=1
            count_idx=1
        print("  {:<2}|".format(count_idx),end='')
    print()
    print("      ",end='')
    count_idx=0
    this_idx=0
    for n in range(col_max):
        count_idx+=1
        if idx_cluster==None:
            print("    |",end='')
        else:
            if count_idx>idx_cluster[this_idx]:
                this_idx+=1
                count_idx=1        
            print(" g{:<2}|".format(idx_cluster[this_idx]),end='')
            
    print()
    
    this_idx=0
    count_idx=0
    for n, center1 in enumerate(centers):
        centers_sim_values.append([])
        count_idx+=1
        if n<col_max:
            if idx_cluster==None:
                print(" {:>2}  ".format(count_idx), end='')
            else:
                if count_idx>idx_cluster[this_idx]:
                    this_idx+=1
                    count_idx=1        
                print("{:>2}g{:<2}".format(count_idx,idx_cluster[this_idx]),end='')
                
        for m, center2 in enumerate(centers):
            sim_val=sim(center1,center2)
            if m!=n:
                centers_sim_values[n].append(sim_val)
            if m==n and n<col_max:
                print("  -->",end='')
                continue
            if m<n and n<col_max: 
                print("  |  ",end='')
                continue
            if m<col_max and n<col_max:
                print("|{:>4.1f}".format(sim_val), end='')
        if n<col_max: print("|")
    print("     "+"  |  "*col_max)
    print("avg  ",end='')        
    for n in range(col_max):
        print("{:4.1f} ".format(np.mean(centers_sim_values[n])),end='')
    print()
    print("std  ",end='') 
    for n in range(col_max):
        print("{:4.1f} ".format(np.std(centers_sim_values[n])),end='')
    print()
    print("max  ",end='') 
    for n in range(col_max):
        print("{:4.1f} ".format(np.max(centers_sim_values[n])),end='')
    print()
    print("min  ",end='') 
    for n in range(col_max):
        print("{:4.1f} ".format(np.min(centers_sim_values[n])),end='')
    print("\n")
    print("   avg tot: ",np.mean(centers_sim_values))
    print("   std tot: ",np.std(centers_sim_values))
    return centers_sim_values


# %%
want_to_tell=True


# %%
# centers_2=[]
# centers_2+=list(CC[str(2)])
# centers_2+=list(CC[str(4)])
# centers_2+=list(CC[str(8)])

# centers_sim_values_2=tab_CC_sim(centers_2,idx_cluster=[2,4,8], tell=want_to_tell)


# %%
# centers_cust=[]

# # idx_cluster=None
# idx_cluster=[4,5,20]
# for n in idx_cluster:
#     centers_cust+=list(CC[str(n)])
# # centers_cust+=list(CC[str(4)])
# # centers_cust+=list(CC[str(7)])

# centers_sim_values_cust=tab_CC_sim(centers_cust,idx_cluster=idx_cluster, tell=want_to_tell)


# %%
# refresh_parameters()
# ANALYSIS OF CLUSTERING SELECTION (suggested max_cl_dimension one)#######
# CHOSE A CLUSTERING TO ANALYZE
CC_cus=CC[str(cls_par['CC'])]



# #########################################################################
centers=CC_cus
# centers=centers_cust
centers_sim_values=tab_CC_sim(centers, tell=True)


# %%
from mpl_toolkits.mplot3d import Axes3D

x=[]
y=[]
Values=[]
N_cluster=len(centers)
    
for n, cent1 in enumerate(centers):
    Values.append([])
    for m, cent2 in enumerate(centers):
        if n<=m:
            Values[n].append(0)
        else:
            Values[n].append(sim(cent1,cent2))
# sorted_values=[]
# for i,raw in enumerate(Values):
#     sorted_values.append(sorted(Values[n]))


x=range(N_cluster)
y=range(N_cluster)
Values=np.array(Values)
hf = plt.figure()
ha = hf.add_subplot(111, projection='3d')

X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
# help(ha.plot_surface)

ha.plot_surface(X, Y, Values)


plt.show()


# %%

hist=[]
bins=10
for n, raw in enumerate(centers_sim_values):
    hist.append([])
#     for m in range(len(Values)):
    for i in range(-bins,bins):
#         count=Values[n].count(lambda x: x>=i/bins and x<(i+1)/bins)
        count=0
        for m, el in enumerate(raw): 
            if el>=i/bins and el<(i+1)/bins: 
                count+=1
        hist[n].append(count)
#         print(i/bins,":", count,end='\t')
#     print()
    
x=range(N_cluster)

y=range(-bins,bins)
hist_data=np.array(hist)
hf = plt.figure(figsize=(20,10))
ha = hf.add_subplot(111, projection='3d')
# for i in range(N_cluster):
#     print(len(x), len(y), len(hist_data), len(hist_data[i]))
Y, X = np.meshgrid(y, x)  # `plot_surface` expects `x` and `y` data to be 2D
# help(ha.plot_surface)

# dY= 1
# dX= 1
# dZ= 0.1
# ha.bar3d(Y,X, hist_data, dY,dX,dZ,)

ha.plot_surface(Y,X, hist_data)
# plt.fullfig(hf)
# plt.show()


# %%

means=[]
for m, raw in enumerate(hist_data[0]):
    means.append(0)
    for n, el in enumerate(hist_data):
        means[m]+=hist_data[n][m]
    means[m]/=len(hist_data)

    
graph_name=" Cluster similarity"    
fig, graph = plt.subplots()

graph.plot(range(-bins,bins),means)
graph.set(xlabel='Cluster Similarity', ylabel=' Average # Occurrences',
       title=graph_name)
graph.grid()

this_pathname_save_img=pathname_save_img+graph_name+'.png'
if CLUSTER_SAVE_IMG:
    fig.savefig(this_pathname_save_img)
    


# %%
print()
print("     The model has learned that {} things are the most important: ".format(len(centers)))
for vecs in centers:
    print(nearest_word(vecs,k=1)+", ", end='')
# print(" are really important things")


# %%
# COMPUTE WORD SIMILARITIES WITH CLUSTER CENTERS

# use to compute best n_top similarities with cluster's centers (using in topn_clust_sim)
def word_vs_clustering(vector,cluster=C_max, n_top=5,sort=True, wv=default_vects):
    if type(vector) is str:
        vector=wv[vector]
    VC_sim=[]
    for key in cluster:
        center=cluster[key]
        sim_with_center=(key, cos_sim(vector,center))
        sim_with_Cwords=[ (word_clust[0], cos_sim(vector,word_clust[0])) for word_clust in topn_sim(center,n=n_top,vects=wv) ]
        VC_sim.append((sim_with_center,sim_with_Cwords))
    if sort:
        VC_sim=sorted(VC_sim,key= lambda x:x[0][1], reverse=True)
    return VC_sim
WvC=word_vs_clustering
# it is a list of tuple of(similarity with cluster center, list of[ tuple of(word, sim)])


# use to display similarity of a word/vector with a cluster's centers (uses WvC)
def display_clust_sim(vector,cluster=C_max,n_cl=5,n_wd=5):
    
    if type(vector) is str:
        print("    \""+vector+"\" projections:\n")
    projections=word_vs_clustering(vector,cluster=cluster,n_top=n_wd, sort=True)
    for n, proj in enumerate(projections):
        if n>=n_cl: break
        print()
        print("     Similarity with cluster",proj[0][0],"=",proj[0][1])
        print("       most similar cluster word: ",end='')
        for word, sim_val in proj[1]:
            print(word+'('+str(sim_val)+')', end=', ')
        print(", ...")
tCs=display_clust_sim

# DEPRECATED use to project word from 
# def projection(vector, centers=centers, normalize=False):
# #     centers needs to be a list of centers array
#     projs=[]
#     for center in centers:
#         projs.append(sim(vector,center))
# #     print(len(projs))
#     if normalize:
#         projs/=norm(projs)
#     return np.array(projs)               
# Pj=projection    

# def cos_sim(a,b):
#     return cosine_similarity([a],[b])

# model1 must be dictionary of {word:vector}
# model2_base must be of same lenght of model1 vectors
# model2_base must be a list of arrays
# return an array of new vectorization of the word
def word_transform(word_or_vec_1, model1, model2_base, metric_f=cos_sim,normalize=False):
    if type(word_or_vec_1) is str:
        vect=model1[word_or_vec_1]
    else:
        vect=word_or_vec_1        
        
    transformed_vector=[]    
    for base2_vector in model2_base:
        value=float(metric_f(vect, base2_vector))
        transformed_vector.append(value)
    if normalize:
        transformed_vector/=norm(transformed_vector)
    return np.array(transformed_vector)
def Pj(word):
    return word_transform(word, 
                          model1=default_vects, 
                          model2_base=CC_cus, 
                          metric_f=cos_sim,
                          normalize=False)

# model2_base must be of same lenght of model1 vectors
# model2_base must be a list of arrays
def model_transform(model2_base, model1=default_vects, metric_f=cos_sim, mode='GS', display=True):

    tot_words=len(model1.vocab)
    if mode=='dict':
        new_vocab={}
    if mode=='GS':
        new_model=Word2VecKeyedVectors(len(model2_base))
    stepping=-1
    for i, word in enumerate(model1.vocab):
        if display:
            stepping=perc_compl(i,tot_words-1,stepping, step=0.01, 
                                text="     {:<6} words transformed:".format(i+1))
        new_vector=word_transform(word, model1, model2_base, metric_f=metric_f,normalize=False)
        if mode=='dict':
            new_vocab[word]=new_vector
        if mode=='GS':
            new_model[word]=new_vector

    if mode=='dict':
        return new_vocab
    if mode=='GS':
        return new_model

# non penso che la userò
# def gensim_to_dict(model, display=False):
#     my_dict
#     for word in model.wv:
#         my_dict[word]=model.wv[word]
#     return my_dict


# %%
# MY VARIABLES NOW
# Cluster_Centers: all clusteringS as {N: (all clustering(N) N centers)} 
# CC=Cluster_Centers
# BackupCC=Cluster_Centers of some iteration
# C_max dictionary as {n: cluster center}

# CC_cus= lista di una base scelta, su cui faremo analisi
# centers=lista dei centri clustering scelto sopra (più grande)
# centers_sim_values=lista di similitudini interne di quel clustering


# %%
# printacose(word_transform('house',model,centers))
display_clust_sim('man')


# %%
if flag_transform is False:
    print("     Transforming emb. words on base derived from Clustering", len(CC_cus) )
    transformed_vects=model_transform(CC_cus)
    flag_transform=True


# %%
if CLUSTER_SAVE_MODEL:
    print("     Saving Cluster projections as",name_save_vec+".vectors" )


# %%
if CLUSTER_SAVE_MODEL:
    transformed_vects.save(pathname_save_vec+".vectors")


# %%
summ=0
for i in range(1,len(C_max)):
    summ+=norm(C_max[str(i)])
summ/=180

summ


# %%
sim_trashold=0.85
found_relations=[]
for i in range(1,len(C_max)):
    for j in range(1,len(C_max)):
        if j<=i: continue
        sim_val=cos_sim(C_max[str(i)],C_max[str(j)])
        if sim_val >= sim_trashold:
            found_relations.append(((i,j),sim_val))


# %%
print("     found ",len(found_relations)," top similarities between clusters (above {:.2f}): \n".format(sim_trashold))

for f_r in found_relations:
    print(f_r[0], "similarity value:", f_r[1])
    center_1=f_r[0][0]
    center_2=f_r[0][1]
    print(" Cluster ",center_1,end=': ')
    for el in topn_sim(C_max[str(center_1)]):
        print(el[0], end=', ')
    print("...")
    print(" Cluster ",center_2,end=': ')
    for el in topn_sim(C_max[str(center_2)]):
        print(el[0], end=', ')
    print("...")
    print()
    


# %%
def_clust=C_max
found_relations=[]

from numpy import random as rand
def random_center(centers=def_clust):
    N_centers=len(C_max)
    r_val=str(rand.randint(1,N_centers+1))
    return r_val


# %%
range_sample=10000
sim_trashold=0.7

for n in range(range_sample):
    i=random_center()
    j=random_center()
    k=random_center()
    l=random_center()
    if i==j or i==k or i==l or j==k or j==l or k==l:
        continue
    relation_1=def_clust[i]-def_clust[j]
    relation_2=def_clust[k]-def_clust[l]
    sim_val=cos_sim(relation_1,relation_2)
    if sim_val>=sim_trashold:
        found_relations.append( ((i,j,k,l),(relation_1,relation_2),sim_val) )


# %%
N_sim_found=len(found_relations)
print("     found ",N_sim_found," top similarities between clusters (above {:.2f}, in sample of {}): \n".format(sim_trashold,range_sample))
for f_r in found_relations:
    print(" ({}-{}) vs ({}-{})".format(f_r[0][0],f_r[0][1],f_r[0][2],f_r[0][3]), "similarity value:", f_r[2])
    for n, idx in enumerate(f_r[0]):
        print(" Cluster {:3}".format(idx),end=': ')
        for el in topn_sim(C_max[idx]):
            print(el[0], end=', ')
        print("...")
    print()


# %%
word='car'
print('     word:',word)
print()
print("   model w2v:")
pp.pprint(topn_sim(word, vects=default_vects))
print()
print("   cluster{} version:".format(len(transformed_vects[word])))
pp.pprint(topn_sim(word, vects=transformed_vects))

print()
print("     norm in new representation:",norm(word_transform(word, model1=default_vects, model2_base=centers)))
tCs(word,n_cl=5,n_wd=5)


# %%
# amazingly slow
# 
# sim_trashold=0.9
# found_relations=[]
# tot_centers=len(CC_cus)


# for i,vect1 in enumerate(CC_cus):
    
#     for j,vect2 in enumerate(CC_cus):
#         if j<=i: continue
#         relation_1=vect1-vect2
#         stepping=-1
#         for k,vect3 in enumerate(CC_cus):
#             if k<=i: continue
#             for l,vect4 in enumerate(CC_cus):
#                 if l<=k: continue
#                 to_write=" i={:3}; j={:3}; k={:3}; l={:3}  ".format(i,j,k,l)
#                 stepping=perc_compl(j,tot_centers,stepping, step=0.01, text=to_write)
#                 relation_2=vect3-vect4
#                 sim_val=cos_sim(relation_1,relation_2)
#                 if sim_val>= sim_trashold:
#                     found_relations.append( ((i,j,k,l),sim_val) )


# %%
if COORDINATE_BLOCKS:
    switch_analysis_to('C')
    


# %%
# WORKING TREE SIMILARITIES OF ALL COMPUTED DIFFERENT CLUSTERIZATION
# compute and memorize conditional (tree) similarity between clusterizations

key_max=0
for key in CC:
    key_max=int(key)
# printacose(key_max,lng=False)
unknown_val=-5
VOID_MEMO=False

def want_value_of(N,M='indifferent'):
    if M=='indifferent':
        if str(N) in CC:
            return True
    else:
        if str(N) in CC and str(M) in CC:
            if M==N-1:               #CONDITION HERE (what you want to keep from all sim values)
                return True
    return False
        
def keyer(GoC,AoB,num, min100=True):
    return GoC+AoB+':'+"{:0>2}".format(num)

tree_CC_sim={}
for N in range(0, key_max+1):
    if VOID_MEMO or want_value_of(N):
        tree_CC_sim[keyer('g','A',N)]={}
    else:
        continue
    for  M in range(0, key_max+1): 
        if VOID_MEMO or want_value_of(N,M):
            tree_CC_sim[keyer('g','A',N)][keyer('g','B',M)]={}
        else:
            continue
        for K_n in range(N):
            tree_CC_sim[keyer('g','A',N)][keyer('g','B',M)][keyer('c','A',K_n+1)]={}
            for K_m in range(M):
                if VOID_MEMO and not want_value_of(N,M):
                    sim_val=unknown_val
                else:
                    sim_val=sim(CC[str(N)][K_n],CC[str(M)][K_m])
#                 if VOID_MEMO or sim_val!= unknown_val:  
                tree_CC_sim[keyer('g','A',N)][keyer('g','B',M)][keyer('c','A',K_n+1)][keyer('c','B',K_m+1)]=sim_val
                


# %%
# VERSION WITH LIST INSTEAD OF DICT 
# dictionary is more friendy for what value you want to recollect from N,M
# 
# tree_centers_sim=[]
# for N, Nkey in enumerate(CC):
#     tree_centers_sim.append([])
#     for M, Mkey in enumerate(CC):
#         if M==N-1:
#             for K_n, Ncenter in enumerate(CC[Nkey]):
#                 tree_centers_sim[N].append([])
#                 for Mcenter in CC[Mkey]:
#                     sim_value=sim(Ncenter,Mcenter)
#                     tree_centers_sim[N][K_n].append(sim_value)
    

# len(tree_centers_sim)


# %%
# g='g'
# c='c'
# a='A'
# b='B'
# pp.pprint(tree_CC_sim[keyer(g,a,5)])
# # printacose(tree_CC_sim,limit=10) 


# %%
print("    ---- fin")


# %%



