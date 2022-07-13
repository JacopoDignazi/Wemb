# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# NB structure of every block will be:
#     - importing from this file.py
#     - possibility of locally customization of params
#     - initializing all internal parameters
#     - loading
#     - working
#     - saving
#     - actions to allow coordination between blocks


# %%
print("     Importing parameters...")


# %%
# ############################################################ GLOBAL (most significant) SETTINGS

# ------------------------ path settings
path_user='C:\\Users\\X556U\\Desktop\\Wemb\\'
USE_SHARED_FOLDER=True

# ------------------------ Dataset and preprocessing settings
Dataset='wiki'
# ^^^^^ pre-built 'lee','wiki','text8'
dataset_format='cor'
dataset_compression=None

# ----------------------- file names
SL_NAME='def'
# ^^^^^ set to 'def' for automatic parameter-based names construction
# ^^^^^ set to some other string for some other name. Some are pre-implemented:
# SL_NAME='google_news' #will load pretrained 3M vectors on google news

# ------------------------ Model and Training settings
min_freq=5
vec_size=100
architecture='CBOW'
# ^^^^^ not yet implemented
window_size=10
n_epochs=10
# ----------------------- Batching settings
BATCHING=True
batch_size=500
# N_batches=1000

# ----------------------- clustering settings
CC_number=None
# ^^^^^ set clustering dimension
# ^^^^^ shared among clustering and analysis

# ----------------------- use some predef. parmeter choice
predef_settings=None    
# ^^^^^ av. settings: 'A6'
customize_from_predef=False

# ----------------------- coordination settings
COORDINATE_BLOCKS = True
SAVE_ALL=True
# set to not bool if unwanted ('unw'), while False means nothing will be saved


# %%
# ############################################################  BUILDER_SETTINGS

# ------------------------- Preprocessing settings
USE_DEF_ALG=True
Alg_default='split&token'
PERC_DATA_PROCESS=100
ENCORPORATE_DOCUMENTS=False

# ------------------------- Vocab settings
LEMMATIZE=False
LOWER=True
NONSTOP=False
NOPUNCT=True
MANUAL_REMOVE=False
mr_set=['\n']

# ------------------------- Corpus settings
LEMMATIZE_CORP_CONDITIONAL=True 
LOWER_CORP_CONDITIONAL=True
# lemm. condition of corpus can be different than vocab cond.
# but not useful and not implemented
PRESERVE_MORPH=True
SUBST_NONVOC_W=True
text_to_subst=''


# %%
# ############################################################ S/L MANUAL SETTINGS
# ^^^^^ if COORDINATE_BLOCKS is True, parameters will be redefined below to allow coordination

BUILDER_LOAD_MODEL = False
BUILDER_SAVE_MODEL = True
# BUILDER_SAVE_IMG

ANALYST_LOAD_CLUSTER=False
ANALYST_SAVE_IMG=False
ANALYST_SAVE_TXT=True

# CLUSTER_LOAD_CLUSTER=False
CLUSTER_SAVE_MODEL = True
CLUSTER_SAVE_IMG = True


# %%
# ############################################################ PATHS MANUAL SETTINGS
# ^^^^^ if USE_SHARED_FOLDER is True, parameters will be all set to path_user below

path_load_dataset=path_user+'Dataset\\'
# ^^^^^ if dataset requires, will reset to datapath(...)

path_load_vectors=path_user+'Vectors\\'
path_save_vectors=path_user+'Vectors\\'
path_save_img=path_user+'Img\\'
path_save_txt=path_user+'Txt\\'
path_manager=path_user+'Coor_mng\\'
# could be decomposed for different blocks {Dnn}


# %%
# ############################################################################# NO SETTINGS BELOW
# ############################################################### just parameters initializations


# %%
# ##################################################### PREDEFINED PARAMETERS
using_params='Custom'
def set_params_alpha6():
    global PERC_DATA_PROCESS
    PERC_DATA_PROCESS=100
    global window_size
    window_size=8   
    global n_epochs
    n_epochs=20  
    global vec_size
    vec_size=80
    global min_freq
    min_freq=5
    global SAVE_FIG
    SAVE_FIG=True
    global using_params
    using_params='alpha6'
    # refresh_parameters()
    # prefix=prefixer()
    print("    ... imported parameters from instance ",using_params)
if predef_settings=='A6':    
    set_params_alpha6()
    
def customize_from_predefined_params():
# insert custom params here ##########
    global vec_size
    vec_size=150  
    global window_size
    window_size=12
# #####################################        
    # global img_text_prefix
    # img_text_prefix=custom_img_text_prefix    
    global using_params
    using_params='Custom__'+using_params  
    # refresh_parameters()
    # prefix=prefixer()
    print("     ... customizing parameters")
if customize_from_predef:
    customize_from_predefined_params()


# %%
# ########################################## COORDINATION MANAGING

if COORDINATE_BLOCKS:
    BUILDER_LOAD_MODEL = False
    BUILDER_SAVE_MODEL = True

    ANALYST_LOAD_CLUSTER=False

    CLUSTER_SAVE_MODEL = True
    
    path_load_vectors=path_save_vectors
    
if type(SAVE_ALL) is bool:
    ANALYST_SAVE_IMG=SAVE_ALL
    CLUSTER_SAVE_IMG=SAVE_ALL
    
    ANALYST_SAVE_TXT=SAVE_ALL


# %%
# ########################################## FOLDER PATH MANAGING

if USE_SHARED_FOLDER:
    path_load_dataset=path_user
#     ^^^^^ will be redefined below if dataset requires datapath(...)
    path_load_vectors=path_user
    path_save_vectors=path_user
    path_save_img=path_user
    path_save_txt=path_user
    path_manager=path_user


# %%
# ########################################## names managing 

sel_par={}
cls_par={}

sel_par['Vsz']=vec_size
sel_par['Wsz']=window_size
sel_par['mfq']=min_freq
sel_par['bsz']=batch_size
sel_par['nEp']=n_epochs    

if CC_number is None:
    cls_par['CC']=sel_par['Vsz']-1
else:
    cls_par['CC']=CC_number

# if imported, can only modify local variables
# def local_reset_names():
#     global sel_par
#     sel_par['Vsz']=vec_size
#     sel_par['Wsz']=window_size
#     sel_par['mfq']=min_freq
#     sel_par['bsz']=batch_size
#     sel_par['nEp']=n_epochs    
    
#     global cls_par
#     if CC_number is None:
#         cls_par['CC']=sel_par['Vsz']-1
#     else:
#         cls_par['CC']=CC_number


imported_sel_par=sel_par
imported_cls_par=cls_par
# Wemb blocks will be modifying sel_par and cls_par
# but importing will keep memory of variables defined here
def reload_par():
    return imported_sel_par, imported_cls_par   

def_pre=Dataset

def general_prefixer(sel_par, sl_name=SL_NAME, cls_par=None, delimit='!',middle='=',extr=['[',']'],pre=def_pre,post=''):# other_string=None):
# ^^^^^ if SL_NAME is 'def' will auto-generate params-based names;
# ^^^^^ otherwise will make a name from SL_NAME and other string
    if sl_name=='def':
        string=pre+'-'
        string+=extr[0]
        for i,word in enumerate(sel_par):
            if i!=0:
                string+=delimit
            string+=word+middle+str(sel_par[word])
        string+=extr[1]
    else:
        string=sl_name
    if cls_par != None:
        string+=extr[0]
        for i,word in enumerate(cls_par):
            if i!=0:
                string+=delimit
            string+=word+middle+str(cls_par[word])
        string+=extr[1]
    string+=post  
    return string
# ^^^^^ sloppy but it works
# could define a give_name_to(who, for_what, sl_name, list_dict_strval, other_string)


# %%
# ########################################## names giver
def give_name_to(who, for_, what, sel_par, sl_name, cls_par,load_cluster=False):
    if who is 'BUILDER':
        if for_ is 'SAVE':
            return general_prefixer(sel_par,sl_name=sl_name)

    if who is 'ANALISYS':
        if not load_cluster:
            cls_par=None
        if for_ is 'LOAD':
            return general_prefixer(sel_par,sl_name=sl_name, cls_par=cls_par)
        if for_ is 'SAVE':
            if what is 'IMG' or what is 'TXT':
                this_post='-'
            else:
                this_post=''
            return general_prefixer(sel_par,sl_name=sl_name,cls_par=cls_par, post=this_post)

    if who is 'CLUSTERING':
        if for_ is 'LOAD':
            return general_prefixer(sel_par,sl_name=sl_name, cls_par=None)
        if for_ is 'SAVE':
            if what is 'IMG' or what is 'TXT':
                this_post='-'
            else:
                this_post=''
            return general_prefixer(sel_par,sl_name=sl_name, cls_par=cls_par, post=this_post)
    else:
        print("     Unexpected name-call \a\a")



# %%
# ########################################## COMUNICATION BETWEEN BLOCKS

text_clustering="analysis_of_clustering_transform"
text_embedding= "analysis_of_word2vec_embedding"

switch_file_name="Toggle_analysis.txt"

def switch_analysis_to(mode):
    if mode=='M':
        to_write=text_embedding
    if mode=='C':
        to_write=text_clustering    
    switch = open(switch_file_name,"w")
    switch.write(to_write)
    switch.close()
    
def get_switch_analysis():
    switch=open(switch_file_name,"r")
    text_found=switch.readline()
    if text_found==text_embedding:
        return 'M'
    if text_found==text_clustering:
        return 'C'
    switch.close()


# %%
# ############################################################ shared functions importing
 
print()
print("     Importing Gensim utils...")
import gensim as gs
from gensim.models import Word2Vec
from gensim.models.word2vec import Word2VecVocab
from gensim.models.keyedvectors import Word2VecKeyedVectors
from gensim.test.utils import datapath
from gensim import utils
import gensim.downloader as api
import gzip

# ############################################################ some utils
import pprint as pp

def printacose(cosa, text=None, lng=True, ppr=False, limit=None):
    print("-"*40)
    if text != None:
        print(text+":")
    if limit == None:
        if ppr==False: print("cosa:",cosa)
        else: 
            print("cosa:")
            pp.pprint(cosa) 
    else:
        for i, element in enumerate(cosa):
            if i==limit: break
            print(i, element)
    print("tipo:",type(cosa))
    if lng: print("lngt:",len(cosa))
    print("\n")
    
def perc_compl(num, den, last_step, text=None,step=0.1):
    perc=((num+1)/den)*100
    if int(perc)==100 or perc>=(last_step+step):
        last_step+=step
        to_write=''
        if text!= None:
            to_write+=text
        to_write+=" [{:-<20.20}]".format("="*int(perc/5)+">")
        to_write+="  {:<2.2f} %  completed".format(perc)
        print('\r'+to_write,end='')
        if int(perc)==100: print(end='\n\n')
    return last_step
#needs to be called as stepping=perc_compl(i,up_to,last_step=stepping)


# %%
# def display_settings():
#     print()
#     print("  {:#^55}".format(" SETTINGS INFO "))
#     print()
#     print("  Using parameters:",using_params)
#     print()
#     print("  Corpus file: ",corpus_path)
#     print("  Number of documents in file: ",tot_doc)
#     print("  Using {:2.2f}% of total data ".format(float(PERC_DATA_PROCESS)))
#     print("  Using strategy: \""+Alg+"\"")
#     print()
#     print("  Number of word in file (by simple splitting) :",tot_word)
#     print("  Number of word will be in use ( {:2.2f}% ) approx.".format(float(PERC_DATA_PROCESS)),
#           int(tot_word*PERC_DATA_PROCESS/100))
#     print()
#     print("  {:-^55}\n".format(" vocabulary building "))
#     print("  Lemmatization of vocabulary words: ",LEMMATIZE)
#     print("  Lowercase tr. of vocabulary words: ",LOWER)
#     print("  Excluding stopwords from vocabulary: ",NONSTOP)
#     print("  Excluding punctuat. from vocabulary: ",NOPUNCT)
#     print("  Manual removing: ", str(MANUAL_REMOVE)+", from set: ",mr_set)
#     print()
#     print("{:-^55}\n".format(" corpus preprocessing "))
#     print("  Lemmatization of corpus: ",LEMMATIZE)
#     print("  Lowercase tr. of corpus: ",LOWER)
#     print("  Preserving syntax of corpus: ", PRESERVE_MORPH)
#     if PRESERVE_MORPH:
#         print("  Substitution of word not in vocabulary:",SUBST_NONVOC_W, end='')
#         if SUBST_NONVOC_W:
#             print(", with text: \'"+text_to_subst+"\'")
#     print()
#     print("  {:-^55}\n".format(" TRAINING'S SETTINGS INFO "))
#     print("  Batching document:", True)
#     print("  Batch size:", batch_size)
#     print("  Number of epochs: ",n_epochs)
#     print()
#     print("  {:-^55}\n".format(" model parameter "))    
#     print("  Minimum frequency of word: ",min_freq)
#     print("  Size of vectors: ", vec_size)
#     print("  Architecture: ",architecture)
#     print("  Size of syntax window: ", window_size)
    
# display_settings()


# %%



# %%
# something to learn
# import io
# import os

# import gensim.models.word2vec
# import gensim.downloader as api
# import smart_open


# def head(path, size):
#     with smart_open.open(path) as fin:
#         return io.StringIO(fin.read(size))


# def generate_input_data():
#     lee_path = datapath('lee_background.cor')
#     ls = gensim.models.word2vec.LineSentence(lee_path)
#     ls.name = '25kB'
#     yield ls

#     text8_path = api.load('text8').fn
#     labels = ('1MB', '10MB', '50MB', '100MB')
#     sizes = (1024 ** 2, 10 * 1024 ** 2, 50 * 1024 ** 2, 100 * 1024 ** 2)
#     for l, s in zip(labels, sizes):
#         ls = gensim.models.word2vec.LineSentence(head(text8_path, s))
#         ls.name = l
#         yield ls


# input_data = list(generate_input_data())


