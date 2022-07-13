# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%


print("\n\n  {:#^55}".format(" Word Embedding "))
print()
print()

ME='BUILDER'


# %%
# ###################################################### SETTINGS

from Corpus_Manager import *


# %%
# ##############################  RUNNING OPTIONS ##################################
# <<<<<<<<<<<<<<<<<<<<<<<<  takes them from Wemb_par import   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# # ------------------------ path settings
# path_user='C:\\Users\\X556U\\Desktop\\Wemb\\'
# USE_SHARED_FOLDER=True

# # ------------------------ Dataset settings
# Dataset='lee'
# dataset_format='cor'
# dataset_compression=None

# ----------------------- file names
# SL_NAME='def'
# ^^^^^ set to 'def' for automatic parameter-based names construction
# ^^^^^ set to some other string for some other name. Some are pre-implemented:
# SL_NAME='google_news' #will load pretrained 3M vectors on google news

# ----------------------- pre-preprocessing
# USE_DEF_ALG=True
# Alg_default='split&token'
# PERC_DATA_PROCESS=100
# ENCORPORATE_DOCUMENTS=False

# # ------------------------- Vocab Options
# LEMMATIZE=False
# LOWER=True
# NONSTOP=False
# NOPUNCT=True
# MANUAL_REMOVE=False
# mr_set=['\n']

# # ------------------------- Corpus Options
# LEMMATIZE_CORP_CONDITIONAL=True 
# LOWER_CORP_CONDITIONAL=True
# # lemm. condition of corpus can be different than vocab cond.
# # but not useful and not implemented
# PRESERVE_MORPH=True
# SUBST_NONVOC_W=True
# text_to_subst=''

# # ------------------------ Model and Training options
# min_freq=5
# vec_size=150
# architecture='CBOW'
# # ^^^^^ not yet implemented
# window_size=12
# n_epochs=20

# # ----------------------- Batching Options
# BATCHING=True
# batch_size=500
# # N_batches=1000

# # ----------------------- use some predef. parmeter choice
# predef_settings=None    
# # ^^^^^  'A6'
# customize_from_predef=False

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ uncomment to manually override imported parameters


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

# -------------------------------------------------save/load settings
# LOAD_MODEL=BUILDER_LOAD_MODEL {Dnn}
SAVE_MODEL=BUILDER_SAVE_MODEL

# --------------------------------------- names settings
name_save_vec=give_name_to(ME,'SAVE','VEC', sel_par,SL_NAME,cls_par)

# --------------------------------------- pathnames settings
pathname_save_vec=path_save_vectors+name_save_vec

# ---------------------------- memory save/safety flag
MEMORIZE_CORPUS=True

flag_corpus=False
flag_batch=False
flag_vocab=False
flag_backup=False
flag_model=False


# %%
print(name_save_vec)
# ^^^^^NAMES CHECKS

# %%
print(pathname_save_vec)
# ^^^^^PATHNAMES CHECKS

# %%
# ####################### initializing other internal params from given ones

print()
print("     Building model from Dataset: ")
print("          ",pathname_corpus)


# %%
def display_settings():
    print()
    print("  {:#^55}".format(" SETTINGS INFO "))
    print()
    print("  Using parameters:",using_params)
    print()
    print("  Corpus file: ",pathname_corpus)
#     print("  Number of documents in file: ",tot_doc)
    print("  Using {:2.2f}% of total data ".format(float(PERC_DATA_PROCESS)))
    print("  Using strategy: \""+Alg+"\"")
    print()
#     print("  Number of word in file (by simple splitting) :",tot_word)
#     print("  Number of word will be in use ( {:2.2f}% ) approx.".format(float(PERC_DATA_PROCESS)),
#           int(tot_word*PERC_DATA_PROCESS/100))
    print("  {:-^55}\n".format(" vocabulary building "))
    print("  Lemmatization of vocabulary words: ",LEMMATIZE)
    print("  Lowercase tr. of vocabulary words: ",LOWER)
    print("  Excluding stopwords from vocabulary: ",NONSTOP)
    print("  Excluding punctuat. from vocabulary: ",NOPUNCT)
    print("  Manual removing: ", str(MANUAL_REMOVE)+", from set: ",mr_set)
    print()
    print("  {:-^55}\n".format(" corpus preprocessing "))
    print("  Lemmatization of corpus: ",LEMMATIZE)
    print("  Lowercase tr. of corpus: ",LOWER)
    print("  Preserving syntax of corpus: ", PRESERVE_MORPH)
    if PRESERVE_MORPH:
        print("  Substitution of word not in vocabulary:",SUBST_NONVOC_W, end='')
        if SUBST_NONVOC_W:
            print(", with text: \'"+text_to_subst+"\'")
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
    
display_settings()
# ---------------------------- image saving path
# potrei automatizzare la costruzione del nome dell'immagine dai parametri
if SAVE_MODEL:
    print()
    print("     At the end of iteration, model and vectors will be saved in\\as:")
    print("          ",pathname_save_vec+".[model/vectors]")


# %%
# tk_tot_word=0
# tokenized_docs=[]
# N_doc=0
# dict_f={}
# i=0
# processed_tot_doc=0
# stepping_data=-1

# print()
# print("     Scanning document: ",end='')
# print("(using {:2d}% of dataset)".format(PERC_DATA_PROCESS))

# Corpus.count(replace=True)
# # print(Corpus.tot_wrd)


# %%
# ############################################### CORPUS MANAGER FOR VOCAB BUILD AND TRAINING

Corpus=Corpus_Manager(pathname_corpus,
                      corp_format=dataset_format,
                      compression=compression)

Corpus.set_tokenizer(tokenizer)
if Alg=='tokenize':
    Corpus.set_preprocess(tokenize=True, split=False)
if Alg=='split&token':
    Corpus.set_preprocess(tokenize=True, split=True)
    
Corpus.set_preprocess(lemmatize=LEMMATIZE,
                      lower=LOWER,
                      remove_stop=NONSTOP,
                      remove_punct=NOPUNCT,
                      remove_manual=MANUAL_REMOVE,
                      remove_manual_set=mr_set,
                      preserve_morphology=PRESERVE_MORPH,
                      substitute_removed=SUBST_NONVOC_W,
                      text_to_substitute=''
                       )                     
Corpus.display_iterator=True  


# %%
# Corpus.count(replace=True)
Corpus.display_doc_propriety()
Corpus.display_preprocess_settings()


# %%

print()
print("     Vocabulary Building: ",end='')
print("(using {:2d}% of dataset)".format(PERC_DATA_PROCESS))



def to_be_invocab(word):
    if word == text_to_subst:
        return False
    return True
# ^^^^^ additional evaluation needed after corpus manager process

corpus_holder=Corpus_Holder()
vocab={}

for doc_line in Corpus:
    if MEMORIZE_CORPUS:
        corpus_holder.hold_doc(doc_line)
    for word in doc_line:
        if not to_be_invocab(word):
            continue
        if word in vocab:
            vocab[word]+=1
        else:
            vocab[word]=1
            
tmp_vocab=sorted(vocab, key= lambda x:vocab[x], reverse=True)
vocab={word: vocab[word] for word in tmp_vocab}
# vocab


# %%
# # ######################################## Preparing Corpus for training 

# def to_be_trained(word):
#     if word not in vocab:
#         return False
#     return True
# # ^^^^^ when vocab is built, will return False if given word not in vocabulary

# Corpus.set_preprocess(keep_word_eval=to_be_trained)

if MEMORIZE_CORPUS:
    corpus_train=corpus_holder
else:
    corpus_train=Corpus


# %%
# older version
# keeping it as reference


# print()
# print("     Training Model:",end='')
# if flag_model:
#     del model
#     print("     (previous iteration deleted)", end='')
# model=Word2Vec(min_count=min_freq,
#                size=vec_size,
#                window=window_size,
#                cbow_mean=1,           #1 for using mean of cbow vects, def=1
#                compute_loss=1,
#                negative=5,           #negative sampling  if >0, def =5
#                hs=0)                 #hierarchical softmax for hs=1, def =0
# # mod.build_vocab(sentences)  # prepare the model vocabulary
# model.build_vocab_from_freq(vocab)
# flag_vocab=True
# # train word vectors
# model.train(sentences=train_corpus, 
#           total_examples=train_n_docs, 
#           total_words=train_n_word,
#           epochs=n_epochs,
# )  
# flag_model=True
# vocab_size=len(model.wv.vocab)
# print(" Done")
# print()
# print("     N word vectors:", vocab_size)
# print("     Total mod. par:", vocab_size*vec_size)


# %%
print()
print("     Initializing Model", end='')
if flag_model:
    # del model
    print(" (previous iteration deleted)", end='')
print(" ...")
model=Word2Vec(min_count=min_freq,
               size=vec_size,
               window=window_size,
               cbow_mean=1,           #1 for using mean of cbow vects, def=1
               compute_loss=True,
               negative=5,           #negative sampling  if >0, def =5
               hs=0)                 #hierarchical softmax for hs=1, def =0


# %%
mod_vocab_size=0
for i,word in enumerate(vocab):
    if vocab[word]<min_freq:
        mod_vocab_size=i+1
        break     
print("     For min_freq of", min_freq,", vector size of",vec_size,", with a vocabulary of", mod_vocab_size, end='')
print(", estimated memory is:\n     ", model.estimate_memory(vocab_size=mod_vocab_size)  )  


# %%
print("     Initializing Model's vocabulary ...")
model.build_vocab_from_freq(vocab)
flag_vocab=True


# %%
if flag_backup is False:
    Backup_M_Init=model
    flag_backup=True
# ^^^^^ empty (but memory costly) just initialized model
# ^^^^^ (for training on different hyperparameter of same instance)


# %%
print()
print("     Training Model:")
print()
model.train(sentences=corpus_train, 
          total_examples=Corpus.tot_doc, 
          total_words=Corpus.tot_wrd,
          epochs=n_epochs)
flag_model=True


# %%
# TO BE TESTED ................................
# lrate_start=0.025
# lrate_end=0.0001
# tot_iterations=n_epochs*N_batches
# DECAY_OVER_BATCHES=False
# # could implement a counting/updating strategy for computing curr_lrate start and end
# # but probably do not need
# print()
# print("     Training Model:")
# stepping=-1
# for curr_epoch in range(n_epochs):
#     curr_lrate_start=lrate_start-(lrate_start-lrate_end)*(curr_epoch/n_epochs)
#     curr_lrate_end=lrate_start-(lrate_start-lrate_end)*(curr_epoch+1/n_epochs)
#     to_write='     Training epoch {:2d} of {:2d} (L.rate is {:.4f}) : '.format(curr_epoch+1, n_epochs,curr_lrate_start)
#     stepping=perc_compl(curr_epoch,n_epochs,stepping, text=to_write)
#     if not DECAY_OVER_BATCHES:
#         model.train(sentences=train_corpus, 
#                     total_examples=train_n_docs, 
#                     total_words=train_n_word,
#                     epochs=1,
#                     start_alpha=curr_lrate_start, 
#                     end_alpha=curr_lrate_end)     

#     if DECAY_OVER_BATCHES:
# # too slow ^^^^^
#         for iteration, batch in enumerate(batches):
#             curr_iteration=curr_epoch*N_batches+iteration
#             curr_lrate_start=lrate_start-(lrate_start-lrate_end)*(curr_iteration/tot_iterations)
#             curr_lrate_end=lrate_start-(lrate_start-lrate_end)*(curr_iteration+1/tot_iterations) 
#             print(curr_iteration, end='\t')
#             model2.train(sentences=batch, 
#                          total_examples=1, 
#                          total_words=batch_size,
#                          epochs=1,
#                          start_alpha=curr_lrate_start, 
#                          end_alpha=curr_lrate_end)

# flag_model=True

# ^^^^^^^^^^^^ TESTING LEARNING RATE CUSTOMIZATION


# %%
vocab_size=len(model.wv.vocab)
print()
print("     N word vectors:", vocab_size)
print("     Total mod. par:", vocab_size*vec_size)
# help(model)
# model.get_latest_training_loss() non funziona o non ho capito qualcosa (???)


# %%
if SAVE_MODEL:
    print("     Saving Model's vectors as",name_save_vec+'.vectors')


# %%
# #################################################### SAVING .model HERE
if SAVE_MODEL:
    model.wv.save(pathname_save_vec+'.vectors')


# %%
if COORDINATE_BLOCKS:
    switch_analysis_to('M')


# %%



