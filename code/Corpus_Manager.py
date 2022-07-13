# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from Wemb_Parameters import *


# %%
# ############################## DATASET PATH INIT SETTINGS
# ^^^^^ defaulted (pre-built datasets below

if Dataset=='lee':
    pathname_corpus = datapath('lee_background.cor')
    dataset_format='cor'
    compression=None

if Dataset=='wiki':
    pathname_corpus= path_load_dataset+'text8'
    dataset_format='txt'
    compression=None
    
if Dataset=='text8':
    pathname_corpus=api.load('text8', return_path=True)
    dataset_format='txt'
    compression='gz'

    


# %%
# ############################## CORPUS MANAGER

class Corpus_Manager(object):
    
    def __init__(self, 
                 path, 
                 corp_format='cor', 
                 compression=None, 
                 batch_size=500, 
                 tokenizer=None, 
                 tot_doc='undefined', 
                 tot_wrd='undefined'):
        
        self.path=path
        self.corp_format=corp_format
        self.compression=compression
        self.batch_size=batch_size
        
        self.reset_preprocess()
        
        self.set_tokenizer(tokenizer)
        self.tot_doc=tot_doc
        self.tot_wrd=tot_wrd
        
        self.display_iterator=True
    
    """An interator that yields sentences (lists of str)."""
    def __iter__(self):
        stepping=-1
        processed_wrd=0
        processed_doc=0

        for line in self.line_generator():
            processed_line, proc_words_inline = self.process(line)
            
            processed_doc+=1
            processed_wrd+=proc_words_inline
            if self.display_iterator:
                stepping=self.display_progress(processed_doc, processed_wrd, stepping)
            yield processed_line
        self.tot_doc=processed_doc
        self.tot_wrd=processed_wrd
            
    def opener(self, newline=None):
        if self.compression==None:
            return open(self.path, newline=newline)
        if self.compression=='gz':
            return gzip.open(self.path,'rt',newline=newline)
        
    def line_generator(self):
        if self.corp_format=='cor': #<--- assumes documents separated by '\n'
            for line in self.opener():
                yield line
        if self.corp_format=='txt': #<--- assumes plain text without '\n'
                                    #     {Bad} if '\n' are presents will treat them as different documents
                                    #       should implement a version with readline
            for text in self.opener():
                end_idx=0
                eof_reached=False
                while eof_reached is False:
                    start_idx=end_idx
                    for _ in range(self.batch_size):
                        end_idx=text.find(' ',end_idx)+1
                        if end_idx==0:
                            eof_reached=True
                            end_idx=len(text)
                            break
                    yield text[start_idx:end_idx]
                    
    def display_doc_propriety(self):
        print("     {:-^55}".format(" doc propriety "))
        print()
        print("     dataset path: ",self.path)
        print("     dataset format: ", self.corp_format)
        print("     dataset compression:",self.compression)
        print("     number of documents: ",self.tot_doc)
        print("     number of words: ",self.tot_wrd)
        print("     batch size (if needed):", self.batch_size)
        print()
        
          
    
    def set_preprocess(self, 
                       split='undefined',
                       lower='undefined',
                       tokenize='undefined',
                       lemmatize='undefined',
                       remove_stop='undefined',
                       remove_punct='undefined',
                       remove_manual='undefined',
                       remove_manual_set='undefined',
                       keep_word_eval='undefined',
                       preserve_morphology='undefined',
                       substitute_removed='undefined',
                       text_to_substitute='undefined'
                      ):
        if split!='undefined':
            self.split=split
        if lower!='undefined':
            self.lower=lower
        if tokenize!='undefined':
            self.tokenize=tokenize
        if lemmatize!='undefined':
            self.lemmatize=lemmatize
        if remove_stop!='undefined':
            self.remove_stop=remove_stop
        if remove_punct!='undefined':
            self.remove_punct=remove_punct
        if remove_manual!='undefined':
            self.remove_manual=remove_manual
        if remove_manual_set!='undefined':
            self.remove_manual_set=remove_manual_set
        if keep_word_eval!='undefined':
            if keep_word_eval is None:
                self.keep_word_eval= lambda x: x not in self.remove_manual_set
            else:
                self.keep_word_eval=keep_word_eval  
        if preserve_morphology!='undefined':
            self.preserve_morphology=preserve_morphology
        if substitute_removed!='undefined':
            self.substitute_removed=substitute_removed
        if text_to_substitute!='undefined':
            self.text_to_substitute=text_to_substitute
    
    def set_tokenizer(self,tokenizer=None):
        self.tokenizer=tokenizer
        
    def reset_preprocess(self):
        self.set_preprocess(split=False,
                            lower=False,
                            tokenize=False,
                            lemmatize=False,
                            remove_stop=False,
                            remove_punct=False,
                            remove_manual=False,
                            remove_manual_set=[],
                            keep_word_eval=None,
                            preserve_morphology=False,
                            substitute_removed=False,
                            text_to_substitute=''
                      )
        
    def display_progress(self, curr_doc, curr_wrd, last_step, mode='perc'):
        to_write="     {:<4} documents and {:<6d} words processed ".format(curr_doc,curr_wrd)
        if self.tot_wrd== 'undefined' or mode!='perc':
            print('\r'+to_write,end='')
        else:
            last_step=perc_compl(curr_wrd,self.tot_wrd+1,last_step=last_step, text=to_write)
        return last_step
            
        
    def count(self, replace=False):
        N_doc=0
        N_wrd=0
        for doc_line in self:
            N_doc+=1
            for word in doc_line:
                N_wrd+=1
        if replace:
            self.tot_doc=N_doc
            self.tot_wrd=N_wrd
        return N_doc,N_wrd
        
    def display_preprocess_settings(self):
        print("     {:-^55}".format("corpus processing settings"))
        print()
        print("     split: ",self.split)
        print("     lower: ", self.lower)
        print("     tokenize: ",self.tokenize)
        if self.tokenize:
            print("     lemmatize: ",self.lemmatize)
            print("     remove_stop: ",self.remove_stop)
            print("     remove_punct: ",self.remove_punct)
            
        print("     remove_manual: ",self.remove_manual)
        if self.remove_manual:
            print("     remove_manual_set: ",self.remove_manual_set)
            print("     preserve_morphology: ",self.preserve_morphology)
            
        print("     substitute_removed: ",self.substitute_removed)
        if self.substitute_removed:
            print("     text_to_substitute: ",self.text_to_substitute)
            
        print("     display iterator: ",self.display_iterator)
        print()
        
    def process(self, line):
#         ^^^^^ process according to above defined settings
        processed_line=[]
        processed_words=0
#         stepping=-1
        if self.lower:
            line=line.lower()
        if self.split:
            line=line.split()
        else:
            line=[line]
        for element in line:
            processed_word=element
            word_to_keep=True
            if self.tokenize:
                for token in self.tokenizer(element):    
                    word_to_keep=True
#                     if self.display_iterator:
#                         to_write="     {:>8d}".format(processed_words+1)+" words (splittings) elaborated;  "
#                         stepping=perc_compl(processed_words,self.tot_word,last_step=stepping, text=to_write)
                    if self.lemmatize:
                        processed_word=token.lemma_
                    else:
                        processed_word=token.text
                    if self.remove_punct and token.is_punct:
                        word_to_keep=False
                    if self.remove_stop and token.is_stop:
                        word_to_keep=False
                    if self.remove_manual and not self.keep_word_eval(token.text):
                        word_to_keep=False  
                    if not word_to_keep:
                        if not self.preserve_morphology:
                            continue
                        if self.substitute_removed:
                            processed_word=self.text_to_substitute
                    processed_line.append(processed_word) 
                    processed_words+=1
            else:
#                 if self.display_iterator:
#                     to_write="     {:>8d}".format(processed_words+1)+" words (splittings) elaborated;  "
#                     stepping=perc_compl(processed_words,self.tot_word,last_step=stepping, text=to_write)
                if self.remove_manual and not self.keep_word_eval(element):
                        word_to_keep=False
                if not word_to_keep:
                    if not self.preserve_morphology:
                        continue
                    if self.substitute_removed:
                        processed_word=self.text_to_substitute
                processed_line.append(processed_word)
                processed_words+=1
        return processed_line, processed_words
                
# ^^^^^^^^^^^^^
# when iterated, generates list of strings, processed according to internal settings
# call corpus.set_preprocessing(...) for internal settings management
# init it like: corpus = Corpus_Manager(path, tokenizer(opt), tot_word (opt))
# iterate like: for line in corpus: ...
# pass to model like: model = gensim.models.Word2Vec(sentences=corpus)


# %%
# ###################################################### CORP HOLDER for keeping it in memory

class Corpus_Holder(object):
    def __init__(self,
                 corpus_manager=None,
                 COPY=False):
        
        self.tot_doc='undefined'
        self.tot_wrd='undefined'
        self.corpus=[]
        if COPY:
            for doc in corpus_manager:
                self.hold_doc(doc)

    
    def hold_doc(self, doc):
        self.corpus.append(doc)
        
        if self.tot_doc == 'undefined':
            self.tot_doc=1
        else:
            self.tot_doc+=1
        if self.tot_wrd == 'undefined':
            self.tot_wrd=len(doc)
        else:
            self.tot_wrd+=len(doc)
            
    def display_progress(self, curr_doc, curr_wrd, last_step, mode='perc'):
        to_write="     {:<4} documents and {:<6d} words processed ".format(curr_doc,curr_wrd)
        if self.tot_wrd== 'undefined' or mode!='perc':
            print('\r'+to_write,end='')
        else:
            last_step=perc_compl(curr_wrd,self.tot_wrd+1,last_step=last_step, text=to_write)
        return last_step
    
    def __iter__(self):
        proc_doc=0
        proc_wrd=0
        stepping=-1
        for doc in self.corpus:
            proc_doc+=1
            proc_wrd+=len(doc)
            stepping=self.display_progress(proc_doc,proc_wrd, stepping)
            yield doc
            
    def display_count(self):
        print("     #docs: {}; #words: {}".format(self.tot_doc,self.tot_wrd))


# %%
print()
print("     Importing spaCy utils...")
import spacy
from spacy.lang.en import English
# from spacy.tokenizer import Tokenizer

nlp = spacy.load('en_core_web_sm')
tokenizer=nlp.Defaults.create_tokenizer(nlp)


# %%
# ################################################################ BUILDER SETTINGS INIT

if USE_DEF_ALG:
    Alg=Alg_default
    
# ---------------------------- Corpus params setting
LEMMATIZE_CORP=LEMMATIZE and LEMMATIZE_CORP_CONDITIONAL
LOWER_CORP=LOWER and LOWER_CORP_CONDITIONAL

# %%

