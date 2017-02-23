import os # ----------------------------------------grabing text files
from os import walk, getcwd, listdir #              grabing text files
import numpy as np #                                text preprocessing
import nltk # --------------------------------------for tokenizer, which is a bad one
#import spacy #could replace nltk tokenizer with spacy, but seems very similar to nltk version
import pandas as pd #                               data wrangling
from sklearn.decomposition import PCA as sklearnPCA # PCA
import matplotlib.pyplot as plt #                   plotting
import matplotlib #---------------------------------to set rcParams to change graph size
import matplotlib.mlab as mlab #                    for histogram generation
import time # --------------------------------------for determining the time to calculate functions
from random import randint #                        random numbers
import scipy as s # --------------------------------Unclear
import colorsys #                                   color coding labels 
from operator import itemgetter # ------------------ 


# WORD COUNTING AND WORD FREQUENCIES ----------

def count_words(list_to_search): #uses single_type_count() to count each token
    unique_words = set(list_to_search)
    word_counts = {}
    for word in unique_words:
        word_counts[word] = single_type_count(word, list_to_search)
    return word_counts # dict w/ word counts

def single_type_count(token_to_count, list_to_search): #counts up all tokens of a type
    number_of_tokens = 0                            
    for token in list_to_search:                   
        if token == token_to_count:                 
            number_of_tokens += 1                   
    return number_of_tokens #returns int 

def total_number_of_words(dict_of_word_counts): #for use with token_counts
    number_of_words = 0
    for word in dict_of_word_counts:
        number_of_words = number_of_words + dict_of_word_counts[word]
    return number_of_words #returns int

def total_number_of_words_in_corpus(list_of_total_word_counts):
    total_number_of_words = 0
    for total in range(0,len(list_of_total_word_counts)):
        total_number_of_words = total_number_of_words + list_of_total_word_counts[total]
    return total_number_of_words #returns int

def get_word_frequencies(dict_of_words_with_counts, total_number_of_words_in_text):
    word_freq = {}
    for word in dict_of_words_with_counts:
        word_freq[word] = dict_of_words_with_counts[word]/total_number_of_words_in_text
    return word_freq # dict with word w/ normalized frequencies





# PLOTTING FUNCTIONS (USING SCI-KIT LEARN) ----------

def generate_colors(num_colors): 
# taken from stackoverflow (http://bit.ly/1Po7eDT); 
# generates random colors for data points in matplotlib
    random_colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        #TEST: print("H/L/S:" + str(hue) + " / " + str(lightness) + " / " + str(saturation))
        random_colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    #TEST: print(random_colors)
    return random_colors

def generate_spectrum_of_colors(num_colors): 
# generates a spectrum of colors--from red to black--for use in matplotlib;
    spectrum_of_colors = []
    hue = 355/360
    for i in np.arange(0., 100., 100. / num_colors):
        lightness = i/100
        saturation = i/100
        spectrum_of_colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return spectrum_of_colors

def histogram_of_word_counts(total_word_counts_numpy_array, output_flag, plot_name):
## generates a historgram of all the texts or text chunks, based on word count;
## number of bins set to 37
    ## (1) generate histogram of data --------------------------------
    n, bin_location_list, patches = plt.hist(total_word_counts_numpy_array, 37, facecolor='red', alpha=0.75) #plt.hist(total_word_counts_array, 2, normed=1, facecolor='red', alpha=0.75)
    
    ## (2) plot graph  and print to file------------------------------
    plt.xlabel('word count')
    plt.ylabel('texts')
    plt.title('Histogram of Word Counts')
    plt.grid(True)
    matplotlib.rcParams['figure.figsize'] = (12.0,12.0)
    
    ## (3) output for articles; need to do this before 'plt.show' -----
    if output_flag == 1:
        plt.savefig(plot_name + '.pdf', dpi=600) #to produce a PDF of plot_PCA
    if output_flag == 2:
        plt.savefig(plot_name + '.png', dpi=600) #to produce a png of plot
    if output_flag != 1 & output_flag != 2 & output_flag != 0:
        print("Warning: output_flag for PCA plot not set to 0, 1, or 2")
        print("see plot_PCA_with_labels function")
    plt.show()
    return

def word_count_binning(total_word_counts_numpy_array):
## generates historgram data for all texts or text chunks, based on word count,
## but does not output histogram plot; number of bins set to 37
    ## (1) generate histogram of data ---------------------------------
    n, bin_location_list, patches = plt.hist(total_word_counts_numpy_array, 37, facecolor='red', alpha=0.75) #plt.hist(total_word_counts_array, 2, normed=1, facecolor='red', alpha=0.75)
    #TEST: print("contents of bins: " + str(n))
    #TEST: print("bin location intervals: " + str(bin_location_list))
    plt.close()
    return n, bin_location_list    

def assign_text_colors_via_word_counts(total_word_counts_nparray, total_word_counts):
## assigns colors for texts based on their relative word counts
## uses the word_count_binning function
    n, bins = word_count_binning(total_word_counts_nparray) #produce binning for all texts or text chunks in corpus
    text_binning = [] #identifies the bin in which a text belongs, where the text index is the same used in total_word_counts
    for text in range(0, len(total_word_counts_nparray)):
        for bin in range (0, len(bins)):
            if (total_word_counts[text] >= bins[bin]) & (total_word_counts[text] <= bins[bin+1]):
                text_binning.append(bin)
                break
    colors = generate_spectrum_of_colors(len(n))
    colors_for_texts = [] #identifies the color of a text, where the text index is the same used in total_word_counts
    for text in range(0, len(text_binning)):
        colors_for_texts.append(colors[text_binning[text]])
    return colors_for_texts

def perform_PCA(MFWlist_array_for_PCA, num_of_PCs):
#perform PCA on using the MFW lists for a set of texts
    pca_results = sklearnPCA(n_components = num_of_PCs)
    pca_coordinates = pca_results.fit_transform(MFWlist_array_for_PCA) #array of x- & y-coordinates 
    return pca_coordinates, pca_results

def plot_PCA_with_legend(pca_coordinates, pca_results, colors_for_texts, textnames, plot_size, output_flag, plot_name):
## plot PCA graph using "arrow" labels; set output_flag = 1 to produce PDF, 2 to produce PNG, else output = 0   
    ## (1) prepare color scheme, data point style, & graph legend ---
    for text in range(0, len(pca_coordinates)):        
        plt.plot(pca_coordinates[text,0],pca_coordinates[text,1], 'o', markersize=7, color=colors_for_texts[text], alpha=0.5, label=textnames[text])
    ## (2) graph display parameters & labels ---
    plt.xlabel('PC 1 ('+str(pca_results.explained_variance_ratio_[0]*100)+'%)') #x-axis title
    plt.ylabel('PC 2('+str(pca_results.explained_variance_ratio_[1]*100)+'%)') #y-axis title
    matplotlib.rcParams['figure.figsize'] = (plot_size, plot_size) #size of graph generated in notebook
    #plt.axis('tight') #OR just fit plot around data automatically; but this usually fits *so* closely that it misses data
    plt.legend()  #generate legend
    plt.title('PCA for ' + str(len(textnames)) + ' novels') #title of plot
    ax = plt.subplot(111) #used in making legend    
    plt.grid(b=True, which='major', color='gray', linestyle='dotted') # Add gridlines
    ## (3) legend placement (outside of plot) ---
    ## Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    # Put a legend below current axis
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -.15), fancybox=True, shadow=False, ncol=5)
    ## (4) "print-ready" plots ---
    ## need to produce these files before 'plt.show' since that command erases graph parameters
    if output_flag == 1:
        plt.savefig(plot_name + '.pdf', dpi=600) #to produce a PDF of plot_PCA
    if output_flag == 2:
        plt.savefig(plot_name + '.png', dpi=600) #to produce a png of plot
    if output_flag != 1 & output_flag != 2 & output_flag != 0:
        print("Warning: output_flag for PCA plot not set to 0, 1, or 2")
        print("see plot_PCA_with_labels function")
    plt.show() ## (5) plot PCA graph to screen --- 
    return

def plot_PCA(pca_coordinates, pca_results, colors_for_texts, textnames, plot_size, output_flag, plot_name):
## plot PCA graph without any description/labeling of texts; 
## set output_flag = 1 to produce PDF, 2 to produce PNG, else output = 0   
    ## (1) prepare color scheme, data point style & labels ---
    for text in range(0, len(pca_coordinates)):        
        plt.plot(pca_coordinates[text,0],pca_coordinates[text,1], 'o', markersize=7, color=colors_for_texts[text], alpha=0.5, label=textnames[text])
    ## (2) graph display parameters & labels ---
    plt.xlabel('PC 1 ('+str(pca_results.explained_variance_ratio_[0]*100)+'%)') #x-axis title
    plt.ylabel('PC 2('+str(pca_results.explained_variance_ratio_[1]*100)+'%)') #y-axis title
    matplotlib.rcParams['figure.figsize'] = (plot_size, plot_size) #size of graph generated in notebook
    ##plt.axis('tight') #OR just fit plot around data automatically; but this usually fits *so* closely that it misses data
    plt.title('PCA for ' + str(len(textnames)) + ' novels') #title of plot
    plt.grid(b=True, which='major', color='gray', linestyle='dotted') # Add gridlines
    ## (3) "print-ready" plots ---
    ## need to produce these files before 'plt.show' since that command erases graph parameters
    if output_flag == 1:
        plt.savefig(plot_name + '.pdf', dpi=600) #to produce a PDF of plot_PCA
    if output_flag == 2:
        plt.savefig(plot_name + '.png', dpi=600) #to produce a png of plot
    if output_flag != 1 & output_flag != 2 & output_flag != 0:
        print("Warning: output_flag for PCA plot not set to 0, 1, or 2")
        print("see plot_PCA_with_labels function")
    plt.show() ## (4) plot PCA graph to screen ---
    return


def plot_PCA_chunked_with_legend(pca_coordinates_for_chunks, pca_results_for_chunks, chunk_index, chunk_size_used, colors_for_texts, textnames, plot_size, output_flag, plot_name):
## plot PCA graph without any description/labeling of texts; 
## set output_flag = 1 to produce PDF, 2 to produce PNG, else output = 0
    ## (1) prepare color scheme, data point style & labels ---
    for text in range(0, len(textnames)):
        plt.plot(pca_coordinates_for_chunks[chunk_index[text][0]:chunk_index[text][1],0], \
            pca_coordinates_for_chunks[chunk_index[text][0]:chunk_index[text][1],1], \
            'o', markersize=7, color=colors_for_texts[text], alpha=0.5, label=textnames[text])
    ## (2) graph display parameters & labels
    plt.xlabel('PC 1 ('+str(pca_results_for_chunks.explained_variance_ratio_[0]*100)+'%)') #x-axis title
    plt.ylabel('PC 2('+str(pca_results_for_chunks.explained_variance_ratio_[1]*100)+'%)') #y-axis title
    matplotlib.rcParams['figure.figsize'] = (plot_size, plot_size) #size of graph generated in notebook
    #plt.axis('tight') #OR just fit plot around data automatically; but this usually fits *so* closely that it excludes data points
    plt.legend()  #generate legend
    plt.title('PCA for ' + str(len(textnames)) + ' novels in chunks of '+str(chunk_size_used)+' words') #title of plot
    ax = plt.subplot(111) #used in making legend
    ##PLACE LEGEND OUTSIDE OF PLOT
    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    # Put a legend below current axis
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -.15), fancybox=True, shadow=False, ncol=5)
    # Add gridlines 
    plt.grid(b=True, which='major', color='gray', linestyle='dotted')

    ## (3) "print-ready" plots ---
    ## need to produce these files before 'plt.show' since that command erases graph parameters
    if plot_name == 0: #must be a string if not a "0"
        plot_name = 'PCA for ' + str(len(textnames)) + ' novels in chunks of '+str(chunk_size_used)+' words'
    if output_flag == 1:
        plt.savefig(plot_name + '.pdf', dpi=600) #to produce a PDF of plot_PCA
    if output_flag == 2:
        plt.savefig(plot_name + '.png', dpi=600) #to produce a png of plot
    if output_flag != 1 & output_flag != 2 & output_flag != 0:
        print("Warning: output_flag for PCA plot not set to 0, 1, or 2")
        print("see plot_PCA_with_labels function")
    plt.show() ## (4) plot PCA graph to screen ---
    return

def plot_PCA_chunked_with_labels(pca_coordinates_for_chunks, pca_results_for_chunks, chunk_index, chunk_size_used, colors_for_texts, textnames, plot_size, output_flag, plot_name):
## plot PCA graph without any description/labeling of texts; 
## set output_flag = 1 to produce PDF, 2 to produce PNG, else output = 0
    ## (1) prepare color scheme, data point style & labels ---
    plt.style.use('seaborn-whitegrid') #works w/ 'seaborn-ticks', 'seaborn-white', classic'
    for text in range(0, len(textnames)):
        plt.plot(pca_coordinates_for_chunks[chunk_index[text][0]:chunk_index[text][1],0], \
            pca_coordinates_for_chunks[chunk_index[text][0]:chunk_index[text][1],1], \
            'o', markersize=7, color=colors_for_texts[text], alpha=0.5, label=textnames[text])
    ## (2) graph display parameters & labels
    plt.xlabel('PC 1 ('+str(pca_results_for_chunks.explained_variance_ratio_[0]*100)+'%)') #x-axis title
    plt.ylabel('PC 2('+str(pca_results_for_chunks.explained_variance_ratio_[1]*100)+'%)') #y-axis title
    matplotlib.rcParams['figure.figsize'] = (plot_size, plot_size) #size of graph generated in notebook
    #plt.axis('tight') #OR just fit plot around data automatically; but this usually fits *so* closely that it excludes data points
    plt.title('PCA w LABELS for ' + str(len(textnames)) + ' texts in chunks of '+str(chunk_size_used)+' words') #title of plot
    ax = plt.subplot(111) #used in making legend
    ##PLACE LEGEND OUTSIDE OF PLOT
    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    # Put a legend below current axis
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -.15), fancybox=True, shadow=False, ncol=5)
    # Add gridlines 
    plt.grid(b=True, which='major', color='gray', linestyle='dotted')

    ## (4) generate arrow labels ---
    chunk_index_number = 0
    #text_loop = 0 #testing
    for text in range (0, len(textnames)):
        #text_loop = text_loop + 1 #testing
        while chunk_index_number <= int(chunk_index[text][1]):
            ax.annotate(chunk_index_number, xy=(pca_coordinates_for_chunks[chunk_index_number,0], pca_coordinates_for_chunks[chunk_index_number,1]), xycoords='data', xytext=(-30, -30), textcoords='offset points', arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
            #print("finished " + str(chunk_index_number) + ", incrementing...")
            chunk_index_number = chunk_index_number + 1
    #print(text_loop) #testing

    ## (5) "print-ready" plots ---
    ## need to produce these files before 'plt.show' since that command erases graph parameters
    if plot_name == 0: #must be a string if not a "0"
        plot_name = 'PCA w LABELS for ' + str(len(textnames)) + ' texts in chunks of '+str(chunk_size_used)+' words'
    if output_flag == 1:
        plt.savefig(plot_name + '.pdf', dpi=600) #to produce a PDF of plot_PCA
    if output_flag == 2:
        plt.savefig(plot_name + '.png', dpi=600) #to produce a png of plot
    if output_flag != 1 & output_flag != 2 & output_flag != 0:
        print("Warning: output_flag for PCA plot not set to 0, 1, or 2")
        print("see plot_PCA_with_labels function")
    plt.show() ## (4) plot PCA graph to screen ---
    return 



def plot_PCA_with_labels(pca_coordinates, pca_results, textnames, colors_for_texts, plot_size, output_flag, plot_name):
## plot PCA graph using "arrow" labels; set output_flag = 1 to produce PDF, 2 to produce PNG, else output = 0
## WARNING: Using some styles for plt.style.use() will *break* arrow labels. For instance,
## plt.style.use('ggplot') breaks arrows. Accordingly, this function sets plt.style.use('classic')
    ## (0) invoke plt.style that works with arrow labels
    plt.style.use('seaborn-whitegrid') #works w/ 'seaborn-ticks', 'seaborn-white', classic'
     ## (1) prepare color scheme, data point style & labels ---
    for text in range(0, len(pca_coordinates)):
        plt.plot(pca_coordinates[text,0],pca_coordinates[text,1], 'o', markersize=7, color=colors_for_texts[text], alpha=0.8, label=textnames[text])
    ## (2) graph display parameters & labels ---
    plt.xlabel('PC 1 ('+str(pca_results.explained_variance_ratio_[0]*100)+'%)') #x-axis title
    plt.ylabel('PC 2 ('+str(pca_results.explained_variance_ratio_[1]*100)+'%)') #y-axis title
    matplotlib.rcParams['figure.figsize'] = (plot_size, plot_size) #size of graph generated in notebook
    plt.title('PCA for ' + str(len(textnames)) + ' novels') #title of plot
    ax = plt.subplot(111) #used in making legend 
    plt.grid(b=True, which='major', color='gray', linestyle='dotted') #grid lines
    ## (3)generate arrow labels ---
    for text in range(0, len(textnames)):
        ax.annotate(textnames[text], xy=(pca_coordinates[text,0], pca_coordinates[text,1]), xycoords='data', xytext=(-30, -30), textcoords='offset points', arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    ## (4) "print-ready" plots ---
    ## need to produce these files before 'plt.show' since that command erases graph parameters
    if output_flag == 1:
        plt.savefig(plot_name + '.pdf', dpi=600) #to produce a PDF of plot_PCA
    if output_flag == 2:
        plt.savefig(plot_name + '.png', dpi=600) #to produce a png of plot
    if output_flag != 1 & output_flag != 2 & output_flag != 0:
        print("Warning: output_flag for PCA plot not set to 0, 1, or 2")
        print("see plot_PCA_with_labels function")
    ## (5) plot PCA graph to screen ---
    plt.show() 
    return



def graph_PC_variance(word_frequencies, plot_size, output_flag, plot_name, drop_last_flag):
## Plots of the variance of all Principal Components as a function of PCs.
## set output_flag = 1 to produce PDF, 2 to produce PNG, else output = 0
## set drop_last_flag = 1 to drop final PC (to make graph more readable), else set equal to 0 
    ## (0) Prepare MFW for PCA
    MFW_for_PCA = pd.DataFrame(word_frequencies).fillna(0).as_matrix()
    
    ## (1) GET ALL PCs (i.e., all the texts or text chunks in corpus)
    pca_allDs = sklearnPCA() #since number of components not specified; calculates all components
    pca_coordinates_allDs = pca_allDs.fit_transform(MFW_for_PCA) #array of x- & y-coordinates 
    PCA_coordinates_for_texts_allDs = pca_coordinates_allDs
    print("number of PCs: " + str(len(PCA_coordinates_for_texts_allDs[0])))
    ## (2) produce list of variance for all PCs
    pca_variance = []
    for PC in range(0, len(PCA_coordinates_for_texts_allDs[0])):
        pca_variance.append(pca_allDs.explained_variance_ratio_[PC]*100)
    ## (3) PLOT PC VARIANCE vs PC
    if drop_last_flag == 1:
        for PC in range(0, len(pca_variance)-1): # the minus one prevents final PC from being plotted
            plt.plot(PC, pca_variance[PC], 'o', markersize=7, alpha=0.5)
    else:
        for PC in range(0, len(pca_variance)): # the minus one prevents final PC from being plotted
            plt.plot(PC, pca_variance[PC], 'o', markersize=7, alpha=0.5)
    ## (4) plot parameters
    matplotlib.rcParams['figure.figsize'] = (15.0,15.0) #graph size
    plt.grid(b=True, which='major', color='gray', linestyle='dotted') # Add gridlines
    plt.title('Percentage Variance of PC vs PC')
    plt.xlabel('PC number') #x-axis title
    plt.ylabel('Percentage of Variance') #y-axis title
    plt.yscale('log') # use logarithmic scale for y-axis
    #plt.yscale('linear')
    ## (5) "print-ready" plots ---
    ## need to produce these files before 'plt.show' since that command erases graph parameters
    if output_flag == 1:
        plt.savefig(plot_name + '.pdf', dpi=600) #to produce a PDF of plot_PCA
    if output_flag == 2:
        plt.savefig(plot_name + '.png', dpi=600) #to produce a png of plot
    if output_flag != 1 & output_flag != 2 & output_flag != 0:
        print("Warning: output_flag for PCA plot not set to 0, 1, or 2")
        print("see plot_PCA_with_labels function")  
    plt.show() ## (6) plot PCA graph to screen ---

## FUNCTION TO CHUNK A TEXT OF ANY SIZE
# Note that "wordlist" needs to be a list of words *in the order* of the original text;
# also, chunk_text returns the number of chunks used for a particular text in addition
# to a list of text chunks
def chunk_text(wordlist, words_per_chunk): #chunk text; where last chunk <= words_per_chunk
    if words_per_chunk == 0:
        print('WARNING: words_per_chunk = 0')
        return
    number_of_chunks, size_of_final_chunk = divmod(len(wordlist), words_per_chunk)
    number_of_chunks = number_of_chunks + 1 #we add one because divmod does include final chunk
    temp = [] #We build a list of chunks using this list
    list_of_text_chunks = [temp + wordlist[words_per_chunk*chunk:words_per_chunk*(chunk+1)] for chunk in range(0, number_of_chunks)]
    return list_of_text_chunks, number_of_chunks

def filenames_of_txts_in_directory(data_location):
## reads in all txt files inputs them into a list of strings
    begin_readin_time = time.clock()
    filenames_of_texts = [ filename for filename in listdir(data_location) if filename.endswith('.txt')] 
    end_readin_time = time.clock()
    print("Examining " + str(len(filenames_of_texts)) + " texts...")
    return filenames_of_texts
    
def tokenized_texts_and_textname_list(filenames_of_texts, data_location):
## NOTE: you can either use NTLK or whitespace tokenizer (or you can add your own);
## both of these tokenizers are bad, but nearly everyone uses them :(
## Ignores any file that is doesn't end with ".txt"
    begin_tokenize_time = time.clock()
    wordlist = []                                                              # element is the word list for a text
    textnames = []                                                             # element is the name of the text
    for document in range(0, len(filenames_of_texts)):                         # Loop through all files in directory
        root, ext = os.path.splitext(filenames_of_texts[document])             # Select file extension for particular file "x" in the list "allFilesInDirectory"
        if (ext == '.txt'):                                                    # redundant check
            text = open(data_location + str(filenames_of_texts[document]), "r").read()   #read file
            textnames.append(root)                                            # create list of text names
        text = text.lower()                                                    # we want everything lowercase
        wordlist.append(nltk.tokenize.word_tokenize(text))                     # tokenize file using NLTK tokenizer (a poor tokenizer)
        #wordlist.append(text.lower().split())                                 # tokenize file using whitespace (a terrible tokenizer)   
    end_tokenize_time = time.clock()
    print("Time to tokenize texts: " + str(end_tokenize_time-begin_tokenize_time))
    return wordlist, textnames

def chunk_all_texts(wordlist, textnames, chunk_size_used):
    begin_chunking_time = time.clock()
    chunk_wordlist = [] # a list of chunk wordlists for entire corpus
    chunk_index = []     # An index of the number of chunks for a text, 
                         # where chunk_index[doc] gives [chunk_wordlists[start_element,end_element]]
    for doc in range(0, len(wordlist)):
            chunk_wordlist_for_single_text, number_of_chunks = chunk_text(wordlist[doc], chunk_size_used)
            chunk_wordlist = chunk_wordlist + chunk_wordlist_for_single_text #add chunks to corpus list of chunks
            if doc == 0:
                chunk_index = [[0, number_of_chunks-1]]    
            else:
                chunk_index.append([chunk_index[doc-1][1]+1,chunk_index[doc-1][1]+number_of_chunks])
    end_chunking_time = time.clock()
    print("Time to chunk texts: " + str(end_chunking_time-begin_chunking_time))
    return chunk_wordlist, chunk_index

def type_counts_and_total_token_counts(wordlist):
## count words for all texts or text chunks, i.e., [{text1_word_counts}, {text2_word_counts},...]
    begin_word_count_time = time.clock() 

    # FIRST, CALCULATE WORD COUNTS FOR INDIVIDUAL TEXTS IN CORPUS
    word_counts = [] # List of dicts with word counts for each text, e.g. "word_counts[0]" will
                     # return a dict of counts for each type for the first "text".
                     # These "texts" could be a novel, a novel chunk, a topic from a topic model, etc.
    total_word_counts = [] # total word counts for each "text", 
                           # i.e., [{text1_word_counts}, {text2_word_counts},...]
    for text in range(0, len(wordlist)):
        word_counts.append(count_words(wordlist[text]))
        total_word_counts.append(total_number_of_words(word_counts[text]))

    # SECOND, WE NEED TO CALCULATE TOTAL WORDS IN CORPUS
    corpus_word_count = total_number_of_words_in_corpus(total_word_counts)
    end_word_count_time = time.clock() 
    print("Time to count words: " + str(end_word_count_time-begin_word_count_time))
    return word_counts, total_word_counts

def word_freq(word_counts, corpus_word_count):
## calculate word frequencies for any set of distinct items (e.g., texts, chunks, topic models, etc.)
    begin_freq_time = time.clock()
    word_frequencies = []
    for text in range(0, len(word_counts)):
        word_frequencies.append(get_word_frequencies(word_counts[text],corpus_word_count)) 
    end_freq_time = time.clock()
    print("Time to compute frequencies: " + str(end_freq_time-begin_freq_time))
    return word_frequencies

def display_MFW(word_frequencies, number_of_MFWs_to_display, text_index_to_compare_MFWs):
    begin_time = time.clock()
    readable_word_frequencies = pd.DataFrame(word_frequencies).T
    text_to_examine = 0 # column identifer for a particular text; full list in "textnames"
    MFW = readable_word_frequencies.sort_values([text_index_to_compare_MFWs], ascending = False)
    MFW = MFW.fillna(0)
    #print(textnames) #uncomment to display list of texts
    display = MFW.head(number_of_MFWs_to_display) #Display first 25 words for MFW
    #MFW #uncomment to display MFW relative to a single text (this list is millions of lines)
    end_time = time.clock()
    print("Time to execute MFW display: " + str(end_time-begin_time))
    return display

def PCAnalysis(word_frequencies, number_of_MFWs_used, corpus_word_count, number_of_components, text_index_to_compare_MFWs):
## APPLY PCA TO WORD FREQUENCIES LIST
## IF number_of_MFWs_used == 0, USE ALL WORDS IN LIST.

    ## prepare word_frequencies for PCA processing
    begin_time = time.clock()
    print("Corpus Word Count:" + str(corpus_word_count))
    if number_of_MFWs_used == 0:
        print("""Using corpus word count (""" + str(corpus_word_count) + """ words) for PCA in """ + str(number_of_components) + """-dimensions...""")
        number_of_MFWs_used = corpus_word_count
        # Note: There's probably a faster way than performing to T operations in the next 3 lines...
        dataframe_word_frequencies = pd.DataFrame(word_frequencies).T
        MFW = dataframe_word_frequencies.sort_values([text_index_to_compare_MFWs], ascending = False)
        MFW_for_PCA = MFW.fillna(0).as_matrix().T #using all MFWs

    else:
        print("Using " + str(number_of_MFWs_used) + " words for PCA in "+ str(number_of_components) + "-dimensions...")  #np.nan_to_num(word_frequencies[number_of_MFWs_used:])
        #Note: There's probably a faster way than performing to T operaions in the next 3 lines
        dataframe_word_frequencies = pd.DataFrame(word_frequencies).T
        MFW = dataframe_word_frequencies.sort_values([text_index_to_compare_MFWs], ascending = False)
        MFW_for_PCA = MFW.head(number_of_MFWs_used).fillna(0).as_matrix().T #using X of MFWs


    ## generate data points of PCA from MFW_for_PCA (this is where PCA is performed)
    pca_coordinates, pca_results = perform_PCA(MFW_for_PCA, number_of_components)   
    end_time = time.clock()

    print("Time to execute PCA: " + str(end_time-begin_time))
    return pca_coordinates, pca_results


def obtain_MFW(word_frequencies, compared_to_which_text, textnames):
#USEFUL FOR EXAMINING MOST FREQUENT WORDS IN WORD LIST
#Uses Pandas to sort word frequencies and fill empty cells
#Returns dataframe "MFW"
    readable_word_frequencies = pd.DataFrame(word_frequencies).T
    #compared_to_which_text = 0 # column identifer for a particular text; full list in "novelnames"
    MFW = readable_word_frequencies.sort_values([compared_to_which_text], ascending = False)
    MFW = MFW.fillna(0) # fill all empty cells with zeros   
    print("MFW list relative to " + textnames[compared_to_which_text])
    print("""Type 'MFW.head(X)' to list the first X most frequent words.""")
    return MFW

def obtain_LFW(word_frequencies, compared_to_which_text, textnames):
#USEFUL FOR EXAMINING MOST FREQUENT WORDS IN WORD LIST
#Uses Pandas to sort word frequencies and fill empty cells
#Returns dataframe "MFW"
    readable_word_frequencies = pd.DataFrame(word_frequencies).T
    #compared_to_which_text = 0 # column identifer for a particular text; full list in "novelnames"
    LFW = readable_word_frequencies.sort_values([compared_to_which_text], ascending = True)
    LFW = LFW.fillna(0) # fill all empty cells with zeros   
    print("LFW list relative to " + textnames[compared_to_which_text])
    print("""Type 'LFW.head(X)' to list the first X most frequent words.""")
    return LFW


# SAVING WORD FREQUENCIES, ETC, FOR FUTURE SESSIONS -----------

def export_PCA_analysis(corpusfolder, word_counts, total_word_counts, corpus_word_count, word_frequencies, wordlist, textnames):
# to SAVE data generated in the "Prepare, Tokenize, and Count Words of Corpus" code block
    import pickle  
    with open('./dat/' + corpusfolder + '/word_counts.p', 'wb') as f:
        pickle.dump(word_counts, f) 
    with open('.dat/' + corpusfolder + '/total_word_counts.p', 'wb') as f:
        pickle.dump(total_word_counts, f) 
    with open('./dat/' + corpusfolder + '/corpus_word_count.p', 'wb') as f:
        pickle.dump(corpus_word_count, f) 
    with open('./dat/'+ corpusfolder + '/word_frequencies.p', 'wb') as f:
        pickle.dump(word_frequencies, f)
    with open('./dat/' + corpusfolder + '/wordlist.p', 'wb') as f:
        pickle.dump(wordlist, f)
    with open('./dat/' + corpusfolder + '/textnames.p', 'wb') as f:
        pickle.dump(textnames, f)

def export_chunked_text_analysis(corpusfolder, chunk_word_counts, chunk_total_word_counts, corpus_word_count_for_chunks, chunk_word_frequencies, chunk_wordlist, chunk_index, wordlist, textnames):
# to SAVE data generated in the "Prepare, Tokenize, and Count Words of Corpus" code block
    import pickle  
    with open('./dat/' + corpusfolder + '/chunk_word_counts.p', 'wb') as f:
        pickle.dump(chunk_word_counts, f) 
    with open('./dat/' + corpusfolder + '/chunk_total_word_counts.p', 'wb') as f:
        pickle.dump(chunk_total_word_counts, f) 
    with open('./dat/' + corpusfolder + '/corpus_word_count_for_chunks.p', 'wb') as f:
        pickle.dump(corpus_word_count_for_chunks, f) 
    with open('./dat/'+ corpusfolder + '/chunk_word_frequencies.p', 'wb') as f:
        pickle.dump(chunk_word_frequencies, f)
    with open('./dat/' + corpusfolder + '/chunk_wordlist.p', 'wb') as f:
        pickle.dump(chunk_wordlist, f)
    with open('./dat/' + corpusfolder + '/chunk_index.p', 'wb') as f:
        pickle.dump(chunk_index, f)
    with open('./dat/' + corpusfolder + '/wordlist.p', 'wb') as f:
        pickle.dump(wordlist, f)
    with open('./dat/' + corpusfolder + '/textnames.p', 'wb') as f:
        pickle.dump(textnames, f)

def import_chunked_text_analysis(corpusfolder):
# to SAVE data generated in the "Prepare, Tokenize, and Count Words of Corpus" code block
    import pickle  
    with open('./dat/' + corpusfolder + '/chunk_word_counts.p', 'rb') as f:
        chunk_word_counts = pickle.load(f)
    with open('./dat/' + corpusfolder + '/chunk_total_word_counts.p', 'rb') as f:
        chunk_total_word_counts = pickle.load(f) 
    with open('./dat/' + corpusfolder + '/corpus_word_count_for_chunks.p', 'rb') as f:
        corpus_word_count_for_chunks = pickle.load(f) 
    with open('./dat/'+ corpusfolder + '/chunk_word_frequencies.p', 'rb') as f:
        chunk_word_frequencies = pickle.load(f)
    with open('./dat/' + corpusfolder + '/chunk_wordlist.p', 'rb') as f:
        chunk_wordlist = pickle.load(f)
    with open('./dat/' + corpusfolder + '/chunk_index.p', 'rb') as f:
        chunk_index = pickle.load(f)
    with open('./dat/' + corpusfolder + '/wordlist.p', 'rb') as f:
        wordlist = pickle.load(f)
    with open('./dat/' + corpusfolder + '/textnames.p', 'rb') as f:
        textnames = pickle.load(f)
    return chunk_word_counts, chunk_total_word_counts, corpus_word_count_for_chunks, chunk_word_frequencies, chunk_wordlist, chunk_index, wordlist, textnames

def import_PCA_analysis(corpusfolder):
# to LOAD data generated in the "Prepare, Tokenize, and Count Words of Corpus" code block that was saved earlier
    import pickle
    with open('./dat/'+ corpusfolder + '/word_counts.p', 'rb') as f:
        word_counts = pickle.load(f)
    with open('./dat/' + corpusfolder + '/total_word_counts.p', 'rb') as f:
        total_word_counts = pickle.load(f)
    with open('./dat/' + corpusfolder + '/corpus_word_count.p', 'rb') as f:
        corpus_word_count = pickle.load(f)
    with open('./dat/' + corpusfolder + '/word_frequencies.p', 'rb') as f:
        word_frequencies = pickle.load(f)
    with open('./dat/' + corpusfolder + '/wordlist.p', 'rb') as f:
        wordlist = pickle.load(f)
    with open('./dat/' + corpusfolder + '/textnames.p', 'rb') as f:
        textnames = pickle.load(f)
    return word_counts, total_word_counts, corpus_word_count, word_frequencies, wordlist, textnames