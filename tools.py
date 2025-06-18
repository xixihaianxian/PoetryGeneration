import  torch
from torch import nn
from torch.utils import data
from torch import optim
import jieba
import config
import re
from collections import defaultdict
from typing import Dict,List
import copy

class LoadData:
    def __init__(self,data_path,max_length:int=config.MAX_LENGTH,seq_length=config.SEQ_LENGTH):
        self.data_path=data_path
        self.pad="PAD"
        self.unk="UNK"
        self.max_length=max_length
        self.seq_length=seq_length

    def sentence_to_wordlist(self,sentence,use_jieba=True):
        if use_jieba:
            return jieba.lcut(sentence)
        else:
            return list(sentence)

    def clean_wordlist(self,wordlist):
        return [word for word in wordlist if word not in config.common_punctuations]

    def transform(self,sentence,word2id:Dict[str,int]):
        return [word2id.get(word) if word in word2id.keys() else word2id.get("UNK") for word in sentence]

    def load_data(self):
        self.word_list=list()
        self.words=list()
        self.text_raws=list()

        with open(self.data_path,"r",encoding="utf-8") as file:
            line=file.readline()
            while line:
                pattern=re.compile(r"[^\u4e00-\u9fa5\u3000-\u303f，。！？]")
                line=re.sub(pattern=pattern,repl="",string=line).replace(" ","")
                self.text_raws.append(line)
                wordlist=self.sentence_to_wordlist(line,use_jieba=False)
                # clean_wordlist=self.clean_wordlist(wordlist)
                self.words.extend(wordlist)
                self.word_list.append(wordlist)
                #TODO 每一句诗转化为列表之后的结果，放到word_list里面，注意要与下面的word_list参数惊醒区分
                line=file.readline()
        self.words=sorted(set(self.words))

    @property
    def sentence_max_length(self):
        return max([len(word_list) for word_list in self.word_list])

    @property
    def word_len(self):
        return len(self.words)

    def word_to_id(self):
        word2id=defaultdict(int)
        word2id.update(config.mask)

        for word in self.words:
            if word not in word2id.keys():
                word2id[word]=len(word2id)

        return word2id

    def adjustment_word_list_no1(self,word_lists,is_id=True):
        """
        第一种填充方式
        :param word_lists: 分词后的列表，列表里面是分词结果
        :param is_id: word_lists是否是id列表
        :return: 返回填充有点结果
        """
        adjust_result=list()

        if self.max_length is None:
            self.max_length=self.sentence_max_length

        for word_list in word_lists:
            if len(word_list)<self.max_length:
                if is_id:
                    word_list.extend(0 * (self.max_length - len(word_list)))
                else:
                    word_list.extend([self.pad] * (self.max_length - len(word_list)))
            elif len(word_list)<self.max_length:
                word_list=word_list[:self.max_length]

            adjust_result.append(word_list)

        return adjust_result

    def adjustment_word_list_no2(self,word_lists,is_id=True):
        """
        第二种填充方式
        :param word_lists: 分词后的列表，列表里面是分词结果
        :param is_id: word_lists是否是id列表
        :return: 返回填充有点结果
        """
        adjust_result=list()

        for word_list in word_lists:
            if len(word_list)<self.sentence_max_length:
                if is_id:
                    word_list.extend([0] * (self.max_length - len(word_list)))
                else:
                    word_list.extend(self.pad * (self.max_length - len(word_list)))
            elif len(word_list)<self.max_length:
                word_list = word_list[:self.max_length]

            adjust_result.append(word_list)

        return adjust_result

    def transform_word_to_id(self,word_list:List[list],word2id:Dict[str,int]):
        """
        :param word_list: 词列表
        :param word2id: word到id的映射
        :return: 返回将词列表的内容转化为id
        """
        transform_result=list()

        for wordlist in word_list:
            wordlist=self.transform(wordlist,word2id)
            transform_result.append(wordlist)

        return transform_result

    def training_pairs(self):
        data_X_P=list()
        data_Y_P=list()
        text="".join(self.text_raws)
        for i in range(0,len(text)-self.seq_length):
            seq_in=text[i:i+self.seq_length]
            seq_out=text[i+self.seq_length]
            data_X_P.append(list(seq_in))
            data_Y_P.append(seq_out)
        return data_X_P,data_Y_P

    def id_to_word(self,word2id:Dict[str,int]):
        return dict(zip(word2id.values(),word2id.keys()))

def cuda_or_cpu():
    return "cuda" if torch.cuda.is_available() else "cpu"

if __name__=="__main__":
    loaddata=LoadData(data_path="./data/poetry.txt")
    loaddata.load_data()
    word2id=loaddata.word_to_id()
    id2word=loaddata.id_to_word(word2id)
    data_X_P,data_Y_P=loaddata.training_pairs()
    adjustment_X_result = loaddata.adjustment_word_list_no1(copy.deepcopy(data_X_P), is_id=False)
    transform_X_result=loaddata.transform_word_to_id(adjustment_X_result,word2id)
    transform_X_original_result=loaddata.transform_word_to_id(data_X_P,word2id)
    transform_Y_result=loaddata.transform_word_to_id(data_Y_P,word2id)
    print(transform_X_result)