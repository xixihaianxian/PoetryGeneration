import torch
from torch import nn
from torch.utils import data
from torch import optim
import config
import tools
import torch.nn.functional as F
from typing import Dict,Tuple
import copy

class PoetryGenerator(nn.Module):
    def __init__(self,vocab_size, embedding_dim=config.EMBEDDING_DIM,
                 hidden_size=config.HIDDEN_SIZE, num_layers=config.NUM_LAYERS):
        super().__init__()
        self.vocab_size=vocab_size
        self.embedding_dim=embedding_dim
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.embedding=nn.Embedding(num_embeddings=self.vocab_size,embedding_dim=self.embedding_dim)
        self.lstm=nn.LSTM(input_size=self.embedding_dim,hidden_size=self.hidden_size,
                          num_layers=self.num_layers,batch_first=True,dropout=0.2 if self.num_layers>1 else 0)
        self.fc=nn.Linear(in_features=self.hidden_size,out_features=self.vocab_size)

    def forward(self,x,hidden:Tuple[torch.Tensor],x_len=None):
        output_x=self.embedding(x)
        output_x,hidden=self.lstm(output_x,hidden)
        output_x=self.fc(output_x[:,-1,:])
        return output_x,hidden

    def init_hidden(self,batch_size):
        weight=next(self.parameters()).data
        hidden=(weight.new(self.num_layers,batch_size,self.hidden_size).zero_().to(device=torch.device(tools.cuda_or_cpu())),
                weight.new(self.num_layers,batch_size,self.hidden_size).zero_().to(device=torch.device(tools.cuda_or_cpu())))
        return hidden

    def poetry_generation(self,word2id:Dict[str,int],id2word:Dict[int,str],start_word="人",generation_max_length=config.GENERATION_MAX_LENGTH,
                          temperature=config.TEMPERATURE,current_hidden=None):
        self.eval()

        log_probs_actions=list()
        generation_idxs=list()

        with torch.no_grad():
            pattern=[word2id.get(word) if word in word2id.keys() else 1 for word in start_word]
            generation_idxs=pattern[:]
            current_input_char_idx = generation_idxs[-1]

            if current_hidden is None:
                hidden=self.init_hidden(batch_size=1)
            else:
                hidden=current_hidden

            for _ in range(generation_max_length-len(start_word)): #TODO 循环生成诗歌

                input_seq=torch.tensor([[current_input_char_idx]],dtype=torch.int64).to(device=torch.device(tools.cuda_or_cpu()))
                out_seq,hidden=self.forward(input_seq,hidden)

                # temperature < 1 → 更确定、保守（适合模仿训练数据）
                # temperature = 1 → 原始行为
                # temperature > 1 → 更随机、多样性高（适合创意生成）

                output_probs=F.softmax(out_seq.div(temperature),dim=-1)

                next_char_idx_tensor=torch.multinomial(output_probs,num_samples=1)
                next_char_idx=next_char_idx_tensor.item()

                log_prob_action=torch.log(output_probs.squeeze()[next_char_idx_tensor.squeeze()])
                log_probs_actions.append(log_prob_action)

                generation_idx=next_char_idx
                generation_idxs.append(generation_idx)

                current_input_char_idx=generation_idx

                if id2word.get(current_input_char_idx) in ["！","。","？"]:
                    break

            if len(generation_idxs)>=generation_max_length:
                generation_idxs=generation_idxs[:generation_max_length]
            elif len(generation_idxs)<generation_max_length:
                generation_idxs.extend([word2id.get("PAD")]*(generation_max_length-len(generation_idxs)))

        return generation_idxs,torch.stack(log_probs_actions) if log_probs_actions else None,hidden

class PoetryDiscriminator(nn.Module):
    def __init__(self, vocab_size,embedding_dim=config.EMBEDDING_DIM, num_filters=config.NUM_FILTERS, filter_sizes=config.FILTER_SIZES, dropout_rate=0.3):
        super().__init__()
        self.vocab_size=vocab_size
        self.embedding_dim=embedding_dim
        self.num_filters=num_filters
        self.filter_sizes=filter_sizes
        self.dropout_rate=dropout_rate
        self.embedding=nn.Embedding(num_embeddings=self.vocab_size,embedding_dim=self.embedding_dim,padding_idx=0)
        self.convs=nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,out_channels=num_filters,kernel_size=ks)
            for ks in self.filter_sizes
        ])
        self.fc=nn.Linear(in_features=len(self.filter_sizes)*num_filters,out_features=1)
        self.dropout=nn.Dropout(p=self.dropout_rate)

    def forward(self,x_indices):
        embedded=self.embedding(x_indices) #TODO (batch_size,seq_len,embedding_dim)
        embedded=embedded.permute(0,2,1) #TODO (batch_size,embedding_dim,seq_len)

        conv_results=[F.relu(conv1d(embedded)) for conv1d in self.convs]
        #TODO (batch_size,num_filters,(seq_len-kernel_size+1))

        pool_results=[F.max_pool1d(conv_result,conv_result.shape[2]).squeeze(2) for conv_result in conv_results]
        #TODO (batch_size,num_filters)

        cat=self.dropout(torch.cat(pool_results,dim=1))
        #TODO (batch_size,num_filters*len(filter_sizes))

        logist=self.fc(cat)
        #TODO (batch_size,1)

        result=F.sigmoid(logist)

        return result

if __name__=="__main__":
    loaddata=tools.LoadData(data_path="./data/poetry.txt")
    loaddata.load_data()
    word2id=loaddata.word_to_id()
    id2word=loaddata.id_to_word(word2id)
    data_X_P,data_Y_P=loaddata.training_pairs()
    adjustment_X_result = loaddata.adjustment_word_list_no1(copy.deepcopy(data_X_P), is_id=False)
    transform_X_result=loaddata.transform_word_to_id(adjustment_X_result,word2id)
    transform_X_original_result=loaddata.transform_word_to_id(data_X_P,word2id)
    transform_Y_result=loaddata.transform_word_to_id(data_Y_P,word2id)
    poetry_dataset=tools.PoetryDataset(transform_X_result,transform_Y_result,transform_X_original_result)
    poetry_data_item=tools.poetry_item(poetry_dataset,batch_size=config.BATCH_SIZE,shuffle=config.SHUFFLE)
    vocab_size=len(word2id)
    poetry_generator=PoetryGenerator(vocab_size=vocab_size)
    poetry_discriminator=PoetryDiscriminator(vocab_size=vocab_size)
    hidden=None
    for x,y,x_prime in poetry_dataset:
        x=x.unsqueeze(0)
        x=x.to(device=torch.device(tools.cuda_or_cpu()))
        poetry_generator=poetry_generator.to(device=torch.device(tools.cuda_or_cpu()))
        if hidden is None:
            hidden=poetry_generator.init_hidden(batch_size=1)
            poetry_generator(x, hidden)
        else:
            poetry_generator(x,hidden)
        break