import torch
from torch import nn
from torch.utils import data
from torch import optim
import tools
import models
import config
import os
import math
from loguru import logger
import copy
from tqdm import tqdm
import random
import numpy as np

os.path.exists("./params") or os.makedirs("./params")

def pretrain_generator(model:models.PoetryGenerator,data_item,epochs,loss_function:nn.Module,word2id,id2word,
                       optimizer:optim.Optimizer):
    model=model.to(device=torch.device(tools.cuda_or_cpu()))
    hidden=None
    loss_list=list()
    loss_min=math.inf
    for epoch in range(epochs):
        loss_sum=0
        model.train()
        for x,y,x_original in tqdm(data_item):
            targets=x_original.to(device=torch.device(tools.cuda_or_cpu()))
            labels=y.to(device=torch.device(tools.cuda_or_cpu()))
            if hidden is None:
                hidden=model.init_hidden(batch_size=targets.size(0))
            predict_target,hidden=model(targets,hidden)
            hidden=(hidden[0].detach(), hidden[1].detach())
            loss=loss_function(predict_target,labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(parameters=model.parameters(),max_norm=5)
            optimizer.step()
            with torch.no_grad():
                loss_sum+=loss.item()
        with torch.no_grad():
            loss_mean=loss_sum/len(data_item)
            sample_indices,_,_=model.poetry_generation(word2id=word2id,id2word=id2word,start_word="风",generation_max_length=10)
            result=[id2word.get(index) for index in sample_indices if index!=0]
            logger.info(f"Generator model train: epoch :{epoch+1} loss:{loss_mean} generator result:{''.join(result)}")
            loss_list.append(loss_mean)
            if loss_mean<loss_min:
                loss_min=loss_mean
                torch.save(model.state_dict(),f"./params/pretrain_best_generator.pth")
                logger.info(f"Save best generator model!")
    return loss_list

def pretrain_discriminator(discriminator_model:models.PoetryDiscriminator, generator_model:models.PoetryGenerator,
                           data_item,word2id,id2word,epochs,loss_function,optimizer:optim.Optimizer,vocab_size):
    discriminator_model=discriminator_model.to(device=torch.device(tools.cuda_or_cpu()))
    generator_model=generator_model.to(device=torch.device(tools.cuda_or_cpu()))
    loss_list=list()
    loss_min=math.inf

    for epoch in range(epochs):
        discriminator_model.train()
        generator_model.eval()
        loss_sum=0

        for x,y,x_original in tqdm(data_item):
            real_targets=x.to(device=torch.device(tools.cuda_or_cpu()))
            real_labels=torch.ones(size=(len(y),1),device=torch.device(tools.cuda_or_cpu()),dtype=torch.int64)

            false_targets=list()
            for _ in range(config.BATCH_SIZE//2):
                seed_char_idx=random.randint(len(config.mask),vocab_size-1)
                start_word=id2word.get(seed_char_idx,"风")
                false_target,_,_=generator_model.poetry_generation(word2id=word2id,id2word=id2word,start_word=start_word,
                                                               generation_max_length=config.MAX_LENGTH)
                false_targets.append(false_target)

            false_targets=torch.tensor(false_targets,dtype=torch.int64,device=torch.device(tools.cuda_or_cpu()))
            false_labels=torch.zeros(size=(len(false_targets),1),dtype=torch.int64,device=torch.device(tools.cuda_or_cpu()))

            targets=torch.cat([real_targets,false_targets],dim=0)
            labels=torch.cat([real_labels,false_labels],dim=0)

            shuffle_index=torch.randperm(targets.size(0))
            targets=targets[shuffle_index]
            labels=labels[shuffle_index].to(dtype=torch.float32)

            predict=discriminator_model(targets)
            loss=loss_function(predict,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                loss_sum+=loss.item()

        with torch.no_grad():
            loss_mean=loss_sum/len(data_item)
            loss_list.append(loss_mean)
            logger.info(f"Discriminator model train: epoch:{epoch + 1} loss:{loss_mean}")
            if loss_mean<loss_min:
                loss_min=loss_mean
                torch.save(discriminator_model.state_dict(),"./params/pretrain_best_discriminator.pth")
    return loss_list

#TODO pretrain_discriminator的数据生成部分，可以替代上面的一部分
def discriminator_data_generator(x,y,x_original,generator_model,vocab_size,word2id,id2word):

    real_targets = x.to(device=torch.device(tools.cuda_or_cpu()))
    real_labels = torch.ones(size=(len(y), 1), device=torch.device(tools.cuda_or_cpu()), dtype=torch.int64)

    false_targets = list()
    for _ in range(config.BATCH_SIZE // 2):
        seed_char_idx = random.randint(len(config.mask), vocab_size - 1)
        start_word = id2word.get(seed_char_idx, "风")
        false_target, _, _ = generator_model.poetry_generation(word2id=word2id, id2word=id2word, start_word=start_word,
                                                               generation_max_length=config.MAX_LENGTH)
        false_targets.append(false_target)

    false_targets = torch.tensor(false_targets, dtype=torch.int64, device=torch.device(tools.cuda_or_cpu()))
    false_labels = torch.zeros(size=(len(false_targets), 1), dtype=torch.int64,
                               device=torch.device(tools.cuda_or_cpu()))

    targets = torch.cat([real_targets, false_targets], dim=0)
    labels = torch.cat([real_labels, false_labels], dim=0)

    shuffle_index = torch.randperm(targets.size(0))
    targets = targets[shuffle_index]
    labels = labels[shuffle_index].to(dtype=torch.float32)
    return targets,labels
#TODO pretrain_discriminator的数据生成部分，可以替代上面的一部分

def train_gan(generator:models.PoetryGenerator,discriminator:models.PoetryDiscriminator,
              data_item,word2id,id2word,vocab_size,epochs=config.EPOCHS_GAN):
    generator_opt=tools.build_optimizer(model=generator,learning_rate=config.LEARNING_RATE_GAN)
    discriminator_opt=tools.build_optimizer(model=discriminator,learning_rate=config.LEARNING_RATE_D)
    loss_function=nn.BCELoss()
    total_loss = None
    generator_sample_number = None
    loss_list=list()

    generator=generator.to(device=torch.device(tools.cuda_or_cpu()))
    discriminator=discriminator.to(device=torch.device(tools.cuda_or_cpu()))

    for epoch in range(epochs):
        #TODO 训练判别模型

        logger.info(f"正在训练判别模型！")
        for _ in range(config.GAN_D_EPOCH):
            discriminator.train()
            generator.eval()

            for x,y,x_original in tqdm(data_item):
                targets,labels=discriminator_data_generator(x,y,x_original,generator,vocab_size,word2id,id2word)
                discriminator_opt.zero_grad()
                predict_d = discriminator(targets)
                loss_d = loss_function(predict_d, labels)
                loss_d.backward()
                discriminator_opt.step()
        logger.info(f"判别模型训练完成！")

        #TODO 训练生成模型

        logger.info(f"正在训练生成模型！")
        for _ in range(config.GAN_GENERATOR_EPOCH):
            generator.train()
            discriminator.eval()
            total_loss=0
            generator_sample_number=config.BATCH_SIZE
            for _ in tqdm(range(generator_sample_number)):
                seed_char_idx = random.randint(len(config.mask), vocab_size - 1)
                start_word = id2word.get(seed_char_idx, "风")
                hidden=generator.init_hidden(batch_size=1)
                gen_indices, log_probs, _=generator.poetry_generation(word2id=word2id,id2word=id2word,start_word=start_word,
                                                                      generation_max_length=config.MAX_LENGTH,log_prob_gradient=True,current_hidden=hidden)
                generator_sample=torch.tensor([gen_indices],dtype=torch.int64).to(device=torch.device(tools.cuda_or_cpu()))
                reward=discriminator(generator_sample).detach()
                loss_g = -torch.sum(log_probs * reward.item())
                generator_opt.zero_grad()
                loss_g.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 5)
                generator_opt.step()
                with torch.no_grad():
                    total_loss+=loss_g.item()
        logger.info(f"生成模型训练完成！")

        with torch.no_grad():
            avg_loss=total_loss/generator_sample_number
            loss_list.append(avg_loss)
            print(f"epoch: {epoch+1}, loss: {avg_loss}")

    torch.save(generator.state_dict(),"./params/gan_best_generator.pth")
    return loss_list


if __name__=="__main__":
    loaddata=tools.LoadData(data_path="./data/sample_poetry.txt")
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
    poetry_generator=models.PoetryGenerator(vocab_size=vocab_size)
    poetry_discriminator=models.PoetryDiscriminator(vocab_size=vocab_size)

    generation_params=torch.load("./params/pretrain_best_generator.pth",weights_only=True)
    discriminator_params=torch.load("./params/pretrain_best_discriminator.pth",weights_only=True)
    poetry_generator.load_state_dict(generation_params)
    poetry_discriminator.load_state_dict(discriminator_params)
    train_gan(poetry_generator,poetry_discriminator,poetry_data_item,word2id,id2word)