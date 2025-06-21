import models
import tools
import config
from pretrain import train_gan
import copy
import torch

if __name__=="__main__":

    #TODO 登录数据
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
    #TODO 登录数据

    poetry_generator=models.PoetryGenerator(vocab_size=vocab_size)
    poetry_discriminator=models.PoetryDiscriminator(vocab_size=vocab_size)

    generation_params=torch.load("./params/pretrain_best_generator.pth",weights_only=True)
    discriminator_params=torch.load("./params/pretrain_best_discriminator.pth",weights_only=True)
    poetry_generator.load_state_dict(generation_params)
    poetry_discriminator.load_state_dict(discriminator_params)
    loss_list=train_gan(poetry_generator,poetry_discriminator,poetry_data_item,word2id,id2word,vocab_size)

    tools.protract_loss(loss_list,config.EPOCHS_GAN,name="gan.png")