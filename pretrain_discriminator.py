import models
import tools
import config
from pretrain import pretrain_discriminator
import copy
import torch

if __name__=="__main__":

    #TODO 数据登录，较为繁琐，可以简化
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
    poetry_data_item=tools.poetry_item(poetry_dataset,batch_size=config.BATCH_SIZE,shuffle=config.SHUFFLE,num_workers=config.NUM_WORKERS)
    vocab_size=len(word2id)
    # TODO 数据登录，较为繁琐，可以简化

    poetry_generator=models.PoetryGenerator(vocab_size=vocab_size)
    poetry_discriminator=models.PoetryDiscriminator(vocab_size=vocab_size)

    generator_params=torch.load("./params/pretrain_best_generator.pth",weights_only=True)
    poetry_generator.load_state_dict(generator_params)

    loss_list=pretrain_discriminator(discriminator_model=poetry_discriminator,generator_model=poetry_generator,
                                      data_item=poetry_data_item,epochs=config.EPOCHS_D,loss_function=tools.build_loss_function(poetry_discriminator),
                                      optimizer=tools.build_optimizer(model=poetry_discriminator,learning_rate=config.LEARNING_RATE_D),
                                      vocab_size=len(word2id),word2id=word2id,id2word=id2word)

    tools.protract_loss(loss_list,config.EPOCHS_D,name="discriminator.png")