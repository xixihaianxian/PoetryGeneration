import tools
import config
from pretrain import train_gan
import copy
import torch
import models

if __name__=="__main__":
    loaddata=tools.LoadData(data_path="./data/sample_poetry.txt")
    loaddata.load_data()
    word2id=loaddata.word_to_id()
    id2word=loaddata.id_to_word(word2id)
    vocab_size=len(word2id)

    poetry_generator = models.PoetryGenerator(vocab_size=vocab_size)

    generation_params=torch.load("./params/gan_best_generator.pth",weights_only=True)
    poetry_generator.load_state_dict(generation_params)

    poetry_generator=poetry_generator.to(device=torch.device(tools.cuda_or_cpu()))

    generation_idxs,_,_=poetry_generator.poetry_generation(
        word2id=word2id,
        id2word=id2word,
        start_word="春风",
        generation_max_length=50,
    )

    result=[id2word.get(id,"UNK") for id in generation_idxs]

    while "PAD" in result:
        result.remove("PAD")

    print("".join(result))