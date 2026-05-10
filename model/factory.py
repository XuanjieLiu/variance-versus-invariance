from model.autoencoder import CSAE


def get_model(dataloader_name, model_config):
    if (
        "phonenums" in dataloader_name
        or "lowercase_letters" in dataloader_name
        or "uppercase_letters" in dataloader_name
    ):
        from model.modules.phonenums import Encoder, Decoder
    elif "insnotes" in dataloader_name:
        from model.modules.insnotes import Encoder, Decoder
    elif "svhn" in dataloader_name:
        from model.modules.svhn import Encoder, Decoder
    elif "sprites" in dataloader_name:
        from model.modules.sprites import Encoder, Decoder
    elif "librispeech" in dataloader_name:
        from model.modules.librispeech import Encoder, Decoder

    return CSAE(model_config, Encoder, Decoder)
