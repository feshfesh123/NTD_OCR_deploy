import string


class ArchitectureConfig:
    """Example config for digit recognition only"""
    IMG_SIZE = (1280, 64)
    INPUT_LENGTH = 70
    MAX_TEXT_LENGTH = 70
    EPOCHS = 5
    BATCH_SIZE = 2
    CHARS =  '\ !%"–&\'()*+,-./0123456789:;?AÁẢÀÃẠÂẤẨẦẪẬĂẮẲẰẴẶBCDĐEÉẺÈẼẸÊẾỂỀỄỆFGHIÍỈÌĨỊJKLMNOÓỎÒÕỌÔỐỔỒỖỘƠỚỞỜỠỢPQRSTUÚỦÙŨỤƯỨỬỪỮỰVWXYÝỶỲỸỴZaáảàãạâấẩầẫậăắẳằẵặbcdđeéẻèẽẹêếểềễệfghiíỉìĩịjklmnoóỏòõọôốổồỗộơớởờỡợpqrstuúủùũụưứửừữựvwxyýỷỳỹỵz'
    CORPUS = ''
    WORD_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÂÊÔàáâãèéêìíòóôõùúýăĐđĩũƠơưạảấầẩậắằẵặẻẽếềểễệỉịọỏốồổỗộớờởỡợụủỨứừửữựỳỵỷỹ"

class FilePaths:
    "filenames and paths to data"
    fnDataset = '../dataset/'
    fnSave = '../model/best_model_CTC.ckpt'
    fnCharList = '../model/charList.txt'
    fnAccuracy = '../model/accuracy.txt'
    fnDataPreProcessed = '../new_data/'
    fnDataCollection = ['../vi_00/', '../vi_01/']
    fnCorpus = '../model/corpus.txt'