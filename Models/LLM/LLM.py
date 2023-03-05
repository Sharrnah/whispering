import settings
llm = None


class LargeLanguageModel:
    model = None

    def __init__(self):
        match self.get_current_model():
            case "flan":
                from Models.LLM import flanLanguageModel
                flanLanguageModel.init()
                self.model = flanLanguageModel.flan
            case "bloomz":
                from Models.LLM import bloomzLanguageModel
                bloomzLanguageModel.init()
                self.model = bloomzLanguageModel.bloomz
            case "gptj":
                from Models.LLM import gptjLanguageModel
                gptjLanguageModel.init()
                self.model = gptjLanguageModel.model
            case "pygmalion":
                from Models.LLM import pygmalionLanguageModel
                pygmalionLanguageModel.init()
                self.model = pygmalionLanguageModel.model

    @staticmethod
    def get_current_model():
        return settings.GetOption("llm_model")

    def whisper_result_prompter(self, whisper_result: str):
        return self.model.whisper_result_prompter(whisper_result)

    def encode(self, input_text):
        return self.model.encode(input_text)


def init():
    global llm
    if settings.GetOption("flan_enabled") and llm is None:
        llm = LargeLanguageModel()
        return True
    else:
        if llm is not None:
            return True
        else:
            return False
