import io
import os
import threading
from pathlib import Path

import numpy as np

import Plugins
import audio_tools
import downloader
import settings
from Models.Singleton import SingletonMeta

cache_path = Path(Path.cwd() / ".cache" / "zonos-tts-cache")
os.makedirs(cache_path, exist_ok=True)
voices_path = Path(cache_path / "voices")
os.makedirs(voices_path, exist_ok=True)


from scipy.io.wavfile import write as write_wav
import torch
import torchaudio
from Models.TTS.zonos.model import Zonos
from Models.TTS.zonos.conditioning import make_cond_dict, supported_language_codes
from Models.TTS.zonos.utils import DEFAULT_DEVICE


failed = False

speed_mapping = {
    "": 15.0,        # Default speed when empty
    "x-slow": 5.0,  # Extra slow speed
    "slow": 10.0,   # Slow speed
    "medium": 15.0,  # Medium speed (default)
    "fast": 20.0,   # Fast speed
    "x-fast": 30.0   # Extra fast speed
}

model_list = {
    "Default": ["v0.1-transformer"],
}


# patching EspeakWrapper of phonemizer library.
# See https://github.com/open-mmlab/Amphion/issues/323#issuecomment-2646709006


TTS_MODEL_LINKS = {
    # Models
    "zonos-v0.1-transformer": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/zonos-tts/v0.1-transformer.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/zonos-tts/v0.1-transformer.zip",
            "https://s3.libs.space:9000/ai-models/zonos-tts/v0.1-transformer.zip",
        ],
        "checksum": "1071ca775f5cb038a5bac3160ad3149b13d495c136dd4315df4d3395486268fe",
        "file_checksums": {
            "config.json": "cd4b7a17bd62a04d19b7a8c92e554f24980fc74b5fb7f51f467f212614d986e9",
            "model.safetensors": "4ac68319d6b8c1b29b94b8801ca41a93a5fc458f4b1d4bbbe5045b2ee77efcf0"
        },
        "path": "v0.1-transformer",
    },
    # espeak
    "eSpeak-NG": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/zonos-tts/eSpeakNG.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/zonos-tts/eSpeakNG.zip",
            "https://s3.libs.space:9000/ai-models/zonos-tts/eSpeakNG.zip",
        ],
        "checksum": "8e1cc4750f6567e5cc276c3abfa5c75f1a8bf279556ca5bca5020bad29e96987",
        "file_checksums": {
            "espeak-ng-data\\af_dict": "25729d3bf4c4a0f08da60aea9eb5a0cf352630f83fab1ab0c3955b7740da1776",
            "espeak-ng-data\\am_dict": "9954365860682c756c8c4459577d3aa0ab77ad0cba6798b746033ada5c3d4bea",
            "espeak-ng-data\\an_dict": "6cf38cedb6693aade49e5a5d27a3c92516a5335125bf85fcae58282e66fb82fd",
            "espeak-ng-data\\ar_dict": "72316426e797777fe4df9420935a3b6a79b37d7e3f3948537ba71cd7b21b2541",
            "espeak-ng-data\\as_dict": "0f9c9f7e98530db66aa6e988a2ee08e8a0f0a7e09693bcecfcb90b5cb67dff10",
            "espeak-ng-data\\az_dict": "e352b2dcf811662d79f4beaaa0c7750706cf669c0409fd815a9bd0244b1adfc7",
            "espeak-ng-data\\ba_dict": "be864c2754fb22093f9265c081a872e46c6dad7a67bb4c9b5cea41d42ba889e7",
            "espeak-ng-data\\be_dict": "ea57c19841b2910727fda8c5982b3c7467ac80d6e84fd032f1d5a19ee743bcd0",
            "espeak-ng-data\\bg_dict": "8fa3adec8e18a3e695b2bbea1d4780182e17222d8b9f8da9908a237828f6b063",
            "espeak-ng-data\\bn_dict": "562a64ec5ea319622ec2286c103d4110222a1712129a3a9f07f2ea14a660038a",
            "espeak-ng-data\\bpy_dict": "6a752c17d300b09eaa6f0768050b62e94420cc2a7b2f1cd2c27f41217014330d",
            "espeak-ng-data\\bs_dict": "d5df564f511af082884719685db86ec4df23467cf012ce25e898540baf9de65b",
            "espeak-ng-data\\ca_dict": "1be203855ecb0bdfe66578ef5552f83289e39972d78c52714d061b7f7493df4a",
            "espeak-ng-data\\chr_dict": "8f9cf3c0b3524af053ffec7c246f2e4c5c431e1e1699f7a3962ec80bedad5ff2",
            "espeak-ng-data\\cmn_dict": "a89070f47954b46f2a33a9128a4a5c113dbd4effd52ef073e43ee2ae1ecff950",
            "espeak-ng-data\\cs_dict": "5ec02aba90465007f20a5c8aeb889275834cfa8c8f621c3d9bee1ca90d21788d",
            "espeak-ng-data\\cv_dict": "8260837c8f18fbe5870905ebb2850dfd18e252150ad6c868b96d9d2c6398ea8a",
            "espeak-ng-data\\cy_dict": "31d74ccad218ab80b79566caeb8d21bcf0fe43326ed7c437fa736d39b85e603e",
            "espeak-ng-data\\da_dict": "fe0c21895b4844868612f9ea846bc6010e30b5dd1f4cb1567459d73d2792a8ac",
            "espeak-ng-data\\de_dict": "e9ac048df5cde03b74907d591c7b29b45ee9ec5fcd4592b0b894f0083572fcfd",
            "espeak-ng-data\\el_dict": "5d9f759750131da777d2fd4f06aa602ed746d38f27b84dd9a2d4a550f2cd452e",
            "espeak-ng-data\\en_dict": "00f82f7503621fa9114e602c210017bc741a8ec50b5524b8e430e71df972562a",
            "espeak-ng-data\\eo_dict": "849f061abddd5388bac112b32e6f7bf747e15e1b3ae53fe91079381effff4cc9",
            "espeak-ng-data\\es_dict": "aac30f66783f7d55fab919de4c2b866bdcb0644108f331f22f737e4bd3a7da18",
            "espeak-ng-data\\et_dict": "1a6734b4e6bc9de9a4c94cfb434873e0a7f4ae4ec89549b3137ae5491cf655da",
            "espeak-ng-data\\eu_dict": "762a6eff63552ef72cd88bf1e92dc5a823eb378d50cb85c7917e8c5bba079137",
            "espeak-ng-data\\fa_dict": "b2f9ef671863d0167dda0e7743cb2dfc9bf2ebccfd3a69a209d2794b06d5be17",
            "espeak-ng-data\\fi_dict": "dda999437bd4b37a7f7400a2230aace1200a52906ab0b1ff3507d44ab19d64b1",
            "espeak-ng-data\\fr_dict": "e399ab924c4d10beef1fc310b30ea56e4ddfd8b4b64b8ed978e9c65394d49b2d",
            "espeak-ng-data\\ga_dict": "b2db03d21f7900b697d7dbb8bec855cd1117e9867f19884faf11c4df626659f5",
            "espeak-ng-data\\gd_dict": "e92b47dcf9a001363ffee1e62863faf264ff5f16e10cd9cf1f1e83ca89ca14aa",
            "espeak-ng-data\\gn_dict": "4e23704b4c5775631bfea9e8a36a380785a49b61023c5433a449daac2d8805d5",
            "espeak-ng-data\\grc_dict": "488779493993d221b09e713e104a221ac1bb35765f826acc26d3d3c1aa3a35ed",
            "espeak-ng-data\\gu_dict": "cb1fbfb53bab2ec6d9e44cd9d12ccb9b80de49f274d772314389c477eef530fd",
            "espeak-ng-data\\hak_dict": "e3075c7d281f95a2ce2264c01edbd09e962cb3ed29400766c5d2116a08a185c8",
            "espeak-ng-data\\haw_dict": "2b4e77b61cadfc5ee622aa8b1d23941e36101d9bdbf8d809e0fa1b442d2c3f74",
            "espeak-ng-data\\he_dict": "98525d2145e12ec29cbc7f179d6fae50e4f4a5df4dfe841766eae4267137b1ef",
            "espeak-ng-data\\hi_dict": "5a68c9532624e57ac845b26ce1e2e5034c4f6353bede46ecbe57e583ec8effd6",
            "espeak-ng-data\\hr_dict": "5f98fcf05f97335a5436a8a9643677da6f158f2a9ee1d41bdda1725b1bb3527e",
            "espeak-ng-data\\ht_dict": "d577219d15c29396db254aed958358456ec1501cdf62cb4ca5413752f2ee7cc1",
            "espeak-ng-data\\hu_dict": "055d563b839d599db40be5b1cc2af9d31793795b328eb8d4f389ec7d38fe7e81",
            "espeak-ng-data\\hy_dict": "8b808ad8f1d4cae32b3b785b96d1e27b3af28e32f4dddfed7c93e129d63b2587",
            "espeak-ng-data\\ia_dict": "1c665603a82beb81e0cef2878c9b44642ae6d519e301fbbd06326e03325febf3",
            "espeak-ng-data\\id_dict": "8fb3f23f80f71f023219cee7032223f3f6886cc5040eb4c8c507f79d1b84e98d",
            "espeak-ng-data\\intonations": "3f8af65fd3eda9759a10f021d61361c120871f463515229c925995c7f90918cc",
            "espeak-ng-data\\io_dict": "0d9d6995fead1ccdd5512f7909ec8f9efc8ccc97f5c9d23ecae6b3ba77ffadbf",
            "espeak-ng-data\\is_dict": "8eeab5b1ca3f75362881fae3de3316fcaff281179a4c51fceb0098f11ead96e0",
            "espeak-ng-data\\it_dict": "ea1455eaa6e043aef7f4aaf7a2661aef565c81bd8b5620cdd0a09c6efabf355a",
            "espeak-ng-data\\ja_dict": "34f849974661fe9b9ea9da1345f2bc78c784629186ac344ff0d0cd628652954d",
            "espeak-ng-data\\jbo_dict": "71298417de1e5fd07df275270a5d47ca8820cb8c9f5dc2156c070b7a3ae7999c",
            "espeak-ng-data\\ka_dict": "a25cefa95a54bc71df0ba8e938bcc2554687589c9d5615b22ee070e8bef7c6fc",
            "espeak-ng-data\\kk_dict": "88dc0106629a57d0aca5073afb11db1a044799185355088a7ad4331c836affd3",
            "espeak-ng-data\\kl_dict": "1121dd806a4ec5105aa8dbf0d858eb606bfd564ed8cd6e3637fc07996c30f93a",
            "espeak-ng-data\\kn_dict": "c51e0385937b407b36c02fd2c85dfecdc5cac411398ee8a74eca15b35617af64",
            "espeak-ng-data\\ko_dict": "93e8fb2972126a3d10185fe177955a81f2fe3a580995f3b8eb9357327d9a9d66",
            "espeak-ng-data\\kok_dict": "00adf88d3d8e5afe19559918f77761f8b99172c33e5aafca342682f0c1371ec6",
            "espeak-ng-data\\ku_dict": "65de1c5429e8d193775b36fcf169a60fbbb9964ef13a4a1bab7ff0ce855cad20",
            "espeak-ng-data\\ky_dict": "8a0f7fac60b086ef00be819e4e0413eb77f5ad3aa557dabeb722e270e1d0f7d5",
            "espeak-ng-data\\la_dict": "17818dd567936fde032404fbbd6177844b9f5fc5a915480f567272147936f917",
            "espeak-ng-data\\lang\\aav\\vi": "3199c980f9e23a88a2aa693cd631bf4fcb0f3408c4272bc01b7ac0ff8e79d778",
            "espeak-ng-data\\lang\\aav\\vi-VN-x-central": "ad6b06e320e20e32d9f82e6eda8af27fcd8fae1603c6a9d603de89c737e4b531",
            "espeak-ng-data\\lang\\aav\\vi-VN-x-south": "18935c79c1c00f732654d3fa713a4b24f26de5219f961844a2aa15e62fcc1ee4",
            "espeak-ng-data\\lang\\art\\eo": "0254b93c8af79d9666a7667ca8b33b21a5453012da4ea0acd79660a9695f89d4",
            "espeak-ng-data\\lang\\art\\ia": "8fb11718030f8b29146f773ca462b838a6c660f166933eb6f1498c361f2e958b",
            "espeak-ng-data\\lang\\art\\io": "8e8391c7f40b7f9d2307faff8018a40dd2b448df4bd581a2950ee8393a00cebf",
            "espeak-ng-data\\lang\\art\\jbo": "d9529de82dccbd95f83c1de401f4ea661609bde644546e9ef421241b7043c00e",
            "espeak-ng-data\\lang\\art\\lfn": "394874fd9022e37cdc307c3386e81f1c0c133a8e9c5fdff74888b519f2c6d648",
            "espeak-ng-data\\lang\\art\\piqd": "d1a559b714cd8dd0ea9e46c4574e06c4f451957fb17bf39b945a89b9586f8b97",
            "espeak-ng-data\\lang\\art\\py": "59e54b95e8571417f43db37a7fd7e351af4e22a8c250cfb66af394f8ed6c4e9c",
            "espeak-ng-data\\lang\\art\\qdb": "8c17442e536a6b83249769db8c98845b400930576be0ce62c9716a79b1bf1212",
            "espeak-ng-data\\lang\\art\\qya": "78ae3f4d70208a9f69e2c26a6a3b5db28d7e6a58740fae937f51dd7fc9faeb1b",
            "espeak-ng-data\\lang\\art\\sjn": "61d85974b6841dfbfcc8cb92ad0742696cb87a6a85d5a28fe99895e46ef9de6f",
            "espeak-ng-data\\lang\\azc\\nci": "7bf70cb18e969398f74428b91c694a4483eaebd6733ecc437eb8714bbf7fff76",
            "espeak-ng-data\\lang\\bat\\lt": "7691c7ca1e79fc9728c70a0d22830eb113a0124bf96807e2a1ace3c6bf0c2f89",
            "espeak-ng-data\\lang\\bat\\ltg": "4d2dff5cddb4c186c7761db9e3d6a09889f20db1024e735d8a273eb358bcb1f9",
            "espeak-ng-data\\lang\\bat\\lv": "f007788963886133b7adaea320e0172c33dbe3ff2ed3f62b0ecdfbb177e09ef0",
            "espeak-ng-data\\lang\\bnt\\sw": "271dd28b3ec41adeea53f1146249ba6cab61572963ab2743041114d95784e368",
            "espeak-ng-data\\lang\\bnt\\tn": "2243a1fc208dfdc37e154e837c0e782afc325d4acfbacc22718b7f05583ace92",
            "espeak-ng-data\\lang\\ccs\\ka": "68788b99dba7b82cdf8fc9dd05134bd72cf21f0b4c07b13b67b6f036e6294cba",
            "espeak-ng-data\\lang\\cel\\cy": "67bc33b4a66041e6343dfb98e1950d1089eecb68731f346fcc4f47cd07e79970",
            "espeak-ng-data\\lang\\cel\\ga": "29c2cc5736ceb2bd9ef745694d4abf7ed363a33569dfbe745811d4703b7b6cce",
            "espeak-ng-data\\lang\\cel\\gd": "7da0ba464579c367089109826ec15c2584a17417cca4a2c66dae0318d33e3128",
            "espeak-ng-data\\lang\\cus\\om": "b7c563f1effd83c08f3d229fb5a13d074513b85165ce4359a674ef30ca71786b",
            "espeak-ng-data\\lang\\dra\\kn": "c988a5226fa27ee6768b45566ce999ced9a4b68f8e79232b323d5903f4bb2258",
            "espeak-ng-data\\lang\\dra\\ml": "aa7e1f1f239ba55a8a79ed1f3544e9a51b94efe53925de60d4ff737273660b5c",
            "espeak-ng-data\\lang\\dra\\ta": "5bc213eb163bbf069cbfe4f2e9f7e7db71a719d8c0a96e95401d94c992960ed7",
            "espeak-ng-data\\lang\\dra\\te": "176ca4a81feb2181393edff3e2d5c0253cb52239392c4b809c61501d60c69811",
            "espeak-ng-data\\lang\\esx\\kl": "d08cf6ff60b99e6c2ceda549bcbb1a57580915d6a4d50c7f0f5d75289d9fb45a",
            "espeak-ng-data\\lang\\eu": "a16e4e4d9b910a6e25a2a55913c812badc4fc3cbece0b8d49a64af112782d5a5",
            "espeak-ng-data\\lang\\gmq\\da": "e8dd5e0d30ff0694a7ee839faa8ed35c015eb927a737aba1f120c00005692eaf",
            "espeak-ng-data\\lang\\gmq\\is": "16c077e6222acfc7702046fa3579bb5d6b57ece98646229e13be2233f9fdcbd7",
            "espeak-ng-data\\lang\\gmq\\nb": "d47966621312fb95570539f446982f6deb98fd46d9b71fec4a0005a6bace8f46",
            "espeak-ng-data\\lang\\gmq\\sv": "02727d3b25fc82409bd2cd1ebb444fdb2d5d3f3de7f76b95fc9ed9164c3e59c3",
            "espeak-ng-data\\lang\\gmw\\af": "fb8ccc05f5eae1754b5aa7e92cb919defdd0810b70eec19f7c77cf836e51db09",
            "espeak-ng-data\\lang\\gmw\\de": "f3cca92f94b70f8c25a29ee0a4c9ce4c7f1022241532e0647fa2b7f698bf104e",
            "espeak-ng-data\\lang\\gmw\\en": "4605d5330801de3641c6e366d15f129ea1f5ffbce8722642aba01ace07ab9c83",
            "espeak-ng-data\\lang\\gmw\\en-029": "faeb8cb201056775f733acbd908c66555fe57bf01cb063920f178f389cbf85d3",
            "espeak-ng-data\\lang\\gmw\\en-GB-scotland": "1ce4282c1f4385dbaf0035e799b2ab1ee9e9a3ab4829100594a3047727b52353",
            "espeak-ng-data\\lang\\gmw\\en-GB-x-gbclan": "2040e176f1f7f27bdce81c32d7e2b1662449d54d264afd3c0ed871b8e42b6e46",
            "espeak-ng-data\\lang\\gmw\\en-GB-x-gbcwmd": "927a5ab891c65b30b4426b151d1623a01024f6b773a38497ba1c90cd77a95747",
            "espeak-ng-data\\lang\\gmw\\en-GB-x-rp": "d0625af7f58561b1b8cf96fd7f93eee6553bcb3eadb9020ae0757bf96e5115e5",
            "espeak-ng-data\\lang\\gmw\\en-US": "41534c2a22df5dd4f1052ff9e1a33a3ea7bff5a26b5c02bdad5ba8ddb7524704",
            "espeak-ng-data\\lang\\gmw\\en-US-nyc": "62b13f9a239fee09b8ddc230a171ebdcd0cd2f14c4b198ebe0b6f9682ba2f372",
            "espeak-ng-data\\lang\\gmw\\lb": "071beb0e769afeba32177796fc0f1386bdc89a3234bed2bf720e76cbf73f5b99",
            "espeak-ng-data\\lang\\gmw\\nl": "56ae04fae28ae9371ed6db28193189e4ee962ba8236f0bcdeef6a06ea4dee374",
            "espeak-ng-data\\lang\\grk\\el": "0b529c998af07c72bf516b5750709464f6258c56c6916b8b8f0c843be9836fca",
            "espeak-ng-data\\lang\\grk\\grc": "319f950b1ad7b1f37efaa390540bf957bddac16d0557a6c3adaff27e0c64e12e",
            "espeak-ng-data\\lang\\inc\\as": "3f58f5628f792b00a5ae6e8fe3bf8696f64af2838e5723799ddbab6027a0b378",
            "espeak-ng-data\\lang\\inc\\bn": "bc3ce23384b9d0bb3bac06a09d90f690fae32a492dcbb030f0e56ba67b5d656c",
            "espeak-ng-data\\lang\\inc\\bpy": "e528d1189b995b94966cef4cc182469cc54f7da8efc5860375c4ecf186ce6146",
            "espeak-ng-data\\lang\\inc\\gu": "f938a755d270745188c2cf2aacba06642a96bb73ba2a0325384121b9d5ec5407",
            "espeak-ng-data\\lang\\inc\\hi": "3c1c3f916f57f2d6d6cbf7923c0f39a36ac2140e2a69ddffce917d7b8bf3ccab",
            "espeak-ng-data\\lang\\inc\\kok": "810ee23fbbe9c186a036bc12f4e5c0de674a132e2e099c9b3dece2a1c9cdb28a",
            "espeak-ng-data\\lang\\inc\\mr": "29063ff3a5257d0d3d7c215dfd4c14deed39e1d48f15894202e6712459720efd",
            "espeak-ng-data\\lang\\inc\\ne": "d53fda082ec174ef33ed3c3468fc973c43b4ef71aca8c1a28cb344668f4838dc",
            "espeak-ng-data\\lang\\inc\\or": "9b52ede7d19659f884854de3dff12597a96f4d6ec7cb4146d2f8dad1453c742f",
            "espeak-ng-data\\lang\\inc\\pa": "eed3cf477f81d13559dbb367256f9532b20dad1432c15b206c777217b8f570ca",
            "espeak-ng-data\\lang\\inc\\sd": "65bfe0e37799029d42a370eb627cca5aa4108b967fba6c8141878d693c903619",
            "espeak-ng-data\\lang\\inc\\si": "8f0993b35ba5b075c38ae76640971256e560abfbd721b608d16a48588932596d",
            "espeak-ng-data\\lang\\inc\\ur": "5b7af7474fac265c7bd7d269111994a09610f954690698f55a8613f3fa973318",
            "espeak-ng-data\\lang\\ine\\hy": "b6ad6129079c49ca915c3005ecb74a02bf5d69cbbc8ec3f8e0f6b949cb3a768f",
            "espeak-ng-data\\lang\\ine\\hyw": "26f8a863c74b3d19c9c42fcf16ae5151c12ac52ef8fab36cae1e5f0d4ad30e6a",
            "espeak-ng-data\\lang\\ine\\sq": "865fde9017b5af82acb844e5f9793998fed422a208ad8aaf9dedacad3bf27af1",
            "espeak-ng-data\\lang\\ira\\fa": "0dc7e93be4d1fbdf7ff52de2fe8bcdc23304e6d86e012b4ab8276d626b5d980c",
            "espeak-ng-data\\lang\\ira\\fa-Latn": "a04cf8efed8db1ac33b306f1f8c6f42733dace87c9fc9f34a2d227b3bb291e28",
            "espeak-ng-data\\lang\\ira\\ku": "57db3a210d73412a49e961dbed9b27077f5c7b967a19a0eb9f9363ea4b16cea5",
            "espeak-ng-data\\lang\\iro\\chr": "f0e2fb8549306c07f571d3e77c89b1d3e9022d448d58e5d424441e01ccce88be",
            "espeak-ng-data\\lang\\itc\\la": "b70d13a264352eba2d22a5d0ef21730849509d18f7277320b008a56decc6efc2",
            "espeak-ng-data\\lang\\jpx\\ja": "6fd980445bc52e0f593062929f5b62831448abaae2a92739beba49df27a79743",
            "espeak-ng-data\\lang\\ko": "c2dfbf88d9aa2ac9bf2f30e92e260120e4f2c21123a02265014ae98d59d5774e",
            "espeak-ng-data\\lang\\map\\haw": "b19f22efb6714354d8d1360f449e05f0f84d50d6fa3c98fcfd619ea1b4ba8996",
            "espeak-ng-data\\lang\\myn\\quc": "76a75b77c672c432b56b84d517310d6aa75c4b3f9de580e290199804366174ea",
            "espeak-ng-data\\lang\\poz\\id": "ce1fcddab3e756a7cdba3dd009ad24cd57e1fde29ce395698355da24e32a6075",
            "espeak-ng-data\\lang\\poz\\mi": "2146cf3868c79b853bb5e41ddbf611d51b4b0a14087199f16194313562b98987",
            "espeak-ng-data\\lang\\poz\\ms": "8a965182ee0c5a41357197c05cf047d21b96744775a57a344d831a961e30351b",
            "espeak-ng-data\\lang\\qu": "3069e443ecf3421bcb3b35b864e1a89fa9465af072f86eedd46a415990aba6e6",
            "espeak-ng-data\\lang\\roa\\an": "ff8aa97c55286cc90531cad5df3cd36165abba11502a84d53b0fcd923f965f23",
            "espeak-ng-data\\lang\\roa\\ca": "787df034320467236292bb46f1f8781a2998a185f117d350c018b6cfcf360854",
            "espeak-ng-data\\lang\\roa\\es": "966aa015ea5646d79f0ca4807cf5da7339aabd3782b55cfa5eb0d8c3fc8fc588",
            "espeak-ng-data\\lang\\roa\\es-419": "a55883208e2c46d0ce51fc9de86523ceaab59c56dbfd7d4a2e752fdf82f4961d",
            "espeak-ng-data\\lang\\roa\\fr": "95f44834b48c075dad13eace54d2c98ff79b81aa0074dd67eebaf66c2909eef8",
            "espeak-ng-data\\lang\\roa\\fr-BE": "72496a316e47b997f893500b8953dc01a10b86a2d4238e13c1260f50b47d792a",
            "espeak-ng-data\\lang\\roa\\fr-CH": "b449b0fd99d9cd93261fe1c6b816448b99b4a10420bee3f301e669568ad0de55",
            "espeak-ng-data\\lang\\roa\\ht": "389bdf3e984a2f93b7060baad82452dac65b83bf4482d3884525a176ae0e0524",
            "espeak-ng-data\\lang\\roa\\it": "0d9069eb9a96db1c55c131b2bb7d1f5255c68fbddc2199ffd3295b52519a3256",
            "espeak-ng-data\\lang\\roa\\pap": "41a5b5e91482a5498a66711c05873a634bbd418786b0ba037032df846344cfc1",
            "espeak-ng-data\\lang\\roa\\pt": "a44060072f790e41912c57b7d52f7b6438c4383e0dabdb052f6fdb7aacc23493",
            "espeak-ng-data\\lang\\roa\\pt-BR": "84dd4e34c970fadbe32211eadecfdfaab52432479c0911e43300da88a33398ba",
            "espeak-ng-data\\lang\\roa\\ro": "b12af10ffc3f89cb87767ca701c6fea0def8a712de63426a9c3d0bf2432467f8",
            "espeak-ng-data\\lang\\sai\\gn": "ff82d2d163d3e06adb580254d89012e689b2615c46579883562838ab401fb5d5",
            "espeak-ng-data\\lang\\sem\\am": "5970b16bf865528ee96a78c08498244bc8f54218fc2add499716b0e0c09969fb",
            "espeak-ng-data\\lang\\sem\\ar": "43f418f8140e5a7bd7096963da20ccba9247f6be1fcdb08cb53dd7967dd72eee",
            "espeak-ng-data\\lang\\sem\\he": "de9195e743b91fb7fbef7493bd5ddccad14c380a58d5aeb7c1d11d3419ef3b86",
            "espeak-ng-data\\lang\\sem\\mt": "d23f38fae3253b6fb42e3336760ab482e676893692455465448bc9fbc0ac3e83",
            "espeak-ng-data\\lang\\sit\\cmn": "b4424b6b0405216bc5d56183744a2b9824869d7c6b01b9919a8c1f55e537f60c",
            "espeak-ng-data\\lang\\sit\\cmn-Latn-pinyin": "14c2fc097db280416b688c2bf97de2e5da532c279754020fde7bf6cd624cdd10",
            "espeak-ng-data\\lang\\sit\\hak": "c03c30b43e0b0abdfa689445baf9f442664a0373154a5d0ca75a33d30f46aeed",
            "espeak-ng-data\\lang\\sit\\my": "4b5ae2566cbf30aa59207020469e09832e66a7883818788f253996c534303dca",
            "espeak-ng-data\\lang\\sit\\yue": "83f3dac92af5b4aaecc119515698a6979524be236b28ea8f1024ec707d99ea5b",
            "espeak-ng-data\\lang\\sit\\yue-Latn-jyutping": "c56d45bb9a9cd97efb38edabe31c7dd3feb0518cc900aac6a891dcb6bd8eb428",
            "espeak-ng-data\\lang\\tai\\shn": "f16bf2de219d3ae9a6aec263ce9c684d9cd3e61258d04c7948db0757c67e2c96",
            "espeak-ng-data\\lang\\tai\\th": "6c8ac809868de659e06673f1fc2b56dbf4637ff9980da7a4726b846478b4cfe6",
            "espeak-ng-data\\lang\\trk\\az": "52b424332de4c443f3e9c5ce4de5aed6fbfd421269fc825756dfc6bcbd7e1ad8",
            "espeak-ng-data\\lang\\trk\\ba": "a0cc3a06293962093c66533f9c1cffbccf083a2eb666751f2778a02a5316d98c",
            "espeak-ng-data\\lang\\trk\\cv": "94baf6bacf581cc20337471c599f85ecbbe211f0b8476351ff19450c6541ea92",
            "espeak-ng-data\\lang\\trk\\kk": "2219f4d6de0fbb1dc9982157c6f3cc5f4c8abe5fcb605908a9d1054b9880cd51",
            "espeak-ng-data\\lang\\trk\\ky": "8239dfe8bdc6d4725880f5192b4eaf645f839e40cadfd145a4409c8387ac2d59",
            "espeak-ng-data\\lang\\trk\\nog": "e0e27fe61d406b6f4a4879fb8f81a14bc65997106666933991b818c7d79fabc1",
            "espeak-ng-data\\lang\\trk\\tk": "d78263398c135119909ad63a3ecb6ffe3aff08383212934dd03695222a9786d0",
            "espeak-ng-data\\lang\\trk\\tr": "490eb5a2f777a2f3396f3a824ed8021bd77a94be801214411461bde6689738c9",
            "espeak-ng-data\\lang\\trk\\tt": "8b9617a9b6c56f4a2fe2f59d9ce55426837196567962716f9330f70948f8f9b0",
            "espeak-ng-data\\lang\\trk\\ug": "65d0cdbc96efab4d9e3cd0095440752a31dc7f425edd3ba0a53c4dd33eac5d63",
            "espeak-ng-data\\lang\\trk\\uz": "04fd711dfda1e00d17dc8b1b250c24d70cc3c498b27e35d31065ac47441a6759",
            "espeak-ng-data\\lang\\urj\\et": "f30cdc672c6f01c309aeb14b0176ca644c4a6667b36feece948d2d3c2bf028f3",
            "espeak-ng-data\\lang\\urj\\fi": "31c40e171eaa7f0a902a2a22d2383db90f4e15e9967b020b1e8f5bcdcbb60db4",
            "espeak-ng-data\\lang\\urj\\hu": "c575421d2ae397065d35af0eb4f79dff4e85f3435ccf45da874f46b3fee1338b",
            "espeak-ng-data\\lang\\urj\\smj": "3d709dec77bb633af0d57b89668dbaaa72d9be8688118d6909f7d457bec12a50",
            "espeak-ng-data\\lang\\zle\\be": "cc437f6bbd7b16705a5725c22a1319cc2302bb2bf39e1c87d7367f0ee747eaa1",
            "espeak-ng-data\\lang\\zle\\ru": "cfda2c25f5e2ae1064046ecec87ea6079d014a48cf110c1a9f18e485cf62c65c",
            "espeak-ng-data\\lang\\zle\\ru-LV": "980e379667ccb94abe33b9c9c660c08800e3f31e36fe4a939551398ab98457ae",
            "espeak-ng-data\\lang\\zle\\uk": "5f339adca7c285c9e3c7afb5518af079dd3205afb8f356a42c2c40e8e3e879b6",
            "espeak-ng-data\\lang\\zls\\bg": "9b91accccb0d53bb2a7bf21922846ba58c74bc082ba71ae32bf12cb0d092f853",
            "espeak-ng-data\\lang\\zls\\bs": "0820e7c78b68fbdaf86d83c782cdde1901b642d1cdbeae4181c6f5f5e21e5e89",
            "espeak-ng-data\\lang\\zls\\hr": "cef3ef0411c66f154e6c82f7decd95beffa8cbfbe953335658df9f125bdd6096",
            "espeak-ng-data\\lang\\zls\\mk": "6dbc2cfb8999141bd0782cd463d4880468161088190c6b4c9412f43552dc39e2",
            "espeak-ng-data\\lang\\zls\\sl": "e1c2bd1f75c0eeb86f64182472d5f597ce23e64ad7478c248c2f75b694655942",
            "espeak-ng-data\\lang\\zls\\sr": "0e076528d3566f6e59b807e16e274dee505e88a6d477da188ab8ec442eae44f4",
            "espeak-ng-data\\lang\\zlw\\cs": "1477cdba5ba3bce85db559accbbd8a24b86b3f6d9d9988bc587e89357c1c4431",
            "espeak-ng-data\\lang\\zlw\\pl": "83c586a2b724b24b9f98d21ade5022112c465ac446ca09930286980ceae1c781",
            "espeak-ng-data\\lang\\zlw\\sk": "f46ff673c7d58c1646096c174d58da34047d893d0d2993dbca260ce54abe77ef",
            "espeak-ng-data\\lb_dict": "609b4ec8edf762661bd0d6512630461883dd16576839f7d70d4638f4037a439d",
            "espeak-ng-data\\lfn_dict": "10281bc404f429165a330b800eb64674e37a135d1b4c689e57fb1d9e827d6924",
            "espeak-ng-data\\lt_dict": "68ef55aba9545d27ce73e7d4aca1c2fa51d2c3d3d626152fbc22bf5a90fbff11",
            "espeak-ng-data\\lv_dict": "8a081cb443b949791348d7d547692d886a596993baa3437b57558305572b5f8e",
            "espeak-ng-data\\mi_dict": "401621fdf61da47d0513e80279134a3515ced898aaadd5205b82cced0f2df042",
            "espeak-ng-data\\mk_dict": "e4960ff4e74e1096ff1c3bccab62ce0429fa4faa725e546cd5d33e43f45225be",
            "espeak-ng-data\\ml_dict": "b12687557c38f9bcd17417854253ddde3781e3b0c16f9470d732205818e7cd38",
            "espeak-ng-data\\mr_dict": "f712a7525db108ef9769aee53fdd1cd141f0df372778924691b89623aa6fc22f",
            "espeak-ng-data\\ms_dict": "74b01897385ba1f018de44c0268048fa5be588bbbb702a900bb35c691a466707",
            "espeak-ng-data\\mt_dict": "1f76761bdfd481a984b15d07522df23c9444f4ac21ea9671917e08bbffa7f8d3",
            "espeak-ng-data\\my_dict": "216f9615246d34d70dced5519b349b61296fd51e7a83cc01620d39102dfdecc5",
            "espeak-ng-data\\nci_dict": "38c7201e7726c8d508f8f8611349cbc44fcc4684a0fee4db7285d83e808a5b69",
            "espeak-ng-data\\ne_dict": "43831c3f53ea6e939cac37b5ae5bd7a2ea8237aa46f0e7f2441ab772a3c51dba",
            "espeak-ng-data\\nl_dict": "d28641049f98c76b460c3dddde3e2edecb7782c4eb9a3f242391301ff526c83f",
            "espeak-ng-data\\no_dict": "43b828f8c1709e7abb23a0cc6416fb32e4814e3f1ff1db4baf2935bf7c77c9ff",
            "espeak-ng-data\\nog_dict": "e83030a54fdce1e85734a5d545e835937edf2d32ffba6ad45f0ca31bccc2ac18",
            "espeak-ng-data\\om_dict": "3624e67ef8d9c20d894db0adccaccc61ee852977cb77d5fe5801e7b4a528fcf5",
            "espeak-ng-data\\or_dict": "e0dd737e9104da63a27d86cd464cf7764837625b1513169b8d28fca1bd3c8c3b",
            "espeak-ng-data\\pa_dict": "cee0e2a2bfb166e602ff84e185351a8ac311571ccdf46e1831c7a25245563c10",
            "espeak-ng-data\\pap_dict": "770039bbf4ea8c5a3ffddd98c4831ca02c9f39e1ff6b3de7729fa71a8f3bc0f1",
            "espeak-ng-data\\phondata": "cf08a27d89c71d11ec20ed26aee3ec0494b81ef85f8193c44f7a8a00231da5c7",
            "espeak-ng-data\\phondata-manifest": "642c8ae8e4aaa8a5e0a0baca97ba12ae6382a7ad249b904c84db1eac7bc21b18",
            "espeak-ng-data\\phonindex": "642d5daa128c0d08828d4352b850b8c2f06d6274b8c46c157b3a635da988aa69",
            "espeak-ng-data\\phontab": "88799c9eedd188a63e63304b86b1abc2dc91a8ad29c19f34e3da4a3d22ab4401",
            "espeak-ng-data\\piqd_dict": "33004a36210c33291aef8a838d477667eed7af52ff5093152891379b1774188d",
            "espeak-ng-data\\pl_dict": "0e0ce6cfe4eba41a1223362a046390dca359a3e597af8243adaac84d624ef18f",
            "espeak-ng-data\\pt_dict": "7e8398f15fa4a21db39a213381f28ec0d302b22663c644e865abb61a09c1b004",
            "espeak-ng-data\\py_dict": "2255348879547940e8b49e0ca1b80452748f142cb6e267e49cab0560c720129a",
            "espeak-ng-data\\qdb_dict": "c9b7f307ffe54f4aaf323290eca1c620572919035b0eb268e5c94f5d7a430058",
            "espeak-ng-data\\qu_dict": "2a35508b6d98457910847cbbacfcfdcde36fa79341cc4ab8cc3b03d9b403b597",
            "espeak-ng-data\\quc_dict": "458c1ca6dcfa288ea2584915d206c5127512501174d82655de2cf031378e6dfd",
            "espeak-ng-data\\qya_dict": "8198581d187f2963a22f8a8e6072290c795fc4f6c5175fd59e02596927cc1bbe",
            "espeak-ng-data\\ro_dict": "b14f68866b0440c4b0d95fb7da3b7ffd3c59b693e8593d188474fabfc9a1117d",
            "espeak-ng-data\\ru_dict": "4f3d9e66d1644f43518ef8dc3b6c350d0ce833b36bfe6709b6a58b1ca4a630cd",
            "espeak-ng-data\\sd_dict": "d655bc0323884376ee7c17c91f1b5864d13b6fde560b91f17cdf0744b93cc1ee",
            "espeak-ng-data\\shn_dict": "173dbb10a1bec6daf347c481899754b0caa8977bb78c585bf2c7a2ac4e9f7730",
            "espeak-ng-data\\si_dict": "a241437d4b64fd328ac4a3e786c9465c60488f6fa31cbba04153f628cdc61f27",
            "espeak-ng-data\\sjn_dict": "eb70510144230f6791574d86bae0918b4191b076dd66c555e6ae3d5637daadf7",
            "espeak-ng-data\\sk_dict": "58b9bc11c9a95d81e96966242367009f8d02ab7a3f8f4fd77e3581ed4159f0a8",
            "espeak-ng-data\\sl_dict": "9cb2bcbf616a5e7dd7ca769e7d222a48c613299ef1d76e44c08b002f8c22861a",
            "espeak-ng-data\\smj_dict": "815270ce38a59c192ef5aa8801119d9b3ed973d9e23dfa6da8d5d27ded5a9cd6",
            "espeak-ng-data\\sq_dict": "6b74bef94c3319407c27e44d944e765d036a88b30b4c17f30bbc182dbffcc4ce",
            "espeak-ng-data\\sr_dict": "770cae9c516e48af7896629faf4abefaf041e1fd4ae184011aad07b97196f613",
            "espeak-ng-data\\sv_dict": "565f058b14f1c89c6bfa44fc3700793886a39c7818f1037ad1f4961fa7517509",
            "espeak-ng-data\\sw_dict": "375fab2ccd0d669177ad406b20b0cdb5851a1e86f093672f24f72c29fec8a815",
            "espeak-ng-data\\ta_dict": "8f3d855c8d7a35ad0e2b22bc7c15a8830e5fc235492894e5db40458655076ecd",
            "espeak-ng-data\\te_dict": "640407c66a5bb975671243325de616c221f90bb2842d3a36278facb68c0772f8",
            "espeak-ng-data\\th_dict": "b5f0b90201530fb22f0607a49a8a1ed2250f9ec3bba9879cf7a361e3fa4d89dc",
            "espeak-ng-data\\tk_dict": "7629e232c79431e118226c90c49db978be8eb7228a010de646ad4bb3e58bc7bf",
            "espeak-ng-data\\tn_dict": "67f68dc336e7c65b946c95414cc8ae96cd10d241783bf857b3f990ac73221b2b",
            "espeak-ng-data\\tr_dict": "0aab44de6c6249f753805cd8ed54584d3fe142f8ae6c4ec2ab65f3d49e0a00a8",
            "espeak-ng-data\\tt_dict": "60d138d38330e44a78b4224cbea452bfca22bce6734c8ce3f640c0216ce14001",
            "espeak-ng-data\\ug_dict": "7c4567ffe6bd8d5224b6b3fd8fab65b1343254636e6228d738ae3a708428e8e7",
            "espeak-ng-data\\uk_dict": "ac06f83885139b25ca05e2233d78fe59508650519f54670ef06275a7b884a4cb",
            "espeak-ng-data\\ur_dict": "c9b152fbfb134f6281de8565bd4f41aef006f57b278c03c557dd5feff6e0dbec",
            "espeak-ng-data\\uz_dict": "5b6701896e599461606c9d6df9eca61fb6863df5f5ed7f027390b67988680a53",
            "espeak-ng-data\\vi_dict": "4b1bdd2a5fbb79514ab71c77422966c1a59276a138241a3d2ee032e977b3b53e",
            "espeak-ng-data\\voices\\!v\\Alex": "1cf9874567f4fa8109329f084e9ad9ed788fb2362298921d0a04a57b24f1141f",
            "espeak-ng-data\\voices\\!v\\Alicia": "a732a74c85215eb03467e52915e9435d99920aad4a98ed773c40c90c70a509cc",
            "espeak-ng-data\\voices\\!v\\Andrea": "6ca52734ec2cf7c419d2d3ad7c3449cd1a1776342c4d152ea12c5b1e72801e89",
            "espeak-ng-data\\voices\\!v\\Andy": "0e13ba1a940cd9df7e65358c36445c9740f3d75f1360c7f7c71ab9f74b45edca",
            "espeak-ng-data\\voices\\!v\\Annie": "40f0f829c5c6e7abf4a02a21f128a572756d6f6693014c26d09a5c29ede5ec33",
            "espeak-ng-data\\voices\\!v\\AnxiousAndy": "7dfe126f517b76cc78269618dbee2c30d324bea9460b400938756ec93995eaf8",
            "espeak-ng-data\\voices\\!v\\Demonic": "241b0e78362bd3208bb586b65bdf52330e1e8322f8f226c4790c4abeb47cfb09",
            "espeak-ng-data\\voices\\!v\\Denis": "a09fe9f7a4e478d2ba1e139c2995633e2b99de9abdaaa82307954599b0e3d6fd",
            "espeak-ng-data\\voices\\!v\\Diogo": "a4a17c453d13242b48783c247816bff8d2cf7bc8931c3f8237887d88a2881b18",
            "espeak-ng-data\\voices\\!v\\Gene": "06258dae680cde2122a4939dbd4f23ed4ea145b7475068bf7dd4d466090afd2a",
            "espeak-ng-data\\voices\\!v\\Gene2": "bcdb3039296eea37398e4423dc9bb6849cda9c5c812b83702ebc49682c649f57",
            "espeak-ng-data\\voices\\!v\\Henrique": "5c6bf1cf5b6bc7c7d48f2a8368ae348c6ff64232c047f40f9d9a6e6824b4ce6a",
            "espeak-ng-data\\voices\\!v\\Hugo": "906acbef890dc4da67cb54b5b0ddc2cab7ee308a95d350c899f0b0cb6a83dd5f",
            "espeak-ng-data\\voices\\!v\\Jacky": "41884ed9b66160aeb5a0dc418370a59fe39ead0013d93e023e6d607ef01c1a6e",
            "espeak-ng-data\\voices\\!v\\Lee": "a1584887cddb40c10e04992fb770025b52e66b70504656094f80975f1d6bbf6b",
            "espeak-ng-data\\voices\\!v\\Marco": "04025677360c0270f094bbb6716b83c8284902f6da9b496bedbbf590963835fb",
            "espeak-ng-data\\voices\\!v\\Mario": "a94ac65d132030df496e8aa364d5cde5f2eafa4b6b3ae361a2901d512c6029d8",
            "espeak-ng-data\\voices\\!v\\Michael": "4409be23ef5fedf4c8bffbfaa8a802a342824d7e97f1d92938336a25f92f0ebe",
            "espeak-ng-data\\voices\\!v\\Mike": "8999b264606c0af23d346a877e10ca05aa757f7ea433e6cfcb3bef50a56ba2cf",
            "espeak-ng-data\\voices\\!v\\Mr serious": "794abd78deb07ce9b03419d20f7744df4c0151d5abff7981217e9d0926a2f240",
            "espeak-ng-data\\voices\\!v\\Nguyen": "4f3d557d9c045080f56022a96e5b3659c4d1caa5fa1d702e12d3ef6bec2e3a45",
            "espeak-ng-data\\voices\\!v\\RicishayMax": "7ddc7a57711ccd7a736540f14271adadf1e2e0422941ecaa7cc43ab87f8e2017",
            "espeak-ng-data\\voices\\!v\\RicishayMax2": "3de5e414d4b2ef082a7b974bb7ede94937e93b5b873d2b2f7928b19d30745180",
            "espeak-ng-data\\voices\\!v\\RicishayMax3": "36d20550b68a8c229d34d956657b7d642929962f7e5d4fddd6396af9e5378b56",
            "espeak-ng-data\\voices\\!v\\Storm": "e817c7d1ce10de527ebc19c0835f580cb584c3ae7d36d30a4f414482602e7b89",
            "espeak-ng-data\\voices\\!v\\Tweaky": "c34d1c1d8b7922ceb886b95ad6ab534ca048f9c47e19698ec2aae5cf0c32cdbc",
            "espeak-ng-data\\voices\\!v\\UniRobot": "23dba52c2d1cb332db6225304818d85488f110a668ddcdbfd7f0438afe37794b",
            "espeak-ng-data\\voices\\!v\\adam": "b21efbf56fd7d58470e7ce4759c6ef9f23949706b269f2d777832204658ee0dd",
            "espeak-ng-data\\voices\\!v\\anika": "228771ce84cad16a089e0b23efe2e69edf25f1ab7c0cb3a50d366ac70e82fbbd",
            "espeak-ng-data\\voices\\!v\\anikaRobot": "3ba724be6aab2fc1a084b25cdb16f469250eb9de4c6c0924a18788d0246fc32f",
            "espeak-ng-data\\voices\\!v\\announcer": "4af6a6748c720cbbaac1e3dcac17bb3bc7a4c553499ec7ee69e054db51a50de3",
            "espeak-ng-data\\voices\\!v\\antonio": "c90477080cd9aa9c384770c0ec18901d01e9851f1bc7d15526a2565a3b0c4da5",
            "espeak-ng-data\\voices\\!v\\aunty": "0aede4d1ca614f32c348f3d728eab58322ccd016e8531234fdc3dfcf00e32bf1",
            "espeak-ng-data\\voices\\!v\\belinda": "9bd791b7115820e25ada82b4911134d5116e8238b4fd4d214dfccf162576141c",
            "espeak-ng-data\\voices\\!v\\benjamin": "3fa10b0d865fc751fd9c8ed1dabac4bb9b011ea2b95064d4d25a8c719f7785d8",
            "espeak-ng-data\\voices\\!v\\boris": "f23d0a6ee2eeb7e3340506d61d422b691fb04f4e98977ea15c1e0ee5715846d5",
            "espeak-ng-data\\voices\\!v\\caleb": "bcde145ec713eae6d3451a6f25dbbdc0b6756fd44e9e092b7aef89f2cff25be8",
            "espeak-ng-data\\voices\\!v\\croak": "2906a2c6f7b7687db2f1c862164a9b80435d5afc3ba16d74619728bf8c3a8ca7",
            "espeak-ng-data\\voices\\!v\\david": "50b5cb5f61a50f5ff5299dd22e5d645446746dbc36909595f6ae47bac0bda9c3",
            "espeak-ng-data\\voices\\!v\\ed": "12ddb6355b40a2e39a3b6cef66a2bf4db15a25045769b330a9a3b344a608026d",
            "espeak-ng-data\\voices\\!v\\edward": "b7a12bcb2af280c40e4be4280b675001024619ed529262dbf3e819db53b5b310",
            "espeak-ng-data\\voices\\!v\\edward2": "59d166f7adfee4515a492993405e95432889050aaa47b6cd7eaeeb457fbca227",
            "espeak-ng-data\\voices\\!v\\f1": "7cdef453c6b0391d50a02b8f35392e6f41249cea65df1021affd68a64917b85f",
            "espeak-ng-data\\voices\\!v\\f2": "62b17f6c8f94add0310db032f5e463c13510c5ff83ef1f21e8894ccbf00b00e3",
            "espeak-ng-data\\voices\\!v\\f3": "d0342ee8ceebe3047831774634d234e90281ec73ce2aed8dd0bdd915ccf500fc",
            "espeak-ng-data\\voices\\!v\\f4": "ff9e2907a818920e3b976232c90eabaf270fff0de0b8a479af9a3633e9b7921f",
            "espeak-ng-data\\voices\\!v\\f5": "98660253b641ccb9a37a361528464cdcb79d523ff555ddae4468fcb5fa6c7ac9",
            "espeak-ng-data\\voices\\!v\\fast": "fa2f9c2918f351499037f6bffb25294d9b781b2785d9772a8ae3293c0c0037fe",
            "espeak-ng-data\\voices\\!v\\grandma": "e033d57ba9b098971fdb995c13830f9c709206add081c586daaec323ff69a392",
            "espeak-ng-data\\voices\\!v\\grandpa": "f25627887b95e8bccac5e6ce99cb97c6e34290ff0bf919febe64a9987138796d",
            "espeak-ng-data\\voices\\!v\\gustave": "5fa89cf8625cd460b9309748eca019ab5b325a04dccbd29f1a9b2a00caa2e848",
            "espeak-ng-data\\voices\\!v\\iven": "10ab9de8365ff7bf5a20621d08f1a2019d4bf5521f3c6f65447c9256057406c8",
            "espeak-ng-data\\voices\\!v\\iven2": "f94830e5abe118dc4c0909630d66637ba2a7c94957279c31fd78e313374054cd",
            "espeak-ng-data\\voices\\!v\\iven3": "27ccf2d2f7bb8f1ff28c705c39cf6ff575eb2458c39a518ed33741ad9c1af252",
            "espeak-ng-data\\voices\\!v\\iven4": "11b3d46c01d4307eedcb60bd8dbcc73a40f3eb86c289176d6fd5bb418223b92f",
            "espeak-ng-data\\voices\\!v\\john": "bfbcc1cec33ef0581fdeb4955fbcd0d824a5d4be5493d731e4a2ef7f8f3ba670",
            "espeak-ng-data\\voices\\!v\\kaukovalta": "b9f6016145835c3999bcd5872b8a53442729bd903591781ea3340746b3b4956b",
            "espeak-ng-data\\voices\\!v\\klatt": "9abb6d05d0ca354831c570e14323dd4e821459b53166d96395875a48bacdbde2",
            "espeak-ng-data\\voices\\!v\\klatt2": "afab662a55953532ff8d6f9c615b824ecec87f7aef6790a58c4ccbc4505a742a",
            "espeak-ng-data\\voices\\!v\\klatt3": "f1c0cd147bd656f6c4830b3341c9694483438c43aea15c982dcc1945dbb334b9",
            "espeak-ng-data\\voices\\!v\\klatt4": "3459c15aa8fa7d7218ac34388dfc0833c850ffc923013949308c2f643f3daad2",
            "espeak-ng-data\\voices\\!v\\klatt5": "44397d8282454a2974078d25509caa470bceb093e484e727843a5147b4193e60",
            "espeak-ng-data\\voices\\!v\\klatt6": "7d3930cd97ef7a0732f2e19f53f84e480f2d441d263390647b92bb0b4c645d05",
            "espeak-ng-data\\voices\\!v\\linda": "df06b4c29f5953034e7815ea7062e356917266f8fab576f76c036045525c8e28",
            "espeak-ng-data\\voices\\!v\\m1": "7f2de023bdfe18651d2d0453d2e7ff5e8ade9f094912a79247336ddc90fd7783",
            "espeak-ng-data\\voices\\!v\\m2": "6eb925eb6691621a6de66c87596c6ed879c80a04215a9a8437dffc28fcfa14dd",
            "espeak-ng-data\\voices\\!v\\m3": "7a4ac872387439814ddd65f5e1ff73017122975911df6b5dc62c5709e6fdb611",
            "espeak-ng-data\\voices\\!v\\m4": "6bef82cea3538da1246b7a2c203206e90155cbb4af86faa8ef03366c3119ceee",
            "espeak-ng-data\\voices\\!v\\m5": "08590ef7ce37630a6875d9cbe200dc473a9054b4184287808a4fd1c8e27dd02a",
            "espeak-ng-data\\voices\\!v\\m6": "fc0e01af458a96d43f55878010d3abcbbda85e846922b25361a78d27c3a9ca00",
            "espeak-ng-data\\voices\\!v\\m7": "bdfc54a3558ca98f9ca082479c5211034a824d8721db91804f9d77867db6fab3",
            "espeak-ng-data\\voices\\!v\\m8": "060130540e2fae5f8acfb19bcd4ef3dc5dc8f8d2134c502bb57af277fd4cc491",
            "espeak-ng-data\\voices\\!v\\marcelo": "3238a933a17c40c61cdd385c0e742bf54c42c008f4110d4b0353a6ae72a5d1a0",
            "espeak-ng-data\\voices\\!v\\max": "c88f70a60a19e76a6774dfbd7664be639b9aa8aa587c680834c305ffa434aaaf",
            "espeak-ng-data\\voices\\!v\\michel": "c0068ad9cb792dfb78aedbdff9c3cec68e5083736bae0c33bf19088186101087",
            "espeak-ng-data\\voices\\!v\\miguel": "9c8c2bbe9fba6d1f06eb63ee67d98d7ebb8846b2bf13847a9b359c90ba9fa8fe",
            "espeak-ng-data\\voices\\!v\\norbert": "35b9923e15caf36280542270282bd2392c6d86ef5db9a8be7be92ce2241a46cb",
            "espeak-ng-data\\voices\\!v\\pablo": "053a85c8a6d7e7a0f6432a3158fe73fa9accf06202e5d69c273d4e5bae8bafc8",
            "espeak-ng-data\\voices\\!v\\paul": "75fb8098bea28859513de6ce033092307382d6f20455e71b051193dfc0793d5e",
            "espeak-ng-data\\voices\\!v\\pedro": "7429792ba833fa2446bec6db790068f5862b9ed099392002cd0e724ec6ab2a66",
            "espeak-ng-data\\voices\\!v\\quincy": "e9945bee7b4f3240221866f497bf2511a886ed7702d8389688a03c2ea120caf0",
            "espeak-ng-data\\voices\\!v\\rob": "fbfffcdb0afb2e3b70f886c02282feae22509d77fe6e7103daf907f7a356595d",
            "espeak-ng-data\\voices\\!v\\robert": "72604f85afe5bb52a5ce7e3427b196e1e3e426a14a64e2d93b6617e248a46417",
            "espeak-ng-data\\voices\\!v\\robosoft": "ce2820edd349f3fbc734bfe94095518f10ecbefdb304560109f37106f9f4d2fa",
            "espeak-ng-data\\voices\\!v\\robosoft2": "848221c7757d58063d86e718dce5db05edd18805f8f15d1ffb1d64119bc4b74b",
            "espeak-ng-data\\voices\\!v\\robosoft3": "f8543efe18019c735ebad99d1cab4e06e0efd565ac269a895415dfebf342c383",
            "espeak-ng-data\\voices\\!v\\robosoft4": "d5f728e0d0a9d27f604f34e835e059590fd01411fd467c91328b573c7597bb2d",
            "espeak-ng-data\\voices\\!v\\robosoft5": "39da0efe1768e14eefe05a7019fab416795bbffb222fce23ad7f963991447f7b",
            "espeak-ng-data\\voices\\!v\\robosoft6": "46426971d76c34266580133fad2c26ab829395a71c1713e6205d9d46b0871672",
            "espeak-ng-data\\voices\\!v\\robosoft7": "e19c1045259d9cfb078c229d5d1372f7c5af971d0e2536a41b89c40ca3e23ede",
            "espeak-ng-data\\voices\\!v\\robosoft8": "389230461647836c1b66c51939c0a493e3fbc81e2b7afc01c6f88b8f9da21e2b",
            "espeak-ng-data\\voices\\!v\\sandro": "1b8c36a3ce35a27cbbabf4679df0264cd37f4c79b66aa1f4c310eb31b9c80490",
            "espeak-ng-data\\voices\\!v\\shelby": "5e96565045bcdd5b42e7ad0eab2bbd670c67cf5c9e3f2ac6b50b30580fb92be4",
            "espeak-ng-data\\voices\\!v\\steph": "78186aeddce4c449348f368046b2665375b1a492e948d739249b0c964f082861",
            "espeak-ng-data\\voices\\!v\\steph2": "ca88414b48553fed1fd409389548a94ea90cf282d6340e3cb968c8538a2ac2ec",
            "espeak-ng-data\\voices\\!v\\steph3": "a7d796211b6969ba92213f89de54b053890306af774424ee38fb02ace45e852d",
            "espeak-ng-data\\voices\\!v\\travis": "7d5a34be05778ee850d97b0491a62db7683c058d660ff2afc71c08c87360f30d",
            "espeak-ng-data\\voices\\!v\\victor": "a880a8083ac5eaadf94f71c63b51f34c1426d4bfe0a26f6761ddec5c6e9a5c76",
            "espeak-ng-data\\voices\\!v\\whisper": "1e3534c843768898cea47a3f83b2fd91720c0775b438ea5c0134676940a66e96",
            "espeak-ng-data\\voices\\!v\\whisperf": "7fea42b10114f1bfd2004181e6b0eb626997cca987893e23d24b5c616aac89d7",
            "espeak-ng-data\\voices\\!v\\zac": "edaf508b2cfbfd6088a163dcc78d5f6a26850a7d39a5a0e5128b09526f56b6db",
            "espeak-ng-data\\voices\\mb\\mb-af1": "62ed7f850ff2871582fbcda1cf88954da64b5338ef11b325a74a2a01cefbbc3c",
            "espeak-ng-data\\voices\\mb\\mb-af1-en": "d04410803885a91d432b41dc8bd8a713cf1c8581f5efbd5c1bcdeb7fee2d922f",
            "espeak-ng-data\\voices\\mb\\mb-ar1": "28c4b82eb47395c7470e029cb700c5fe567b5ba1c208c9e552bd5b2651f398c7",
            "espeak-ng-data\\voices\\mb\\mb-ar2": "d3e11e4a83c50e8b3c3d66d6b818209e634ab4dd3db362fd6c3c37af12f88447",
            "espeak-ng-data\\voices\\mb\\mb-br1": "4a9f4144bc1101b94f520eb6d6ad0028773647106dcdf77b54008fe85e505629",
            "espeak-ng-data\\voices\\mb\\mb-br2": "153206f9271447ca284b40053ab0d30c10795f6cb16a241530690650ea925ae2",
            "espeak-ng-data\\voices\\mb\\mb-br3": "a8465e29038e2cb96b82610f25cb0105f55f41e35e6ce20399908839a230bfb4",
            "espeak-ng-data\\voices\\mb\\mb-br4": "95a9be32283b208e20b03446b14d0432c62310e376c081a161ff1a3d7ebb0c42",
            "espeak-ng-data\\voices\\mb\\mb-ca1": "23b563ebca77f6393bf12df6393876983cba6794b56e9d5604c766aaf5ec7073",
            "espeak-ng-data\\voices\\mb\\mb-ca2": "9795b4f881eee8aac3bd0d932f091461f3720196f10529cead114812aa771dc2",
            "espeak-ng-data\\voices\\mb\\mb-cn1": "9a7c485ff7945464d98614e30fc75c2f3f6b773148b3ed49b2de438f37d21650",
            "espeak-ng-data\\voices\\mb\\mb-cr1": "972853cf9f24ebee6889931d6c4968334f7212a479883bb3e87e3505e4f9aa90",
            "espeak-ng-data\\voices\\mb\\mb-cz1": "bdd9824997eecdcdda1721ca046c0416fccbd472c2ba3cc9a2148c875e4d1a74",
            "espeak-ng-data\\voices\\mb\\mb-cz2": "ddbeb9b7db45dc7ab5b01f93f3d651e1e017e46f0cc30f26386a1ed37090edf3",
            "espeak-ng-data\\voices\\mb\\mb-de1": "7707cfbf66a06d8579769385b88f3f2e8ecb0e15756e2703ecaabdfbaaa0b581",
            "espeak-ng-data\\voices\\mb\\mb-de1-en": "34eb90c4091ee4f804504045511533ccbe5fadba9971771740af8c16815dd7d1",
            "espeak-ng-data\\voices\\mb\\mb-de2": "dfd105df0953bd969de9a91388739598e87ad64135f4f74d5dad4c09a3f6a403",
            "espeak-ng-data\\voices\\mb\\mb-de2-en": "4bfb2708141fb24b0a52fd5f7b6bc2b18d4acc925e7fda2adda8a190c4bcda6f",
            "espeak-ng-data\\voices\\mb\\mb-de3": "42bfce7cea2a5f31e08f29a46885b741909ff2fb60badab5988b114a3bd81064",
            "espeak-ng-data\\voices\\mb\\mb-de3-en": "92ba6f23ff7ccecfbe74b0c62ccf7dc50c9e980602cc9159c233330aef0fb521",
            "espeak-ng-data\\voices\\mb\\mb-de4": "d737db3b975723c21e71e8632ec3e242cd15974b6f6de89b4ce2e565df98ee5e",
            "espeak-ng-data\\voices\\mb\\mb-de4-en": "9316da75a9ecfcbeaa9dbba5cd668faab9564405043d8e995004e30bedd4570b",
            "espeak-ng-data\\voices\\mb\\mb-de5": "bd53618e85139fe926ca5cd156c2bfb000e540098a85c428e9a6c315951b9428",
            "espeak-ng-data\\voices\\mb\\mb-de5-en": "fcbf3863ac3b6bf117145a50699138de5a3426487eb2f69535300136aa6ff6f1",
            "espeak-ng-data\\voices\\mb\\mb-de6": "a931d05983b101ed27171ba1797c5f54d2485e86d732f6f4bb4da45ec41728e8",
            "espeak-ng-data\\voices\\mb\\mb-de6-en": "123f0c331d757235b6d384ac118a97b472917440d8fe130ab2cc7d9f62a0a0e0",
            "espeak-ng-data\\voices\\mb\\mb-de6-grc": "c69b453e273548c33a856d0f7c1eb93c8b84395a73a664630cd5e685a6b142e9",
            "espeak-ng-data\\voices\\mb\\mb-de7": "cb9068f9789af30aa1e0f647a7ac12e829894c671b8c1c3387b51144468feae1",
            "espeak-ng-data\\voices\\mb\\mb-de8": "e3948b0fb5e695908d9d10015a0feab890feecee620fe700a7e7d3df151aad0e",
            "espeak-ng-data\\voices\\mb\\mb-ee1": "a8fded3877b1dd140509b830d758d06b1cbf18e0c47abd0270aa6c7bfeb56bc9",
            "espeak-ng-data\\voices\\mb\\mb-en1": "b02d193825fc3a5e9d7a89294fe5a10dbd83decc81f2a23c0c3e945692c84cc2",
            "espeak-ng-data\\voices\\mb\\mb-es1": "ab76ae896d0ae17da47c26a0a9a17a42f2e6923cdbd63a8164df2eb88dc710b2",
            "espeak-ng-data\\voices\\mb\\mb-es2": "f64e33f6951e5fbabfc23fd502a56b279b78445dfbcccbc1290cb1892997066e",
            "espeak-ng-data\\voices\\mb\\mb-es3": "7d7cfc7868785a15a2e95ab918fb492d0410eaad9d3de7c4484d248734fdfcfe",
            "espeak-ng-data\\voices\\mb\\mb-es4": "f2457af6556daef565f00022941cefce5168910a43c9a6b7b1d3207c3122378f",
            "espeak-ng-data\\voices\\mb\\mb-fr1": "b02dc3ea4fe36490143711ea5096943be599dbe5f035eb770be252358367152f",
            "espeak-ng-data\\voices\\mb\\mb-fr1-en": "cd0d2b5f2a233dbc618779150c0ced03d0cf35d0155c2d8b76c88faa2dc36ec3",
            "espeak-ng-data\\voices\\mb\\mb-fr2": "add6fcd4c1ff5c08b3fafc1fdbfa9239e9f6be1dc10f1c7755f137a60052834d",
            "espeak-ng-data\\voices\\mb\\mb-fr3": "17a9db1b1efaa1bdeecbf215c88fcb5f9d73f690ada4cf9ce357478a8e1a8f8d",
            "espeak-ng-data\\voices\\mb\\mb-fr4": "6d258f7d60d94f84c8f25b3f30f9de02d0939076e692ebd375de4ef9b855a25d",
            "espeak-ng-data\\voices\\mb\\mb-fr4-en": "9e8feba6f9b322d0c8fc29ce6f6428a928d2ea0299fe41a32ed480d4a29d09d2",
            "espeak-ng-data\\voices\\mb\\mb-fr5": "39113dab3ed1505690d67d415708fbb2c0b9f7a75688fae4acf8e1f364d5fdd5",
            "espeak-ng-data\\voices\\mb\\mb-fr6": "9cf9058fec07afeb53597d0c6bc5ae2147ab7ed7768881028e7dc929f233a483",
            "espeak-ng-data\\voices\\mb\\mb-fr7": "b1a99967024319bcaf9c6bc4d04074e094524af67159059127c9129c8d99df93",
            "espeak-ng-data\\voices\\mb\\mb-gr1": "a46d1558b7411b082d87680ca98dc21db99b5aebbd9fdb2f211a944c614f2026",
            "espeak-ng-data\\voices\\mb\\mb-gr2": "a4d4edab2b956079dc5fef7358e6539d13563263f123e3adb6c646d746f475e8",
            "espeak-ng-data\\voices\\mb\\mb-gr2-en": "b69b92a118b367edff28839e49dac2608da570ae3c9f5d670481ab1c595e1264",
            "espeak-ng-data\\voices\\mb\\mb-hb1": "ae2a32038609230e7223dc7e55dd0a0ff3d5b8b19089c99bb5169c625bfa4383",
            "espeak-ng-data\\voices\\mb\\mb-hb2": "e1ac0934029ce20d6784700797a9a31957b4b2008f1ac298edff6b8a72eb6093",
            "espeak-ng-data\\voices\\mb\\mb-hu1": "1db8241a935ecd6a6b5e4ced906021a86535e68047fbbd91b20adbfbc5c68f81",
            "espeak-ng-data\\voices\\mb\\mb-hu1-en": "4a36dfce2aee32e2675feddc71ef8cd253a504cbe1ea81024867a2d7de158fac",
            "espeak-ng-data\\voices\\mb\\mb-ic1": "2130a394e3f2941399b704d64a314afa8b634c86d963c708eadd928e3be3248d",
            "espeak-ng-data\\voices\\mb\\mb-id1": "351d13dee787052313d55f3762f3633b35c736348ecbb3e76914c9893f816e81",
            "espeak-ng-data\\voices\\mb\\mb-in1": "c5b5f5221379e06fa21b99bb0ed8ea9ffe28355616fecc38d772d74941cb131d",
            "espeak-ng-data\\voices\\mb\\mb-in2": "7fa8a9bfd82c29302ed7e7af5a5d827cfed3d1b5193d3d467f8cf8174b428627",
            "espeak-ng-data\\voices\\mb\\mb-ir1": "8cebd3cd81694646011d39354b3e3d45343a96d7858e4bd42e3fce9be381a7fe",
            "espeak-ng-data\\voices\\mb\\mb-it1": "f37fba434dd4c0a2e5da268f596067dcd619ebe249d6d81d79d68acc5793638c",
            "espeak-ng-data\\voices\\mb\\mb-it2": "67a10fd596b6f1a770002709646a68efb06a1ad77b04118f8332b39a4489f58d",
            "espeak-ng-data\\voices\\mb\\mb-it3": "ac301735de92fdf11d9f76a013eb5a25042cafa40fe7f276fe27326b1cf20f56",
            "espeak-ng-data\\voices\\mb\\mb-it4": "6d6f27286476035298926ef3caf242ff37d6fdf8dda3e95d4e2b5776468d3004",
            "espeak-ng-data\\voices\\mb\\mb-jp1": "a3e9c93c3fa497ee8309d904512a2f36350f266aa7ad68886df20eb82ea6337c",
            "espeak-ng-data\\voices\\mb\\mb-jp2": "5a1a7530de791a66d922f28fb9322c579ac9afa4b2446114698bcbef2d9c9992",
            "espeak-ng-data\\voices\\mb\\mb-jp3": "37a903aa6d0d36e3f6252e8666f3a5bd6a34fd731846ba85d58ce2967e2eefca",
            "espeak-ng-data\\voices\\mb\\mb-la1": "540788a3f4ee5abe38cdf31d52c7cf531e913d3af5d21d5b2d7a5270b6bb7841",
            "espeak-ng-data\\voices\\mb\\mb-lt1": "f269e9c9b0f1c026bcfeee33d68a7f366a5c4c81dbb5f82482d1527cfa30df69",
            "espeak-ng-data\\voices\\mb\\mb-lt2": "438b914e1591e7483356427bc4e32b85ccfb1c8b8a306ad9fd71f8e17aab7c53",
            "espeak-ng-data\\voices\\mb\\mb-ma1": "b734fa925230ec9f103ac5b493baca30010e5128cf84a782a78f9d101014fe0e",
            "espeak-ng-data\\voices\\mb\\mb-mx1": "4130e2c6168a5ebcb98a595e8a7c569d0d5ebb1f8da0622f07accc86894181d2",
            "espeak-ng-data\\voices\\mb\\mb-mx2": "b20de9d7df7fd34a8f462245e78156e4b3d2fafdd3db1da04ac5d4d2e35070c4",
            "espeak-ng-data\\voices\\mb\\mb-nl1": "abe393ff2f4ce9d6bb91d024038044d282ac07445675ef7b65cc3ebec9f36b47",
            "espeak-ng-data\\voices\\mb\\mb-nl2": "04c3b361ab3f1fda090f39e17a8b5ef7acdf9de20a3d332f30802cfc3d95877b",
            "espeak-ng-data\\voices\\mb\\mb-nl2-en": "a7428804c19a26ba3340a235506341ed212ae7bdd7c87fc9a6e52a2eed7c25cf",
            "espeak-ng-data\\voices\\mb\\mb-nl3": "e7185ed167f719325f252b16f28ea6c1fc8fe30bac5392a4ce8c2194146860da",
            "espeak-ng-data\\voices\\mb\\mb-nz1": "b372cc01633cc6a5ef6f125fd2843b4d3fbc290f37e8b4035a7d11410f1eb8ef",
            "espeak-ng-data\\voices\\mb\\mb-pl1": "1b092a60c374081f2c9bac3d09bdeea6f97e821d7261fbb530db161505c63783",
            "espeak-ng-data\\voices\\mb\\mb-pl1-en": "1ce94b38560e30b33c0efb7d0424f3fa7ff86e913eb7f943feb88911616f0ba6",
            "espeak-ng-data\\voices\\mb\\mb-pt1": "3295baee44757c31bb25e0a53fd7886fac51d3fe87a3bd63f5b25d7797ca12a9",
            "espeak-ng-data\\voices\\mb\\mb-ro1": "3b0923ea8affeaa73a7c658d7c7f44beb7f47ff32ba48be3a71fd469c69359dd",
            "espeak-ng-data\\voices\\mb\\mb-ro1-en": "9552cef4c612a6a6701ad287542607c87a40bd666a196b014e0a6c47fc855b0b",
            "espeak-ng-data\\voices\\mb\\mb-sw1": "6263ef96f906154d2a4b9d7ab385239a3d48a7f929932beb8342a45e1c46aea0",
            "espeak-ng-data\\voices\\mb\\mb-sw1-en": "cbbcb9184cc8d2b6193e29a27b428a4860d647eeda8c04cbfb2f6b21ad2024a1",
            "espeak-ng-data\\voices\\mb\\mb-sw2": "75a3b9f171f2e6681cb6656e1b9198fee6d1a6ad5c025dc52e056b37a0b5a40e",
            "espeak-ng-data\\voices\\mb\\mb-sw2-en": "49cf98f88639db8c000fddd24af4dc6c48da571fce063d41a39a165b02b72166",
            "espeak-ng-data\\voices\\mb\\mb-tl1": "3b932ee736f71fef55010f1670062e822e61271504d46d1806d5cbc9d5ae7eb8",
            "espeak-ng-data\\voices\\mb\\mb-tr1": "3ea261addfce816922db38556aabb0cb83b5ca3d0e4a666c30bb875ec5f30397",
            "espeak-ng-data\\voices\\mb\\mb-tr2": "a0f57dfa06053452d3cc2a343e00e2717a7de11f6ff516507e7406cba70fc277",
            "espeak-ng-data\\voices\\mb\\mb-us1": "03b30d4c8b571c48df3be931dae2bb6f10d683821df0e0dac612ebbae351ba32",
            "espeak-ng-data\\voices\\mb\\mb-us2": "63ea6b18456f022ae2cf80c1d0c0ec9d324575cf388eb4d6057f507b433bc669",
            "espeak-ng-data\\voices\\mb\\mb-us3": "796fc7e7cb73322dd26102e270c55d1ce754702d9f6f76ba2abf3d2397783b94",
            "espeak-ng-data\\voices\\mb\\mb-vz1": "4bd9cd1244c71b583fe8279bb8c092ddaca61873ce76b19a9d78a8731b97eedc",
            "espeak-ng-data\\yue_dict": "1d26afa203034698772107abfff1b53acbee30434700a4d2b75dee98951588f8",
            "espeak-ng.exe": "e6a7f1db017d838a917d79450f77138a391a31114dd6ec0855919e20db4a5149",
            "libespeak-ng.dll": "b80ecdf4402991ea3977587bc756c8f04072e7dbb1821f0f376969ff0735bd49"
        },
        "path": "eSpeak NG",
    },
    # DAC
    "dac_44khz": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/zonos-tts/dac_44khz.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/zonos-tts/dac_44khz.zip",
            "https://s3.libs.space:9000/ai-models/zonos-tts/dac_44khz.zip",
        ],
        "checksum": "c7da3a820e5600a45dc0bb22ff7327a1c91f669b14151b55319afb584e8b5787",
        "file_checksums": {
            "config.json": "4eb55fb9af1990b8d608184ad29b70e358589719af7ea8d3c06998f7c2264a64",
            "model.safetensors": "6128ebff483a41422b0164d079a3773b0d8d82e64c4293d775994cbf8baf913a",
            "preprocessor_config.json": "c7d295758ce5777d6d88fef1996e94adc8ef3e2237ddfc5ecc24d1407aaddd7d"
        },
        "path": "dac_44khz",
    },
    # Default Voices
    "voices": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/zonos-tts/voices.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/zonos-tts/voices.zip",
            "https://s3.libs.space:9000/ai-models/zonos-tts/voices.zip",
        ],
        "checksum": "3997e6e1c7a7c0255bac3fd6ad9493098cfda410091755399e873e848459ac96",
        "file_checksums": {
            "Announcer_Ahri.txt": "65cdbe885b89037dc651bea9bb7c41077471a2d7168e25c905c7034da7de285d",
            "Announcer_Ahri.wav": "2a3fd17d45b3c5633dd64e2d80a6e3fc924fa829da405e3b591a5bacdc88f9fc",
            "Attenborough.txt": "4c617f7adc60b992de93abd18bd447da5c882db7d04d9d51b241cdf79cbda6a1",
            "Attenborough.wav": "358540c89932baf1103960d99949b78ea7466f95b2225fdcd8f8bb8b976f09ee",
            "Jane.txt": "58e939100b6422f76e631d445a957047fa915ba6727f984ebdcecfa3418f5d08",
            "Jane.wav": "d1d2235af1a4408c641a765427978486f5cca9b369fc6456d8086449f1f92fe3",
            "Justin.txt": "6ce2802c88bd83ef12ecb3338f1bf6f8bc5bc12212b3cd1d2863d0d3ab93632b",
            "Justin.wav": "a83c37f408b53efaeb9189f166c6669d1a0dc6cf779e85913fa9cbbbbe0d5aaf",
            "Xiaochen.txt": "1316b1e27871565b1d7cd4f64b0521a37632cc15d1ea0944d18394bdaf76d8e2",
            "Xiaochen.wav": "7f0b735e188a06dc9f104eeb3fd71a3ef580d1f2133c95630c92a244dd253732",
            "en_0.txt": "3bb999d455ca88b8eca589bd16d3b99db8b324b5f8c57e3283e4bb4db8593243",
            "en_0.wav": "f006e2e9c76523bde4f5bbe67a7be9a600786d7432cbcc9486bc9501053298b7",
            "en_1.txt": "79cccada817b316fa855dc8ca04823f59a11c956b5780fbb3267ddf684c8e145",
            "en_1.wav": "b0e22048e72414fcc1e6b6342e47a774d748a195ed34e4a5b3fcf416707f2b71",
            "test_zh_1_ref_short.wav": "96724a113240d1f82c6ded1334122f0176b96c9226ccd3c919e625bcfd2a3ede"
        },
        "path": "voices",
    },
}


class ZonosTTS(metaclass=SingletonMeta):
    model = None
    sample_rate = 44100
    last_generation = {"audio": None, "sample_rate": None}
    voice_list = []
    audio_streamer = None

    last_speaker_embedding = None
    last_speaker_audio = None

    compute_device = "cpu"
    download_state = {"is_downloading": False}

    stop_flag = False
    stop_flag_lock = threading.Lock()

    special_settings = {
        "language": "en-us",

        "happiness": 0.3077,
        "sadness": 0.0256,
        "disgust": 0.0256,
        "fear": 0.0256,
        "surprise": 0.0256,
        "anger": 0.0256,
        "other": 0.2564,
        "neutral": 0.3077,

        "ignore_list": [],
        "seed": -1,
    }

    def __init__(self):
        self.compute_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        # model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device=device)
        # self.model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=DEFAULT_DEVICE)
        model_type = "v0.1-transformer"
        #model_type = "v0.1-hybrid"

        if self.model is None:
            self.download_model("eSpeak-NG")
            self.download_model("dac_44khz")
            self.download_model("zonos-"+model_type)
            self.download_model("voices")
            #os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = "C:\Program Files\eSpeak NG\libespeak-ng.dll"
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = str(Path(cache_path / "eSpeak NG" / "libespeak-ng.dll").resolve())
            os.environ["PHONEMIZER_ESPEAK_PATH"] = str(Path(cache_path / "eSpeak NG").resolve())
            os.environ['PATH'] = ';'.join([str(Path(cache_path / "eSpeak NG").resolve())] + [os.environ['PATH']])

            self.model = Zonos.from_local(
                config_path=str(Path(cache_path / model_type / "config.json").resolve()),
                model_path=str(Path(cache_path / model_type / "model.safetensors").resolve()),
                dac_path=str(Path(cache_path / "dac_44khz").resolve()),
                device=DEFAULT_DEVICE
            )
            self.target_sample_rate = self.model.autoencoder.sampling_rate

            if not self.voice_list:
                self.update_voices()
        pass

    def stop(self):
        print("TTS Stop requested")
        if self.audio_streamer is not None:
            self.audio_streamer.stop()
            self.audio_streamer = None

    def set_compute_device(self, device):
        self.compute_device_str = device
        if device is None or device == "cuda" or device == "auto" or device == "":
            self.compute_device_str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            #device = torch.device(self.compute_device_str)
            device = self.compute_device_str
        self.compute_device = device

    def list_models(self):
        return model_list

    def list_models_indexed(self):
        return tuple([{"language": language, "models": models} for language, models in self.list_models().items()])

    def download_model(self, model_name):
        downloader.download_model({
            "model_path": cache_path,
            "model_link_dict": TTS_MODEL_LINKS,
            "model_name": model_name,
            "title": "Text 2 Speech (Zonos TTS)",

            "alt_fallback": False,
            "force_non_ui_dl": False,
            "extract_format": "zip",
        }, self.download_state)
        pass

    def load(self):
        pass

    def _get_voices(self):
        return self.voice_list

    def update_voices(self):
        # find all voices that have a .wav or .mp3 file
        voice_files = [f.stem for f in voices_path.iterdir() if f.is_file() and (f.suffix == ".wav" or f.suffix == ".mp3")]

        voice_list = []
        for voice_id in voice_files:
            wav_file = voices_path / f"{voice_id}.wav"
            mp3_file = voices_path / f"{voice_id}.mp3"

            if wav_file.exists() or mp3_file.exists():
                audio_file = wav_file if wav_file.exists() else mp3_file
                voice_list.append({"name": voice_id, "audio_filename": str(audio_file.resolve())})
        self.voice_list = voice_list

    def list_voices(self):
        self.update_voices()
        return [voice["name"] for voice in self._get_voices()]

    def get_voice_by_name(self, voice_name):
        for voice in self._get_voices():
            if voice["name"] == voice_name:
                return voice
        return None

    def get_last_generation(self):
        return self.last_generation["audio"], self.last_generation["sample_rate"]

    @staticmethod
    def generate_random_seed():
        #min_seed = -0x8000_0000_0000_0000
        #max_seed = 0xffff_ffff_ffff_ffff
        #seed = random.randint(min_seed, max_seed)
        #if seed < 0:
        #    seed = max_seed + seed + 1
        #return seed
        return torch.randint(0, 2 ** 32 - 1, (1,)).item()

    def set_special_setting(self, special_settings):
        self.special_settings = special_settings

    def tts(self, text, ref_audio=None, remove_silence=True, silence_after_segments=0.2, normalize=True):
        #with self.stop_flag_lock:
        #    self.stop_flag = False
        print("TTS requested Zonos TTS")

        self.set_compute_device(settings.GetOption('tts_ai_device'))

        tts_volume = settings.GetOption("tts_volume")
        tts_normalize = settings.GetOption("tts_normalize")

        if ref_audio is None:
            voice_name = settings.GetOption('tts_voice')
            selected_voice = self.get_voice_by_name(voice_name)
            if selected_voice is None:
                print("No voice selected or does not exist. Using default voice 'en_1'.")
                voice_name = "en_1"
                selected_voice = self.get_voice_by_name(voice_name)
            ref_audio = selected_voice["audio_filename"]

        # reuse speaker embedding
        if self.last_speaker_audio != ref_audio or self.last_speaker_embedding is None:
            wav, sampling_rate = torchaudio.load(ref_audio)
            speaker = self.model.make_speaker_embedding(wav, sampling_rate)
            self.last_speaker_audio = ref_audio
            self.last_speaker_embedding = speaker
        else:
            speaker = self.last_speaker_embedding

        seed = self.special_settings["seed"]
        # convert string to integer. If its invalid, default to -1
        try:
            seed = int(seed)
        except ValueError:
            seed = -1
        if seed <= -1:
            seed = self.generate_random_seed()
        torch.manual_seed(seed)

        print("Using seed:", seed)

        tts_speed = speed_mapping.get(settings.GetOption('tts_prosody_rate'), 1)

        language = self.special_settings["language"]
        emotion = [
            self.special_settings["happiness"],
            self.special_settings["sadness"],
            self.special_settings["disgust"],
            self.special_settings["fear"],
            self.special_settings["surprise"],
            self.special_settings["anger"],
            self.special_settings["other"],
            self.special_settings["neutral"],
        ]

        # fix if all emotions are either set to 1.0 or 0.0, or all non-zero emotions are set to 1.0
        if all(emotion_value == 0.0 for emotion_value in emotion) or all(emotion_value == 1.0 for emotion_value in emotion):
            emotion = [0.0001 if emotion_value == 0.0 else 0.9999 for emotion_value in emotion]
        elif all(emotion_value in (0.0, 1.0) for emotion_value in emotion) and any(emotion_value == 1.0 for emotion_value in emotion):
            emotion = [0.9999 if emotion_value == 1.0 else 0.0001 for emotion_value in emotion]

        unconditional_keys = self.special_settings["ignore_list"]

        cond_dict = make_cond_dict(text=text, speaker=speaker, language=language, emotion=emotion, speaking_rate=tts_speed, unconditional_keys=unconditional_keys, device=self.compute_device)
        conditioning = self.model.prepare_conditioning(cond_dict)

        codes = self.model.generate(conditioning, batch_size=1, disable_torch_compile=True) # €todo with torch_compile Not yet supported. Probably when updating to pytorch 2.6.0 which does not support Direct-ML (yet)
        wavs = self.model.autoencoder.decode(codes).cpu()

        final_wave = wavs[0]

        if tts_normalize:
            final_wave, _ = audio_tools.normalize_audio_lufs(
                final_wave, self.sample_rate, -24.0, -16.0,
                1.3, verbose=True
            )

        # change volume
        if tts_volume != 1.0:
            final_wave = audio_tools.change_volume(wavs[0], tts_volume)

        # call custom plugin event method
        plugin_audio = Plugins.plugin_custom_event_call('plugin_tts_after_audio', {'audio': final_wave, 'sample_rate': self.sample_rate})
        if plugin_audio is not None and 'audio' in plugin_audio and plugin_audio['audio'] is not None:
            final_wave = plugin_audio['audio']

        final_wave = final_wave.unsqueeze(0)
        # save last generation in memory
        self.last_generation = {"audio": final_wave, "sample_rate": self.sample_rate}

        print("TTS generation finished")

        return final_wave, self.sample_rate

    def tts_streaming(self, text, ref_audio=None, normalize=True):
        #self.stop_flag = False
        print("TTS requested Zonos TTS (Streaming)")
        self.load()
        self.set_compute_device(settings.GetOption('tts_ai_device'))

        chunk_size = settings.GetOption("tts_streamed_chunk_size")

        self.init_audio_stream_playback()

        tts_volume = settings.GetOption("tts_volume")
        tts_normalize = settings.GetOption("tts_normalize")

        if ref_audio is None:
            voice_name = settings.GetOption('tts_voice')
            selected_voice = self.get_voice_by_name(voice_name)
            if selected_voice is None:
                print("No voice selected or does not exist. Using default voice 'en_1'.")
                voice_name = "en_1"
                selected_voice = self.get_voice_by_name(voice_name)
            ref_audio = selected_voice["audio_filename"]

        # reuse speaker embedding
        if self.last_speaker_audio != ref_audio or self.last_speaker_embedding is None:
            wav, sampling_rate = torchaudio.load(ref_audio)
            speaker = self.model.make_speaker_embedding(wav, sampling_rate)
            self.last_speaker_audio = ref_audio
            self.last_speaker_embedding = speaker
        else:
            speaker = self.last_speaker_embedding

        seed = self.generate_random_seed()
        torch.manual_seed(seed)

        tts_speed = speed_mapping.get(settings.GetOption('tts_prosody_rate'), 1)

        language = self.special_settings["language"]
        emotion = [
            self.special_settings["happiness"],
            self.special_settings["sadness"],
            self.special_settings["disgust"],
            self.special_settings["fear"],
            self.special_settings["surprise"],
            self.special_settings["anger"],
            self.special_settings["other"],
            self.special_settings["neutral"],
        ]

        # fix if all emotions are either set to 1.0 or 0.0, or all non-zero emotions are set to 1.0
        if all(emotion_value == 0.0 for emotion_value in emotion) or all(emotion_value == 1.0 for emotion_value in emotion):
            emotion = [0.0001 if emotion_value == 0.0 else 0.9999 for emotion_value in emotion]
        elif all(emotion_value in (0.0, 1.0) for emotion_value in emotion) and any(emotion_value == 1.0 for emotion_value in emotion):
            emotion = [0.9999 if emotion_value == 1.0 else 0.0001 for emotion_value in emotion]

        unconditional_keys = self.special_settings["ignore_list"]

        cond_dict = make_cond_dict(text=text, speaker=speaker, language=language, emotion=emotion, speaking_rate=tts_speed, unconditional_keys=unconditional_keys, device=self.compute_device)
        conditioning = self.model.prepare_conditioning(cond_dict)

        stream_generator = self.model.stream(
            prefix_conditioning=conditioning,
            audio_prefix_codes=None,
            chunk_size=chunk_size,
            batch_size=1,
            disable_torch_compile=True,
        )

        audio_chunks = []
        for sr_out, codes_chunk in stream_generator:
            if self.audio_streamer is None:
                break
            audio_chunk = self.model.autoencoder.decode(codes_chunk).cpu()
            return_audio_chunk = audio_chunk[0]

            if tts_normalize:
                return_audio_chunk, _ = audio_tools.normalize_audio_lufs(
                    return_audio_chunk, self.sample_rate, -24.0, -16.0,
                    1.3, verbose=False
                )

            # change volume
            if tts_volume != 1.0:
                return_audio_chunk = audio_tools.change_volume(return_audio_chunk, tts_volume)

            audio_chunks.append(return_audio_chunk)
            # torch tensor to pcm bytes
            wav_bytes = self.return_pcm_audio(return_audio_chunk)
            if self.audio_streamer is not None:
                self.audio_streamer.add_audio_chunk(wav_bytes)

        full_audio = np.concatenate(audio_chunks, axis=-1)
        # numpy array to torch.Tensor
        full_audio = torch.from_numpy(full_audio).float()
        full_audio = full_audio.unsqueeze(0)

        self.last_generation = {"audio": full_audio, "sample_rate": self.sample_rate}

        print("TTS generation finished")

        return full_audio, self.sample_rate


    def init_audio_stream_playback(self):
        audio_device = settings.GetOption("device_out_index")
        if audio_device is None or audio_device == -1:
            audio_device = settings.GetOption("device_default_out_index")

        chunk_size = settings.GetOption("tts_streamed_chunk_size")
        #if self.audio_streamer is not None:
        #    self.audio_streamer.stop()
        #    self.audio_streamer = None
        #else:
        if self.audio_streamer is None:
            self.audio_streamer = audio_tools.AudioStreamer(audio_device,
                                                            source_sample_rate=self.sample_rate,
                                                            playback_channels=2,
                                                            buffer_size=chunk_size,
                                                            input_channels=1,
                                                            dtype="float32",
                                                            tag="tts",
                                                            )

    def play_audio(self, audio, device=None):
        source_channels = 1

        if device is None:
            device = settings.GetOption("device_default_out_index")

        secondary_audio_device = None
        if settings.GetOption("tts_use_secondary_playback") and (
                (settings.GetOption("tts_secondary_playback_device") == -1 and device != settings.GetOption("device_default_out_index")) or
                (settings.GetOption("tts_secondary_playback_device") > -1 and device != settings.GetOption("tts_secondary_playback_device"))):
            secondary_audio_device = settings.GetOption("tts_secondary_playback_device")
            if secondary_audio_device == -1:
                secondary_audio_device = settings.GetOption("device_default_out_index")

        allow_overlapping_audio = settings.GetOption("tts_allow_overlapping_audio")
        #audio = np.int16(audio * 32767)  # Convert to 16-bit PCM
        #audio = audio_tools.convert_audio_datatype_to_integer(audio)

        # play audio tensor
        audio_tools.play_audio(audio, device,
                               source_sample_rate=int(self.target_sample_rate),
                               audio_device_channel_num=1,
                               target_channels=1,
                               input_channels=source_channels,
                               dtype="float32",
                               tensor_sample_with=4,
                               tensor_channels=1,
                               secondary_device=secondary_audio_device,
                               stop_play=not allow_overlapping_audio,
                               tag="tts"
                               )

    def return_wav_file_binary(self, audio, sample_rate=sample_rate):
        # convert pytorch tensor to numpy array
        np_arr = audio.detach().cpu().numpy()

        # convert numpy array to wav file
        buff = io.BytesIO()
        write_wav(buff, sample_rate, np_arr)

        return buff.read()

    def return_pcm_audio(self, audio):
        # convert pytorch tensor to numpy array
        np_arr = audio.detach().cpu().numpy()

        # convert numpy array to raw PCM bytes
        pcm_bytes = np_arr.tobytes()

        return pcm_bytes
